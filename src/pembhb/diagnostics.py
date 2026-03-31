"""Non-intrusive training diagnostics for the DenoisingAutoencoder.

Usage
-----
Append ``AutoencoderDiagnosticsCallback()`` to the Trainer's callback list.
No modifications to ``autoencoder.py`` or ``model.py`` are required.

    >>> from pembhb.diagnostics import AutoencoderDiagnosticsCallback
    >>> cb = AutoencoderDiagnosticsCallback()
    >>> trainer = Trainer(callbacks=[..., cb])
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Any

import torch
import torch.nn as nn
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_autoencoder(pl_module: LightningModule):
    """Extract the DenoisingAutoencoder from a LightningModule.

    Works for both standalone ``DenoisingAutoencoder`` and
    ``JointAEInferenceNetwork`` (which stores it at ``.autoencoder``).
    """
    from pembhb.autoencoder import DenoisingAutoencoder

    if isinstance(pl_module, DenoisingAutoencoder):
        return pl_module
    if hasattr(pl_module, "autoencoder"):
        return pl_module.autoencoder
    raise TypeError(
        f"Cannot extract DenoisingAutoencoder from {type(pl_module).__name__}"
    )


def _enumerate_layers(autoencoder) -> dict[str, nn.Module]:
    """Build a dict mapping human-readable names to trainable layer modules.

    Handles the unnamed ``nn.Sequential`` containers in ``ConvEncoder.conv``
    and ``ConvDecoder.conv`` by assigning indexed names.
    """
    layers: dict[str, nn.Module] = {}

    # --- Encoder convolutions ---
    conv_idx = bn_idx = 0
    for module in autoencoder.encoder.conv:
        if isinstance(module, nn.Conv1d):
            layers[f"enc_conv{conv_idx}"] = module
            conv_idx += 1
        elif isinstance(module, nn.BatchNorm1d):
            layers[f"enc_bn{bn_idx}"] = module
            bn_idx += 1

    # Encoder FC
    if hasattr(autoencoder.encoder, "fc"):
        layers["enc_fc"] = autoencoder.encoder.fc

    # --- Decoder FC ---
    if hasattr(autoencoder.decoder, "fc"):
        layers["dec_fc"] = autoencoder.decoder.fc

    # --- Decoder convolutions ---
    deconv_idx = bn_idx = 0
    for module in autoencoder.decoder.conv:
        if isinstance(module, nn.ConvTranspose1d):
            layers[f"dec_deconv{deconv_idx}"] = module
            deconv_idx += 1
        elif isinstance(module, nn.BatchNorm1d):
            layers[f"dec_bn{bn_idx}"] = module
            bn_idx += 1

    return layers


def _compute_effective_rank(activations: torch.Tensor) -> float:
    """Exponential entropy of singular values — measures bottleneck utilisation.

    Parameters
    ----------
    activations : (B, D) tensor
        Batch of bottleneck vectors.

    Returns
    -------
    float
        Effective rank in [1, D].  If close to D the full capacity is used.
    """
    s = torch.linalg.svdvals(activations.float())
    s = s[s > 1e-12]
    p = s / s.sum()
    return torch.exp(-torch.sum(p * torch.log(p))).item()


# ---------------------------------------------------------------------------
# Main callback
# ---------------------------------------------------------------------------

class AutoencoderDiagnosticsCallback(Callback):
    """Lightweight per-layer diagnostics for ``DenoisingAutoencoder`` training.

    Tracks gradient norms, parameter update norms, weight statistics,
    BatchNorm drift, bottleneck effective rank, and the generalisation gap.
    All scalars are logged to the existing TensorBoard logger **and** saved
    as a structured ``.pt`` file at the end of training.

    Parameters
    ----------
    log_every_n_epochs : int
        How often to write scalars to TensorBoard (default: every epoch).
    spectral_every_n_epochs : int
        How often to compute the expensive bottleneck SVD (default: every
        10 epochs).
    save_path : str or None
        Explicit path for the diagnostics ``.pt`` file.  If *None* it is
        auto-derived from the trainer's logger directory.
    """

    def __init__(
        self,
        log_every_n_epochs: int = 1,
        spectral_every_n_epochs: int = 10,
        save_path: str | None = None,
    ):
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs
        self.spectral_every_n_epochs = spectral_every_n_epochs
        self.save_path = save_path

        # Populated in on_fit_start
        self._named_layers: dict[str, nn.Module] = {}
        self._prev_params: dict[str, torch.Tensor] = {}
        self._finetuned: bool = False

        # Per-step accumulators (reset each epoch)
        self._grad_accum: dict[str, list[float]] = defaultdict(list)

        # Per-epoch history
        self._history: dict[str, Any] = {
            "grad_norms_mean": defaultdict(list),
            "grad_norms_max": defaultdict(list),
            "update_norms": defaultdict(list),
            "relative_updates": defaultdict(list),
            "weight_norms": defaultdict(list),
            "grad_ratio": [],
            "enc_dec_grad_ratio": [],
            "bottleneck_eff_rank": [],
            "generalization_gap": [],
            "bn_stats": defaultdict(lambda: {"mean_norm": [], "var_mean": []}),
        }

    # ------------------------------------------------------------------
    # on_fit_start — build layer map, snapshot initial weights
    # ------------------------------------------------------------------

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        ae = _get_autoencoder(pl_module)
        self._named_layers = _enumerate_layers(ae)

        # Detect finetuning: if any parameter has non-zero values the model
        # was likely loaded from a previous round's checkpoint.
        total_norm = sum(
            p.data.norm().item()
            for layer in self._named_layers.values()
            for p in layer.parameters()
        )
        self._finetuned = total_norm > 1.0  # heuristic

        # Snapshot current weights for first epoch's update-norm computation
        self._snapshot_params()

    # ------------------------------------------------------------------
    # on_after_backward — accumulate gradient norms (runs every step)
    # ------------------------------------------------------------------

    def on_after_backward(self, trainer: Trainer, pl_module: LightningModule) -> None:
        for name, layer in self._named_layers.items():
            for pname, param in layer.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.data.norm(2).item()
                    self._grad_accum[f"{name}.{pname}"].append(grad_norm)

    # ------------------------------------------------------------------
    # on_train_epoch_end — reduce, log, snapshot
    # ------------------------------------------------------------------

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        epoch = trainer.current_epoch
        should_log = (epoch % self.log_every_n_epochs == 0)

        # --- 1. Gradient statistics ---
        all_layer_grad_means: dict[str, float] = {}
        for key, norms in self._grad_accum.items():
            if not norms:
                continue
            mean_gn = sum(norms) / len(norms)
            max_gn = max(norms)
            self._history["grad_norms_mean"][key].append(mean_gn)
            self._history["grad_norms_max"][key].append(max_gn)
            all_layer_grad_means[key] = mean_gn
            if should_log and trainer.logger:
                trainer.logger.log_metrics(
                    {f"diag/grad_mean/{key}": mean_gn, f"diag/grad_max/{key}": max_gn},
                    step=trainer.global_step,
                )

        # Gradient norm ratio (max / min across layers)
        if all_layer_grad_means:
            vals = list(all_layer_grad_means.values())
            min_gn = min(vals)
            max_gn_val = max(vals)
            grad_ratio = max_gn_val / (min_gn + 1e-30)
            self._history["grad_ratio"].append(grad_ratio)
            if should_log and trainer.logger:
                trainer.logger.log_metrics(
                    {"diag/grad_ratio": grad_ratio}, step=trainer.global_step
                )

        # Encoder vs decoder gradient ratio
        enc_total = sum(v for k, v in all_layer_grad_means.items() if k.startswith("enc_"))
        dec_total = sum(v for k, v in all_layer_grad_means.items() if k.startswith("dec_"))
        enc_dec_ratio = enc_total / (dec_total + 1e-30)
        self._history["enc_dec_grad_ratio"].append(enc_dec_ratio)
        if should_log and trainer.logger:
            trainer.logger.log_metrics(
                {"diag/enc_dec_grad_ratio": enc_dec_ratio}, step=trainer.global_step
            )

        # Reset per-step accumulator
        self._grad_accum.clear()

        # --- 2. Parameter update norms & weight norms ---
        for name, layer in self._named_layers.items():
            for pname, param in layer.named_parameters():
                key = f"{name}.{pname}"
                current = param.data.detach()
                w_norm = current.norm(2).item()
                self._history["weight_norms"][key].append(w_norm)

                if key in self._prev_params:
                    update_norm = (current - self._prev_params[key]).norm(2).item()
                    rel_update = update_norm / (w_norm + 1e-30)
                    self._history["update_norms"][key].append(update_norm)
                    self._history["relative_updates"][key].append(rel_update)

                    if should_log and trainer.logger:
                        trainer.logger.log_metrics(
                            {
                                f"diag/update_norm/{key}": update_norm,
                                f"diag/relative_update/{key}": rel_update,
                                f"diag/weight_norm/{key}": w_norm,
                            },
                            step=trainer.global_step,
                        )

        # --- 3. BatchNorm running stats ---
        for name, layer in self._named_layers.items():
            if isinstance(layer, nn.BatchNorm1d):
                if layer.running_mean is not None:
                    mn = layer.running_mean.norm(2).item()
                    vn = layer.running_var.mean().item()
                    self._history["bn_stats"][name]["mean_norm"].append(mn)
                    self._history["bn_stats"][name]["var_mean"].append(vn)
                    if should_log and trainer.logger:
                        trainer.logger.log_metrics(
                            {
                                f"diag/bn_mean_norm/{name}": mn,
                                f"diag/bn_var_mean/{name}": vn,
                            },
                            step=trainer.global_step,
                        )

        # Snapshot weights for next epoch
        self._snapshot_params()

    # ------------------------------------------------------------------
    # on_validation_epoch_end — expensive metrics (bottleneck SVD, gen gap)
    # ------------------------------------------------------------------

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        epoch = trainer.current_epoch

        # --- Generalization gap ---
        train_loss = trainer.callback_metrics.get("train_loss_epoch")
        val_loss = trainer.callback_metrics.get("val_loss")
        if train_loss is not None and val_loss is not None:
            tl = train_loss.item() if hasattr(train_loss, "item") else float(train_loss)
            vl = val_loss.item() if hasattr(val_loss, "item") else float(val_loss)
            gap = (vl - tl) / (tl + 1e-30)
            self._history["generalization_gap"].append(gap)
            if trainer.logger:
                trainer.logger.log_metrics(
                    {"diag/generalization_gap": gap}, step=trainer.global_step
                )

        # --- Bottleneck effective rank (expensive, run sparingly) ---
        if epoch % self.spectral_every_n_epochs != 0:
            return

        ae = _get_autoencoder(pl_module)
        if not hasattr(ae, "encode"):
            return

        # Grab one validation batch
        val_dl = trainer.val_dataloaders
        if val_dl is None:
            return

        try:
            batch = next(iter(val_dl))
        except StopIteration:
            return

        device = next(ae.parameters()).device
        with torch.no_grad():
            noisy = batch["wave_fd"].to(device) + batch["noise_fd"].to(device)
            x_norm = ae.preprocess(noisy)
            bottleneck = ae.encode(x_norm)
            # For unet, encode returns (bottleneck, skips)
            if isinstance(bottleneck, tuple):
                bottleneck = bottleneck[0]
            # Flatten if needed
            if bottleneck.ndim > 2:
                bottleneck = bottleneck.reshape(bottleneck.shape[0], -1)

            eff_rank = _compute_effective_rank(bottleneck)
            self._history["bottleneck_eff_rank"].append(eff_rank)
            if trainer.logger:
                trainer.logger.log_metrics(
                    {"diag/bottleneck_eff_rank": eff_rank}, step=trainer.global_step
                )

    # ------------------------------------------------------------------
    # on_fit_end — save structured diagnostics to disk
    # ------------------------------------------------------------------

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        ae = _get_autoencoder(pl_module)

        save_path = self.save_path
        if save_path is None and trainer.logger and hasattr(trainer.logger, "log_dir"):
            save_path = os.path.join(trainer.logger.log_dir, "ae_diagnostics.pt")

        if save_path is None:
            return

        # Build metadata
        metadata = {
            "architecture": getattr(ae, "architecture", "unknown"),
            "bottleneck_dim": getattr(ae, "bottleneck_dim", None),
            "hidden_channels": list(ae.hparams.get("hidden_channels", []))
            if hasattr(ae, "hparams")
            else [],
            "total_epochs": trainer.current_epoch + 1,
            "layer_names": list(self._named_layers.keys()),
            "finetuned": self._finetuned,
        }

        # Convert defaultdicts to plain dicts for clean serialisation
        output = {"metadata": metadata}
        for key, val in self._history.items():
            if isinstance(val, defaultdict):
                output[key] = dict(val)
            else:
                output[key] = val

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(output, save_path)
        print(f"[AutoencoderDiagnostics] Saved diagnostics to {save_path}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _snapshot_params(self) -> None:
        """Clone current parameter values for next epoch's update-norm computation."""
        self._prev_params.clear()
        for name, layer in self._named_layers.items():
            for pname, param in layer.named_parameters():
                self._prev_params[f"{name}.{pname}"] = param.data.detach().clone()
