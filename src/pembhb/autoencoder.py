"""
Unet-based denoising autoencoder for frequency-domain MBHB data.

The encoder compresses noisy input (wave_fd + noise_fd) to a low-dimensional
bottleneck, and the decoder reconstructs the clean signal (wave_fd).
The bottleneck representation can be extracted separately for use as a
data summarizer in InferenceNetwork.
"""

import torch
from torch import nn
from torch.nn import functional as F
from lightning import LightningModule
from torch.utils.data import DataLoader

from pembhb.model import DoubleConv, Down, Up, OutConv


# ---------------------------------------------------------------------------
# Encoder / Decoder split of the Unet
# ---------------------------------------------------------------------------

class UnetEncoder(nn.Module):
    """Encoder (contracting) path of a 1-D Unet.

    Produces skip-connection feature maps *and* the bottleneck tensor.
    """

    def __init__(
        self,
        n_in_channels: int,
        sizes: tuple = (16, 32, 64, 128, 256),
        down_sampling: tuple = (2, 2, 2, 2),
    ):
        super().__init__()
        self.inc = DoubleConv(n_in_channels, sizes[0])
        self.down1 = Down(sizes[0], sizes[1], down_sampling[0])
        self.down2 = Down(sizes[1], sizes[2], down_sampling[1])
        self.down3 = Down(sizes[2], sizes[3], down_sampling[2])
        self.down4 = Down(sizes[3], sizes[4], down_sampling[3])

    def forward(self, x):
        """Return (bottleneck, [skip1, skip2, skip3, skip4])."""
        x1 = self.inc(x)       # (B, sizes[0], L)
        x2 = self.down1(x1)    # (B, sizes[1], L/d0)
        x3 = self.down2(x2)    # (B, sizes[2], L/d0/d1)
        x4 = self.down3(x3)    # (B, sizes[3], L/d0/d1/d2)
        x5 = self.down4(x4)    # (B, sizes[4], L/d0/d1/d2/d3)  ← bottleneck
        return x5, [x1, x2, x3, x4]


class UnetDecoder(nn.Module):
    """Decoder (expanding) path of a 1-D Unet."""

    def __init__(
        self,
        n_out_channels: int,
        sizes: tuple = (16, 32, 64, 128, 256),
    ):
        super().__init__()
        self.up1 = Up(sizes[4], sizes[3])
        self.up2 = Up(sizes[3], sizes[2])
        self.up3 = Up(sizes[2], sizes[1])
        self.up4 = Up(sizes[1], sizes[0])
        self.outc = OutConv(sizes[0], n_out_channels)

    def forward(self, bottleneck, skips):
        """Reconstruct from bottleneck + skip connections.

        :param bottleneck: tensor of shape (B, sizes[-1], L_bottleneck)
        :param skips: list [x1, x2, x3, x4] from encoder
        :return: reconstructed tensor of shape (B, n_out_channels, L_input)
        """
        x1, x2, x3, x4 = skips
        x = self.up1(bottleneck, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


# ---------------------------------------------------------------------------
# Denoising Autoencoder (LightningModule)
# ---------------------------------------------------------------------------

class DenoisingAutoencoder(LightningModule):
    """Unet-based denoising autoencoder for frequency-domain gravitational-wave data.

    Training
    --------
    * **Input** : ``batch["wave_fd"] + batch["noise_fd"]``  (noisy complex signal)
    * **Target**: ``batch["wave_fd"]``                       (clean complex signal)
    * Both are normalised before being fed to the network:
        1. Convert complex → real by stacking two channels per TDI channel
           along the channel axis (``C`` → ``2*C``).
           The representation is controlled by ``representation``:
           - ``"amp_phase"``: ``[log|amplitude|, phase]``
           - ``"real_imag"``: ``[real, imag]``
        2. Center by subtracting a running mean.
        3. Scale by dividing by a running global max.
    * The loss is MSE between the Unet output and the normalised-scaled
      clean signal.

    Inference / data-summary mode
    -----------------------------
    After training, wrap this module in :class:`AutoencoderWrapper` and pass
    it to ``InferenceNetwork`` as ``data_summarizer``.  Calling
    ``AutoencoderWrapper.forward(d_f, d_t)`` returns the **flattened
    bottleneck** (not the reconstructed signal).
    """

    VALID_REPRESENTATIONS = ("amp_phase", "real_imag")

    def __init__(
        self,
        n_channels: int = 3,
        n_freqs: int = 1024,
        sizes: tuple = (16, 32, 64, 128, 256),
        down_sampling: tuple = (2, 2, 2, 2),
        lr: float = 1e-3,
        scheduler_patience: int = 10,
        scheduler_factor: float = 0.5,
        representation: str = "amp_phase",
    ):
        super().__init__()
        if representation not in self.VALID_REPRESENTATIONS:
            raise ValueError(
                f"representation must be one of {self.VALID_REPRESENTATIONS}, "
                f"got '{representation}'"
            )
        self.save_hyperparameters()

        self.n_channels = n_channels
        self.n_freqs = n_freqs
        self.lr = lr
        self.scheduler_patience = scheduler_patience
        self.scheduler_factor = scheduler_factor
        self.sizes = sizes
        self.representation = representation

        # Complex → real representation doubles the channels
        n_real_channels = n_channels * 2

        self.encoder = UnetEncoder(
            n_in_channels=n_real_channels,
            sizes=sizes,
            down_sampling=down_sampling,
        )
        self.decoder = UnetDecoder(
            n_out_channels=n_real_channels,
            sizes=sizes,
        )

        # ---- normalisation buffers (computed from training data) -----------
        # These are registered as buffers so they are saved/loaded with the
        # checkpoint and moved to the correct device automatically.
        self.register_buffer("mean_vec", torch.zeros(n_real_channels, n_freqs))
        self.register_buffer("global_scale_factor", torch.tensor(1.0))
        self._normalisation_fitted = False

    # ------------------------------------------------------------------
    # Complex → real conversion
    # ------------------------------------------------------------------

    def _complex_to_real(self, z: torch.Tensor) -> torch.Tensor:
        """Convert a complex tensor (B, C, F) → real tensor (B, 2C, F).

        The channel layout depends on ``self.representation``:

        * ``"amp_phase"``:  [log|amp_ch0|, …, phase_ch0, …]
        * ``"real_imag"``:  [re_ch0, …, im_ch0, …]
        """
        if self.representation == "amp_phase":
            amplitude = torch.abs(z)
            phase = torch.angle(z)
            log_amplitude = torch.log(amplitude + 1e-33)
            return torch.cat([log_amplitude, phase], dim=1)
        else:  # real_imag
            return torch.cat([z.real, z.imag], dim=1)

    # ------------------------------------------------------------------
    # Normalisation helpers  (mirror ReducedOrderModel._normalize / _denormalize)
    # ------------------------------------------------------------------

    def fit_normalisation(self, dataloader: DataLoader):
        """Compute mean and global scale from the **clean** training signals.

        Call this **once** before training.  The statistics are stored as
        buffers and will be saved together with the model checkpoint.

        :param dataloader: training DataLoader whose batches contain ``wave_fd``.
        """
        device = next(self.parameters()).device
        running_sum = torch.zeros(self.n_channels * 2, self.n_freqs, device=device)
        n_samples = 0
        max_val = 0.0

        with torch.no_grad():
            for batch in dataloader:
                wave_fd = batch["wave_fd"].to(device)
                real = self._complex_to_real(wave_fd)  # (B, 2C, F)
                running_sum += real.sum(dim=0)
                n_samples += real.shape[0]
                max_val = max(max_val, real.abs().max().item())

        self.mean_vec.copy_(running_sum / n_samples)
        self.global_scale_factor.fill_(max_val)
        self._normalisation_fitted = True
        print(
            f"[AutoEncoder] normalisation fitted on {n_samples} samples  "
            f"(representation={self.representation}, "
            f"global_scale={self.global_scale_factor.item():.4e})"
        )

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Center and scale: ``(x - mean) / scale``."""
        return (x - self.mean_vec) / self.global_scale_factor

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Undo normalisation."""
        return x * self.global_scale_factor + self.mean_vec

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def encode(self, x_norm: torch.Tensor):
        """Run the encoder on **already-normalised** real input.

        :param x_norm: (B, 2C, F) normalised real tensor.
        :return: (bottleneck, skips)
        """
        return self.encoder(x_norm)

    def decode(self, bottleneck: torch.Tensor, skips: list[torch.Tensor]) -> torch.Tensor:
        """Run the decoder.

        :param bottleneck: (B, sizes[-1], L_bottleneck)
        :param skips: list of skip-connection tensors from the encoder
        :return: (B, 2C, F) reconstructed normalised signal
        """
        return self.decoder(bottleneck, skips)

    def forward(self, x_norm: torch.Tensor) -> torch.Tensor:
        """Full autoencoder pass (encode → decode).

        :param x_norm: (B, 2C, F) normalised real tensor.
        :return: (B, 2C, F) reconstructed normalised signal.
        """
        bottleneck, skips = self.encode(x_norm)
        return self.decode(bottleneck, skips)

    # ------------------------------------------------------------------
    # Preprocessing helper (raw complex batch → normalised real tensor)
    # ------------------------------------------------------------------

    def preprocess(self, z: torch.Tensor) -> torch.Tensor:
        """Convert complex FD data to normalised real representation.

        :param z: complex tensor (B, C, F)
        :return: normalised real tensor (B, 2C, F)
        """
        return self._normalize(self._complex_to_real(z))

    # ------------------------------------------------------------------
    # Lightning training / validation steps
    # ------------------------------------------------------------------

    def _compute_extra_metrics(self, reconstructed: torch.Tensor, target: torch.Tensor):
        """Compute absolute error (MAE) and relative error.

        :param reconstructed: model output (B, 2C, F)
        :param target: normalised clean signal (B, 2C, F)
        :return: (mae, relative_error) scalar tensors
        """
        abs_error = (reconstructed - target).abs().mean()
        target_norm = target.norm(dim=(1, 2)).mean()  # mean L2 norm across batch
        relative_error = abs_error / (target_norm + 1e-30)
        return abs_error, relative_error

    def training_step(self, batch, batch_idx):
        noisy = batch["wave_fd"] + batch["noise_fd"]
        clean = batch["wave_fd"]

        # normalise
        noisy_norm = self.preprocess(noisy)
        clean_norm = self.preprocess(clean)

        # forward (full autoencoder)
        reconstructed = self(noisy_norm)

        loss = F.mse_loss(reconstructed, clean_norm)
        mae, rel_err = self._compute_extra_metrics(reconstructed, clean_norm)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_mae", mae, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_rel_err", rel_err, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        noisy = batch["wave_fd"] + batch["noise_fd"]
        clean = batch["wave_fd"]

        noisy_norm = self.preprocess(noisy)
        clean_norm = self.preprocess(clean)

        reconstructed = self(noisy_norm)

        loss = F.mse_loss(reconstructed, clean_norm)
        mae, rel_err = self._compute_extra_metrics(reconstructed, clean_norm)

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_mae", mae, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_rel_err", rel_err, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.scheduler_factor,
            patience=self.scheduler_patience,
            min_lr=1e-7,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }


# ---------------------------------------------------------------------------
# Wrapper for InferenceNetwork.data_summary
# ---------------------------------------------------------------------------

class AutoencoderWrapper(nn.Module):
    """Wraps a trained :class:`DenoisingAutoencoder` so it can be used as
    ``InferenceNetwork.data_summary``.

    Calling ``forward(d_f, d_t)`` returns the **flattened bottleneck**
    representation (not the reconstructed signal).

    The encoder weights are frozen by default (``freeze=True``).
    """

    def __init__(self, autoencoder: DenoisingAutoencoder, freeze: bool = True, device: str = "cuda"):
        super().__init__()
        self.autoencoder = autoencoder
        self.device = device

        if freeze:
            for p in self.autoencoder.parameters():
                p.requires_grad = False

        # Pre-compute the number of bottleneck features so
        # InferenceNetwork can query it via get_n_features().
        self._n_features = self._compute_n_features()

    def _compute_n_features(self) -> int:
        """Run a dummy forward pass through the encoder to find the
        flattened bottleneck size."""
        n_real_ch = self.autoencoder.n_channels * 2
        dummy = torch.zeros(1, n_real_ch, self.autoencoder.n_freqs)
        with torch.no_grad():
            dummy = dummy.to(self.device)
            bottleneck, _ = self.autoencoder.encoder(dummy)
        return bottleneck.numel()  # sizes[-1] * L_bottleneck

    def get_n_features(self) -> int:
        """Return the dimensionality of the bottleneck (flattened)."""
        return self._n_features

    def forward(self, d_f: torch.Tensor, d_t: torch.Tensor):
        """Encode noisy FD data and return the flattened bottleneck.

        :param d_f: complex tensor (B, C, F)  — raw frequency-domain data
                    (typically ``wave_fd + noise_fd``)
        :param d_t: ignored (kept for API compatibility with other data
                    summarizers like ROMWrapper)
        :return: (bottleneck_flat, d_t) where ``bottleneck_flat`` has shape
                 ``(B, n_features)``
        """
        x_norm = self.autoencoder.preprocess(d_f)
        bottleneck, _skips = self.autoencoder.encode(x_norm)
        bottleneck_flat = bottleneck.reshape(bottleneck.shape[0], -1)  # (B, n_features)
        return bottleneck_flat, d_t
