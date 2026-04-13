import os
import numpy as np
import torch
from pembhb import ROOT_DIR, get_numpy_dtype
from pembhb.utils import (
    _ORDERED_PRIOR_KEYS,
    get_widest_interval_1d,
    get_widest_box_2d,
    get_logratios_grid,
    get_logratios_grid_2d,
)
# from pembhb.data import MBHBDataset, mbhb_collate_fn
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import  Callback

from datetime import datetime, timedelta

import matplotlib.pyplot as plt


class PeriodicProgressCallback(Callback):
    """Print a one-line training status every *print_every* epochs.

    Produces logfile-friendly output (no ANSI codes, no carriage returns)
    suitable for piping through ``tee``.  Complements TensorBoard logging:
    only the most essential scalars are printed here.
    """

    def __init__(self, print_every: int = 20, label: str = ""):
        super().__init__()
        self.print_every = print_every
        self.label = label

    def _print_status(self, trainer, pl_module, suffix: str = ""):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        epoch = trainer.current_epoch
        m = trainer.callback_metrics

        parts = [f"[{ts}]"]
        if self.label:
            parts.append(self.label)
        parts.append(f"epoch {epoch}/{trainer.max_epochs}")

        for key in ("train_loss", "val_loss", "val_accuracy", "train_accuracy"):
            if key in m:
                v = float(m[key])
                fmt = f"{v:.4e}" if "loss" in key else f"{v:.4f}"
                parts.append(f"{key}={fmt}")

        try:
            opt = pl_module.optimizers()
            if isinstance(opt, list):
                opt = opt[0]
            if len(opt.param_groups) >= 2:
                lr_ae = opt.param_groups[0]["lr"]
                lr_nre = opt.param_groups[1]["lr"]
                parts.append(f"lr_ae={lr_ae:.2e}")
                parts.append(f"lr_nre={lr_nre:.2e}")
            else:
                lr = opt.param_groups[0]["lr"]
                parts.append(f"lr={lr:.2e}")
        except Exception:
            pass

        if suffix:
            parts.append(suffix)

        print(" | ".join(parts), flush=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        # Log learning rates to TensorBoard every epoch
        try:
            opt = pl_module.optimizers()
            if isinstance(opt, list):
                opt = opt[0]
            if trainer.logger is not None:
                if len(opt.param_groups) >= 2:
                    trainer.logger.log_metrics({
                        "lr/autoencoder": opt.param_groups[0]["lr"],
                        "lr/nre": opt.param_groups[1]["lr"],
                    }, step=trainer.current_epoch)
                else:
                    trainer.logger.log_metrics({
                        "lr": opt.param_groups[0]["lr"],
                    }, step=trainer.current_epoch)
        except Exception:
            pass

        if trainer.current_epoch % self.print_every == 0:
            self._print_status(trainer, pl_module)

    def on_train_end(self, trainer, pl_module):
        self._print_status(trainer, pl_module, suffix="[done]")


class PlotPosteriorCallback(Callback):
    def __init__(self, timestamp: str, obs_loader: DataLoader, input_idx_list: list, output_idx_list: list, round_idx: int , call_every_n_epochs=1, training_start_time: datetime = None, print_every: int = 20):
        self.epochs_elapsed = 0
        self.call_every_n_epochs = call_every_n_epochs
        self.print_every = print_every
        self.timestamp = timestamp
        self.obs_loader = obs_loader
        self.input_idx_list = input_idx_list
        self.output_idx_list = output_idx_list
        self.n_marginals = len(input_idx_list)
        self.init_time = datetime.now()
        self.training_start_time = training_start_time if training_start_time is not None else self.init_time
        self.round_idx = round_idx
        # Storage for volume ratio diagnostics
        self.volume_ratios = {}
        # Storage for differential entropy diagnostics
        self.differential_entropies = {}
    
    def _compute_posterior_volume_2d(self, widest_box):
        """
        Compute the area/volume of the posterior from the widest contour box.
        
        Parameters:
        -----------
        widest_box : tuple
            The bounding box of the 99.99% contour.
            Currently: (x_min, x_max, y_min, y_max) for axis-aligned boxes.
            
        Returns:
        --------
        float
            Area enclosed by the posterior contour.
            
        Notes:
        ------
        FUTURE EXTENSION FOR TILTED BOXES:
        - If posterior contours become non-axis-aligned, widest_box format may change
          to a list of vertices [(x1,y1), (x2,y2), ...]
        - In that case, use Shoelace formula or similar for polygon area:
          area = 0.5 * abs(sum(x[i]*y[i+1] - x[i+1]*y[i] for i in range(n)))
        - Consider using shapely.geometry.Polygon for robust area calculation
        """
        # Current implementation: axis-aligned box
        # widest_box = (x_min, x_max, y_min, y_max)
        posterior_area = (widest_box[1] - widest_box[0]) * (widest_box[3] - widest_box[2])
        return posterior_area
    
    def _compute_prior_volume_2d(self, pl_module, in_param_idx):
        """
        Compute the area/volume of the prior for a 2D marginal.
        
        Parameters:
        -----------
        pl_module : LightningModule
            The model containing prior information in hparams.
        in_param_idx : tuple
            Indices of the two parameters defining the 2D marginal.
            
        Returns:
        --------
        float
            Area of the prior region.
            
        Notes:
        ------
        **MODIFY THIS METHOD WHEN SWITCHING TO TILTED BOUNDING BOXES**
        
        Current implementation assumes axis-aligned rectangular priors.
        Prior bounds are stored as:
            prior_dict[param_name] = [min_value, max_value]
        
        For tilted/rotated bounding boxes:
        1. Prior specification will change (e.g., vertices, rotation matrix, etc.)
        2. Access prior from: pl_module.hparams["dataset_info"]["conf"]["prior"]
        3. Compute area based on new representation:
           - If vertices: use Shoelace formula or shapely.geometry.Polygon
           - If rotation + bounds: compute area of rotated rectangle
           - Example with vertices:
             ```python
             vertices = prior_dict[marginal_key]  # [(x1,y1), (x2,y2), ...]
             from shapely.geometry import Polygon
             prior_area = Polygon(vertices).area
             ```
        4. Ensure consistency with sampler_init_kwargs format in sampler.py
        
        Potential issues to address:
        - Normalization: If grid evaluation doesn't align with tilted prior,
          posterior normalization may be affected
        - Grid coverage: Axis-aligned grids may inefficiently cover tilted regions
        - Coordinate transforms: May need to transform between rotated and
          canonical coordinate systems
        """
        # Current implementation: axis-aligned rectangular prior
        # Use the actual sampling prior (sampler_init_kwargs) as the
        # authoritative source.  Fall back to conf["prior"] for backward compat.
        _sik = pl_module.hparams["dataset_info"].get("sampler_init_kwargs", {})
        if "prior_bounds" in _sik:
            prior_dict = _sik["prior_bounds"]
        else:
            prior_dict = pl_module.hparams["dataset_info"]["conf"]["prior"]
        
        # Get bounds for each parameter
        param_name_0 = _ORDERED_PRIOR_KEYS[in_param_idx[0]]
        param_name_1 = _ORDERED_PRIOR_KEYS[in_param_idx[1]]
        
        prior_bounds_0 = prior_dict[param_name_0]
        prior_bounds_1 = prior_dict[param_name_1]
        
        # Compute area as product of widths
        prior_area = (prior_bounds_0[1] - prior_bounds_0[0]) * (prior_bounds_1[1] - prior_bounds_1[0])
        
        return prior_area

    def _compute_posterior_volume_1d(self, widest_interval):
        """Compute the width of the posterior credible interval for a 1D marginal.
        
        Parameters:
        -----------
        widest_interval : list
            [low, high] bounds of the credible interval.
            
        Returns:
        --------
        float
            Width of the posterior interval.
        """
        return widest_interval[1] - widest_interval[0]
    
    def _compute_prior_volume_1d(self, pl_module, in_param_idx):
        """Compute the width of the prior for a 1D marginal.
        
        Parameters:
        -----------
        pl_module : LightningModule
            The model containing prior information in hparams.
        in_param_idx : int
            Index of the parameter.
            
        Returns:
        --------
        float
            Width of the prior range.
        """
        # Use the actual sampling prior (sampler_init_kwargs) as the
        # authoritative source.  Fall back to conf["prior"] for backward compat.
        _sik = pl_module.hparams["dataset_info"].get("sampler_init_kwargs", {})
        if "prior_bounds" in _sik:
            prior_dict = _sik["prior_bounds"]
        else:
            prior_dict = pl_module.hparams["dataset_info"]["conf"]["prior"]
        param_name = _ORDERED_PRIOR_KEYS[in_param_idx]
        prior_bounds = prior_dict[param_name]
        return prior_bounds[1] - prior_bounds[0]

    def _log_sky_contour_ratios(self, norm2d, gx, gy, dp0, dp1, trainer, param_tag):
        """Log width/height ratios between 95.5% and wider HPD contour boxes.

        Compares the bounding rectangle of the 95.5% HPD region to those
        at 1-1e-3 and 1-1e-4 levels.  Logs the width and height ratios
        so the user can gauge how sensitive the box size is to the
        credible level chosen for sky truncation.
        """
        from pembhb.sky_truncation import _hpd_threshold

        ref_level = 0.9545
        comparison_levels = [1.0 - 1e-3, 1.0 - 1e-4]

        def _hpd_bbox(density_2d, level):
            """Axis-aligned bounding box of the HPD region at *level*."""
            thresh = _hpd_threshold(density_2d, level)
            mask = density_2d >= thresh
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            if not rows.any() or not cols.any():
                return None
            row_idx = np.where(rows)[0]
            col_idx = np.where(cols)[0]
            width = float(gx[0, col_idx[-1]] - gx[0, col_idx[0]])
            height = float(gy[row_idx[-1], 0] - gy[row_idx[0], 0])
            return width, height

        ref_box = _hpd_bbox(norm2d, ref_level)
        if ref_box is None:
            return
        ref_w, ref_h = ref_box

        for eps_level in comparison_levels:
            comp_box = _hpd_bbox(norm2d, eps_level)
            if comp_box is None or ref_w == 0 or ref_h == 0:
                continue
            comp_w, comp_h = comp_box
            eps_tag = f"{1.0 - eps_level:.0e}"  # e.g. "1e-03"
            w_ratio = comp_w / ref_w
            h_ratio = comp_h / ref_h
            if trainer.logger is not None:
                trainer.logger.log_metrics({
                    f"sky_box_ratio/width_{eps_tag}_vs_955": w_ratio,
                    f"sky_box_ratio/height_{eps_tag}_vs_955": h_ratio,
                }, step=trainer.current_epoch)
            if trainer.current_epoch % self.print_every == 0:
                print(f"  [sky_box] 1-eps={eps_level:.4f} vs 95.5%: "
                      f"width_ratio={w_ratio:.3f}, height_ratio={h_ratio:.3f}",
                      flush=True)

    @staticmethod
    def _differential_entropy_1d(norm1d, dp):
        """Differential entropy H = -int p log p dx (nats) via Riemann sum."""
        with np.errstate(divide="ignore"):
            log_p = np.where(norm1d > 0, np.log(norm1d), 0.0)
        return -float(np.sum(norm1d * log_p * dp))

    @staticmethod
    def _differential_entropy_2d(norm2d, dp0, dp1):
        """Joint differential entropy H = -int int p log p dx dy (nats)."""
        with np.errstate(divide="ignore"):
            log_p = np.where(norm2d > 0, np.log(norm2d), 0.0)
        return -float(np.sum(norm2d * log_p * dp0 * dp1))

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.epochs_elapsed == 0:
            os.makedirs(os.path.join(ROOT_DIR, "plots", self.timestamp), exist_ok=True)

        self.epochs_elapsed += 1
        if (self.epochs_elapsed-2) % self.call_every_n_epochs == 0:
            #print("plotting posteriors on observed data")
            train_time = datetime.now() - self.training_start_time
            td_trunc = train_time - timedelta(microseconds=train_time.microseconds)
            title_plot = f"training time={td_trunc}s"
            # plot the posterior on the observed data , using the current model
            for i in range(self.n_marginals):
                in_param_idx = self.input_idx_list[i]
                out_param_idx = self.output_idx_list[i]

                # Initialize widest_boxes dict if not present
                if not hasattr(pl_module, 'widest_boxes'):
                    pl_module.widest_boxes = {}
                marginal_key = tuple(in_param_idx)

                if len(in_param_idx) == 1:
                    # Handle 1D marginals
                    param_idx = in_param_idx[0]
                    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
                    fig.suptitle(
                        f"Round {self.round_idx} - Epoch {trainer.current_epoch} - {title_plot}",
                        fontsize=10,
                    )
                    
                    try:
                        epsilon_value = 1e-4
                        widest_interval, norm1d, grid, inj_params = get_widest_interval_1d(
                            pl_module,
                            self.obs_loader,
                            in_param_idx=param_idx,
                            out_param_idx=out_param_idx,
                            eps=epsilon_value
                        )
                        
                        # Plot
                        ax.plot(grid.flatten(), norm1d, 'b-', linewidth=1.5)
                        ax.axvline(inj_params[0], color='r', linestyle='--', label='Injection')
                        ax.axvline(widest_interval[0], color='g', linestyle=':', label=f'{100*(1-epsilon_value):.2f}% CI')
                        ax.axvline(widest_interval[1], color='g', linestyle=':')
                        ax.fill_between(grid.flatten(), 0, norm1d, 
                                       where=(grid.flatten() >= widest_interval[0]) & (grid.flatten() <= widest_interval[1]),
                                       alpha=0.3, color='green')
                        ax.set_xlabel(_ORDERED_PRIOR_KEYS[param_idx])
                        ax.set_ylabel('Posterior density')
                        ax.legend()
                        
                        # Store the widest interval
                        pl_module.widest_boxes[marginal_key] = widest_interval
                        
                        # Compute posterior-to-prior volume (width) ratio for 1D marginal
                        posterior_width = self._compute_posterior_volume_1d(widest_interval)
                        prior_width = self._compute_prior_volume_1d(pl_module, param_idx)
                        volume_ratio = posterior_width / prior_width
                        
                        # Store and log the volume ratio
                        if marginal_key not in self.volume_ratios:
                            self.volume_ratios[marginal_key] = []
                        self.volume_ratios[marginal_key].append({
                            'epoch': trainer.current_epoch,
                            'ratio': volume_ratio,
                            'posterior_width': posterior_width,
                            'prior_width': prior_width
                        })
                        
                        # Compute differential entropy for 1D marginal
                        dp = float(grid[1, 0] - grid[0, 0])
                        entropy = self._differential_entropy_1d(norm1d, dp)
                        if marginal_key not in self.differential_entropies:
                            self.differential_entropies[marginal_key] = []
                        self.differential_entropies[marginal_key].append({
                            'epoch': trainer.current_epoch,
                            'entropy': entropy,
                        })

                        # Log metrics to tensorboard if logger exists
                        if trainer.logger is not None:
                            metric_name = f"volume_ratio/{_ORDERED_PRIOR_KEYS[param_idx]}"
                            trainer.logger.log_metrics({metric_name: volume_ratio}, step=trainer.current_epoch)
                            entropy_metric = f"diff_entropy/{_ORDERED_PRIOR_KEYS[param_idx]}"
                            trainer.logger.log_metrics({entropy_metric: entropy}, step=trainer.current_epoch)

                        # Print diagnostic
                        param_name = _ORDERED_PRIOR_KEYS[param_idx]
                        if trainer.current_epoch % self.print_every == 0:
                            print(f"Round {self.round_idx}, Epoch {trainer.current_epoch}, {param_name}: "
                                  f"vol_ratio={volume_ratio:.4f} "
                                  f"(post={posterior_width:.3e}, prior={prior_width:.3e}), "
                                  f"H={entropy:.4f} nats", flush=True)
                        
                        out = os.path.join(ROOT_DIR, "plots", self.timestamp, 
                                          f"posterior_round_{self.round_idx}_epoch_{trainer.current_epoch}_{_ORDERED_PRIOR_KEYS[param_idx]}.pdf")
                        fig.savefig(out, bbox_inches="tight")
                    except Exception as e:
                        print(f"Error plotting 1D marginal for {_ORDERED_PRIOR_KEYS[param_idx]}: {e}")
                    finally:
                        plt.close(fig)

                elif len(in_param_idx) == 2:
                    # Handle 2D marginals
                    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
                    fig.tight_layout()
                    fig.suptitle(
                        f"Round {self.round_idx} - Epoch {trainer.current_epoch} - {title_plot}",
                        fontsize=10,
                    )

                    try:
                        widest_box, inj_params, norm2d, dp0, dp1, gx, gy = get_widest_box_2d(
                            pl_module,
                            self.obs_loader,
                            in_param_idx=in_param_idx,
                            out_param_idx=out_param_idx,
                            ax_buffer=ax,
                            do_plot=True,
                            return_norm2d=True,
                        )

                        # Store widest_box keyed by the marginal (tuple of input parameter indices)
                        pl_module.widest_boxes[marginal_key] = widest_box

                        # Compute posterior-to-prior volume ratio for 2D marginal
                        posterior_area = self._compute_posterior_volume_2d(widest_box)
                        prior_area = self._compute_prior_volume_2d(pl_module, in_param_idx)
                        volume_ratio = posterior_area / prior_area

                        # Store and log the volume ratio
                        if marginal_key not in self.volume_ratios:
                            self.volume_ratios[marginal_key] = []
                        self.volume_ratios[marginal_key].append({
                            'epoch': trainer.current_epoch,
                            'ratio': volume_ratio,
                            'posterior_area': posterior_area,
                            'prior_area': prior_area
                        })

                        # Compute differential entropy for 2D marginal
                        entropy = self._differential_entropy_2d(norm2d, dp0, dp1)
                        if marginal_key not in self.differential_entropies:
                            self.differential_entropies[marginal_key] = []
                        self.differential_entropies[marginal_key].append({
                            'epoch': trainer.current_epoch,
                            'entropy': entropy,
                        })

                        # Log metrics to tensorboard if logger exists
                        param_tag = f"{_ORDERED_PRIOR_KEYS[in_param_idx[0]]}_{_ORDERED_PRIOR_KEYS[in_param_idx[1]]}"
                        if trainer.logger is not None:
                            trainer.logger.log_metrics({f"volume_ratio/{param_tag}": volume_ratio}, step=trainer.current_epoch)
                            trainer.logger.log_metrics({f"diff_entropy/{param_tag}": entropy}, step=trainer.current_epoch)

                        # --- Sky dilation ratio diagnostics (7, 8) = (lambda, sin_beta) ---
                        if marginal_key == (7, 8):
                            self._log_sky_contour_ratios(
                                norm2d, gx, gy, dp0, dp1,
                                trainer, param_tag,
                            )

                        # Print diagnostic
                        param_names = f"{_ORDERED_PRIOR_KEYS[in_param_idx[0]]}-{_ORDERED_PRIOR_KEYS[in_param_idx[1]]}"
                        if trainer.current_epoch % self.print_every == 0:
                            print(f"Round {self.round_idx}, Epoch {trainer.current_epoch}, {param_names}: "
                                  f"vol_ratio={volume_ratio:.4f} "
                                  f"(post={posterior_area:.3e}, prior={prior_area:.3e}), "
                                  f"H={entropy:.4f} nats", flush=True)
                        
                        out = os.path.join(ROOT_DIR, "plots", self.timestamp,
                                          f"posterior_round_{self.round_idx}_epoch_{trainer.current_epoch}_{_ORDERED_PRIOR_KEYS[in_param_idx[0]]}_{_ORDERED_PRIOR_KEYS[in_param_idx[1]]}.pdf")
                        fig.savefig(out, bbox_inches="tight")
                    except ValueError as ve:
                        print(f"caught ValueError: {ve} during contour plotting, skipping this plot")
                    finally:
                        plt.close(fig)

    def on_train_end(self, trainer, pl_module):
        self.on_validation_epoch_end(trainer, pl_module)
        print(f"Total training time: {datetime.now() - self.init_time}")


class VolumeRatioEarlyStopping(Callback):
    """Per-marginal early stopping based on posterior/prior volume ratio.

    Tracks an independent EMA and stall counter for **each** marginal.
    Training stops as soon as **any** marginal satisfies either:

    * its volume ratio drops to ``min_ratio_threshold``, **or**
    * its EMA has stalled for ``patience`` consecutive evaluations.

    The ``stop_reason`` attribute (str) records which marginal triggered
    the stop and why.
    """

    def __init__(
        self,
        warmup_epochs: int = 50,
        patience: int = 10,
        rel_tol: float = 0.02,
        ema_alpha: float = 0.3,
        min_ratio_threshold: float = 0.5,
        print_every: int = 20,
    ):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.patience = patience
        self.rel_tol = rel_tol
        self.ema_alpha = ema_alpha
        self.min_ratio_threshold = min_ratio_threshold
        self.print_every = print_every

        # per-marginal state: keyed by marginal_key (tuple)
        self._ema: dict[tuple, float] = {}
        self._prev_ema: dict[tuple, float] = {}
        self._stall_count: dict[tuple, int] = {}
        self.ema_history: list[dict] = []
        self.stop_reason: str = ""

    def _find_plot_callback(self, trainer) -> "PlotPosteriorCallback | None":
        for cb in trainer.callbacks:
            if isinstance(cb, PlotPosteriorCallback):
                return cb
        return None

    @staticmethod
    def _marginal_label(key):
        names = [_ORDERED_PRIOR_KEYS[i] for i in key]
        return "-".join(names)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch < self.warmup_epochs:
            return

        plot_cb = self._find_plot_callback(trainer)
        if plot_cb is None or not plot_cb.volume_ratios:
            return

        # Collect the latest volume ratio for each marginal at this epoch
        current = {}  # marginal_key -> ratio
        for marginal_key, history in plot_cb.volume_ratios.items():
            if history and history[-1]["epoch"] == trainer.current_epoch:
                current[marginal_key] = history[-1]["ratio"]

        if not current:
            return

        triggered_key = None
        triggered_reason = ""

        for mkey, ratio in current.items():
            label = self._marginal_label(mkey)

            # Hard threshold
            if ratio <= self.min_ratio_threshold:
                triggered_key = mkey
                triggered_reason = (
                    f"volume_ratio_threshold: {label} ratio={ratio:.4f} "
                    f"<= {self.min_ratio_threshold}"
                )
                break

            # Update per-marginal EMA
            if mkey not in self._ema:
                self._ema[mkey] = ratio
                self._stall_count[mkey] = 0
            else:
                self._ema[mkey] = self.ema_alpha * ratio + (1 - self.ema_alpha) * self._ema[mkey]

            # Check plateau
            if mkey in self._prev_ema:
                rel_change = abs(self._ema[mkey] - self._prev_ema[mkey]) / (abs(self._prev_ema[mkey]))
                if rel_change < self.rel_tol:
                    self._stall_count[mkey] = self._stall_count.get(mkey, 0) + 1
                else:
                    self._stall_count[mkey] = 0

            self._prev_ema[mkey] = self._ema[mkey]

            if self._stall_count.get(mkey, 0) >= self.patience:
                triggered_key = mkey
                triggered_reason = (
                    f"volume_ratio_plateau: {label} ema={self._ema[mkey]:.4f}, "
                    f"stall={self._stall_count[mkey]}/{self.patience}"
                )

        # Log to TensorBoard
        if trainer.logger is not None:
            metrics = {}
            for mkey in current:
                label = self._marginal_label(mkey)
                metrics[f"volume_ratio_ema/{label}"] = self._ema.get(mkey, current[mkey])
            trainer.logger.log_metrics(metrics, step=trainer.current_epoch)

        # Diagnostics printing
        if trainer.current_epoch % self.print_every == 0:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            parts = []
            for mkey in sorted(current):
                label = self._marginal_label(mkey)
                ema_val = self._ema.get(mkey, current[mkey])
                stall = self._stall_count.get(mkey, 0)
                parts.append(f"{label}(r={current[mkey]:.4f},ema={ema_val:.4f},s={stall})")
            print(f"[{ts}] [VolumeRatioES] epoch {trainer.current_epoch}: {', '.join(parts)}", flush=True)

        if triggered_key is not None:
            self.stop_reason = triggered_reason
            print(f"[VolumeRatioES] Stopping at epoch {trainer.current_epoch}: {triggered_reason}")
            trainer.should_stop = True


class DifferentialEntropyEarlyStopping(Callback):
    """Per-marginal early stopping based on differential entropy convergence.

    Tracks an independent EMA and stall counter for **each** marginal.
    Training stops as soon as **any** marginal satisfies either:

    * its entropy drops to a hard threshold (the entropy of a uniform
      distribution over half the prior domain width), **or**
    * its EMA has stalled for ``patience`` consecutive evaluations.

    Threshold derivation
    --------------------
    1-D marginal with prior [A, B]:
        H_thresh = (1/2) log(B - A)
    2-D marginal with priors [A₁,B₁] × [A₂,B₂]:
        H_thresh = (1/4)( log(B₁-A₁) + log(B₂-A₂) )

    The ``stop_reason`` attribute (str) records which marginal triggered
    the stop and why.
    """

    def __init__(
        self,
        warmup_epochs: int = 50,
        patience: int = 10,
        rel_tol: float = 0.02,
        ema_alpha: float = 0.3,
        print_every: int = 20,
    ):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.patience = patience
        self.rel_tol = rel_tol
        self.ema_alpha = ema_alpha
        self.print_every = print_every

        # per-marginal state
        self._ema: dict[tuple, float] = {}
        self._prev_ema: dict[tuple, float] = {}
        self._stall_count: dict[tuple, int] = {}
        self._thresholds: dict[tuple, float] = {}  # computed once per marginal
        self.ema_history: list[dict] = []
        self.stop_reason: str = ""

    def _find_plot_callback(self, trainer) -> "PlotPosteriorCallback | None":
        for cb in trainer.callbacks:
            if isinstance(cb, PlotPosteriorCallback):
                return cb
        return None

    @staticmethod
    def _marginal_label(key):
        names = [_ORDERED_PRIOR_KEYS[i] for i in key]
        return "-".join(names)

    def _get_threshold(self, pl_module, marginal_key):
        """Compute the entropy threshold for *marginal_key* from prior bounds.

        1-D: H = log((B-A)/2)
        2-D: H = log((B1-A1)/2) + log((B2-A2)/2)
        """
        if marginal_key in self._thresholds:
            return self._thresholds[marginal_key]

        _sik = pl_module.hparams["dataset_info"].get("sampler_init_kwargs", {})
        if "prior_bounds" in _sik:
            prior_dict = _sik["prior_bounds"]
        else:
            prior_dict = pl_module.hparams["dataset_info"]["conf"]["prior"]

        d = len(marginal_key)
        h = 0.0
        for idx in marginal_key:
            pname = _ORDERED_PRIOR_KEYS[idx]
            lo, hi = prior_dict[pname]
            h += np.log(hi - lo)
        h /= (2 * d)
        self._thresholds[marginal_key] = h
        return h

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch < self.warmup_epochs:
            return

        plot_cb = self._find_plot_callback(trainer)
        if plot_cb is None or not plot_cb.differential_entropies:
            return

        current = {}
        for marginal_key, history in plot_cb.differential_entropies.items():
            if history and history[-1]["epoch"] == trainer.current_epoch:
                current[marginal_key] = history[-1]["entropy"]

        if not current:
            return

        triggered_key = None
        triggered_reason = ""

        for mkey, entropy in current.items():
            label = self._marginal_label(mkey)
            threshold = self._get_threshold(pl_module, mkey)

            # Hard threshold
            if entropy <= threshold:
                triggered_key = mkey
                triggered_reason = (
                    f"entropy_threshold: {label} H={entropy:.4f} "
                    f"<= thresh={threshold:.4f}"
                )
                break

            # Update per-marginal EMA
            if mkey not in self._ema:
                self._ema[mkey] = entropy
                self._stall_count[mkey] = 0
            else:
                self._ema[mkey] = self.ema_alpha * entropy + (1 - self.ema_alpha) * self._ema[mkey]

            # Check plateau
            if mkey in self._prev_ema:
                rel_change = abs(self._ema[mkey] - self._prev_ema[mkey]) / (abs(self._prev_ema[mkey]))
                if rel_change < self.rel_tol:
                    self._stall_count[mkey] = self._stall_count.get(mkey, 0) + 1
                else:
                    self._stall_count[mkey] = 0

            self._prev_ema[mkey] = self._ema[mkey]

            if self._stall_count.get(mkey, 0) >= self.patience:
                triggered_key = mkey
                triggered_reason = (
                    f"entropy_plateau: {label} ema={self._ema[mkey]:.4f}, "
                    f"stall={self._stall_count[mkey]}/{self.patience}"
                )

        # Log to TensorBoard
        if trainer.logger is not None:
            metrics = {}
            for mkey in current:
                label = self._marginal_label(mkey)
                metrics[f"diff_entropy_ema/{label}"] = self._ema.get(mkey, current[mkey])
            trainer.logger.log_metrics(metrics, step=trainer.current_epoch)

        # Diagnostics printing
        if trainer.current_epoch % self.print_every == 0:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            parts = []
            for mkey in sorted(current):
                label = self._marginal_label(mkey)
                ema_val = self._ema.get(mkey, current[mkey])
                stall = self._stall_count.get(mkey, 0)
                thresh = self._get_threshold(pl_module, mkey)
                parts.append(f"{label}(H={current[mkey]:.3f},ema={ema_val:.3f},s={stall},th={thresh:.3f})")
            print(f"[{ts}] [EntropyES] epoch {trainer.current_epoch}: {', '.join(parts)}", flush=True)

        if triggered_key is not None:
            self.stop_reason = triggered_reason
            print(f"[EntropyES] Stopping at epoch {trainer.current_epoch}: {triggered_reason}")
            trainer.should_stop = True