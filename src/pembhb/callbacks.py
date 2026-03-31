import os 
from pembhb import ROOT_DIR
from pembhb.utils import _ORDERED_PRIOR_KEYS, get_widest_interval_1d, get_widest_box_2d
# from pembhb.data import MBHBDataset, mbhb_collate_fn
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import  Callback

from datetime import datetime, timedelta

import matplotlib.pyplot as plt


class PlotPosteriorCallback(Callback):
    def __init__(self, timestamp: str, obs_loader: DataLoader, input_idx_list: list, output_idx_list: list, round_idx: int , call_every_n_epochs=1): 
        self.epochs_elapsed = 0
        self.call_every_n_epochs = call_every_n_epochs
        self.timestamp = timestamp
        self.obs_loader = obs_loader
        self.input_idx_list = input_idx_list
        self.output_idx_list = output_idx_list
        self.n_marginals = len(input_idx_list)
        self.init_time = datetime.now()
        self.round_idx = round_idx
        # Storage for volume ratio diagnostics
        self.volume_ratios = {}
    
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

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.epochs_elapsed == 0: 
            os.makedirs(os.path.join(ROOT_DIR, "plots", self.timestamp), exist_ok=True)

        self.epochs_elapsed += 1
        if (self.epochs_elapsed-2) % self.call_every_n_epochs == 0:
            #print("plotting posteriors on observed data")
            train_time = datetime.now() - self.init_time
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
                        
                        # Log metric to tensorboard if logger exists
                        if trainer.logger is not None:
                            metric_name = f"volume_ratio/{_ORDERED_PRIOR_KEYS[param_idx]}"
                            trainer.logger.log_metrics({metric_name: volume_ratio}, step=trainer.current_epoch)
                        
                        # Print diagnostic
                        param_name = _ORDERED_PRIOR_KEYS[param_idx]
                        print(f"Round {self.round_idx}, Epoch {trainer.current_epoch}, {param_name}: "
                              f"Volume ratio (posterior/prior) = {volume_ratio:.6f} "
                              f"(posterior width: {posterior_width:.6e}, prior width: {prior_width:.6e})")
                        
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
                        widest_box, inj_params = get_widest_box_2d(
                            pl_module,
                            self.obs_loader,
                            in_param_idx=in_param_idx,
                            out_param_idx=out_param_idx,
                            ax_buffer=ax,
                            do_plot=True
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
                        
                        # Log metric to tensorboard if logger exists
                        if trainer.logger is not None:
                            metric_name = f"volume_ratio/{_ORDERED_PRIOR_KEYS[in_param_idx[0]]}_{_ORDERED_PRIOR_KEYS[in_param_idx[1]]}"
                            trainer.logger.log_metrics({metric_name: volume_ratio}, step=trainer.current_epoch)
                        
                        # Print diagnostic
                        param_names = f"{_ORDERED_PRIOR_KEYS[in_param_idx[0]]}-{_ORDERED_PRIOR_KEYS[in_param_idx[1]]}"
                        print(f"Round {self.round_idx}, Epoch {trainer.current_epoch}, {param_names}: "
                              f"Volume ratio (posterior/prior) = {volume_ratio:.6f} "
                              f"(posterior area: {posterior_area:.6e}, prior area: {prior_area:.6e})")
                        
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
    """Early stopping based on convergence of the posterior/prior volume ratio.

    Reads volume ratio history from a companion :class:`PlotPosteriorCallback`
    (which must appear **before** this callback in the trainer's callback list)
    and stops training when the smoothed maximum volume ratio across all
    marginals has plateaued.

    Parameters
    ----------
    warmup_epochs : int
        Ignore the first *warmup_epochs* epochs (volume ratios are unreliable
        while the classifier is still learning).
    patience : int
        Number of consecutive evaluations with relative change below
        *rel_tol* before stopping.
    rel_tol : float
        Relative-change threshold on the EMA of the max volume ratio.
    ema_alpha : float
        Smoothing factor for exponential moving average (0 < alpha <= 1).
        Higher values weight recent observations more.
    """

    def __init__(
        self,
        warmup_epochs: int = 50,
        patience: int = 10,
        rel_tol: float = 0.02,
        ema_alpha: float = 0.3,
        min_ratio_threshold: float = 0.5,
    ):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.patience = patience
        self.rel_tol = rel_tol
        self.ema_alpha = ema_alpha
        self.min_ratio_threshold = min_ratio_threshold

        # internal state
        self._ema: float | None = None
        self._prev_ema: float | None = None
        self._stall_count: int = 0
        self._n_evaluations: int = 0
        # full history for diagnostics
        self.ema_history: list[dict] = []

    # ------------------------------------------------------------------
    def _find_plot_callback(self, trainer) -> "PlotPosteriorCallback | None":
        for cb in trainer.callbacks:
            if isinstance(cb, PlotPosteriorCallback):
                return cb
        return None

    # ------------------------------------------------------------------
    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch < self.warmup_epochs:
            return

        plot_cb = self._find_plot_callback(trainer)
        if plot_cb is None or not plot_cb.volume_ratios:
            return

        # Collect the latest volume ratio for each marginal at this epoch
        current_ratios = []
        for marginal_key, history in plot_cb.volume_ratios.items():
            if history and history[-1]["epoch"] == trainer.current_epoch:
                current_ratios.append(history[-1]["ratio"])

        if not current_ratios:
            return  # PlotPosteriorCallback didn't run this epoch

        # Hard threshold: if any marginal has already shrunk to min_ratio_threshold, stop.
        min_ratio = min(current_ratios)
        if min_ratio <= self.min_ratio_threshold:
            print(
                f"[VolumeRatioES] Stopping at epoch {trainer.current_epoch}: "
                f"min volume ratio {min_ratio:.4f} <= threshold {self.min_ratio_threshold:.4f}."
            )
            if trainer.logger is not None:
                trainer.logger.log_metrics(
                    {"volume_ratio/min": min_ratio},
                    step=trainer.current_epoch,
                )
            trainer.should_stop = True
            return

        max_ratio = max(current_ratios)
        self._n_evaluations += 1

        # Update EMA
        if self._ema is None:
            self._ema = max_ratio
        else:
            self._ema = self.ema_alpha * max_ratio + (1 - self.ema_alpha) * self._ema

        # Store for diagnostics
        self.ema_history.append({
            "epoch": trainer.current_epoch,
            "max_ratio": max_ratio,
            "ema": self._ema,
        })

        # Log to TensorBoard
        if trainer.logger is not None:
            trainer.logger.log_metrics(
                {
                    "volume_ratio/ema_max": self._ema,
                    "volume_ratio/raw_max": max_ratio,
                    "volume_ratio/raw_min": min_ratio,
                },
                step=trainer.current_epoch,
            )

        # Need at least 2 EMA values to check plateau
        if self._prev_ema is not None:
            rel_change = abs(self._ema - self._prev_ema) / (abs(self._prev_ema) + 1e-10)
            if rel_change < self.rel_tol:
                self._stall_count += 1
            else:
                self._stall_count = 0

            print(
                f"[VolumeRatioES] epoch {trainer.current_epoch}: "
                f"max_ratio={max_ratio:.4f}, ema={self._ema:.4f}, "
                f"rel_change={rel_change:.4f}, stall={self._stall_count}/{self.patience}"
            )
        else:
            print(
                f"[VolumeRatioES] epoch {trainer.current_epoch}: "
                f"max_ratio={max_ratio:.4f}, ema={self._ema:.4f} (first evaluation)"
            )

        self._prev_ema = self._ema

        if self._stall_count >= self.patience:
            print(
                f"[VolumeRatioES] Stopping: volume ratio EMA plateaued at {self._ema:.4f} "
                f"for {self.patience} consecutive evaluations."
            )
            trainer.should_stop = True