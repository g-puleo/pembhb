import os
import yaml
import numpy as np
import torch
from pembhb import ROOT_DIR, get_numpy_dtype
from pembhb.utils import (
    _ORDERED_PRIOR_KEYS,
    ordered_prior_keys,
    get_widest_interval_1d,
    get_widest_box_2d,
    get_logratios_grid,
    get_logratios_grid_2d,
    get_pvalues_1d,
    grid_posterior_moments_1d,
    grid_posterior_moments_2d,
    compute_fisher_sigmas_for_testset,
    eval_posterior_2d,
    contour_levels,
    posterior_contours_2d,
    posterior_heatmap_2d,
    contour_boxes,
    pp_plot_overlay,
)
# from pembhb.data import MBHBDataset, mbhb_collate_fn
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from pembhb.sky_truncation import get_main_mode_box
from pembhb.regions import (
    Region, region_from_equal_tailed, region_from_main_mode, region_from_hpd,
    prev_keep_mask,
)
from pembhb.mask_truncation import _MASK_PERIOD_BY_NAME
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Single-colour map for the §2 "suppressed region" wash in 2D posterior plots.
# NaN cells (the kept region) render transparent, so only ~keep is tinted.
_SUPPRESSED_CMAP = ListedColormap(["red"])


class StreamReuseLogger(Callback):
    """Per-epoch reuse accounting for the streaming ring-buffer data path.

    Arms ``ring.count_seen`` only for the fit loop (so the pre-fit normalisation
    and GPU-reserve passes are not counted), tallies pre-noise training examples,
    and logs examples / distinct sims / reuse each epoch to TB. If ``summary_path``
    is given, also appends one line per epoch. All counters are per round (the
    datamodule and ring are rebuilt each round).
    """

    def __init__(self, summary_path=None):
        self.summary_path = summary_path

    def on_train_start(self, trainer, pl_module):
        trainer.datamodule.ring.count_seen = True

    def on_train_end(self, trainer, pl_module):
        trainer.datamodule.ring.count_seen = False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        dm = trainer.datamodule
        dm._examples_consumed += batch["wave_fd"].shape[0] // dm.n_train_noise_realisations

    def on_train_epoch_end(self, trainer, pl_module):
        dm = trainer.datamodule
        examples, distinct = dm._examples_consumed, dm.ring.distinct_sims_seen
        reuse = examples / max(distinct, 1)
        if trainer.logger:
            trainer.logger.log_metrics(
                {"stream/examples_consumed": examples,
                 "stream/distinct_sims_seen": distinct,
                 "stream/reuse_factor": reuse}, step=trainer.global_step)
        if self.summary_path:
            os.makedirs(os.path.dirname(self.summary_path), exist_ok=True)
            with open(self.summary_path, "a") as f:
                f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S}  [epoch {trainer.current_epoch}] "
                        f"examples={examples} distinct_sims={distinct} reuse={reuse:.2f}\n")


def _param_keys(pl_module):
    """Parameter names for the run's spin basis (slots 2,3), from dataset_info.

    Reads ``sampler_init_kwargs["spin_param_basis"]`` (persisted in the sidecar);
    falls back to the legacy chi1/chi2 basis when absent.
    """
    sik = pl_module.hparams["dataset_info"].get("sampler_init_kwargs", {})
    return ordered_prior_keys(sik.get("spin_param_basis", "chi1chi2"))


class _StepCadence:
    """Fire at most once per ``interval`` global steps.

    ``interval=None`` -> disabled (the caller keeps its epoch logic). Driven from
    ``on_train_batch_end`` with ``trainer.global_step`` so the cadence is honoured
    exactly, independent of epoch size (small streaming buffers no longer inflate
    callback frequency).
    """

    def __init__(self, interval):
        self.interval = interval
        self._last = None

    @property
    def enabled(self):
        return self.interval is not None

    def should_fire(self, global_step):
        if self._last is None or global_step - self._last >= self.interval:
            self._last = global_step
            return True
        return False


class PeriodicProgressCallback(Callback):
    """Print a one-line training status every *print_every* epochs.

    Produces logfile-friendly output (no ANSI codes, no carriage returns)
    suitable for piping through ``tee``.  Complements TensorBoard logging:
    only the most essential scalars are printed here.
    """

    def __init__(self, print_every: int = 20, label: str = "", print_every_n_steps=None):
        super().__init__()
        self.print_every = print_every
        self.label = label
        # Opt-in global-step cadence; None -> unchanged epoch behaviour.
        self._step = _StepCadence(print_every_n_steps)

    def _log_lrs(self, trainer, pl_module, step):
        """Log learning rates to TensorBoard at the given x-axis ``step``."""
        try:
            opt = pl_module.optimizers()
            if isinstance(opt, list):
                opt = opt[0]
            if trainer.logger is not None:
                if len(opt.param_groups) >= 2:
                    trainer.logger.log_metrics({
                        "lr/autoencoder": opt.param_groups[0]["lr"],
                        "lr/nre": opt.param_groups[1]["lr"],
                    }, step=step)
                else:
                    trainer.logger.log_metrics({
                        "lr": opt.param_groups[0]["lr"],
                    }, step=step)
        except Exception:
            pass

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
        if self._step.enabled:
            return  # step-mode: handled in on_train_batch_end
        # Log learning rates to TensorBoard every epoch
        self._log_lrs(trainer, pl_module, trainer.current_epoch)
        if trainer.current_epoch % self.print_every == 0:
            self._print_status(trainer, pl_module)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self._step.enabled:
            return
        if self._step.should_fire(trainer.global_step):
            self._log_lrs(trainer, pl_module, trainer.global_step)
            self._print_status(trainer, pl_module, suffix=f"step {trainer.global_step}")

    def on_train_end(self, trainer, pl_module):
        self._print_status(trainer, pl_module, suffix="[done]")


class PlotPosteriorCallback(Callback):
    def __init__(self, timestamp: str, obs_loader: DataLoader, input_idx_list: list, output_idx_list: list, round_idx: int , call_every_n_epochs=1, training_start_time: datetime = None, print_every: int = 20, warmup_epochs: int = 0, call_every_n_steps=None, warmup_steps=None, 
                 box_dilation_1d: float = 1.0, 
                 eps_1d: float = 1e-4, 
                 sky_credible_level: float = 0.999, 
                 sky_dilation: float = 1.1,
                 truncation_mode: str = "rectangle",
                credible_level_1d: float = 0.997,
                dilation_1d: float = 1.2,
                credible_level_2d: float = 0.997,
                dilation_2d: float = 1.2,
                prev_accepted: dict = None,
                zero_policy: str = "off"):
        self.epochs_elapsed = 0
        # §2 — the previous round's accepted set per marginal ({marginal_key:
        # {"kind": "1d"/"2d", ...}}) and the active zeroing policy. When set (and
        # not "off"), the posterior plots shade the region OUTSIDE the previous
        # mask, where _mask_truncate_marginal zeroes / down-weights the density
        # before this round's HPD analysis (see pembhb.regions.apply_prev_mask).
        self.prev_accepted = prev_accepted or {}
        self.zero_policy = str(zero_policy)
        # Box-construction params — single source of truth is the run's
        # truncation_veto config, so the boxes written into widest_boxes match
        # the ones the coverage veto measures (see _compute_round_coverage).
        self.box_dilation_1d = box_dilation_1d
        self.eps_1d = eps_1d
        self.sky_credible_level = sky_credible_level
        self.sky_dilation = sky_dilation
        # HPD level + dilation used to build the 2D posterior region for the
        # volume ratio — matched to the truncation (analyse_posterior_2d) so the
        # numerator is the SAME accepted set the sampler will draw from next round.
        self.credible_level_2d = credible_level_2d
        self.dilation_2d = dilation_2d
        self.call_every_n_epochs = call_every_n_epochs
        # Opt-in global-step cadence; None -> unchanged epoch behaviour.
        self._step = _StepCadence(call_every_n_steps)
        self.warmup_steps = warmup_steps
        self.print_every = print_every
        self.timestamp = timestamp
        self.obs_loader = obs_loader
        self.input_idx_list = input_idx_list
        self.output_idx_list = output_idx_list
        self.n_marginals = len(input_idx_list)
        self.init_time = datetime.now()
        self.training_start_time = training_start_time if training_start_time is not None else self.init_time
        self.round_idx = round_idx
        self.warmup_epochs = warmup_epochs
        # Storage for volume ratio diagnostics
        self.volume_ratios = {}
        # Storage for differential entropy diagnostics
        self.differential_entropies = {}
    
    # Posterior/prior volume ratios are now `Region.volume_fraction()` — the
    # posterior grid spans exactly the trained prior box, so the accepted-pixel
    # fraction *is* the posterior/prior ratio, with one implementation for 1D and
    # 2D (rectangle or mask). The four `_compute_*_volume_*` helpers this replaced
    # are gone; see `pembhb.regions`.

    # ------------------------------------------------------------------
    # §2 — visualise where the posterior is zeroed / down-weighted before
    # this round's HPD analysis (outside the previous round's trained mask).
    # ------------------------------------------------------------------
    def _suppressed_label(self):
        if self.zero_policy == "hysteresis":
            return "§2 down-weighted (outside prev mask)"
        return "§2 zeroed (outside prev mask)"

    def _prev_keep_or_none(self, marginal_key, grids):
        """Boolean 'inside the previous round's mask' on ``grids``, or None when
        §2 is off / there is no previous mask (round 1) for this marginal."""
        if self.zero_policy == "off":
            return None
        prev = self.prev_accepted.get(marginal_key)
        if prev is None:
            return None
        return prev_keep_mask(prev, grids)

    def _proposal_pixels(self, marginal_key, grids, grid_total):
        """Pixel count of the mask actually sampled from this round.

        The volume ratio is the early-stop metric ``posterior / proposal``.  In
        mask mode the sampling proposal is the PREVIOUS round's accepted set, an
        irregular mask that can fill only a fraction of its own bounding box — so
        dividing by the full grid (``mask.size``) understates the ratio and it
        never rises to 1 even when the posterior has converged to the proposal.
        Here the denominator is the proposal mask itself, resampled onto the
        current grid.  Falls back to the full grid when there is no mask proposal
        (round 1, rectangle mode, or a previous mask that does not overlap this
        grid), where posterior/box is the correct measure.  This is independent
        of the §2 zero policy: the sampler uses the mask regardless.
        """
        prev = self.prev_accepted.get(marginal_key)
        if prev is None:
            return grid_total
        n = int(np.count_nonzero(prev_keep_mask(prev, grids)))
        return n if n > 0 else grid_total

    def _shade_suppressed_1d(self, ax, grid1d, marginal_key):
        keep = self._prev_keep_or_none(marginal_key, (grid1d,))
        if keep is None:
            return
        supp = ~np.asarray(keep, dtype=bool)
        if not supp.any():
            return
        g = np.asarray(grid1d, dtype=float)
        half = 0.5 * (g[1] - g[0])
        # contiguous runs of suppressed cells [a, b); shade to the cell edges
        padded = np.concatenate(([False], supp, [False]))
        trans = np.flatnonzero(padded[1:] != padded[:-1])
        labelled = False
        for a, b in zip(trans[0::2], trans[1::2]):
            ax.axvspan(g[a] - half, g[b - 1] + half,
                       color="red", alpha=0.12, hatch="//", lw=0,
                       label=None if labelled else self._suppressed_label())
            labelled = True

    def _shade_suppressed_2d(self, ax, gx, gy, marginal_key):
        keep = self._prev_keep_or_none(marginal_key, (gx[0, :], gy[:, 0]))
        if keep is None or keep.all():
            return
        # translucent red wash over the suppressed pixels (~keep); the mask is
        # (n_y, n_x) so imshow with the grid extent lines up with the heatmap.
        supp = np.where(keep, np.nan, 1.0)
        extent = (float(gx[0, 0]), float(gx[0, -1]), float(gy[0, 0]), float(gy[-1, 0]))
        ax.imshow(supp, origin="lower", extent=extent, aspect="auto",
                  cmap=_SUPPRESSED_CMAP, alpha=0.30, interpolation="nearest", zorder=3)

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
        if self._step.enabled:
            return  # step-mode: handled in on_train_batch_end
        if trainer.current_epoch < self.warmup_epochs:
            return

        if self.epochs_elapsed == 0:
            # Posterior-evolution PDFs go in a subfolder so the run's plots
            # root stays scannable when the cadence (call_every_n_epochs)
            # produces many figures.
            os.makedirs(
                os.path.join(ROOT_DIR, "plots", self.timestamp,
                             "posterior_evolution"),
                exist_ok=True,
            )

        self.epochs_elapsed += 1
        if (self.epochs_elapsed-2) % self.call_every_n_epochs == 0:
            self._compute_and_plot(trainer, pl_module, trainer.current_epoch, False)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self._step.enabled:
            return
        if self.warmup_steps is not None and trainer.global_step < self.warmup_steps:
            return
        if not self._step.should_fire(trainer.global_step):
            return
        os.makedirs(
            os.path.join(ROOT_DIR, "plots", self.timestamp, "posterior_evolution"),
            exist_ok=True,
        )
        # mid-batch the model is in train mode (dropout active) -> eval for the
        # diagnostic forward passes, then restore.
        was_training = pl_module.training
        pl_module.eval()
        try:
            with torch.no_grad():
                self._compute_and_plot(trainer, pl_module, trainer.global_step, True)
        finally:
            if was_training:
                pl_module.train()

    def _compute_and_plot(self, trainer, pl_module, tag, tag_is_step):
        """Posterior diagnostics + plots on the observation, tagged by ``tag``
        (epoch or global step). Sets ``pl_module.widest_boxes`` and appends
        step/epoch-keyed entries to volume_ratios / differential_entropies."""
        tag_kind = "step" if tag_is_step else "epoch"
        do_print = tag_is_step or (trainer.current_epoch % self.print_every == 0)
        if True:
            #print("plotting posteriors on observed data")
            train_time = datetime.now() - self.training_start_time
            td_trunc = train_time - timedelta(microseconds=train_time.microseconds)
            title_plot = f"training time={td_trunc}s"
            keys = _param_keys(pl_module)  # basis-aware parameter names
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
                        f"Round {self.round_idx} - {tag_kind} {tag} - {title_plot}",
                        fontsize=10,
                    )
                    
                    try:
                        epsilon_value = self.eps_1d
                        widest_interval, norm1d, grid, inj_params = get_widest_interval_1d(
                            pl_module,
                            self.obs_loader,
                            in_param_idx=param_idx,
                            out_param_idx=out_param_idx,
                            eps=epsilon_value,
                            dilation=self.box_dilation_1d,
                        )
                        
                        # Plot
                        ax.plot(grid.flatten(), norm1d, 'b-', linewidth=1.5)
                        ax.axvline(inj_params[0], color='r', linestyle='--', label='Injection')
                        ax.axvline(widest_interval[0], color='g', linestyle=':', label=f'{100*(1-epsilon_value):.2f}% CI')
                        ax.axvline(widest_interval[1], color='g', linestyle=':')
                        ax.fill_between(grid.flatten(), 0, norm1d,
                                       where=(grid.flatten() >= widest_interval[0]) & (grid.flatten() <= widest_interval[1]),
                                       alpha=0.3, color='green')
                        # §2: shade the band outside the previous round's mask,
                        # where the density is zeroed / down-weighted before the
                        # HPD analysis (only when §2 is active and a prev mask
                        # exists — i.e. from round 2 onward).
                        self._shade_suppressed_1d(ax, grid[:, 0], marginal_key)
                        ax.set_xlabel(keys[param_idx])
                        ax.set_ylabel('Posterior density')
                        ax.legend()
                        
                        # Store the widest interval
                        pl_module.widest_boxes[marginal_key] = widest_interval

                        # Posterior-to-prior volume ratio for the 1D marginal.
                        # The grid spans exactly the trained prior box, so the
                        # accepted-pixel fraction of the equal-tailed region IS
                        # the posterior/prior width ratio (one implementation for
                        # every marginal; see pembhb.regions).
                        region1d = region_from_equal_tailed(
                            norm1d, (grid[:, 0],), eps=epsilon_value,
                            dilation=self.box_dilation_1d)
                        # ratio = posterior pixels / sampling-proposal pixels
                        # (NOT / full grid — see _proposal_pixels). Matters for a
                        # multi-interval (gapped) 1D proposal, e.g. cos-ι.
                        posterior_pixels = int(np.count_nonzero(region1d.mask))
                        proposal_pixels = self._proposal_pixels(
                            marginal_key, (grid[:, 0],), int(region1d.mask.size))
                        volume_ratio = posterior_pixels / proposal_pixels
                        _dp = float(grid[1, 0] - grid[0, 0])
                        posterior_width = posterior_pixels * _dp
                        prior_width = proposal_pixels * _dp
                        
                        # Store and log the volume ratio
                        if marginal_key not in self.volume_ratios:
                            self.volume_ratios[marginal_key] = []
                        self.volume_ratios[marginal_key].append({
                            'epoch': trainer.current_epoch,
                            'step': trainer.global_step,
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
                            'step': trainer.global_step,
                            'entropy': entropy,
                        })

                        # Log metrics to tensorboard if logger exists
                        if trainer.logger is not None:
                            metric_name = f"volume_ratio/{keys[param_idx]}"
                            trainer.logger.log_metrics({metric_name: volume_ratio}, step=tag)
                            entropy_metric = f"diff_entropy/{keys[param_idx]}"
                            trainer.logger.log_metrics({entropy_metric: entropy}, step=tag)

                        # Print diagnostic
                        param_name = keys[param_idx]
                        if do_print:
                            print(f"Round {self.round_idx}, {tag_kind} {tag}, {param_name}: "
                                  f"vol_ratio={volume_ratio:.4f} "
                                  f"(post={posterior_width:.3e}, prior={prior_width:.3e}), "
                                  f"H={entropy:.4f} nats", flush=True)
                        
                        out = os.path.join(ROOT_DIR, "plots", self.timestamp,
                                          "posterior_evolution",
                                          f"posterior_round_{self.round_idx}_{tag_kind}_{tag}_{keys[param_idx]}.pdf")
                        fig.savefig(out, bbox_inches="tight")
                    except Exception as e:
                        print(f"Error plotting 1D marginal for {keys[param_idx]}: {e}")
                    finally:
                        plt.close(fig)

                elif len(in_param_idx) == 2:
                    # Handle 2D marginals
                    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
                    fig.tight_layout()
                    fig.suptitle(
                        f"Round {self.round_idx} - {tag_kind} {tag} - {title_plot}",
                        fontsize=10,
                    )

                    out = os.path.join(ROOT_DIR, "plots", self.timestamp,
                                      "posterior_evolution",
                                      f"posterior_round_{self.round_idx}_{tag_kind}_{tag}_{keys[in_param_idx[0]]}_{keys[in_param_idx[1]]}.pdf")
                    param_names = [keys[in_param_idx[0]], keys[in_param_idx[1]]]
                    param_label = f"{keys[in_param_idx[0]]}-{keys[in_param_idx[1]]}"

                    try:
                        # Baseline heatmap — always produced, independent of contour levels.
                        norm_2d, inj_params, gx, gy, dp0, dp1 = eval_posterior_2d(pl_module, self.obs_loader, in_param_idx, out_param_idx)
                        posterior_heatmap_2d(gx, gy, norm_2d, inj_params[0], ax, param_names)
                        # §2: overlay the region outside the previous round's mask
                        # (zeroed / down-weighted before this round's analysis).
                        self._shade_suppressed_2d(ax, gx, gy, marginal_key)

                        # Contour overlay + contour-derived diagnostics (best effort).
                        # matplotlib raises ValueError when contour thresholds are
                        # non-strictly-increasing (flat / degenerate posteriors); in
                        # that case we keep the heatmap and skip the rest.
                        try:
                            levels, labels = contour_levels(norm_2d)
                            boxes_overlay, cs = contour_boxes(gx, gy, norm_2d, levels, ax=ax)
                            fmt = {lev: f"{p:.3f}" for lev, p in zip(levels, labels)}
                            ax.clabel(cs, fmt=fmt, fontsize=8)

                            if marginal_key == (7, 8):
                                box = get_main_mode_box(gx, gy, norm_2d, credible_level=self.sky_credible_level, dilation_factor=self.sky_dilation)
                                widest_box = (box['lam'][0], box['lam'][1], box['beta'][0], box['beta'][1])
                                is_wrapped = box["is_wrapped"]
                            else:
                                widest_box = boxes_overlay[0]
                                is_wrapped = False

                            pl_module.widest_boxes[marginal_key] = widest_box

                            # Posterior-to-prior area ratio via Region. This is
                            # the true accepted-MASK fraction of the dominant
                            # mode — correct for wrapped / non-rectangular modes,
                            # unlike the bounding-box area the old helpers used
                            # (a change of measure, not just ±1 px; it slightly
                            # shifts the 2D early-stop timing).
                            # Full HPD posterior (ALL modes), built the same way
                            # as the truncation/proposal (analyse_posterior_2d at
                            # credible_level_2d/dilation_2d) — NOT the main mode
                            # only, which undercounts a multimodal posterior and
                            # trips the 0.5 stop early.
                            region2d = region_from_hpd(
                                norm_2d, (gx[0, :], gy[:, 0]),
                                credible_level=self.credible_level_2d,
                                dilation_factor=self.dilation_2d,
                                periods=(_MASK_PERIOD_BY_NAME.get(keys[in_param_idx[0]]),
                                         _MASK_PERIOD_BY_NAME.get(keys[in_param_idx[1]])))
                            # ratio = posterior pixels / sampling-proposal pixels
                            # (NOT / full grid — see _proposal_pixels).
                            posterior_pixels = int(np.count_nonzero(region2d.mask))
                            proposal_pixels = self._proposal_pixels(
                                marginal_key, (gx[0, :], gy[:, 0]), int(region2d.mask.size))
                            volume_ratio = posterior_pixels / proposal_pixels
                            cell_area = float(dp0 * dp1)
                            posterior_area = posterior_pixels * cell_area
                            prior_area = proposal_pixels * cell_area

                            if marginal_key not in self.volume_ratios:
                                self.volume_ratios[marginal_key] = []
                            self.volume_ratios[marginal_key].append({
                                'epoch': trainer.current_epoch,
                                'step': trainer.global_step,
                                'ratio': volume_ratio,
                                'posterior_area': posterior_area,
                                'prior_area': prior_area
                            })

                            entropy = self._differential_entropy_2d(norm_2d, dp0, dp1)
                            if marginal_key not in self.differential_entropies:
                                self.differential_entropies[marginal_key] = []
                            self.differential_entropies[marginal_key].append({
                                'epoch': trainer.current_epoch,
                                'step': trainer.global_step,
                                'entropy': entropy,
                            })

                            param_tag = f"{keys[in_param_idx[0]]}_{keys[in_param_idx[1]]}"
                            if trainer.logger is not None:
                                trainer.logger.log_metrics({f"volume_ratio/{param_tag}": volume_ratio}, step=tag)
                                trainer.logger.log_metrics({f"diff_entropy/{param_tag}": entropy}, step=tag)

                            if marginal_key == (7, 8):
                                self._log_sky_contour_ratios(
                                    norm_2d, gx, gy, dp0, dp1,
                                    trainer, param_tag,
                                )

                            if do_print:
                                print(f"Round {self.round_idx}, {tag_kind} {tag}, {param_label}: "
                                      f"vol_ratio={volume_ratio:.4f} "
                                      f"(post={posterior_area:.3e}, prior={prior_area:.3e}), "
                                      f"H={entropy:.4f} nats", flush=True)
                        except ValueError as ve:
                            print(f"Round {self.round_idx}, {tag_kind} {tag}, {param_label}: "
                                  f"contour overlay failed ({ve}); saving heatmap without contours.",
                                  flush=True)

                        fig.savefig(out, bbox_inches="tight")
                    finally:
                        plt.close(fig)

    def on_train_end(self, trainer, pl_module):
        # Final diagnostic pass. In step-mode on_validation_epoch_end is a no-op,
        # so compute directly (with the eval()/no_grad() wrap).
        if self._step.enabled:
            was_training = pl_module.training
            pl_module.eval()
            try:
                with torch.no_grad():
                    self._compute_and_plot(trainer, pl_module, trainer.global_step, True)
            finally:
                if was_training:
                    pl_module.train()
        else:
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
        plateau_grace_epochs: int = 0,
        print_every: int = 20,
        warmup_steps=None,
        step_mode: bool = False,
    ):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        # Opt-in: consume plot-cb entries on a global-step cadence instead of
        # matching the epoch. When step_mode is False, behaviour is unchanged.
        self.warmup_steps = warmup_steps
        self.step_mode = step_mode
        self._last_seen_step = -1
        self.patience = patience
        self.rel_tol = rel_tol
        self.ema_alpha = ema_alpha
        self.min_ratio_threshold = min_ratio_threshold
        # The plateau (stabilisation) stop is the FALLBACK: it may fire only
        # after the 0.5 fast-truncation threshold has had this many epochs past
        # warmup to trigger and didn't. In the search phase the threshold wins;
        # the plateau only takes over once the prior can no longer be halved.
        self.plateau_grace_epochs = plateau_grace_epochs
        self.print_every = print_every

        # per-marginal state: keyed by marginal_key (tuple)
        self._ema: dict[tuple, float] = {}
        self._prev_ema: dict[tuple, float] = {}
        self._stall_count: dict[tuple, int] = {}
        self.ema_history: list[dict] = []
        self.stop_reason: str = ""
        # "threshold" (fast 0.5 path) or "plateau" (saturated fallback) — read
        # by the campaign monitor: consecutive "plateau" rounds signal that
        # truncation has stopped shrinking the prior.
        self.stopped_via: str = ""

    def _find_plot_callback(self, trainer) -> "PlotPosteriorCallback | None":
        for cb in trainer.callbacks:
            if isinstance(cb, PlotPosteriorCallback):
                return cb
        return None

    @staticmethod
    def _marginal_label(key, keys):
        names = [keys[i] for i in key]
        return "-".join(names)

    def _collect_current_by_step(self, plot_cb):
        """Step-mode: return the newest unconsumed set of per-marginal ratios
        (one plot-cb fire), advancing ``_last_seen_step``. Empty if nothing new."""
        latest_step = -1
        for history in plot_cb.volume_ratios.values():
            if history:
                latest_step = max(latest_step, history[-1].get("step", -1))
        if latest_step <= self._last_seen_step:
            return {}
        self._last_seen_step = latest_step
        current = {}
        for mkey, history in plot_cb.volume_ratios.items():
            if history and history[-1].get("step", -1) == latest_step:
                current[mkey] = history[-1]["ratio"]
        return current

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.step_mode:
            return  # step-mode: handled in on_train_batch_end
        if trainer.current_epoch < self.warmup_epochs:
            return
        plot_cb = self._find_plot_callback(trainer)
        if plot_cb is None or not plot_cb.volume_ratios:
            return
        keys = _param_keys(pl_module)
        current = {}
        for marginal_key, history in plot_cb.volume_ratios.items():
            if history and history[-1]["epoch"] == trainer.current_epoch:
                current[marginal_key] = history[-1]["ratio"]
        if not current:
            return
        plateau_allowed = (
            trainer.current_epoch >= self.warmup_epochs + self.plateau_grace_epochs
        )
        self._evaluate(trainer, pl_module, current, keys, plateau_allowed,
                       trainer.current_epoch, "epoch")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.step_mode:
            return
        if self.warmup_steps is not None and trainer.global_step < self.warmup_steps:
            return
        plot_cb = self._find_plot_callback(trainer)
        if plot_cb is None or not plot_cb.volume_ratios:
            return
        current = self._collect_current_by_step(plot_cb)
        if not current:
            return
        keys = _param_keys(pl_module)
        # plateau allowed once past warmup_steps (already gated above). patience
        # now counts plot-cb fires (= patience * call_every_n_steps steps).
        self._evaluate(trainer, pl_module, current, keys, True,
                       trainer.global_step, "step")

    def _evaluate(self, trainer, pl_module, current, keys, plateau_allowed, tag, tag_kind):
        triggered_key = None
        triggered_reason = ""

        for mkey, ratio in current.items():
            label = self._marginal_label(mkey, keys)

            # Fast path: hard 0.5 truncation threshold.
            if ratio <= self.min_ratio_threshold:
                triggered_key = mkey
                self.stopped_via = "threshold"
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

            if plateau_allowed and self._stall_count.get(mkey, 0) >= self.patience:
                triggered_key = mkey
                self.stopped_via = "plateau"
                triggered_reason = (
                    f"volume_ratio_plateau: {label} ema={self._ema[mkey]:.4f}, "
                    f"stall={self._stall_count[mkey]}/{self.patience}"
                )

        # Log to TensorBoard
        if trainer.logger is not None:
            metrics = {}
            for mkey in current:
                label = self._marginal_label(mkey, keys)
                metrics[f"volume_ratio_ema/{label}"] = self._ema.get(mkey, current[mkey])
            trainer.logger.log_metrics(metrics, step=tag)

        # Diagnostics printing
        if tag_kind == "step" or trainer.current_epoch % self.print_every == 0:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            parts = []
            for mkey in sorted(current):
                label = self._marginal_label(mkey, keys)
                ema_val = self._ema.get(mkey, current[mkey])
                stall = self._stall_count.get(mkey, 0)
                parts.append(f"{label}(r={current[mkey]:.4f},ema={ema_val:.4f},s={stall})")
            print(f"[{ts}] [VolumeRatioES] {tag_kind} {tag}: {', '.join(parts)}", flush=True)

        if triggered_key is not None:
            self.stop_reason = triggered_reason
            print(f"[VolumeRatioES] Stopping at {tag_kind} {tag}: {triggered_reason}")
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
        warmup_steps=None,
        step_mode: bool = False,
    ):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.patience = patience
        self.rel_tol = rel_tol
        self.ema_alpha = ema_alpha
        self.print_every = print_every
        # Opt-in step cadence (consume plot-cb entries by global step).
        self.warmup_steps = warmup_steps
        self.step_mode = step_mode
        self._last_seen_step = -1

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
    def _marginal_label(key, keys):
        names = [keys[i] for i in key]
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

        keys = _param_keys(pl_module)
        d = len(marginal_key)
        h = 0.0
        for idx in marginal_key:
            pname = keys[idx]
            lo, hi = prior_dict[pname]
            h += np.log(hi - lo)
        h /= (2 * d)
        self._thresholds[marginal_key] = h
        return h

    def _collect_current_by_step(self, plot_cb):
        """Step-mode: newest unconsumed per-marginal entropies (one plot fire)."""
        latest_step = -1
        for history in plot_cb.differential_entropies.values():
            if history:
                latest_step = max(latest_step, history[-1].get("step", -1))
        if latest_step <= self._last_seen_step:
            return {}
        self._last_seen_step = latest_step
        current = {}
        for mkey, history in plot_cb.differential_entropies.items():
            if history and history[-1].get("step", -1) == latest_step:
                current[mkey] = history[-1]["entropy"]
        return current

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.step_mode:
            return  # step-mode: handled in on_train_batch_end
        if trainer.current_epoch < self.warmup_epochs:
            return
        plot_cb = self._find_plot_callback(trainer)
        if plot_cb is None or not plot_cb.differential_entropies:
            return
        keys = _param_keys(pl_module)
        current = {}
        for marginal_key, history in plot_cb.differential_entropies.items():
            if history and history[-1]["epoch"] == trainer.current_epoch:
                current[marginal_key] = history[-1]["entropy"]
        if not current:
            return
        self._evaluate(trainer, pl_module, current, keys, trainer.current_epoch, "epoch")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.step_mode:
            return
        if self.warmup_steps is not None and trainer.global_step < self.warmup_steps:
            return
        plot_cb = self._find_plot_callback(trainer)
        if plot_cb is None or not plot_cb.differential_entropies:
            return
        current = self._collect_current_by_step(plot_cb)
        if not current:
            return
        keys = _param_keys(pl_module)
        self._evaluate(trainer, pl_module, current, keys, trainer.global_step, "step")

    def _evaluate(self, trainer, pl_module, current, keys, tag, tag_kind):
        triggered_key = None
        triggered_reason = ""

        for mkey, entropy in current.items():
            label = self._marginal_label(mkey, keys)
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
                label = self._marginal_label(mkey, keys)
                metrics[f"diff_entropy_ema/{label}"] = self._ema.get(mkey, current[mkey])
            trainer.logger.log_metrics(metrics, step=tag)

        # Diagnostics printing
        if tag_kind == "step" or trainer.current_epoch % self.print_every == 0:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            parts = []
            for mkey in sorted(current):
                label = self._marginal_label(mkey, keys)
                ema_val = self._ema.get(mkey, current[mkey])
                stall = self._stall_count.get(mkey, 0)
                thresh = self._get_threshold(pl_module, mkey)
                parts.append(f"{label}(H={current[mkey]:.3f},ema={ema_val:.3f},s={stall},th={thresh:.3f})")
            print(f"[{ts}] [EntropyES] {tag_kind} {tag}: {', '.join(parts)}", flush=True)

        if triggered_key is not None:
            self.stop_reason = triggered_reason
            print(f"[EntropyES] Stopping at {tag_kind} {tag}: {triggered_reason}")
            trainer.should_stop = True


def compute_truncation_coverage(
    model, loader, marginals_1d_info, marginals_2d_info,
    eps_1d=1e-4, sky_credible_level=0.999, sky_dilation=1.1,
    ngrid_1d=100, ngrid_2d=100, dilation_1d=1.0,
):
    """Empirical coverage of each marginal's truncation region over a labelled set.

    For every marginal, per sample, build the accepted :class:`~pembhb.regions.Region`
    the round-end truncation would build and count how many samples' ground truths
    fall inside their own region (``Region.contains``). Returns
    ``{marginal_key_tuple: (coverage, n_inside, n_total)}``.

    This is the signal for the coverage-gated truncation veto (see
    ``tmnre_joint.SequentialTrainerJoint.run``): a region that is nominally
    ``1 - eps`` credible but empirically covers far fewer truths means the network
    is overconfident for that parameter, so its truncation should be vetoed rather
    than discard the true value.

    * 1D: equal-tailed ``[eps_1d/2, 1-eps_1d/2]`` interval
      (:func:`~pembhb.regions.region_from_equal_tailed`).
    * 2D: highest-mass HPD component (:func:`~pembhb.regions.region_from_main_mode`),
      with each axis's periodicity resolved from its parameter name — so this now
      works for **any** 2D marginal, not just the sky ``(7,8)`` pair (the old code
      applied sky main-mode logic to every 2D marginal, the bug its own docstring
      admitted).

    Membership is the true accepted-mask test, not a bounding-box test: a truth
    that lands in a gap between components (inside the box but outside the mask)
    correctly counts as NOT covered.
    """
    coverage = {}

    for _label, in_idx, out_idx in marginals_1d_info:
        logratios, inj, grid = get_logratios_grid(
            loader, model, ngrid_1d, in_param_idx=in_idx, out_param_idx=out_idx)
        g1d = grid[:, 0]
        n_total = logratios.shape[0]
        n_inside = 0
        for s in range(n_total):
            region = region_from_equal_tailed(
                np.exp(logratios[s]), (g1d,), eps=eps_1d, dilation=dilation_1d)
            n_inside += int(region.contains(np.array([float(inj[s])]))[0])
        coverage[(in_idx,)] = (n_inside / n_total, n_inside, n_total)

    for _label, in_pair, out_idx in marginals_2d_info:
        norm2d, inj, gx, gy = eval_posterior_2d(
            model, loader, in_pair, out_idx,
            ngrid_points=ngrid_2d, keep_batch_dim=True)
        gx_1d, gy_1d = gx[0, :], gy[:, 0]
        # Periodic axes resolved per parameter from the "<name0>__<name1>" label.
        names = str(_label).split("__")
        periods = (_MASK_PERIOD_BY_NAME.get(names[0]) if len(names) > 0 else None,
                   _MASK_PERIOD_BY_NAME.get(names[1]) if len(names) > 1 else None)
        n_total = norm2d.shape[0]
        n_inside = 0
        for s in range(n_total):
            region = region_from_main_mode(
                norm2d[s], (gx_1d, gy_1d),
                credible_level=sky_credible_level, dilation_factor=sky_dilation,
                periods=periods)
            n_inside += int(region.contains(
                np.array([[float(inj[s, 0])], [float(inj[s, 1])]]))[0])
        coverage[tuple(in_pair)] = (n_inside / n_total, n_inside, n_total)

    return coverage


class ChainConvergenceMonitor:
    """Across-round (campaign) convergence, tracked **per marginal**.

    Not a Lightning callback — a plain helper the trainer carries across rounds
    and queries at each round boundary. Each marginal carries its own absolute
    differential-entropy series and stall counter; a marginal is *converged*
    once its round-over-round change has flattened,

        |H_{n-1} − H_n| < eps_nats   for ``patience`` consecutive candidate rounds,

    and the **chain stops only when every reported marginal is converged**
    (AND-reduction). Summing the marginals would be wrong — a flat sum can hide
    one marginal still shrinking while another drifts up. Because H is a
    log-volume, ``eps_nats`` is a *relative* per-round width change, so the test
    is scale-free.

    A round is a candidate only if it did NOT stop via the within-round 0.5
    volume threshold (``stopped_via == "threshold"`` means the prior just halved
    — still actively shrinking, so every stall resets). An optional Fisher gate
    additionally requires the round's median τ to sit inside ``tau_gate``.

    ``converged_at`` records, per marginal, the round at which it first reached
    ``patience`` — surfaced at chain end so you can see when each marginal
    settled. State is persisted to ``state_path`` so ``--resume`` continues the
    decision.
    """

    def __init__(self, eps_nats: float = 0.05, patience: int = 2,
                 state_path: str | None = None,
                 tau_gate: tuple | None = None,
                 require_not_threshold: bool = True,
                 log_path: str | None = None):
        self.eps_nats = eps_nats
        self.patience = patience
        self.state_path = state_path
        self.tau_gate = tau_gate
        self.require_not_threshold = require_not_threshold
        self.log_path = log_path
        self.history: list[dict] = []
        self._prev_H: dict[str, float] = {}
        self._stall: dict[str, int] = {}
        self.converged_at: dict[str, int] = {}   # marginal -> round first converged
        self.converged = False
        self.stop_reason = ""
        self._load()

    def log(self, msg):
        """Tee a ChainConv message to stdout and, if configured, append it
        (timestamped) to the dedicated chain-convergence log."""
        print(msg, flush=True)
        if self.log_path:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}\n")

    def _load(self):
        if self.state_path and os.path.exists(self.state_path):
            with open(self.state_path) as f:
                s = yaml.safe_load(f) or {}
            self.history = s.get("history", []) or []
            self._prev_H = dict(s.get("prev_H", {}) or {})
            self._stall = {k: int(v) for k, v in (s.get("stall", {}) or {}).items()}
            self.converged_at = {k: int(v) for k, v in (s.get("converged_at", {}) or {}).items()}
            self.converged = bool(s.get("converged", False))

    def _save(self):
        if not self.state_path:
            return
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        tmp = self.state_path + ".tmp"
        with open(tmp, "w") as f:
            yaml.safe_dump({
                "history": self.history,
                "prev_H": {k: float(v) for k, v in self._prev_H.items()},
                "stall": {k: int(v) for k, v in self._stall.items()},
                "converged_at": {k: int(v) for k, v in self.converged_at.items()},
                "converged": bool(self.converged),
            }, f, sort_keys=False)
        os.replace(tmp, self.state_path)

    def update(self, round_idx, entropies: dict, stopped_via="", median_tau=None):
        """Record a round given ``{marginal_label: H}``; return chain-converged."""
        candidate = (not self.require_not_threshold) or (stopped_via != "threshold")
        tau_ok = self.tau_gate is None or (
            median_tau is not None
            and self.tau_gate[0] <= median_tau <= self.tau_gate[1]
        )

        per = {}
        for label, H in entropies.items():
            prev = self._prev_H.get(label)
            dH = None if prev is None else abs(prev - H)
            flat = dH is not None and dH < self.eps_nats
            if candidate and flat and tau_ok:
                self._stall[label] = self._stall.get(label, 0) + 1
            else:
                self._stall[label] = 0
            self._prev_H[label] = float(H)
            # First time this marginal reaches patience: stamp the round.
            if self._stall[label] >= self.patience and label not in self.converged_at:
                self.converged_at[label] = int(round_idx)
                self.log(f"[ChainConv] marginal '{label}' CONVERGED at round {round_idx} "
                         f"(H={H:.4f}, |ΔH|={dH:.2e} < {self.eps_nats})")
            per[label] = {"H": float(H), "dH": None if dH is None else float(dH),
                          "stall": self._stall[label]}

        self.history.append({
            "round": int(round_idx), "stopped_via": stopped_via,
            "median_tau": None if median_tau is None else float(median_tau),
            "per_marginal": per,
        })

        all_converged = bool(entropies) and all(
            self._stall.get(l, 0) >= self.patience for l in entropies
        )
        if all_converged:
            self.converged = True
            self.stop_reason = (
                f"all {len(entropies)} marginals flattened (|ΔH|<{self.eps_nats}, "
                f"patience={self.patience}); converged_at={self.converged_at}"
            )
        self._save()
        return self.converged

    def summary(self) -> str:
        """One-line-per-marginal record of when each marginal converged."""
        if not self.converged_at:
            return "[ChainConv] no marginal reached convergence."
        lines = ["[ChainConv] per-marginal convergence rounds:"]
        for label in sorted(self.converged_at):
            lines.append(f"    {label}: converged at round {self.converged_at[label]}")
        never = sorted(set(self._stall) - set(self.converged_at))
        for label in never:
            lines.append(f"    {label}: never converged "
                         f"(final stall {self._stall.get(label, 0)}/{self.patience})")
        return "\n".join(lines)


class WarmupEarlyStopping(EarlyStopping):
    """EarlyStopping that ignores the first ``warmup_epochs`` epochs.

    During AE warm-up in joint training, the NRE accuracy is logged as a
    constant ~0.5 (the heads are not being trained yet); a stock EarlyStopping
    on ``val_accuracy`` would either trigger immediately or burn its patience
    budget before the NRE phase even starts. This wrapper short-circuits the
    check until ``warmup_epochs`` have elapsed, then behaves identically to
    Lightning's :class:`EarlyStopping`.
    """

    def __init__(self, warmup_epochs: int, **kwargs):
        super().__init__(**kwargs)
        self._warmup_epochs = warmup_epochs

    def _run_early_stopping_check(self, trainer):
        if trainer.current_epoch < self._warmup_epochs:
            return
        super()._run_early_stopping_check(trainer)


class PPKSTestEarlyStopping(Callback):
    """Per-marginal PP-plot KS test + tail-mass overconfidence detector.

    Every ``run_every_n_epochs`` epochs (default 1) the callback evaluates the
    1-D marginal posteriors on a small held-out subset of the test split. For
    each 1-D marginal it computes two scalars from the rank distribution
    ``r_i = F̂_post(θ_true^i)``:

    * ``D_t`` — one-sample Kolmogorov-Smirnov statistic against ``Uniform(0,1)``
      (any miscalibration direction).
    * ``T_t`` — tail mass ``P[r ∈ [0, q] ∪ [1-q, 1]]``. Expected value under
      perfect calibration is ``2q``. Overconfident (undercovered) posteriors
      pile ranks at the extremes → ``T_t`` rises above ``2q``. Overcovered
      posteriors push ranks centrally → ``T_t`` falls below ``2q``. Combining
      ``D_t`` (deviation) with ``T_t`` (direction) isolates overconfidence
      specifically, not generic miscalibration.

    Both are EMA-smoothed per marginal. When ``trigger_on_overconfidence`` is
    true, ``trainer.should_stop`` is set as soon as the EMA-smoothed ``D`` and
    ``T`` exceed ``d_threshold`` and ``t_threshold`` for ``patience``
    consecutive evaluations on **any** marginal. With the default
    ``trigger_on_overconfidence=False`` (monitoring mode) the callback only
    logs ``pp_ks/D/{label}`` and ``pp_ks/T/{label}`` to TensorBoard.

    The ``stop_reason`` attribute records which marginal triggered the stop.
    """

    def __init__(
        self,
        test_loader: DataLoader,
        marginals_1d_info: list,
        ngrid_points: int = 50,
        warmup_epochs: int = 50,
        run_every_n_epochs: int = 1,
        patience: int = 40,
        ema_alpha: float = 0.3,
        d_threshold: float = 0.15,
        t_threshold: float = 0.15,
        t_quantile: float = 0.05,
        trigger_on_overconfidence: bool = False,
        print_every: int = 20,
        state_path: str | None = None,
        round_idx: int | None = None,
        plots_dir: str | None = None,
        compute_lambda_tau: bool = False,
        marginals_2d_info: list | None = None,
        datagen_conf: dict | None = None,
        fisher_varying_params: list | None = None,
        fisher_backend: str = "cpu",
        lt_h5_path: str | None = None,
    ):
        super().__init__()
        # ``test_loader`` is built once by the caller (typically wrapping a
        # ``Subset`` of the data module's test split) and reused every epoch
        # so the rank distributions are comparable across time.
        self.test_loader = test_loader
        self.marginals_1d_info = marginals_1d_info
        self.ngrid_points = ngrid_points
        self.warmup_epochs = warmup_epochs
        self.run_every_n_epochs = max(1, int(run_every_n_epochs))
        self.patience = patience
        self.ema_alpha = ema_alpha
        self.d_threshold = d_threshold
        self.t_threshold = t_threshold
        self.t_quantile = t_quantile
        self.trigger_on_overconfidence = trigger_on_overconfidence
        self.print_every = print_every

        # Per-marginal EMA and stall buffers are seeded from ``state_path`` if
        # present (multi-round TMNRE), else fresh.  ``cumulative_epoch_offset``
        # holds the sum of ``trainer.current_epoch + 1`` across all previously
        # completed rounds — it is what makes warmup and patience meaningful
        # across the campaign rather than reset per round.
        self.state_path = state_path
        # ``plots_dir`` is the run's plots root (typically
        # ``ROOT_DIR/plots/{TIME_OF_EXECUTION}``).  Overlay PP plots land in
        # the ``ppks_traces`` subdirectory; at trigger time a TRIGGER copy is
        # additionally placed at ``plots_dir`` for visibility.  None disables
        # PP-plot output (state/log only).
        self.plots_dir = plots_dir
        self._ema_d: dict[str, float] = {}
        self._ema_t: dict[str, float] = {}
        self._stall: dict[str, int] = {}
        self.history: list[dict] = []
        self.stop_reason: str = ""
        self.cumulative_epoch_offset: int = 0
        self.triggered: bool = False
        self.trigger_metadata: dict | None = None
        # ``round_idx`` is human-readable bookkeeping only — it's the current
        # round number the callback is attached to.  Saved state can carry a
        # stale value from the previous round, so the ctor arg always wins
        # when explicitly provided.
        self._round_idx: int | None = round_idx
        self._load_state()
        if round_idx is not None:
            self._round_idx = round_idx

        # λ_i / τ_i statistics (optional). When enabled, every evaluation
        # appends per-test-sample posterior moments to an HDF5 (one per round);
        # the Fisher denominator of τ is epoch-independent, computed once.
        self.compute_lambda_tau = bool(compute_lambda_tau)
        self.marginals_2d_info = marginals_2d_info or []
        self.datagen_conf = datagen_conf
        self.fisher_varying_params = fisher_varying_params or []
        self.fisher_backend = fisher_backend
        self.lt_h5_path = lt_h5_path
        self._lt_keys = _ORDERED_PRIOR_KEYS  # basis-aware names, set on first eval
        self._lt_truth_full = None        # (n_test, 11) true params, lazy
        self._lt_fisher_sigmas = None     # (n_test, n_varying), lazy
        self._lt_fisher_order = None      # varying-param name order
        self._lt_means: dict = {}         # {label: {param: [per-eval (n_test,)]}}
        self._lt_stds: dict = {}
        self._lt_cum_eps: list = []

    # ------------------------------------------------------------------ state
    def _load_state(self) -> None:
        """Seed cross-round state from ``self.state_path`` (if present)."""
        if not self.state_path or not os.path.exists(self.state_path):
            return
        with open(self.state_path) as f:
            s = yaml.safe_load(f) or {}
        self._ema_d = dict(s.get("ema_d", {}) or {})
        self._ema_t = dict(s.get("ema_t", {}) or {})
        self._stall = {k: int(v) for k, v in (s.get("stall", {}) or {}).items()}
        self.cumulative_epoch_offset = int(s.get("cumulative_epoch_offset", 0))
        self.triggered = bool(s.get("triggered", False))
        self.trigger_metadata = s.get("trigger_metadata") or None
        self._round_idx = s.get("round_idx")

    def _save_state(self) -> None:
        if not self.state_path:
            return
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        payload = {
            "ema_d": {k: float(v) for k, v in self._ema_d.items()},
            "ema_t": {k: float(v) for k, v in self._ema_t.items()},
            "stall": {k: int(v) for k, v in self._stall.items()},
            "cumulative_epoch_offset": int(self.cumulative_epoch_offset),
            "triggered": bool(self.triggered),
            "trigger_metadata": self.trigger_metadata,
            "round_idx": self._round_idx,
        }
        # Atomic write: tmp + rename so a kill mid-write can't corrupt state.
        tmp = self.state_path + ".tmp"
        with open(tmp, "w") as f:
            yaml.safe_dump(payload, f, sort_keys=False)
        os.replace(tmp, self.state_path)

    def _cum_ep(self, trainer) -> int:
        return int(trainer.current_epoch) + int(self.cumulative_epoch_offset)

    def on_validation_epoch_end(self, trainer, pl_module):
        cum_ep = self._cum_ep(trainer)
        if cum_ep < self.warmup_epochs:
            return
        # ``run_every_n_epochs`` is keyed off the cumulative axis so the
        # cadence is stable across rounds.
        if cum_ep % self.run_every_n_epochs != 0:
            return

        # Local imports keep the module light at import time.
        from scipy.stats import kstest

        was_training = pl_module.training
        try:
            pl_module.eval()
            self._evaluate_and_maybe_stop(trainer, pl_module, kstest)
        finally:
            if was_training:
                pl_module.train()

    def _evaluate_and_maybe_stop(self, trainer, pl_module, kstest):
        keys = _param_keys(pl_module)  # basis-aware parameter names
        self._lt_keys = keys  # cache for _write_lt_h5 (has no pl_module)
        per_marginal: dict[str, dict] = {}
        # All marginals whose stall counter has reached patience this
        # evaluation — we record every one of them, not just the first.
        triggered_labels: list[str] = []
        # Collected for the overlay PP plot at end of the evaluation.
        ranks_per_marginal: dict[str, np.ndarray] = {}
        # {label: (param_name, mean (n_test,), std (n_test,))} for λ/τ.
        moments_1d: dict = {}

        for label, in_idx, out_idx in self.marginals_1d_info:
            logratios, inj_params, grid = get_logratios_grid(
                self.test_loader, pl_module, self.ngrid_points,
                in_param_idx=in_idx, out_param_idx=out_idx,
            )
            # get_pvalues_1d expects grid shape (ngrid, 1) (as returned by
            # get_logratios_grid).  Pass through unchanged.
            ranks = get_pvalues_1d(logratios, grid, inj_params)
            ranks_per_marginal[label] = np.asarray(ranks)

            if self.compute_lambda_tau:
                mean, std = grid_posterior_moments_1d(logratios, grid)
                moments_1d[label] = (keys[in_idx], mean, std)

            D = float(kstest(ranks, "uniform").statistic)
            q = self.t_quantile
            T = float(np.mean((ranks < q) | (ranks > 1.0 - q)))

            if label not in self._ema_d:
                self._ema_d[label] = D
                self._ema_t[label] = T
                self._stall[label] = 0
            else:
                a = self.ema_alpha
                self._ema_d[label] = a * D + (1.0 - a) * self._ema_d[label]
                self._ema_t[label] = a * T + (1.0 - a) * self._ema_t[label]

            d_ema = self._ema_d[label]
            t_ema = self._ema_t[label]

            # Symmetric tail-mass deviation: |T - 2q| > t_threshold catches
            # narrow-biased (T → 1) AND narrow-unbiased (T → 0) overconfidence,
            # not just the high-T direction.
            t_baseline = 2.0 * self.t_quantile
            d_violates = d_ema > self.d_threshold
            t_violates = abs(t_ema - t_baseline) > self.t_threshold
            if d_violates or t_violates:
                self._stall[label] += 1
                violations = []
                if d_violates:
                    violations.append("D")
                if t_violates:
                    violations.append("T")
                violation_str = "|".join(violations)
            else:
                self._stall[label] = 0
                violation_str = ""

            per_marginal[label] = {
                "D": D, "T": T, "D_ema": d_ema, "T_ema": t_ema,
                "stall": self._stall[label],
                "violation": violation_str,
            }

            if (self.trigger_on_overconfidence
                    and self._stall[label] >= self.patience):
                triggered_labels.append(label)

        cum_ep = self._cum_ep(trainer)

        # Snapshot for offline analysis (cumulative axis).
        self.history.append({
            "cum_ep": cum_ep,
            "round": self._round_idx,
            "epoch_in_round": int(trainer.current_epoch),
            "per_marginal": per_marginal,
        })

        if self.compute_lambda_tau:
            self._update_lambda_tau(cum_ep, moments_1d, pl_module)
            self._check_tau_trigger(cum_ep, trainer=trainer)

        if trainer.logger is not None and per_marginal:
            metrics = {}
            for label, v in per_marginal.items():
                metrics[f"pp_ks/D/{label}"] = v["D"]
                metrics[f"pp_ks/T/{label}"] = v["T"]
                metrics[f"pp_ks/D_ema/{label}"] = v["D_ema"]
                metrics[f"pp_ks/T_ema/{label}"] = v["T_ema"]
            metrics["pp_ks/cumulative_epoch"] = float(cum_ep)
            trainer.logger.log_metrics(metrics, step=cum_ep)

        if cum_ep % self.print_every == 0 and per_marginal:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            parts = [
                f"{label}(D={v['D']:.3f},T={v['T']:.3f},s={v['stall']})"
                for label, v in sorted(per_marginal.items())
            ]
            mode = "trigger" if self.trigger_on_overconfidence else "monitor"
            print(f"[{ts}] [PPKS-{mode}] cumep {cum_ep} "
                  f"(round={self._round_idx}, epoch_in_round="
                  f"{int(trainer.current_epoch)}): "
                  f"{', '.join(parts)}", flush=True)

        # Persist state at every evaluation so a kill mid-round still leaves
        # a coherent record of EMA / stall on the cumulative axis.
        self._save_state()

        # Overlay PP plot: one figure per evaluation, all 1D marginals on
        # the same axes.  Filename keyed by the cumulative-epoch axis so a
        # single sorted listing reads chronologically across the campaign.
        overlay_path = None
        if self.plots_dir and ranks_per_marginal:
            overlay_path = os.path.join(
                self.plots_dir, "ppks_traces",
                f"cumep_{cum_ep:04d}_pp.png",
            )
            title = (f"P-P overlay — cum_ep={cum_ep} "
                     f"round={self._round_idx} "
                     f"epoch_in_round={int(trainer.current_epoch)}")
            pp_plot_overlay(ranks_per_marginal, overlay_path, title=title)

        # MONITOR + CHECKPOINT mode: when ``trigger_on_overconfidence`` is
        # true and at least one marginal reaches patience, we DO NOT stop
        # training (the user's other early-stop criteria — volume_ratio,
        # truth-missed — own the actual termination decision).  Instead we:
        #   * write a checkpoint at the trigger point so the model state is
        #     preserved for post-hoc inspection,
        #   * record structured trigger metadata,
        #   * log one clear line to stdout,
        #   * set ``self.triggered = True`` to gate against re-firing.
        if triggered_labels and not self.triggered:
            t_baseline = 2.0 * self.t_quantile
            parts = []
            marginal_records = []
            for lbl in triggered_labels:
                v = per_marginal[lbl]
                parts.append(
                    f"{lbl}(violation={v['violation']}, "
                    f"D_ema={v['D_ema']:.4f}/thr={self.d_threshold}, "
                    f"T_ema={v['T_ema']:.4f} (|T-{t_baseline:.2f}|="
                    f"{abs(v['T_ema'] - t_baseline):.4f})/thr={self.t_threshold}, "
                    f"stall={v['stall']}/{self.patience})"
                )
                marginal_records.append({
                    "name": lbl,
                    "violation": v["violation"],
                    "D_ema": float(v["D_ema"]),
                    "T_ema": float(v["T_ema"]),
                    "stall": int(v["stall"]),
                })
            triggered_reason = "; ".join(parts)
            self.stop_reason = triggered_reason
            self.triggered = True
            self.trigger_metadata = {
                "cumulative_epoch": int(cum_ep),
                "round": self._round_idx,
                "epoch_in_round": int(trainer.current_epoch),
                "d_threshold": float(self.d_threshold),
                "t_threshold": float(self.t_threshold),
                "t_baseline": float(t_baseline),
                "patience": int(self.patience),
                "marginals": marginal_records,
            }
            # Save the model exactly at the would-stop point.  Path lives
            # next to the regular round checkpoints so it surfaces in any
            # standard checkpoint scan.
            ckpt_dir = None
            if trainer.checkpoint_callback is not None:
                ckpt_dir = getattr(trainer.checkpoint_callback, "dirpath", None)
            if ckpt_dir is None and trainer.logger is not None:
                ckpt_dir = os.path.join(trainer.logger.log_dir, "checkpoints")
            if ckpt_dir is not None:
                ckpt_path = os.path.join(
                    ckpt_dir, f"ppks_trigger_cumep_{cum_ep}.ckpt",
                )
                os.makedirs(ckpt_dir, exist_ok=True)
                trainer.save_checkpoint(ckpt_path)
                self.trigger_metadata["checkpoint_path"] = ckpt_path
            # At trigger time, also drop a TRIGGER-named copy of the overlay
            # PP plot at the plots root (not buried in ppks_traces/).
            if self.plots_dir and ranks_per_marginal:
                trigger_overlay_path = os.path.join(
                    self.plots_dir,
                    f"TRIGGER_cumep_{cum_ep:04d}_pp.png",
                )
                title = (f"P-P overlay at PPKS trigger — cum_ep={cum_ep} "
                         f"round={self._round_idx} "
                         f"epoch_in_round={int(trainer.current_epoch)}")
                pp_plot_overlay(ranks_per_marginal, trigger_overlay_path,
                                title=title)
                self.trigger_metadata["overlay_path"] = trigger_overlay_path
            print(f"[PPKS-TRIGGER] cumep={cum_ep} round={self._round_idx} "
                  f"epoch_in_round={int(trainer.current_epoch)} — "
                  f"would-stop reasons: {triggered_reason}", flush=True)
            # Persist now so the trigger info is on disk before any further
            # training churn.
            self._save_state()
    
    def _check_tau_trigger(self, cum_ep, trainer):
        for label, params in self._lt_stds.items():
            for param, std_list in params.items():
                std = std_list[-1]                       # this eval, (n_test,)
                fis = self._lt_fisher_sigma_for(param)   # (n_test,) maybe NaN
                tau = np.abs(std / fis)                          # or std**2/fis**2
                # fraction below 1 → stall → marker
                frac_below_1 = np.mean(tau<1)
                
    # ------------------------------------------------------------ λ / τ stats
    def _lt_ensure_fisher(self):
        """Gather the test-set truths and the epoch-independent Fisher σ once."""
        if self._lt_truth_full is not None:
            return
        truths, wave0 = [], None
        for batch in self.test_loader:
            truths.append(np.asarray(batch["source_parameters"], dtype=np.float64))
            if wave0 is None:
                wave0 = np.asarray(batch["wave_fd"][0])
        self._lt_truth_full = np.concatenate(truths, axis=0)   # (n_test, 11)

        if self.fisher_varying_params and self.datagen_conf is not None:
            print(f"[λτ] computing Fisher σ for {self._lt_truth_full.shape[0]} "
                  f"test points (backend={self.fisher_backend}) ...", flush=True)
            self._lt_fisher_sigmas, self._lt_fisher_order = (
                compute_fisher_sigmas_for_testset(
                    self.datagen_conf, self._lt_truth_full,
                    self.fisher_varying_params, wave_fd_check=wave0,
                    backend=self.fisher_backend,
                )
            )
        else:
            self._lt_fisher_order = []

    def _lt_fisher_sigma_for(self, param):
        """Fisher σ column for *param* (NaN if it was not in the Fisher set)."""
        n_test = self._lt_truth_full.shape[0]
        if self._lt_fisher_sigmas is not None and param in self._lt_fisher_order:
            return self._lt_fisher_sigmas[:, self._lt_fisher_order.index(param)]
        return np.full(n_test, np.nan)

    def _lt_append(self, label, param, mean, std):
        # equivalent to self._lt_means[label][param].append(mean|std) , but include key existence checks. 
        self._lt_means.setdefault(label, {}).setdefault(param, []).append(mean)
        self._lt_stds.setdefault(label, {}).setdefault(param, []).append(std)

    def _update_lambda_tau(self, cum_ep, moments_1d, pl_module):
        self._lt_ensure_fisher()
        self._lt_cum_eps.append(int(cum_ep))
        keys = _param_keys(pl_module)  # basis-aware parameter names

        for label, (param, mean, std) in moments_1d.items():
            self._lt_append(label, param, mean, std)

        for label, in_idx, out_idx in self.marginals_2d_info:
            logratios, _, gx, gy = get_logratios_grid_2d(
                self.test_loader, pl_module, self.ngrid_points,
                out_param_idx=out_idx, in_param_idx=in_idx,
            )
            m0, s0, m1, s1 = grid_posterior_moments_2d(logratios, gx, gy)
            self._lt_append(label, keys[in_idx[0]], m0, s0)
            self._lt_append(label, keys[in_idx[1]], m1, s1)

        if self.lt_h5_path:
            self._write_lt_h5()

    def _write_lt_h5(self):
        import h5py
        os.makedirs(os.path.dirname(self.lt_h5_path), exist_ok=True)
        tmp = self.lt_h5_path + ".tmp"
        with h5py.File(tmp, "w") as f:
            f.attrs["round"] = -1 if self._round_idx is None else int(self._round_idx)
            f.create_dataset("cum_ep", data=np.asarray(self._lt_cum_eps))
            for label, params in self._lt_means.items():
                g = f.create_group(label)
                for param, mean_list in params.items():
                    pg = g.create_group(param)
                    pg.create_dataset("posterior_mean", data=np.stack(mean_list))
                    pg.create_dataset("posterior_std",
                                      data=np.stack(self._lt_stds[label][param]))
                    idx = self._lt_keys.index(param)
                    pg.create_dataset("ground_truth", data=self._lt_truth_full[:, idx])
                    pg.create_dataset("fisher_sigma",
                                      data=self._lt_fisher_sigma_for(param))
        os.replace(tmp, self.lt_h5_path)

    # --------------------------------------------------------- round-boundary
    def on_train_end(self, trainer, pl_module):
        """Bump the cumulative offset by this round's epoch count and save."""
        # Lightning sets ``trainer.current_epoch`` to the last completed epoch
        # before ``on_train_end``, so we add +1 to convert "last index" to
        # "epoch count" (matches what _cum_ep would have returned next round).
        self.cumulative_epoch_offset += int(trainer.current_epoch) + 1
        self._save_state()