import os
import yaml
import numpy as np
import torch
from pembhb import ROOT_DIR, get_numpy_dtype, PLOTS_ROOT_DIR
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
from pembhb.regions import (
    Region, region_from_equal_tailed, region_from_main_mode, region_from_hpd,
    prev_keep_mask, truncation_region
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


def _cupy_pool_gb():
    try:
        import cupy
        return cupy.get_default_memory_pool().total_bytes() / 2**30
    except Exception:
        return float("nan")


class GPUMemoryProfiler(Callback):
    """Per-phase peak of ``torch.cuda.max_memory_allocated`` (GB), printed each epoch.

    Phases: the train step, the val step, and every epoch-level hook of the other
    callbacks (wrapped in place). bbhx allocates through CuPy's pool, which torch
    does not see, so that pool's size is printed alongside. Place first in the list.
    """

    WRAPPED_HOOKS = ("on_validation_epoch_end", "on_train_epoch_end", "on_train_end")

    def __init__(self, callbacks):
        self.peaks = {}
        for cb in callbacks:
            for hook in self.WRAPPED_HOOKS:
                if getattr(type(cb), hook) is not getattr(Callback, hook):
                    setattr(cb, hook, self._wrap(f"{type(cb).__name__}.{hook}", getattr(cb, hook)))

    def _record(self, name):
        gb = torch.cuda.max_memory_allocated() / 2**30
        self.peaks[name] = max(self.peaks.get(name, 0.0), gb)

    def _wrap(self, name, fn):
        def wrapped(*args, **kwargs):
            torch.cuda.reset_peak_memory_stats()
            try:
                return fn(*args, **kwargs)
            finally:
                self._record(name)
        return wrapped

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        torch.cuda.reset_peak_memory_stats()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._record("train_step")

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        torch.cuda.reset_peak_memory_stats()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self._record("val_step")

    def _dump(self, label):
        if self.peaks:
            parts = ", ".join(f"{k}={v:.2f}" for k, v in sorted(self.peaks.items(), key=lambda kv: -kv[1]))
            print(f"[GPUMem] {label} peak alloc GB: {parts} | "
                  f"torch reserved={torch.cuda.memory_reserved() / 2**30:.2f} "
                  f"cupy pool={_cupy_pool_gb():.2f}", flush=True)
        self.peaks = {}

    def on_train_epoch_start(self, trainer, pl_module):
        self._dump(f"epoch {trainer.current_epoch - 1}")

    def on_train_end(self, trainer, pl_module):
        self._dump(f"epoch {trainer.current_epoch} (last)")


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


class _CadencedCallback(Callback):
    """Fires ``_fire(trainer, pl_module, tag, tag_is_step)`` every
    ``call_every_n_epochs`` validation epochs (after ``warmup_epochs``), or every
    ``call_every_n_steps`` global steps when step cadence is on, and once more at
    the end of training."""

    def __init__(self, call_every_n_epochs=1, warmup_epochs=0,
                 call_every_n_steps=None, warmup_steps=None):
        self.call_every_n_epochs = call_every_n_epochs
        self.warmup_epochs = warmup_epochs
        self._step = _StepCadence(call_every_n_steps)
        self.warmup_steps = warmup_steps
        self.epochs_elapsed = 0

    def _fire(self, trainer, pl_module, tag, tag_is_step):
        raise NotImplementedError

    def on_validation_epoch_end(self, trainer, pl_module):
        if self._step.enabled:
            return  # step-mode: handled in on_train_batch_end
        if trainer.current_epoch < self.warmup_epochs:
            return
        self.epochs_elapsed += 1
        if (self.epochs_elapsed - 2) % self.call_every_n_epochs == 0:
            self._fire(trainer, pl_module, trainer.current_epoch, False)

    def _fire_in_eval(self, trainer, pl_module):
        # mid-batch the model is in train mode (dropout active) -> eval for the
        # diagnostic forward passes, then restore.
        was_training = pl_module.training
        pl_module.eval()
        try:
            with torch.no_grad():
                self._fire(trainer, pl_module, trainer.global_step, True)
        finally:
            if was_training:
                pl_module.train()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self._step.enabled:
            return
        if self.warmup_steps is not None and trainer.global_step < self.warmup_steps:
            return
        if not self._step.should_fire(trainer.global_step):
            return
        self._fire_in_eval(trainer, pl_module)

    def on_train_end(self, trainer, pl_module):
        if self._step.enabled:
            self._fire_in_eval(trainer, pl_module)
        else:
            self.on_validation_epoch_end(trainer, pl_module)


class PlotPosteriorCallback(_CadencedCallback):
    """Figures of the posterior at the observation during training.

    Writes ``posterior_evolution/posterior_round_<r>_<epoch|step>_<tag>_<param>.pdf``
    and shades where the density is zeroed outside the previous round's accepted
    set (§2). Metrics and the proposal live in :class:`TruncationMonitor`.
    """

    def __init__(self, timestamp: str, obs_loader: DataLoader, input_idx_list: list,
                 output_idx_list: list, round_idx: int, call_every_n_epochs=1,
                 training_start_time: datetime = None, print_every: int = 20,
                 warmup_epochs: int = 0, call_every_n_steps=None, warmup_steps=None,
                 box_dilation_1d: float = 1.0, eps_1d: float = 1e-4,
                 prev_accepted: dict = None, zero_policy: str = "off"):
        super().__init__(call_every_n_epochs, warmup_epochs, call_every_n_steps, warmup_steps)
        # §2 — the domain per marginal ({marginal_key: {"kind": "1d"/"2d", ...}}),
        # shared with the Truncator, and the active zeroing policy.
        self.prev_accepted = prev_accepted if prev_accepted is not None else {}
        self.zero_policy = str(zero_policy)
        self.box_dilation_1d = box_dilation_1d
        self.eps_1d = eps_1d
        self.print_every = print_every
        self.timestamp = timestamp
        self.obs_loader = obs_loader
        self.input_idx_list = input_idx_list
        self.output_idx_list = output_idx_list
        self.n_marginals = len(input_idx_list)
        self.init_time = datetime.now()
        self.training_start_time = training_start_time if training_start_time is not None else self.init_time
        self.round_idx = round_idx

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

    def _fire(self, trainer, pl_module, tag, tag_is_step):
        tag_kind = "step" if tag_is_step else "epoch"
        outdir = os.path.join(PLOTS_ROOT_DIR, self.timestamp, "posterior_evolution")
        os.makedirs(outdir, exist_ok=True)
        train_time = datetime.now() - self.training_start_time
        td_trunc = train_time - timedelta(microseconds=train_time.microseconds)
        title = f"Round {self.round_idx} - {tag_kind} {tag} - training time={td_trunc}s"
        keys = _param_keys(pl_module)  # basis-aware parameter names
        for in_param_idx, out_param_idx in zip(self.input_idx_list, self.output_idx_list):
            marginal_key = tuple(in_param_idx)
            if len(in_param_idx) == 1:
                param_idx = in_param_idx[0]
                fig, ax = plt.subplots(1, 1, figsize=(6, 4))
                fig.suptitle(title, fontsize=10)
                try:
                    interval, norm1d, grid, inj_params = get_widest_interval_1d(
                        pl_module, self.obs_loader, in_param_idx=param_idx,
                        out_param_idx=out_param_idx, eps=self.eps_1d,
                        dilation=self.box_dilation_1d)
                    g = grid.flatten()
                    ax.plot(g, norm1d, 'b-', linewidth=1.5)
                    ax.axvline(inj_params[0], color='r', linestyle='--', label='Injection')
                    ax.axvline(interval[0], color='g', linestyle=':', label=f'{100*(1-self.eps_1d):.2f}% CI')
                    ax.axvline(interval[1], color='g', linestyle=':')
                    ax.fill_between(g, 0, norm1d, where=(g >= interval[0]) & (g <= interval[1]),
                                    alpha=0.3, color='green')
                    self._shade_suppressed_1d(ax, grid[:, 0], marginal_key)
                    ax.set_xlabel(keys[param_idx])
                    ax.set_ylabel('Posterior density')
                    ax.legend()
                    fig.savefig(os.path.join(
                        outdir, f"posterior_round_{self.round_idx}_{tag_kind}_{tag}_{keys[param_idx]}.pdf"),
                        bbox_inches="tight")
                except Exception as e:
                    print(f"Error plotting 1D marginal for {keys[param_idx]}: {e}")
                finally:
                    plt.close(fig)
            elif len(in_param_idx) == 2:
                fig, ax = plt.subplots(1, 1, figsize=(4, 4))
                fig.tight_layout()
                fig.suptitle(title, fontsize=10)
                names = [keys[in_param_idx[0]], keys[in_param_idx[1]]]
                label = "-".join(names)
                try:
                    norm_2d, inj_params, gx, gy, dp0, dp1 = eval_posterior_2d(
                        pl_module, self.obs_loader, in_param_idx, out_param_idx)
                    posterior_heatmap_2d(gx, gy, norm_2d, inj_params[0], ax, names)
                    self._shade_suppressed_2d(ax, gx, gy, marginal_key)
                    # matplotlib raises ValueError when contour levels are not
                    # strictly increasing (flat / degenerate posteriors): keep
                    # the heatmap and skip the contours.
                    try:
                        levels, labels = contour_levels(norm_2d)
                        _, cs = contour_boxes(gx, gy, norm_2d, levels, ax=ax)
                        ax.clabel(cs, fmt={lev: f"{p:.3f}" for lev, p in zip(levels, labels)},
                                  fontsize=8)
                        if marginal_key == (7, 8):
                            self._log_sky_contour_ratios(norm_2d, gx, gy, dp0, dp1,
                                                         trainer, "_".join(names))
                    except ValueError as ve:
                        print(f"Round {self.round_idx}, {tag_kind} {tag}, {label}: "
                              f"contour overlay failed ({ve}); saving heatmap without contours.",
                              flush=True)
                    fig.savefig(os.path.join(
                        outdir, f"posterior_round_{self.round_idx}_{tag_kind}_{tag}_{names[0]}_{names[1]}.pdf"),
                        bbox_inches="tight")
                finally:
                    plt.close(fig)

    def on_train_end(self, trainer, pl_module):
        super().on_train_end(trainer, pl_module)
        print(f"Total training time: {datetime.now() - self.init_time}")


class TruncationMonitor(_CadencedCallback):
    """The proposal the round would make now, and the metrics that end a round.

    At each fire: ``truncator.propose`` (the same call the round end makes, not
    committed), the ``volume_ratio`` of each marginal's proposal against its
    domain (:func:`pembhb.truncator.volume_ratio`) and the differential entropy
    of each marginal at the observation. ``volume_ratios`` and
    ``differential_entropies`` (lists of per-fire dicts, keyed by marginal) are
    what ``VolumeRatioEarlyStopping`` / ``DifferentialEntropyEarlyStopping`` and
    the trainer read; ``latest`` is the last proposal.
    """

    def __init__(self, truncator, obs_loader, marginals, prior, round_idx,
                 call_every_n_epochs=1, warmup_epochs=0, call_every_n_steps=None,
                 warmup_steps=None, print_every=20, eps_1d=1e-4, box_dilation_1d=1.0):
        super().__init__(call_every_n_epochs, warmup_epochs, call_every_n_steps, warmup_steps)
        self.truncator = truncator
        self.obs_loader = obs_loader
        self.marginals = [(tuple(k), int(o)) for k, o in marginals]   # (marginal_key, out_idx)
        self.prior = prior
        self.round_idx = round_idx
        self.print_every = print_every
        self.eps_1d = eps_1d
        self.box_dilation_1d = box_dilation_1d
        self.volume_ratios = {}
        self.differential_entropies = {}
        self.latest = None

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

    def _coarse_entropy(self, pl_module, key, out_idx):
        """Differential entropy on the coarse box-wide grid, and that density."""
        if len(key) == 1:
            _, norm1d, grid, _ = get_widest_interval_1d(
                pl_module, self.obs_loader, in_param_idx=key[0], out_param_idx=out_idx,
                eps=self.eps_1d, dilation=self.box_dilation_1d)
            return (self._differential_entropy_1d(norm1d, float(grid[1, 0] - grid[0, 0])),
                    norm1d, (grid[:, 0],))
        norm_2d, _, gx, gy, dp0, dp1 = eval_posterior_2d(pl_module, self.obs_loader, key, out_idx)
        return (self._differential_entropy_2d(norm_2d, dp0, dp1),
                norm_2d, (gx[0, :], gy[:, 0]))

    def _fire(self, trainer, pl_module, tag, tag_is_step):
        from pembhb.truncator import box_volume, volume_ratio
        tag_kind = "step" if tag_is_step else "epoch"
        do_print = tag_is_step or (trainer.current_epoch % self.print_every == 0)
        keys = _param_keys(pl_module)
        tr = self.truncator
        proposal = tr.propose(pl_module, self.obs_loader, self.marginals, self.prior)
        self.latest = proposal
        for key, out_idx in self.marginals:
            name = "-".join(keys[i] for i in key)
            tb_name = name if len(key) == 1 else "_".join(keys[i] for i in key)
            try:
                entropy, density, grids = self._coarse_entropy(pl_module, key, out_idx)
                region = proposal.regions.get(key)
                if region is None:
                    # rectangle mode: the posterior's HPD region on the coarse grid
                    names = [keys[i] for i in key]
                    region, _ = truncation_region(
                        density, grids, None, policy="off", hysteresis_weight=None,
                        credible=float(tr.conf.get("credible_level_1d" if len(key) == 1
                                                   else "credible_level_2d", 0.997)),
                        dilation=float(tr.conf.get("dilation_1d" if len(key) == 1
                                                   else "dilation_2d", 1.2)),
                        periods=tuple(_MASK_PERIOD_BY_NAME.get(n) for n in names),
                        clip=False)
                ratio, post_vol, prop_vol, mode_ratios = volume_ratio(
                    region, tr.prev_accepted.get(key), box_volume(self.prior, key, keys))
            except ValueError as ve:
                print(f"Round {self.round_idx}, {tag_kind} {tag}, {name}: "
                      f"truncation metric failed ({ve}).", flush=True)
                continue
            self.volume_ratios.setdefault(key, []).append({
                'epoch': trainer.current_epoch, 'step': trainer.global_step,
                'ratio': ratio, 'posterior_volume': post_vol, 'prior_volume': prop_vol,
                'mode_ratios': mode_ratios})
            self.differential_entropies.setdefault(key, []).append({
                'epoch': trainer.current_epoch, 'step': trainer.global_step,
                'entropy': entropy})
            kept = proposal.kept.get(key)
            if trainer.logger is not None:
                metrics = {f"volume_ratio/{tb_name}": ratio, f"diff_entropy/{tb_name}": entropy}
                if kept is not None:
                    metrics[f"truncation/kept/{tb_name}"] = kept
                metrics.update({f"volume_ratio_mode/{tb_name}/{k}": r
                                for k, r in enumerate(mode_ratios)})
                trainer.logger.log_metrics(metrics, step=tag)
            if do_print:
                print(f"Round {self.round_idx}, {tag_kind} {tag}, {name}: "
                      f"vol_ratio={ratio:.4f} (post={post_vol:.3e}, prior={prop_vol:.3e}), "
                      f"modes={[round(r, 3) for r in mode_ratios]}, "
                      + ("" if kept is None else f"kept={kept:.5f}, ")
                      + f"H={entropy:.4f} nats", flush=True)


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

    def _find_plot_callback(self, trainer) -> "TruncationMonitor | None":
        for cb in trainer.callbacks:
            if isinstance(cb, TruncationMonitor):
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

    def _find_plot_callback(self, trainer) -> "TruncationMonitor | None":
        for cb in trainer.callbacks:
            if isinstance(cb, TruncationMonitor):
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
    * 2D: all components of the HPD level set (:func:`~pembhb.regions.region_from_hpd`),
      as mask-mode truncation keeps them, with each axis's periodicity resolved
      from its parameter name.

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
            region = region_from_hpd(
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


class CalibrationMonitor(Callback):
    """Calibration monitor on a fixed test pool (no early stopping).

    Every ``run_every_n_epochs`` epochs it evaluates the 1-D marginal posteriors
    on ``test_n`` simulations of the round's test split and, per marginal, the
    ranks ``r_i = F̂_post(θ_true^i)`` of the true values:

    * ``D`` — Kolmogorov-Smirnov distance of the ranks from ``Uniform(0, 1)``
      (0 = calibrated; any miscalibration raises it).
    * ``T`` — tail mass ``P[r ∈ [0, q] ∪ [1-q, 1]]``; ``2q`` when calibrated,
      larger when overconfident, smaller when underconfident.

    Both are EMA-smoothed after ``warmup_epochs`` (cumulative over rounds) and
    logged to TensorBoard as ``pp_ks/{D,T,D_ema,T_ema}/<param>``.
    ``d_threshold``/``t_threshold`` are reference levels for the plots only.
    Optionally also records the λ/τ statistics (see :meth:`_update_lambda_tau`).
    """

    def __init__(
        self,
        test_loader: DataLoader,
        marginals_1d_info: list,
        ngrid_points: int = 50,
        warmup_epochs: int = 50,
        run_every_n_epochs: int = 1,
        ema_alpha: float = 0.3,
        d_threshold: float = 0.15,
        t_threshold: float = 0.15,
        t_quantile: float = 0.05,
        print_every: int = 20,
        state_path: str | None = None,
        round_idx: int | None = None,
        plots_dir: str | None = None,
        compute_lambda_tau: bool = False,
        marginals_2d_info: list | None = None,
        datagen_conf: dict | None = None,
        fisher_varying_params: list | None = None,
        fisher_backend: str = "cpu",
        fisher_chunk_size: int = 25,
        lt_h5_path: str | None = None,
        lt_warmup_epochs: int | None = None,
    ):
        super().__init__()
        # ``test_loader`` is built once by the caller (typically wrapping a
        # ``Subset`` of the data module's test split) and reused every epoch
        # so the rank distributions are comparable across time.
        self.test_loader = test_loader
        self.marginals_1d_info = marginals_1d_info
        self.ngrid_points = ngrid_points
        self.warmup_epochs = warmup_epochs
        # λ/τ stats may start earlier than the EMA; below ``warmup_epochs``
        # the callback runs in stats-only mode.
        self.lt_warmup_epochs = (warmup_epochs if lt_warmup_epochs is None
                                 else int(lt_warmup_epochs))
        self.run_every_n_epochs = max(1, int(run_every_n_epochs))
        self.ema_alpha = ema_alpha
        self.d_threshold = d_threshold
        self.t_threshold = t_threshold
        self.t_quantile = t_quantile
        self.print_every = print_every

        # Per-marginal EMAs are seeded from ``state_path`` if present
        # (multi-round TMNRE), else fresh.  ``cumulative_epoch_offset`` holds
        # the sum of ``trainer.current_epoch + 1`` across all previously
        # completed rounds, so warmup and the EMA run on the campaign axis.
        self.state_path = state_path
        # ``plots_dir`` is the run's plots root; overlay PP plots land in its
        # ``ppks_traces`` subdirectory.  None disables PP-plot output.
        self.plots_dir = plots_dir
        self._ema_d: dict[str, float] = {}
        self._ema_t: dict[str, float] = {}
        self.history: list[dict] = []
        self.cumulative_epoch_offset: int = 0
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
        self.fisher_chunk_size = int(fisher_chunk_size)
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
        self.cumulative_epoch_offset = int(s.get("cumulative_epoch_offset", 0))
        self._round_idx = s.get("round_idx")

    def _save_state(self) -> None:
        if not self.state_path:
            return
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        payload = {
            "ema_d": {k: float(v) for k, v in self._ema_d.items()},
            "ema_t": {k: float(v) for k, v in self._ema_t.items()},
            "cumulative_epoch_offset": int(self.cumulative_epoch_offset),
            "d_threshold": float(self.d_threshold),
            "t_threshold": float(self.t_threshold),
            "t_quantile": float(self.t_quantile),
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
        if trainer.sanity_checking:     # would evaluate twice at the round's first step
            return
        cum_ep = self._cum_ep(trainer)
        gate = (min(self.warmup_epochs, self.lt_warmup_epochs)
                if self.compute_lambda_tau else self.warmup_epochs)
        if cum_ep < gate:
            return
        # Below warmup_epochs only the raw statistics (and λ/τ) are collected:
        # no EMA, no overlay plot.
        stats_only = cum_ep < self.warmup_epochs
        # ``run_every_n_epochs`` is keyed off the cumulative axis so the
        # cadence is stable across rounds.
        if cum_ep % self.run_every_n_epochs != 0:
            return

        # Local imports keep the module light at import time.
        from scipy.stats import kstest

        was_training = pl_module.training
        try:
            pl_module.eval()
            self._evaluate(trainer, pl_module, kstest, stats_only=stats_only)
        finally:
            if was_training:
                pl_module.train()

    def _evaluate(self, trainer, pl_module, kstest, stats_only=False):
        keys = _param_keys(pl_module)  # basis-aware parameter names
        self._lt_keys = keys  # cache for _write_lt_h5 (has no pl_module)
        per_marginal: dict[str, dict] = {}
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

            if stats_only:
                # Raw statistics only; the EMA still starts from the first
                # post-warmup evaluation.
                per_marginal[label] = {"D": D, "T": T}
                continue

            if label not in self._ema_d:
                self._ema_d[label] = D
                self._ema_t[label] = T
            else:
                a = self.ema_alpha
                self._ema_d[label] = a * D + (1.0 - a) * self._ema_d[label]
                self._ema_t[label] = a * T + (1.0 - a) * self._ema_t[label]

            per_marginal[label] = {"D": D, "T": T,
                                   "D_ema": self._ema_d[label],
                                   "T_ema": self._ema_t[label]}

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

        if trainer.logger is not None and per_marginal:
            metrics = {}
            for label, v in per_marginal.items():
                metrics[f"pp_ks/D/{label}"] = v["D"]
                metrics[f"pp_ks/T/{label}"] = v["T"]
                if "D_ema" in v:
                    metrics[f"pp_ks/D_ema/{label}"] = v["D_ema"]
                    metrics[f"pp_ks/T_ema/{label}"] = v["T_ema"]
            metrics["pp_ks/cumulative_epoch"] = float(cum_ep)
            trainer.logger.log_metrics(metrics, step=cum_ep)

        if cum_ep % self.print_every == 0 and per_marginal:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            parts = [f"{label}(D={v['D']:.3f},T={v['T']:.3f})"
                     for label, v in sorted(per_marginal.items())]
            print(f"[{ts}] [calibration] cumep {cum_ep} "
                  f"(round={self._round_idx}, epoch_in_round="
                  f"{int(trainer.current_epoch)}): "
                  f"{', '.join(parts)}", flush=True)

        # Persist state at every evaluation so a kill mid-round still leaves
        # a coherent record of the EMA on the cumulative axis.
        self._save_state()

        # Overlay PP plot: one figure per evaluation, all 1D marginals on
        # the same axes.  Filename keyed by the cumulative-epoch axis so a
        # single sorted listing reads chronologically across the campaign.
        if self.plots_dir and ranks_per_marginal and not stats_only:
            overlay_path = os.path.join(
                self.plots_dir, "ppks_traces",
                f"cumep_{cum_ep:04d}_pp.png",
            )
            title = (f"P-P overlay — cum_ep={cum_ep} "
                     f"round={self._round_idx} "
                     f"epoch_in_round={int(trainer.current_epoch)}")
            pp_plot_overlay(ranks_per_marginal, overlay_path, title=title)

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
                  f"test points (backend={self.fisher_backend}, chunk={self.fisher_chunk_size}) ...", flush=True)
            self._lt_fisher_sigmas, self._lt_fisher_order = (
                compute_fisher_sigmas_for_testset(
                    self.datagen_conf, self._lt_truth_full,
                    self.fisher_varying_params, wave_fd_check=wave0,
                    backend=self.fisher_backend,
                    chunk_size=self.fisher_chunk_size,
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