"""TMNRE with joint Autoencoder + Inference Network training.

This script mirrors ``tmnre.py`` but replaces the sequential
"train AE → freeze encoder → train NRE" pipeline with a single
``JointAEInferenceNetwork`` that trains both objectives simultaneously
under one Lightning ``Trainer``.

Key difference from ``tmnre.py``:
- The AE reconstruction loss and the NRE contrastive loss share the
  same training loop.
- The encoder bottleneck is **detached** before it reaches the NRE
  heads, so the BCE loss never back-propagates through the encoder.
- A warm-up phase (configurable via ``ae_warmup_epochs``) trains only
  the AE for the first N epochs of round 1; in subsequent rounds the
  warm-up is skipped because the encoder is already trained.
"""

import os, shutil, copy
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from glob import glob

from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint, Callback
from torch.utils.data import DataLoader, Subset

from pembhb.simulator import MBHBSimulatorFD_TD, MBHBSimulatorFD
from pembhb.model import JointAEInferenceNetwork
from pembhb.autoencoder import DenoisingAutoencoder, AutoencoderWrapper, MarginalEncoderTrainer, resolve_loss_band
from pembhb.data import MBHBDataModule, MBHBDataset, mbhb_collate_fn
from lightning.pytorch import seed_everything

from pembhb import ROOT_DIR, DATA_ROOT_DIR, set_precision
from pembhb import utils

# ---- reuse helpers from tmnre.py ----------------------------------------
from pembhb.utils import (
    validate_marginals,
    resolve_marginals_for_round,
    transfer_classifier_weights,
    get_widest_interval_1d,
    get_widest_box_2d,
    apply_pipeline_section,
)
from pembhb.callbacks import (
    PlotPosteriorCallback, VolumeRatioEarlyStopping,
    DifferentialEntropyEarlyStopping, PeriodicProgressCallback,
    WarmupEarlyStopping, PPKSTestEarlyStopping, ChainConvergenceMonitor,
    compute_truncation_coverage,
)
from pembhb.utils import _ORDERED_PRIOR_KEYS as _PPKS_ORDERED_PRIOR_KEYS
from pembhb.diagnostics import AutoencoderDiagnosticsCallback

def get_timestamp():
    return datetime.now().strftime("%Y/%m/%d")


def _round_marginal_entropies(plot_cb, keys=_PPKS_ORDERED_PRIOR_KEYS):
    """Round-final absolute differential entropy per marginal: ``{label: H}``.

    ``keys`` is the basis-aware parameter-name list for the run.
    """
    if plot_cb is None or not getattr(plot_cb, "differential_entropies", None):
        return {}
    out = {}
    for key, hist in plot_cb.differential_entropies.items():
        if hist:
            label = "-".join(keys[i] for i in key)
            out[label] = hist[-1]["entropy"]
    return out


def _round_volume_ratios(plot_cb):
    """Round-final posterior/prior volume ratio per marginal: ``{tuple: ratio}``.

    Keyed by the marginal's parameter-index tuple (e.g. ``(0,)``, ``(7, 8)``),
    matching ``model.marginals_dict`` so it can drive per-marginal reinit.
    """
    if plot_cb is None or not getattr(plot_cb, "volume_ratios", None):
        return {}
    out = {}
    for key, hist in plot_cb.volume_ratios.items():
        if hist:
            out[tuple(key)] = hist[-1]["ratio"]
    return out


def _round_entropies(plot_cb):
    """Round-final differential entropy per marginal: ``{tuple: H}`` (nats).

    Keyed by the marginal's parameter-index tuple, matching
    ``_round_volume_ratios`` so it can drive per-marginal entropy-plateau reinit.
    """
    if plot_cb is None or not getattr(plot_cb, "differential_entropies", None):
        return {}
    out = {}
    for key, hist in plot_cb.differential_entropies.items():
        if hist:
            out[tuple(key)] = hist[-1]["entropy"]
    return out


def _entropy_plateau_keys(history, min_delta, patience, warmup_rounds, round_idx):
    """Marginal tuples whose end-of-round differential entropy has plateaued.

    A marginal plateaus when its last ``patience`` round-to-round changes in H
    are *all* smaller than ``min_delta`` nats. Rounds up to ``warmup_rounds`` are
    skipped: early on many heads sit flat at the prior entropy because they have
    not started learning yet (e.g. ``phi`` stays at ~1.85 nats until it suddenly
    drops) — that is a false plateau we must not mistake for convergence.

    ``history`` is ``{tuple: [H_1, H_2, ...]}`` accumulated one value per round.
    """
    if round_idx <= warmup_rounds:
        return set()
    out = set()
    for key, hist in history.items():
        if len(hist) < patience + 1:
            continue
        diffs = np.abs(np.diff(hist[-(patience + 1):]))
        if np.all(diffs < min_delta):
            out.add(tuple(key))
    return out


def _marginal_prior_volume(marginal, prior, ordered_keys):
    """Prior-box volume of a marginal = product of its parameters' widths.

    For a 1D marginal ``(i,)`` this is just the width of parameter ``i``; for a
    2D marginal ``(i, j)`` it is the box area. ``prior`` is ``{name: [lo, hi]}``
    and ``ordered_keys`` maps a parameter index to its name for the run's spin
    basis (see ``utils.ordered_prior_keys``).
    """
    vol = 1.0
    for idx in marginal:
        lo, hi = prior[ordered_keys[idx]]
        vol *= abs(float(hi) - float(lo))
    return vol


def _round_median_tau(lt_h5_path):
    """Median τ = σ_post²/σ_Fisher² over all params/samples at the round's last eval."""
    if not lt_h5_path or not os.path.exists(lt_h5_path):
        return None
    import h5py
    taus = []
    with h5py.File(lt_h5_path, "r") as f:
        for label in f:
            if label == "cum_ep":
                continue
            for param in f[label]:
                pg = f[label][param]
                std = pg["posterior_std"][-1]          # last eval, (n_test,)
                fis = pg["fisher_sigma"][:]
                with np.errstate(divide="ignore", invalid="ignore"):
                    taus.append(std ** 2 / fis ** 2)
    if not taus:
        return None
    allt = np.concatenate(taus)
    allt = allt[np.isfinite(allt)]
    return float(np.median(allt)) if allt.size else None


def _check_obs_matches_datagen(obs_path, datagen_conf):
    """Verify the obs sidecar YAML's waveform_params matches the training datagen config.

    The obs HDF5 file is expected to ship with a sibling ``.yaml`` describing the
    grid it was generated on. If that grid disagrees with the current
    ``datagen_config.yaml``, the obs lives on a different frequency basis than
    the training data and inference is silently wrong — so we abort.
    """
    import yaml as _yaml
    sidecar = obs_path[:-3] + ".yaml" if obs_path.endswith(".h5") else obs_path + ".yaml"
    if not os.path.exists(sidecar):
        raise RuntimeError(
            f"[Obs] sidecar YAML not found at {sidecar}. The observation file must be "
            f"accompanied by a .yaml describing how it was generated (waveform_params, prior, ...)."
        )
    with open(sidecar) as f:
        obs_meta = _yaml.safe_load(f)
    obs_wp = (obs_meta or {}).get("conf", {}).get("waveform_params", {}) or {}
    train_wp = datagen_conf.get("waveform_params", {}) or {}

    diffs = []
    for key in sorted(set(obs_wp) | set(train_wp)):
        if obs_wp.get(key) != train_wp.get(key):
            diffs.append((key, obs_wp.get(key), train_wp.get(key)))

    if diffs:
        lines = [f"  {k}: obs={ov!r}  train={tv!r}" for k, ov, tv in diffs]
        raise RuntimeError(
            f"[Obs] waveform_params mismatch between observation sidecar ({sidecar}) "
            f"and configs/datagen_config.yaml:\n" + "\n".join(lines) +
            "\nThe observation must be generated on the same waveform grid as the training data."
        )
    print(f"[Obs] waveform_params verified against {sidecar}")


def _discover_last_round(run_name, data_root_dir):
    """Return the index of the last completed round for *run_name*.

    Scans ``{data_root_dir}/logs/{run_name}/round_{i}/version_*/checkpoints/``
    for increasing *i* and returns the highest index for which a checkpoint
    file exists, or 0 if none are found.
    """
    log_root = os.path.join(data_root_dir, "logs")
    round_idx = 0
    while True:
        dirs = sorted(glob(os.path.join(log_root, run_name, f"round_{round_idx + 1}", "version_*")))
        if not dirs:
            break
        ckpts = sorted(glob(os.path.join(dirs[-1], "checkpoints", "*.ckpt")))
        if not ckpts:
            break
        round_idx += 1
    return round_idx




# -------------------------------------------------------------------------
# SequentialTrainerJoint  (mirrors SequentialTrainer from tmnre.py)
# -------------------------------------------------------------------------

class SequentialTrainerJoint:
    """Round-based TMNRE trainer using joint AE+NRE training."""

    def __init__(self, train_conf, datagen_conf, dataset_obs_path, resume=False, seed=42):
        self.train_conf = train_conf
        self.datagen_conf = datagen_conf
        # Untruncated initial prior — the reference for the very first
        # prior-shrink cold-restart comparison (see _train_joint).
        self._initial_prior = copy.deepcopy(datagen_conf["prior"])
        self.dataset_obs_path = dataset_obs_path
        self.seed = seed
        self.training_start = datetime.now()

        _check_obs_matches_datagen(dataset_obs_path, datagen_conf)

        validate_marginals(train_conf["marginals"])

        self.dataset_observation = Subset(
            MBHBDataset(dataset_obs_path, cache_in_memory=True), indices=[0]
        )
        obs_noise_scale = self.dataset_observation.dataset.noise_scale
        obs_td_params = self.dataset_observation.dataset.td_params
        # The obs HDF5 file MUST contain a stored ``noise_fd`` dataset
        # (see scripts/add_noise_to_obs.py).  Without it, the collate fn
        # draws a fresh torch.randn() realisation on every forward pass,
        # which makes PP-KS, posterior eval, and truncation non-reproducible.
        if not self.dataset_observation.dataset.has_stored_noise:
            raise RuntimeError(
                f"[Obs] '{dataset_obs_path}' has no stored 'noise_fd' dataset. "
                f"Training would draw fresh random noise on every call, which "
                f"breaks reproducibility of PP-KS, posterior eval and truncation. "
                f"Run scripts/add_noise_to_obs.py first, then point --obs-path "
                f"at the *_withnoise.h5 file."
            )

        # Persist the obs file used for this run so it can be recovered later
        # (the path is otherwise only in the launch command's stdout).
        try:
            import yaml as _yaml
            _obs_log_dir = os.path.join(DATA_ROOT_DIR, TIME_OF_EXECUTION)
            os.makedirs(_obs_log_dir, exist_ok=True)
            _obs_sidecar = (
                dataset_obs_path[:-3] + ".yaml"
                if dataset_obs_path.endswith(".h5") else dataset_obs_path + ".yaml"
            )
            with open(os.path.join(_obs_log_dir, "observation_used.yaml"), "w") as _f:
                _yaml.safe_dump(
                    {
                        "obs_path": os.path.abspath(dataset_obs_path),
                        "obs_sidecar": os.path.abspath(_obs_sidecar),
                        "has_stored_noise": True,
                    },
                    _f,
                )
            print(f"[Obs] recorded obs path → {_obs_log_dir}/observation_used.yaml")
        except Exception as _exc:
            print(f"[Obs] WARNING: could not write observation_used.yaml: {_exc}")
        self.dataloader_obs = DataLoader(
            self.dataset_observation,
            batch_size=1,
            shuffle=False,
            collate_fn=lambda b: mbhb_collate_fn(
                b, obs_noise_scale,
                noise_shuffling=False,
                noise_factor=self.train_conf["noise_factor"],
                td_params=obs_td_params,
            ),
        )
        self.logMchirp_lower = [datagen_conf["prior"]["logMchirp"][0]]
        self.logMchirp_upper = [datagen_conf["prior"]["logMchirp"][1]]
        self.q_lower = [datagen_conf["prior"]["q"][0]]
        self.q_upper = [datagen_conf["prior"]["q"][1]]
        self._setup_plot()

        # ---- baseline model (optional, skipped on resume) ---------------
        if not resume and self.train_conf["baseline_model"]["use"]:
            from pembhb.model import InferenceNetwork
            self.model = InferenceNetwork.load_from_checkpoint(
                self.train_conf["baseline_model"]["filename"]
            )
            out_idx = 0
            prior_keys = utils.ordered_prior_keys(
                self.datagen_conf.get("spin_param_basis", "chi1chi2"))
            for key, marginal_list in self.train_conf["marginals"].items():
                for marginal in marginal_list:
                    if len(marginal) == 1:
                        widest_interval, _, _, _ = get_widest_interval_1d(
                            self.model, self.dataloader_obs,
                            in_param_idx=marginal[0], out_param_idx=out_idx, eps=1e-4,
                        )
                        param_name = prior_keys[marginal[0]]
                        self.datagen_conf["prior"][param_name] = widest_interval
                    elif len(marginal) == 2:
                        widest_box, _ = get_widest_box_2d(
                            self.model, self.dataloader_obs,
                            in_param_idx=tuple(marginal), out_param_idx=out_idx,
                        )
                        self.datagen_conf["prior"][prior_keys[marginal[0]]] = [widest_box[0], widest_box[1]]
                        self.datagen_conf["prior"][prior_keys[marginal[1]]] = [widest_box[2], widest_box[3]]
                    out_idx += 1
            print(f"Updated prior after baseline model: {self.datagen_conf['prior']}")

        # ---- Fisher prior (optional, skipped on resume) -----------------
        self.fisher_prior_bounds = None
        fp_conf = self.train_conf.get("fisher_prior", {})
        if not resume and fp_conf.get("enabled", False):
            print("[Fisher] Computing Fisher Information Matrix for prior initialisation ...")
            self.fisher_prior_bounds = utils.compute_fisher_prior_bounds(
                datagen_config=self.datagen_conf,
                observation_file=dataset_obs_path,
                event_idx=fp_conf["event_idx"],
                varying_params=fp_conf["varying_params"],
                fixed_params=fp_conf["fixed_params"],
                n_sigma=fp_conf.get("n_sigma", 5.0),
                param_n_sigma=fp_conf.get("param_n_sigma", None),
                spin_param_basis=self.datagen_conf.get("spin_param_basis", "chi1chi2"),
            )

        # ---- Autoencoder instance (persists across rounds) --------------
        self._autoencoder = None

        # ---- Campaign-level convergence monitor (optional) --------------
        # Round-final signals captured at the end of each _train_joint, read by
        # the chain monitor in run() to decide when to stop the whole chain.
        self._last_marginal_entropies = {}
        self._last_stopped_via = ""
        self._last_median_tau = None
        # Per-marginal final volume ratios from the previous round, used to
        # decide which classifier heads to reinit (selective transfer). Empty
        # before round 1. The persistent merged parameter-normalisation vector
        # is rebuilt selectively each round from this.
        self._last_volume_ratios = {}
        # Cross-round end-of-round differential entropy per marginal tuple,
        # ``{tuple: [H_round1, H_round2, ...]}``. Drives entropy-plateau reinit
        # and is persisted/restored for --resume.
        self._entropy_history = {}
        # Prior-shrink cold restart: per-marginal reference box volume (the box
        # at the last cold restart, or the initial prior before the first) and
        # the number of cold restarts fired so far. A marginal is reset once its
        # current box has shrunk by >= shrink_factor relative to its reference,
        # after which the reference ratchets down to the current box.
        self._cr_reference_vol = {}   # {marginal_tuple: float}
        self._cr_level = {}           # {marginal_tuple: int}
        self._cr_history = []         # provenance: list of reset events
        # Per-marginal empirical coverage from the previous round's final model,
        # {marginal_tuple: (coverage, n_inside, n_total)}; drives the truncation
        # veto. Recomputed each round, so no resume state needed.
        self._last_coverage = {}
        self._current_normalisation = None
        cc_conf = self.train_conf.get("chain_convergence", {})
        self.chain_monitor = None
        if cc_conf.get("enabled", False):
            _tg = cc_conf.get("tau_gate")
            self.chain_monitor = ChainConvergenceMonitor(
                eps_nats=cc_conf.get("eps_nats", 0.05),
                patience=cc_conf.get("patience", 2),
                state_path=os.path.join(
                    DATA_ROOT_DIR, TIME_OF_EXECUTION, "chain_convergence_state.yaml"),
                tau_gate=tuple(_tg) if _tg else None,
                require_not_threshold=cc_conf.get("require_not_threshold", True),
            )
            print(f"[ChainConv] enabled: eps_nats={cc_conf.get('eps_nats', 0.05)}, "
                  f"patience={cc_conf.get('patience', 2)}, tau_gate={_tg}")

    # -----------------------------------------------------------------
    # Plot helpers (unchanged from tmnre.py)
    # -----------------------------------------------------------------

    def _setup_plot(self):
        self.fig, self.axes = plt.subplots(1, 2, figsize=(12, 6))
        for ax, title, ylabel in zip(
            self.axes,
            ["logMchirp Prior Bounds", "q Prior Bounds"],
            ["logMchirp", "q"],
        ):
            ax.set_title(title)
            ax.set_xlabel("Iteration")
            ax.set_ylabel(ylabel)

    def _plot_updated_prior_bounds(self, updated_prior):
        self.logMchirp_lower.append(updated_prior["logMchirp"][0])
        self.logMchirp_upper.append(updated_prior["logMchirp"][1])
        self.q_lower.append(updated_prior["q"][0])
        self.q_upper.append(updated_prior["q"][1])
        for ax in self.axes:
            ax.cla()
        self.axes[0].set_title("logMchirp Prior Bounds")
        self.axes[0].set_xlabel("Iteration")
        self.axes[0].set_ylabel("logMchirp")
        self.axes[1].set_title("q Prior Bounds")
        self.axes[1].set_xlabel("Iteration")
        self.axes[1].set_ylabel("q")
        self.axes[0].plot(range(len(self.logMchirp_lower)), self.logMchirp_lower, label="Lower Bound", color="blue")
        self.axes[0].plot(range(len(self.logMchirp_upper)), self.logMchirp_upper, label="Upper Bound", color="orange")
        self.axes[1].plot(range(len(self.q_lower)), self.q_lower, label="Lower Bound", color="blue")
        self.axes[1].plot(range(len(self.q_upper)), self.q_upper, label="Upper Bound", color="orange")
        self.axes[0].legend()
        self.axes[1].legend()
        self.fig.tight_layout()
        self.fig.savefig(
            os.path.join(ROOT_DIR, "plots", TIME_OF_EXECUTION,
                         f"prior_bounds_iteration_{len(self.logMchirp_lower)-1}.png")
        )

    # -----------------------------------------------------------------
    # Data generation (same as SequentialTrainer)
    # -----------------------------------------------------------------

    def _generate_data(self, round_idx, sampler_init_kwargs):
        if self.train_conf.get("streaming", {}).get("enabled", False):
            self._setup_streaming(round_idx, sampler_init_kwargs)
            return
        fname_base = f"simulation_round_{round_idx}"
        fname_h5 = os.path.join(DATA_ROOT_DIR, TIME_OF_EXECUTION, f"{fname_base}.h5")
        os.makedirs(os.path.dirname(fname_h5), exist_ok=True)

        # Disk hygiene: delete the previous round's .h5 (kept the sidecar yaml
        # for the audit trail) once we are about to generate the next one.
        # Opt-in via train_conf["delete_prev_round_h5"] so existing campaigns
        # keep their datasets by default.
        if self.train_conf.get("delete_prev_round_h5", False) and round_idx > 1:
            prev_h5 = os.path.join(
                DATA_ROOT_DIR, TIME_OF_EXECUTION,
                f"simulation_round_{round_idx - 1}.h5",
            )
            # Resolve symlinks: if the previous round's file is a symlink (e.g.
            # to a canonical dataset shared across runs) we should NOT delete
            # the canonical target. Skip the unlink in that case.
            if os.path.islink(prev_h5):
                print(f"[disk] previous round h5 is a symlink, leaving target intact: {prev_h5}")
            elif os.path.exists(prev_h5):
                size_gb = os.path.getsize(prev_h5) / 1024**3
                os.remove(prev_h5)
                print(f"[disk] deleted previous round dataset ({size_gb:.1f} GiB): {prev_h5}")
        domain = self.datagen_conf.get("waveform_params", {}).get("domain", "fd_td")
        # Each round gets a distinct but deterministic seed
        round_seed = self.seed + round_idx
        if domain == "fd":
            wp = self.datagen_conf["waveform_params"]
            sim = MBHBSimulatorFD(
                self.datagen_conf,
                sampler_init_kwargs=sampler_init_kwargs,
                seed=round_seed,
                n_freq_bins=wp.get("n_freq_bins", 4096),
                freq_spacing=wp.get("freq_spacing", "linear"),
            )
        else:
            sim = MBHBSimulatorFD_TD(self.datagen_conf, sampler_init_kwargs=sampler_init_kwargs, seed=round_seed)
        N_simulations = 50000
        batch_size_generation = 100
        if not os.path.exists(fname_h5):
            sim.sample_and_store(fname_h5, N=N_simulations, batch_size=batch_size_generation)
        else:
            try:
                resp = input(f"Dataset file {fname_h5} already exists. Resample and overwrite? [y/N]: ").strip().lower()
            except Exception:
                resp = "n"
            if resp in ("y", "yes"):
                os.remove(fname_h5)
                sim.sample_and_store(fname_h5, N=N_simulations, batch_size=batch_size_generation)
                print(f"Resampled dataset and saved to {fname_h5}")
            else:
                print(f"Using existing dataset at {fname_h5}")
        self.data_fname_yaml = fname_h5.replace(".h5", ".yaml")
        self.datagen_info = utils.read_config(self.data_fname_yaml)
        self.data_module = MBHBDataModule(
            fname_h5, self.train_conf["batch_size"],
            num_workers=4, cache_in_memory=False,
            noise_factor=self.train_conf["noise_factor"],
            seed=self.seed,
            n_train_noise_realisations=self.train_conf.get(
                "n_train_noise_realisations", 1
            ),
        )
        # Training datasets must NOT contain stored noise — the collate fn
        # draws fresh noise realisations at runtime (data augmentation).
        # Only the observation file should carry a fixed noise_fd.
        import h5py as _h5
        with _h5.File(fname_h5, "r") as _f:
            assert "noise_fd" not in _f, (
                f"Training dataset {fname_h5} contains a stored 'noise_fd' dataset. "
                "Training data should be noise-free so that mbhb_collate_fn generates "
                "fresh noise at runtime. Remove 'noise_fd' or regenerate the dataset "
                "with store_noise=False."
            )
        assert self.data_module.median_snr > 8, "Median SNR lower than 8."
        self.data_module.setup(stage="fit")
        self.test_dataloader = self.data_module.test_dataloader()

    # -----------------------------------------------------------------
    # Streaming data generation (producer thread + GPU ring buffer)
    # -----------------------------------------------------------------

    def _setup_streaming(self, round_idx, sampler_init_kwargs):
        """Round-start setup for the streaming path: build the simulator for
        this round's box, allocate + seed-fill the GPU ring buffer, generate a
        frozen validation pool, start the background producer, and expose a
        :class:`StreamingDataModule` as ``self.data_module``.

        The producer keeps refreshing the ring during ``trainer.fit``; it is
        stopped and joined at the end of ``_train_joint``.
        """
        import time
        import torch
        import yaml
        from pembhb import get_torch_complex_dtype, get_torch_dtype
        from pembhb.streaming import RingBuffer, Producer, StreamingDataModule

        self._stream_t0 = time.time()

        sconf = self.train_conf["streaming"]
        n_buffers = int(sconf.get("n_buffers", 5))
        M = int(sconf.get("buffer_size", 10000))
        val_size = int(sconf.get("val_size", 2000))
        device = self.train_conf["device"]

        wp = self.datagen_conf["waveform_params"]
        assert wp.get("domain", "fd_td") == "fd", (
            "streaming requires the FD simulator (set waveform_params.domain='fd')"
        )
        round_seed = self.seed + round_idx
        sim = MBHBSimulatorFD(
            self.datagen_conf, sampler_init_kwargs=sampler_init_kwargs, seed=round_seed,
            n_freq_bins=wp.get("n_freq_bins", 4096),
            freq_spacing=wp.get("freq_spacing", "linear"),
        )

        # Discover per-sample shapes from a tiny probe.
        probe = sim.sample(2, keep_on_gpu=True)
        C, F = probe["wave_fd"].shape[1], probe["wave_fd"].shape[2]
        n_params = probe["parameters"].shape[0]

        ring = RingBuffer(
            n_buffers=n_buffers, buffer_size=M,
            sample_shapes={"wave_fd": (C, F), "params": (n_params,)},
            dtypes={"wave_fd": get_torch_complex_dtype(), "params": get_torch_dtype()},
            device=device,
            host_fields=("params",),  # keep params on CPU (matches HDF5 path)
        )
        producer = Producer(ring, sim, gen_batch_size=int(sconf.get("gen_batch_size", 250)))
        producer.seed_fill_all()  # blocking: give the trainer data on step 0

        # Frozen validation pool (generated once, never refreshed this round).
        vs = sim.sample(val_size, keep_on_gpu=True)
        # Frozen val/test pool on CPU (small; matches HDF5 raw-batch semantics
        # so callbacks reading it via np.asarray work unchanged).
        val_pool = {
            "wave_fd": vs["wave_fd"].cpu(),
            "params": torch.as_tensor(vs["parameters"]).t().contiguous().to(get_torch_dtype()),
        }

        producer.start()
        self._ring = ring
        self._producer = producer
        self.data_module = StreamingDataModule(
            ring, sim, val_pool,
            batch_size=self.train_conf["batch_size"],
            noise_factor=self.train_conf["noise_factor"],
            n_train_noise_realisations=self.train_conf.get("n_train_noise_realisations", 1),
            device=device,
            # Steps/epoch = samples_per_epoch / batch_size. Set this to a large
            # HDF5-like epoch so per-epoch validation/callbacks fire at the same
            # cadence as the non-streaming path (otherwise a small buffer makes
            # them fire ~steps_per_epoch_seq/steps_per_epoch_stream times more).
            samples_per_epoch=sconf.get("samples_per_epoch", None),
            # "iterable" (default, fresh chunks/epoch) or "mapstyle" (legacy
            # July-6 behaviour: one buffer/epoch cycled over, producer throttled).
            dataset_style=sconf.get("dataset_style", "iterable"),
        )
        self.data_module.setup(stage="fit")
        self.test_dataloader = self.data_module.test_dataloader()

        # Audit sidecar (same path the HDF5 path writes) so the round-end
        # shutil.copy and resume bookkeeping keep working.
        self.data_fname_yaml = os.path.join(
            DATA_ROOT_DIR, TIME_OF_EXECUTION, f"simulation_round_{round_idx}.yaml")
        os.makedirs(os.path.dirname(self.data_fname_yaml), exist_ok=True)
        with open(self.data_fname_yaml, "w") as _f:
            yaml.safe_dump(
                {"conf": self.datagen_conf, "sampler_init_kwargs": sampler_init_kwargs,
                 "streaming": dict(sconf)}, _f)
        self.datagen_info = utils.read_config(self.data_fname_yaml)
        assert self.data_module.median_snr > 8, "Median SNR lower than 8."

    # -----------------------------------------------------------------
    # Build or update autoencoder
    # -----------------------------------------------------------------

    def _transfer_data_summary(self):
        """Whether encoder weights + data-normalisation carry across rounds."""
        return self.train_conf.get("joint_training", {}).get(
            "transfer_data_summary_across_rounds", False)

    def _reset_whitening(self, model):
        """Re-set the (physics-derived, round-invariant) whitening scale only.
        Used when reusing a transferred encoder so its fitted amplitude/mean
        stay frozen at their round-1 values."""
        if getattr(model, "whiten", False):
            model.set_whitening(self.data_module.get_noise_scale())

    def _fit_data_normalisation(self, model):
        """Fit the encoder's input normalisation on the current round's data:
        whitening scale (if whiten) + amplitude scale / mean (if either flag)."""
        self._reset_whitening(model)
        if getattr(model, "amplitude_normalise", False) or getattr(
                model, "subtract_mean_whitened", False):
            norm_loader = self.data_module.train_dataloader(shuffle=False, num_workers=0,
                                                            single_chunk=True)
            model.fit_white_normalisation(norm_loader)

    def _build_autoencoder(self, round_idx):
        """Create or reuse the DenoisingAutoencoder for this round.

        Round 1: fresh autoencoder, fit normalisation from training data.
        Round >1: reuse the autoencoder from the previous round (already
        normalisation-fitted).
        """
        ae_conf = self.train_conf["architecture"]["data_summary"]["Autoencoder"]
        device = ae_conf.get("device", self.train_conf["device"])

        prior_bounds = self.datagen_info.get("conf", {}).get("prior", None)

        # Auto-derive n_freqs from the current round's HDF5 grid for every round,
        # not only round 1.  On resume (round_idx >= 2) the round-1 update doesn't
        # run, so saved hparams would otherwise keep the YAML fallback value and
        # break diagnostic-time model reconstruction.
        try:
            _freqs = self.data_module.get_freqs()
            _n_freqs = len(_freqs)
            if ae_conf.get("n_freqs") != _n_freqs:
                print(f"[autoencoder] sync ae_conf.n_freqs -> {_n_freqs} from HDF5 "
                      f"(round {round_idx})")
            ae_conf["n_freqs"] = _n_freqs
        except Exception:
            pass

        # Build a fresh encoder unless we are transferring weights across rounds.
        if self._autoencoder is None or round_idx == 1 or not self._transfer_data_summary():
            hidden_channels = ae_conf.get("hidden_channels", (32, 64, 128, 256, 256))
            if isinstance(hidden_channels, list):
                hidden_channels = tuple(hidden_channels)

            # Auto-derive n_freqs from the HDF5 grid; fall back to ae_conf if
            # the data module isn't available. Resolve fmin_loss/fmax_loss (Hz)
            # → idx_lowerbound/idx_upperbound using the same grid. Write the
            # resolved values back into ae_conf so the checkpoint snapshot
            # records the indices (resume path stays grid-agnostic).
            try:
                freqs = self.data_module.get_freqs()
                n_freqs = len(freqs)
                if "n_freqs" in ae_conf and ae_conf["n_freqs"] != n_freqs:
                    print(f"[autoencoder] auto-derived n_freqs={n_freqs} from HDF5 "
                          f"(config had {ae_conf['n_freqs']}); using HDF5 value.")
            except Exception as e:
                freqs = None
                n_freqs = ae_conf.get("n_freqs", 4096)
                print(f"[autoencoder] warning: failed to read freqs from data module "
                      f"({e}); falling back to ae_conf.n_freqs={n_freqs}.")
            if freqs is not None:
                idx_lo, idx_hi = resolve_loss_band(freqs, ae_conf)
            else:
                idx_lo = ae_conf.get("idx_lowerbound", None)
                idx_hi = ae_conf.get("idx_upperbound", None)
            ae_conf["n_freqs"] = n_freqs
            ae_conf["idx_lowerbound"] = idx_lo
            ae_conf["idx_upperbound"] = idx_hi

            autoencoder = DenoisingAutoencoder(
                n_channels=ae_conf.get("n_channels", 2),
                n_freqs=n_freqs,
                architecture=ae_conf.get("architecture", "conv"),
                bottleneck_dim=ae_conf.get("bottleneck_dim", 128),
                hidden_channels=hidden_channels,
                kernel_size=ae_conf.get("kernel_size", 4),
                stride=ae_conf.get("stride", 2),
                dropout=ae_conf.get("dropout", 0.0),
                decoder_post_fc_bn=ae_conf.get("decoder_post_fc_bn", True),
                lr=ae_conf.get("lr", 1e-3),
                weight_decay=ae_conf.get("weight_decay", 1e-5),
                scheduler_patience=ae_conf.get("scheduler_patience", 10),
                scheduler_factor=ae_conf.get("scheduler_factor", 0.3),
                representation=ae_conf.get("representation", "real_imag"),
                high_freq_only=ae_conf.get("high_freq_only", False),
                freq_split_idx=ae_conf.get("freq_split_idx", 2048),
                idx_lowerbound=idx_lo,
                idx_upperbound=idx_hi,
                amplitude_normalise=ae_conf.get("amplitude_normalise", True),
                subtract_mean_whitened=ae_conf.get("subtract_mean_whitened", True),
                prior_bounds=prior_bounds,
                whiten=ae_conf.get("whiten", True),
                compressor_window=ae_conf.get("compressor_window", None),
                reconstruct_std=ae_conf.get("reconstruct_std", True),
            )
            autoencoder = autoencoder.to(device)

            self._autoencoder = autoencoder
            self._fit_data_normalisation(self._autoencoder)
        else:
            # Reuse the transferred encoder with its round-1 data-normalisation
            # frozen (symmetric with classifier transfer / param-norm). Only the
            # physics-derived whitening scale is re-set defensively.
            print(f"[Joint] Reusing autoencoder from previous round for round {round_idx}")
            self._reset_whitening(self._autoencoder)

        return self._autoencoder

    def _build_channelized_mlp_compressor(self, round_idx):
        """Create or reuse the :class:`ChannelizedMLPCompressor` for this round.

        Round 1: fresh module, fit normalisation from training data.
        Round >1: reuse the module from the previous round; re-fit whitening
        defensively (cheap; depends only on ASD + T_obs).
        """
        from pembhb.autoencoder import ChannelizedMLPCompressor

        cfg = self.train_conf["architecture"]["data_summary"]["ChannelizedMLP"]
        device = cfg.get("device", self.train_conf["device"])

        try:
            n_freqs = len(self.data_module.get_freqs())
            if cfg.get("n_freqs") != n_freqs:
                print(f"[ChannelizedMLP] sync n_freqs -> {n_freqs} from HDF5 "
                      f"(round {round_idx})")
            cfg["n_freqs"] = n_freqs
        except Exception:
            n_freqs = cfg.get("n_freqs", 4096)

        if self._autoencoder is None or round_idx == 1 or not self._transfer_data_summary():
            compressor = ChannelizedMLPCompressor(
                n_channels=cfg.get("n_channels", 2),
                n_freqs=n_freqs,
                hidden_dim_per_channel=cfg.get("hidden_dim_per_channel", 256),
                out_dim_per_channel=cfg.get("out_dim_per_channel", 64),
                representation=cfg.get("representation", "real_imag"),
                whiten=cfg.get("whiten", True),
                amplitude_normalise=cfg.get("amplitude_normalise", True),
                subtract_mean_whitened=cfg.get("subtract_mean_whitened", True),
                dropout=cfg.get("dropout", 0.0),
            )
            compressor = compressor.to(device)
            self._autoencoder = compressor
            self._fit_data_normalisation(self._autoencoder)
        else:
            print(f"[Joint] Reusing ChannelizedMLP compressor from previous round for round {round_idx}")
            self._reset_whitening(self._autoencoder)
        return self._autoencoder

    def _build_marginal_encoder(self, round_idx):
        """Create or reuse the MarginalEncoderTrainer for this round.

        Round 1: fresh ME, fit normalisation from training data.
        Round >1: reuse the ME from the previous round, re-fit normalisation.
        """
        raise NotImplementedError("still have to whiten the data properly")
        me_conf = self.train_conf["architecture"]["data_summary"]["MarginalEncoder"]
        device = me_conf.get("device", self.train_conf["device"])

        marginals_flat = [
            m for mlist in self.train_conf["marginals"].values() for m in mlist
        ]
        prior_bounds = self.datagen_info.get("conf", {}).get("prior", None)

        if self._autoencoder is None or round_idx == 1:
            hidden_channels = me_conf.get("hidden_channels", (32, 64, 128, 256, 256))
            if isinstance(hidden_channels, list):
                hidden_channels = tuple(hidden_channels)
            regressor_hidden = me_conf.get("regressor_hidden_sizes", (128, 64))
            if isinstance(regressor_hidden, list):
                regressor_hidden = tuple(regressor_hidden)

            encoder = MarginalEncoderTrainer(
                n_channels=me_conf.get("n_channels", 2),
                n_freqs=me_conf.get("n_freqs", 4096),
                marginals=marginals_flat,
                bottleneck_dim=me_conf.get("bottleneck_dim", 200),
                hidden_channels=hidden_channels,
                kernel_size=me_conf.get("kernel_size", 5),
                stride=me_conf.get("stride", 2),
                dropout=me_conf.get("dropout", 0.0),
                residual=me_conf.get("residual", False),
                regressor_hidden_sizes=regressor_hidden,
                representation=me_conf.get("representation", "real_imag"),
                amplitude_normalise=me_conf.get("amplitude_normalise", False),
                prior_bounds=prior_bounds,
            )
            encoder = encoder.to(device)

            encoder.set_whitening(self.data_module.get_noise_scale())
            if encoder.amplitude_normalise:
                norm_loader = self.data_module.train_dataloader(shuffle=False, num_workers=0,
                                                                single_chunk=True)
                encoder.fit_amplitude_normalisation(norm_loader)
            self._autoencoder = encoder
        else:
            print(f"[Joint-ME] Reusing ME from previous round for round {round_idx}")
            self._autoencoder.set_whitening(self.data_module.get_noise_scale())
            if self._autoencoder.amplitude_normalise:
                norm_loader = self.data_module.train_dataloader(shuffle=False, num_workers=0,
                                                                single_chunk=True)
                self._autoencoder.fit_amplitude_normalisation(norm_loader)

        return self._autoencoder

    # -----------------------------------------------------------------
    # Resume helpers
    # -----------------------------------------------------------------

    def _restore_from_checkpoint(self, run_name, last_round):
        """Restore model and prior state from the last completed round.

        Loads ``truncation.ckpt`` (final-epoch weights saved at the end of
        ``_train_joint``) under
        ``{DATA_ROOT_DIR}/logs/{run_name}/round_{last_round}/version_*/``.
        The ``checkpoints/*.ckpt`` files produced by ``ModelCheckpoint`` are
        kept for debugging only — they track ``val_nre_loss`` and are not
        suitable as the starting point for the next round's joint training.

        The prior is loaded from
        ``{DATA_ROOT_DIR}/{run_name}/prior_after_round_{last_round}.yaml``
        which is written at the end of each round by ``run()``.
        """
        import yaml as _yaml

        log_root = os.path.join(DATA_ROOT_DIR, "logs")
        dirs = sorted(glob(os.path.join(log_root, run_name, f"round_{last_round}", "version_*")))
        if not dirs:
            raise RuntimeError(f"[Resume] No log directory found for round {last_round} of run '{run_name}'")
        ckpt_path = os.path.join(dirs[-1], "truncation.ckpt")
        if not os.path.exists(ckpt_path):
            raise RuntimeError(
                f"[Resume] truncation.ckpt not found at {ckpt_path}. "
                f"This file is saved at the end of each round's joint training; "
                f"either the round did not finish or the file was deleted."
            )
        print(f"[Resume] Loading final-epoch checkpoint: {ckpt_path}")

        device = self.train_conf.get("device", "cpu")
        self.model = JointAEInferenceNetwork.load_from_checkpoint(ckpt_path, map_location=device)
        self._autoencoder = self.model.autoencoder
        print(f"[Resume] Model and autoencoder restored from round {last_round}.")

        prior_path = os.path.join(DATA_ROOT_DIR, run_name, f"prior_after_round_{last_round}.yaml")
        if os.path.exists(prior_path):
            with open(prior_path) as _f:
                saved = _yaml.safe_load(_f)
            self.datagen_conf["prior"] = saved["prior"]
            print(f"[Resume] Prior restored from {prior_path}: {self.datagen_conf['prior']}")
            # Re-initialise plot tracking lists from the restored prior
            self.logMchirp_lower = [self.datagen_conf["prior"]["logMchirp"][0]]
            self.logMchirp_upper = [self.datagen_conf["prior"]["logMchirp"][1]]
            self.q_lower = [self.datagen_conf["prior"]["q"][0]]
            self.q_upper = [self.datagen_conf["prior"]["q"][1]]
        else:
            print(f"[Resume] Warning: {prior_path} not found; using prior from config. "
                  f"First resumed round may use a slightly wider prior.")

        # Selective-transfer state (merged normalisation + last volume ratios).
        # Absent for runs that pre-date the selective path; harmless otherwise.
        base = os.path.join(DATA_ROOT_DIR, run_name)
        norm_p = os.path.join(base, f"normalisation_after_round_{last_round}.yaml")
        vr_p = os.path.join(base, f"volume_ratios_after_round_{last_round}.yaml")
        ent_p = os.path.join(base, f"diff_entropies_after_round_{last_round}.yaml")
        cr_p = os.path.join(base, f"cold_restart_state_after_round_{last_round}.yaml")
        if os.path.exists(norm_p):
            with open(norm_p) as _f:
                payload = _yaml.safe_load(_f)
            self._current_normalisation = {k: np.asarray(v) for k, v in payload.items()}
            print(f"[Resume] Restored merged normalisation from {norm_p}")
        if os.path.exists(vr_p):
            with open(vr_p) as _f:
                vr_payload = _yaml.safe_load(_f) or {}
            self._last_volume_ratios = {
                tuple(int(x) for x in k.split(",")): float(v)
                for k, v in vr_payload.items()
            }
            print(f"[Resume] Restored last volume ratios: {self._last_volume_ratios}")
        if os.path.exists(ent_p):
            with open(ent_p) as _f:
                ent_payload = _yaml.safe_load(_f) or {}
            self._entropy_history = {
                tuple(int(x) for x in k.split(",")): [float(v) for v in vals]
                for k, vals in ent_payload.items()
            }
            print(f"[Resume] Restored entropy history for {len(self._entropy_history)} marginals "
                  f"({sum(len(v) for v in self._entropy_history.values())} values)")
        if os.path.exists(cr_p):
            with open(cr_p) as _f:
                cr_payload = _yaml.safe_load(_f) or {}
            self._cr_reference_vol = {
                tuple(int(x) for x in k.split(",")): float(v)
                for k, v in (cr_payload.get("reference_vol") or {}).items()
            }
            self._cr_level = {
                tuple(int(x) for x in k.split(",")): int(v)
                for k, v in (cr_payload.get("level") or {}).items()
            }
            self._cr_history = cr_payload.get("history") or []
            print(f"[Resume] Restored cold-restart state: "
                  f"{sum(self._cr_level.values())} resets across "
                  f"{len(self._cr_reference_vol)} marginals")

    # -----------------------------------------------------------------
    # Round-1 parameter normalisation (frozen across rounds)
    # -----------------------------------------------------------------

    def _normalisation_yaml_path(self):
        return os.path.join(
            DATA_ROOT_DIR, TIME_OF_EXECUTION, "normalisation_round_1.yaml"
        )

    def _save_round1_normalisation(self, normalisation):
        import yaml as _yaml
        path = self._normalisation_yaml_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {k: np.asarray(v).tolist() for k, v in normalisation.items()}
        with open(path, "w") as f:
            _yaml.safe_dump(payload, f)
        print(f"[Norm] Saved round-1 parameter normalisation to {path}")

    def _load_round1_normalisation(self):
        import yaml as _yaml
        path = self._normalisation_yaml_path()
        if not os.path.exists(path):
            raise RuntimeError(
                f"[Resume] normalisation_round_1.yaml not found at {path}. "
                f"This run pre-dates the frozen-normalisation change; resume "
                f"is not supported. Restart from round 1 to capture round-1 "
                f"normalisation."
            )
        with open(path) as f:
            payload = _yaml.safe_load(f)
        return {k: np.asarray(v) for k, v in payload.items()}

    # -----------------------------------------------------------------
    # Prior-shrink cold restart
    # -----------------------------------------------------------------

    def _prior_shrink_reinit_keys(self, round_idx, joint_conf):
        """Marginals whose prior box shrank by >= shrink_factor since last reset.

        The current box is ``self.datagen_conf["prior"]`` — the prior the current
        round's data was drawn from (i.e. ``prior_after_round_{round_idx-1}``).
        Each triggered marginal is reset (its head is left out of the weight
        transfer and its params re-standardised) and its reference box ratchets
        down to the current box, so the next reset needs another shrink_factor.
        """
        ps_conf = joint_conf.get("prior_shrink_reinit", {})
        factor = float(ps_conf.get("shrink_factor", 100))
        ordered_keys = utils.ordered_prior_keys(
            self.datagen_conf.get("spin_param_basis", "chi1chi2"))
        cur_prior = self.datagen_conf["prior"]
        marg_dict = utils.resolve_marginals_for_round(self.train_conf, round_idx)
        all_marginals = [tuple(m) for lst in marg_dict.values() for m in lst]

        shrink_keys = set()
        for key in all_marginals:
            cur_vol = _marginal_prior_volume(key, cur_prior, ordered_keys)
            # First sight of this marginal: reference is the initial prior box.
            if key not in self._cr_reference_vol:
                self._cr_reference_vol[key] = _marginal_prior_volume(
                    key, self._initial_prior, ordered_keys)
                self._cr_level.setdefault(key, 0)
            ref_vol = self._cr_reference_vol[key]
            if ref_vol > 0 and cur_vol <= ref_vol / factor:
                self._cr_level[key] = self._cr_level.get(key, 0) + 1
                level = self._cr_level[key]
                self._cr_history.append({
                    "round": int(round_idx), "marginal": list(key),
                    "level": level, "ref_vol": float(ref_vol),
                    "cur_vol": float(cur_vol),
                    "ratio": float(cur_vol / ref_vol) if ref_vol else 0.0,
                })
                self._cr_reference_vol[key] = cur_vol   # ratchet reference down
                shrink_keys.add(key)
                names = "-".join(ordered_keys[i] for i in key)
                print(f"[ColdRestart] marginal {key} ({names}) cold_restart_{level}: "
                      f"box vol {cur_vol:.3e} <= ref/{factor:g} ({ref_vol/factor:.3e}); "
                      f"resetting head + param standardisation.")
        return shrink_keys

    # -----------------------------------------------------------------
    # Selective (per-marginal) parameter normalisation
    # -----------------------------------------------------------------

    def _compute_full_normalisation(self, periodic_bc_params):
        """Fresh normalisation from the current round's data (all params)."""
        mean, std = self.data_module.get_params_mean_std()
        sincos_mean, sincos_std = self.data_module.get_sincos_mean_std(periodic_bc_params)
        return {
            "td_normalisation": np.array(self.data_module.get_max_td()),
            "param_mean": np.asarray(mean),
            "param_std": np.asarray(std),
            "sincos_mean": np.asarray(sincos_mean),
            "sincos_std": np.asarray(sincos_std),
        }

    def _build_selective_normalisation(self, reinit_param_idxs, periodic_bc_params):
        """Merge fresh stats for *reinit_param_idxs* into the frozen baseline.

        Params whose marginal was reinitialised are re-standardised on the new
        (truncated) data; every other param keeps the normalisation locked from
        the round in which its head was last (re)initialised. Round 1 (no
        baseline yet) returns fully-fresh stats. Updates
        ``self._current_normalisation`` in place.
        """
        fresh = self._compute_full_normalisation(periodic_bc_params)
        if self._current_normalisation is None:
            self._current_normalisation = {k: np.array(v) for k, v in fresh.items()}
            return self._current_normalisation

        merged = {k: np.array(v) for k, v in self._current_normalisation.items()}
        # td_normalisation is a global data scale, not tied to a head — refresh.
        merged["td_normalisation"] = fresh["td_normalisation"]
        for i in reinit_param_idxs:
            merged["param_mean"][i] = fresh["param_mean"][i]
            merged["param_std"][i] = fresh["param_std"][i]
        # sincos entries are laid out [sin(p),cos(p)] per periodic param, in the
        # order of periodic_bc_params.
        for j, pidx in enumerate(periodic_bc_params):
            if pidx in reinit_param_idxs and 2 * j + 1 < len(merged["sincos_mean"]):
                merged["sincos_mean"][2 * j:2 * j + 2] = fresh["sincos_mean"][2 * j:2 * j + 2]
                merged["sincos_std"][2 * j:2 * j + 2] = fresh["sincos_std"][2 * j:2 * j + 2]
        self._current_normalisation = merged
        return merged

    def _round_state_paths(self, round_idx):
        base = os.path.join(DATA_ROOT_DIR, TIME_OF_EXECUTION)
        return (
            os.path.join(base, f"normalisation_after_round_{round_idx}.yaml"),
            os.path.join(base, f"volume_ratios_after_round_{round_idx}.yaml"),
            os.path.join(base, f"diff_entropies_after_round_{round_idx}.yaml"),
            os.path.join(base, f"cold_restart_state_after_round_{round_idx}.yaml"),
        )

    def _persist_round_state(self, round_idx):
        """Persist merged normalisation + volume ratios + entropy history + cold-restart state."""
        import yaml as _yaml
        norm_path, vr_path, ent_path, cr_path = self._round_state_paths(round_idx)
        os.makedirs(os.path.dirname(norm_path), exist_ok=True)
        if self._current_normalisation is not None:
            payload = {k: np.asarray(v).tolist() for k, v in self._current_normalisation.items()}
            with open(norm_path, "w") as f:
                _yaml.safe_dump(payload, f)
        # keys are param-index tuples → stringify for YAML
        vr_payload = {",".join(map(str, k)): float(v) for k, v in self._last_volume_ratios.items()}
        with open(vr_path, "w") as f:
            _yaml.safe_dump(vr_payload, f)
        # full cross-round entropy history (one list per marginal) so --resume
        # can restore the series the plateau detector reads.
        ent_payload = {",".join(map(str, k)): [float(x) for x in v]
                       for k, v in self._entropy_history.items()}
        with open(ent_path, "w") as f:
            _yaml.safe_dump(ent_payload, f)
        # prior-shrink cold-restart state (reference box volume + level per
        # marginal, plus the reset-event history) so --resume keeps ratcheting.
        cr_payload = {
            "reference_vol": {",".join(map(str, k)): float(v)
                              for k, v in self._cr_reference_vol.items()},
            "level": {",".join(map(str, k)): int(v)
                      for k, v in self._cr_level.items()},
            "history": self._cr_history,
        }
        with open(cr_path, "w") as f:
            _yaml.safe_dump(cr_payload, f)
        # round-end truncation-veto coverage per marginal (provenance/review).
        if self._last_coverage:
            base = os.path.dirname(norm_path)
            cov_payload = {
                ",".join(map(str, k)): {
                    "coverage": float(c), "n_inside": int(n_in), "n_total": int(n_tot)}
                for k, (c, n_in, n_tot) in self._last_coverage.items()
            }
            with open(os.path.join(
                    base, f"truncation_coverage_after_round_{round_idx}.yaml"), "w") as f:
                _yaml.safe_dump(cov_payload, f)

    # -----------------------------------------------------------------
    # Truncation-veto coverage
    # -----------------------------------------------------------------

    def _compute_round_coverage(self, round_idx):
        """Empirical coverage of each marginal's truncation box at round end.

        For the coverage-gated truncation veto: evaluate the final model on up to
        ``truncation_veto.n_coverage`` held-out val-pool samples and, per marginal,
        count how many ground truths fall inside their own credible box (same box
        the truncation builds). Returns ``{marginal_tuple: (coverage, n_in, n_tot)}``
        or ``{}`` when the veto is disabled.
        """
        veto_conf = self.train_conf.get("truncation_veto", {})
        if not veto_conf.get("enabled", False):
            return {}
        from torch.utils.data import Subset, DataLoader as _DL
        test_ds = self.data_module.test
        n_cov = min(int(veto_conf.get("n_coverage", 1000)), len(test_ds))
        base_loader = self.data_module.test_dataloader()
        cov_loader = _DL(
            Subset(test_ds, list(range(n_cov))),
            batch_size=base_loader.batch_size, shuffle=False,
            num_workers=0, collate_fn=base_loader.collate_fn,
        )
        keys = utils.ordered_prior_keys(
            self.datagen_conf.get("spin_param_basis", "chi1chi2"))
        m1d, m2d = [], []
        for out_idx, marginal in enumerate(self.model.marginals_list):
            if len(marginal) == 1:
                m1d.append((keys[marginal[0]], marginal[0], out_idx))
            elif len(marginal) == 2:
                m2d.append((f"{keys[marginal[0]]}__{keys[marginal[1]]}",
                            (marginal[0], marginal[1]), out_idx))
        was_training = self.model.training
        self.model.eval()
        try:
            cov = compute_truncation_coverage(
                self.model, cov_loader, m1d, m2d,
                eps_1d=float(veto_conf.get("eps_1d", 1e-4)),
                sky_credible_level=float(veto_conf.get("sky_credible_level", 0.999)),
                sky_dilation=float(veto_conf.get("sky_dilation", 1.1)),
            )
        finally:
            if was_training:
                self.model.train()
        for key, (c, n_in, n_tot) in sorted(cov.items()):
            name = "-".join(keys[i] for i in key)
            print(f"[TruncVeto] round {round_idx} coverage {name}: "
                  f"{c:.3f} ({n_in}/{n_tot})", flush=True)
        return cov

    # -----------------------------------------------------------------
    # Joint training
    # -----------------------------------------------------------------

    def _train_joint(self, round_idx):
        """Train encoder + NRE jointly for this round (AE or ME mode)."""
        ds_type = self.train_conf["architecture"]["data_summary"].get("type", "Autoencoder")
        joint_conf = self.train_conf.get("joint_training", {})
        # Two independent cross-round transfers, both off by default. Each also
        # freezes the input-normalisation of the subnetwork it carries.
        transfer_classifier = joint_conf.get("transfer_classifiers_across_rounds", False)
        transfer_data_summary = joint_conf.get("transfer_data_summary_across_rounds", False)
        # Selective per-marginal transfer: carry over every head except those
        # flagged for reinit (those are reinitialised and their params
        # re-standardised). Two independent triggers, both taking precedence
        # over the all-or-nothing transfer_classifier flag:
        #   * reinit_truncated_classifiers — head's volume ratio <= threshold
        #   * reinit_plateaued_classifiers — head's entropy has plateaued
        #     (|dH| < min_delta for `patience` rounds; a "cold restart").
        #   * reinit_on_prior_shrink — head's prior box has shrunk by
        #     >= shrink_factor since its last reset (a "cold restart").
        reinit_truncated = joint_conf.get("reinit_truncated_classifiers", False)
        reinit_plateau = joint_conf.get("reinit_plateaued_classifiers", False)
        reinit_prior_shrink = joint_conf.get("reinit_on_prior_shrink", False)
        reinit_selective = reinit_truncated or reinit_plateau or reinit_prior_shrink

        if ds_type == "MarginalEncoder":
            enc_conf = self.train_conf["architecture"]["data_summary"]["MarginalEncoder"]
            encoder_model = self._build_marginal_encoder(round_idx)
        elif ds_type == "ChannelizedMLP":
            enc_conf = self.train_conf["architecture"]["data_summary"]["ChannelizedMLP"]
            encoder_model = self._build_channelized_mlp_compressor(round_idx)
        else:
            enc_conf = self.train_conf["architecture"]["data_summary"]["Autoencoder"]
            encoder_model = self._build_autoencoder(round_idx)

        device = enc_conf.get("device", self.train_conf["device"])

        # Warm-up runs whenever the encoder is freshly built — round 1, or any
        # round where the data-summary is not transferred (fresh untrained
        # encoder). It is skipped only when reusing a transferred encoder.
        fresh_encoder = (round_idx == 1) or not transfer_data_summary
        ae_warmup_epochs = joint_conf.get("ae_warmup_epochs", 50) if fresh_encoder else 0

        # Parameter normalisation is paired with the classifier heads: when the
        # heads are transferred we freeze the round-1 stats (persisted to YAML)
        # so carried-over heads keep seeing the same input distribution; when
        # they are not transferred we recompute the stats every round.
        periodic_bc_params = self.train_conf.get("periodic_bc_params", [])
        reinit_keys = set()
        if reinit_selective:
            # (a) volume-ratio trigger: heads whose final volume ratio last round
            #     was <= threshold get fresh heads + re-standardised params.
            if reinit_truncated:
                threshold = self.train_conf.get("volume_ratio_early_stop", {}).get(
                    "min_ratio_threshold", 0.5)
                vr_keys = {k for k, r in self._last_volume_ratios.items()
                           if r <= threshold}
                reinit_keys |= vr_keys
                if round_idx > 1:
                    print(f"[Transfer] Volume-ratio reinit: {len(vr_keys)} head(s) "
                          f"below vr<= {threshold}: {sorted(vr_keys)}.")
            # (b) entropy-plateau trigger (cold restart): heads whose end-of-round
            #     entropy stopped moving over the last `patience` rounds.
            if reinit_plateau:
                pl_conf = joint_conf.get("diff_entropy_plateau_reinit", {})
                pl_keys = _entropy_plateau_keys(
                    self._entropy_history,
                    min_delta=pl_conf.get("min_delta", 0.1),
                    patience=pl_conf.get("patience", 3),
                    warmup_rounds=pl_conf.get("warmup_rounds", 10),
                    round_idx=round_idx,
                )
                reinit_keys |= pl_keys
                if round_idx > 1 and pl_keys:
                    print(f"[Transfer] Entropy-plateau reinit (cold restart): "
                          f"{sorted(pl_keys)} (min_delta={pl_conf.get('min_delta', 0.1)}, "
                          f"patience={pl_conf.get('patience', 3)}).")
            # (c) prior-shrink trigger (cold restart): heads whose prior box has
            #     shrunk by >= shrink_factor since their last reset. The current
            #     box is the prior that generated this round's data
            #     (self.datagen_conf["prior"] == prior_after_round_{round-1}).
            if reinit_prior_shrink:
                ps_keys = self._prior_shrink_reinit_keys(round_idx, joint_conf)
                reinit_keys |= ps_keys
            reinit_param_idxs = {i for k in reinit_keys for i in k}
            normalisation = self._build_selective_normalisation(
                reinit_param_idxs, periodic_bc_params)
            if round_idx > 1:
                print(f"[Transfer] Selective reinit total: {len(reinit_keys)} head(s) "
                      f"{sorted(reinit_keys)}; re-standardised params "
                      f"{sorted(reinit_param_idxs)}.")
        elif not transfer_classifier:
            mean, std = self.data_module.get_params_mean_std()
            sincos_mean, sincos_std = self.data_module.get_sincos_mean_std(periodic_bc_params)
            normalisation = {
                "td_normalisation": np.array(self.data_module.get_max_td()),
                "param_mean": np.array(mean),
                "param_std": np.array(std),
                "sincos_mean": np.array(sincos_mean),
                "sincos_std": np.array(sincos_std),
            }
        elif round_idx == 1 and not os.path.exists(self._normalisation_yaml_path()):
            mean, std = self.data_module.get_params_mean_std()
            sincos_mean, sincos_std = self.data_module.get_sincos_mean_std(periodic_bc_params)
            normalisation = {
                "td_normalisation": np.array(self.data_module.get_max_td()),
                "param_mean": np.array(mean),
                "param_std": np.array(std),
                "sincos_mean": np.array(sincos_mean),
                "sincos_std": np.array(sincos_std),
            }
            self._save_round1_normalisation(normalisation)
        else:
            normalisation = self._load_round1_normalisation()
            print(f"[Norm] Reusing round-1 parameter normalisation from {self._normalisation_yaml_path()}")

        # Build joint model
        old_model = getattr(self, "model", None)
        # Per-group scheduler configs (new API).  When absent we fall back to
        # the deprecated scalar keys below via JointAEInferenceNetwork's
        # backward-compat path.
        ae_sched_cfg = joint_conf.get("ae_scheduler")
        nre_sched_cfg = joint_conf.get("nre_scheduler")
        # If nre_scheduler is given without an explicit start_epoch, default
        # it to the end of the AE warmup so the NRE scheduler doesn't count
        # epochs where val_nre_loss is logged as 0.
        if nre_sched_cfg is not None and "start_epoch" not in nre_sched_cfg:
            nre_sched_cfg = {**nre_sched_cfg, "start_epoch": ae_warmup_epochs}
        
        self.model = JointAEInferenceNetwork(
            train_conf=self.train_conf,
            dataset_info=self.datagen_info,
            normalisation=normalisation,
            encoder_model=encoder_model,
            ae_warmup_epochs=ae_warmup_epochs,
            lr_ae=joint_conf.get("lr_ae", enc_conf.get("lr", 1e-3)),
            lr_nre=joint_conf.get("lr_nre", self.train_conf.get("learning_rate", 1e-4)),
            ae_weight_decay=joint_conf.get("ae_weight_decay", enc_conf.get("weight_decay", 1e-5)),
            ae_scheduler_patience=joint_conf.get("ae_scheduler_patience", enc_conf.get("scheduler_patience", 10)),
            ae_scheduler_factor=joint_conf.get("ae_scheduler_factor", enc_conf.get("scheduler_factor", 0.3)),
            ae_scheduler=ae_sched_cfg,
            nre_scheduler=nre_sched_cfg,
            periodic_bc_params=self.train_conf.get("periodic_bc_params", []),
            freeze_ae_after_warmup=joint_conf.get("freeze_ae_after_warmup", False),
            encoder_trains_via_nre=(ds_type == "ChannelizedMLP"),
        )
        self.model.to(device)
        if reinit_selective and old_model is not None:
            transfer_classifier_weights(old_model, self.model, skip_keys=reinit_keys)
        elif transfer_classifier and old_model is not None:
            transfer_classifier_weights(old_model, self.model)
        self.model.train()

        # ---- Callbacks --------------------------------------------------
        logger = TensorBoardLogger(
            os.path.join(DATA_ROOT_DIR, "logs"),
            name=f"{TIME_OF_EXECUTION}/round_{round_idx}",
        )

        # Monitor val_nre_loss (not val_loss): the combined val_loss is
        # dominated by the AE term and its minimum is almost always reached
        # during the warmup (where val_nre_loss = 0), yielding a checkpoint
        # whose NRE heads are effectively untrained.  These ModelCheckpoint
        # files are kept for debugging only — the next round is restored
        # from truncation.ckpt (final-epoch weights), not from here.
        # ``WarmupModelCheckpoint`` ignores epochs before ``ae_warmup_epochs``
        # so the 0-valued warmup readings don't anchor best=0.
        class WarmupModelCheckpoint(ModelCheckpoint):
            def __init__(self, warmup_epochs: int, **kwargs):
                super().__init__(**kwargs)
                self._warmup_epochs = warmup_epochs

            def _save_topk_checkpoint(self, trainer, monitor_candidates):
                if trainer.current_epoch < self._warmup_epochs:
                    return
                super()._save_topk_checkpoint(trainer, monitor_candidates)

        checkpoint_callback = WarmupModelCheckpoint(
            warmup_epochs=ae_warmup_epochs,
            monitor="val_nre_loss",
            mode="min",
        )

        periodic_checkpoint_callback = ModelCheckpoint(
            every_n_epochs=joint_conf.get("checkpoint_every_n_epochs", 10),
            save_top_k=-1,
            save_on_train_epoch_end=True,
            filename="joint-periodic-{epoch:03d}",
        )

        # Early stopping on NRE accuracy (only meaningful after warmup)
        # early_stopping_callback = EarlyStopping(
        #     monitor="val_accuracy",
        #     patience=self.train_conf["early_stop_patience"],
        #     mode="max",
        #     min_delta=self.train_conf["early_stop_min_delta"],
        #     stopping_threshold=self.train_conf["early_stop_threshold"],
        # )

        # --- Opt-in global-step callback cadence (see the plan / STREAMING_DATAGEN).
        # Unset -> epoch behaviour unchanged. Auto-enabled for streaming so small
        # ring-buffer epochs don't inflate plot/log/early-stop frequency.
        streaming_enabled = self.train_conf.get("streaming", {}).get("enabled", False)
        sc = self.train_conf.get("step_cadence", {})
        call_every_n_steps = sc.get("call_every_n_steps")
        print_every_n_steps = sc.get("print_every_n_steps")
        steps_per_seq_epoch = None
        if streaming_enabled and call_every_n_steps is None and sc.get("seq_samples_per_epoch"):
            steps_per_seq_epoch = max(
                1, int(sc["seq_samples_per_epoch"]) // int(self.train_conf["batch_size"]))
            call_every_n_steps = sc.get("call_every_n_epochs_seq", 10) * steps_per_seq_epoch
            if print_every_n_steps is None:
                print_every_n_steps = sc.get("print_every_seq", 20) * steps_per_seq_epoch
            print(f"[StepCadence] streaming auto: steps_per_seq_epoch={steps_per_seq_epoch}, "
                  f"call_every_n_steps={call_every_n_steps}, print_every_n_steps={print_every_n_steps}")
        step_mode = call_every_n_steps is not None

        def _warmup_steps_for(warmup_epochs_value):
            """Explicit step_cadence.warmup_steps wins; else derive from the seq
            epoch size when auto-enabled; else None (epoch warmup)."""
            if sc.get("warmup_steps") is not None:
                return sc["warmup_steps"]
            if steps_per_seq_epoch is not None:
                return int(warmup_epochs_value) * steps_per_seq_epoch
            return None

        plot_posterior_callback = PlotPosteriorCallback(
            timestamp=TIME_OF_EXECUTION,
            obs_loader=self.dataloader_obs,
            input_idx_list=self.model.marginals_list,
            output_idx_list=list(range(len(self.model.marginals_list))),
            round_idx=round_idx,
            call_every_n_epochs=10,
            training_start_time=self.training_start,
            warmup_epochs=ae_warmup_epochs,
            call_every_n_steps=call_every_n_steps,
            warmup_steps=_warmup_steps_for(ae_warmup_epochs),
        )
        if ae_warmup_epochs > 0:
            print(f"[PlotPosterior] skipping first {ae_warmup_epochs} epochs (AE warmup)")

        # Early stopping callback (reads from PlotPosteriorCallback)
        callbacks_list = [checkpoint_callback,
                          periodic_checkpoint_callback,
                          plot_posterior_callback,
                          PeriodicProgressCallback(print_every=20, label="Joint",
                                                   print_every_n_steps=print_every_n_steps)]
        # AE-specific gradient/weight diagnostics only make sense when the
        # encoder is an actual DenoisingAutoencoder (has .encoder.conv etc.).
        if ds_type not in ("ChannelizedMLP", "MarginalEncoder"):
            callbacks_list.insert(3, AutoencoderDiagnosticsCallback())

        # Joint-only knobs live under joint_training: (fall back to top level
        # for old configs where they were at the root).
        es_criterion = joint_conf.get(
            "early_stop_criterion",
            self.train_conf.get("early_stop_criterion", "volume_ratio"),
        )

        if es_criterion == "volume_ratio":
            vr_conf = self.train_conf.get("volume_ratio_early_stop", {})
            vr_warmup = vr_conf.get("warmup_epochs", 50) + ae_warmup_epochs
            vr_callback = VolumeRatioEarlyStopping(
                warmup_epochs=vr_warmup,
                patience=vr_conf.get("patience", 10),
                rel_tol=vr_conf.get("rel_tol", 0.02),
                ema_alpha=vr_conf.get("ema_alpha", 0.3),
                min_ratio_threshold=vr_conf.get("min_ratio_threshold", 0.5),
                plateau_grace_epochs=vr_conf.get("plateau_grace_epochs", 0),
                warmup_steps=_warmup_steps_for(vr_warmup),
                step_mode=step_mode,
            )
            print(f"[EarlyStopping] criterion=volume_ratio, warmup_epochs={vr_warmup} "
                  f"(ae_warmup={ae_warmup_epochs} + vr_warmup={vr_conf.get('warmup_epochs', 50)})")
            callbacks_list.append(vr_callback)

        elif es_criterion == "differential_entropy":
            de_conf = joint_conf.get(
                "diff_entropy_early_stop",
                self.train_conf.get("diff_entropy_early_stop", {}),
            )
            de_warmup = de_conf.get("warmup_epochs", 50) + ae_warmup_epochs
            de_callback = DifferentialEntropyEarlyStopping(
                warmup_epochs=de_warmup,
                patience=de_conf.get("patience", 10),
                rel_tol=de_conf.get("rel_tol", 0.02),
                ema_alpha=de_conf.get("ema_alpha", 0.3),
                warmup_steps=_warmup_steps_for(de_warmup),
                step_mode=step_mode,
            )
            print(f"[EarlyStopping] criterion=differential_entropy, warmup_epochs={de_warmup} "
                  f"(ae_warmup={ae_warmup_epochs} + de_warmup={de_conf.get('warmup_epochs', 50)})")
            callbacks_list.append(de_callback)

        elif es_criterion == "accuracy":
            acc_conf = joint_conf.get("accuracy_early_stop", {})
            callbacks_list.append(WarmupEarlyStopping(
                warmup_epochs=ae_warmup_epochs,
                monitor="val_accuracy",
                mode="max",
                patience=acc_conf.get("patience", 90),
                min_delta=acc_conf.get("min_delta", 0.001),
                stopping_threshold=acc_conf.get("threshold", 0.999),
            ))
            print(f"[EarlyStopping] criterion=accuracy, "
                  f"patience={acc_conf.get('patience', 90)} starts after "
                  f"ae_warmup_epochs={ae_warmup_epochs}")

        else:
            raise ValueError(
                f"Unknown early_stop_criterion '{es_criterion}'. "
                "Expected one of: 'volume_ratio', 'differential_entropy', 'accuracy'."
            )

        # Optional PP-plot KS / tail-mass overconfidence monitor (can also
        # trigger early stopping when trigger_on_overconfidence: true).
        pp_conf = self.train_conf.get("pp_ks_early_stop")
        if pp_conf and pp_conf.get("enabled", False):
            from torch.utils.data import Subset, DataLoader as _DL
            test_ds = self.data_module.test
            n_eval = int(pp_conf.get("test_n", 100))
            n_eval = min(n_eval, len(test_ds))
            subset = Subset(test_ds, list(range(n_eval)))
            # Reuse the data module's collate so noise is drawn consistently
            # with how the test loader was constructed elsewhere.
            test_batch_size = int(pp_conf.get("test_batch_size", 100))
            base_loader = self.data_module.test_dataloader()
            ppks_loader = _DL(
                subset, batch_size=test_batch_size, shuffle=False,
                num_workers=0, collate_fn=base_loader.collate_fn,
            )

            # Parameter names follow the run's spin basis (slots 2,3).
            ppks_keys = utils.ordered_prior_keys(
                self.datagen_conf.get("spin_param_basis", "chi1chi2"))
            # 1-D marginals only — label, in_param_idx (model input), out_idx
            # (column in model output).
            marginals_1d_info = []
            marginals_2d_info = []
            for out_idx, marginal in enumerate(self.model.marginals_list):
                if len(marginal) == 1:
                    in_idx = marginal[0]
                    marginals_1d_info.append(
                        (ppks_keys[in_idx], in_idx, out_idx)
                    )
                elif len(marginal) == 2:
                    i0, i1 = marginal[0], marginal[1]
                    label = f"{ppks_keys[i0]}__{ppks_keys[i1]}"
                    marginals_2d_info.append((label, (i0, i1), out_idx))

            # λ/τ statistics config. Fisher set = every non-fixed (inferred)
            # parameter in the round's prior, in canonical order.
            lt_conf = pp_conf.get("lambda_tau", {})
            lt_enabled = bool(lt_conf.get("enabled", False))
            fisher_varying_params = [
                k for k in ppks_keys
                if self.datagen_conf["prior"][k][0] != self.datagen_conf["prior"][k][1]
            ]
            lt_h5_path = os.path.join(
                DATA_ROOT_DIR, TIME_OF_EXECUTION,
                f"lambda_tau_stats_round_{round_idx}.h5",
            )
            lt_backend = lt_conf.get("fisher_backend", self.datagen_conf.get("backend", "cpu"))

            # Warmup is keyed off the *cumulative* epoch axis inside the
            # callback, so AE-warmup epochs already count — no need to
            # offset by ae_warmup_epochs here.
            ppks_warmup = int(pp_conf.get("warmup_epochs", 50))
            ppks_state_path = os.path.join(
                DATA_ROOT_DIR, TIME_OF_EXECUTION, "ppks_state.yaml",
            )
            ppks_plots_dir = os.path.join(
                ROOT_DIR, "plots", TIME_OF_EXECUTION,
            )
            callbacks_list.append(PPKSTestEarlyStopping(
                test_loader=ppks_loader,
                marginals_1d_info=marginals_1d_info,
                ngrid_points=int(pp_conf.get("ngrid_points", 50)),
                warmup_epochs=ppks_warmup,
                run_every_n_epochs=int(pp_conf.get("run_every_n_epochs", 1)),
                patience=int(pp_conf.get("patience", 40)),
                ema_alpha=float(pp_conf.get("ema_alpha", 0.3)),
                d_threshold=float(pp_conf.get("d_threshold", 0.15)),
                t_threshold=float(pp_conf.get("t_threshold", 0.15)),
                t_quantile=float(pp_conf.get("t_quantile", 0.05)),
                trigger_on_overconfidence=bool(
                    pp_conf.get("trigger_on_overconfidence", False)
                ),
                print_every=int(pp_conf.get("print_every", 20)),
                state_path=ppks_state_path,
                round_idx=round_idx,
                plots_dir=ppks_plots_dir,
                compute_lambda_tau=lt_enabled,
                marginals_2d_info=marginals_2d_info,
                datagen_conf=self.datagen_conf,
                fisher_varying_params=fisher_varying_params,
                fisher_backend=lt_backend,
                lt_h5_path=lt_h5_path,
            ))
            if lt_enabled:
                print(f"[λτ] enabled: 2d_marginals={len(marginals_2d_info)}, "
                      f"fisher_params={fisher_varying_params}, backend={lt_backend}, "
                      f"out={lt_h5_path}")
            mode = ("trigger" if pp_conf.get("trigger_on_overconfidence", False)
                    else "monitor")
            print(f"[PPKS] enabled in {mode} mode "
                  f"(test_n={n_eval}, ngrid={pp_conf.get('ngrid_points', 50)}, "
                  f"cumulative_warmup={ppks_warmup}, "
                  f"state={ppks_state_path})")

        streaming = self.train_conf.get("streaming", {}).get("enabled", False)
        trainer = Trainer(
            logger=logger,
            max_epochs=self.train_conf["epochs"],
            accelerator=self.train_conf["device"],
            devices=1,
            enable_progress_bar=False,
            callbacks=callbacks_list,
            gradient_clip_val=enc_conf.get("gradient_clip_val", None),
            # Streaming reads one ring buffer per epoch: reload the dataloader
            # every epoch so StreamingDataModule can rotate to the next buffer.
            reload_dataloaders_every_n_epochs=1 if streaming else 0,
        )

        # Reserve GPU memory on the first round so other processes cannot
        # steal it during the overnight run. Skipped under streaming: the probe
        # loader would claim a ring buffer without releasing it.
        if (device == "cuda" and not streaming and not getattr(self, "_gpu_reserved", False)
                and not self.train_conf.get("skip_gpu_reserve", False)):
            from pembhb.gpu_utils import reserve_gpu_memory
            safety = self.train_conf.get("gpu_reserve_safety_factor", 1.25)
            reserve_gpu_memory(self.model, self.data_module.train_dataloader(),
                               safety_factor=safety)
            self._gpu_reserved = True

        trainer.fit(self.model, self.data_module)

        # Stop the background producer and free the ring buffers before the
        # round-end bookkeeping/truncation read.
        if streaming:
            import time
            self.data_module.release_active()
            self._ring.stop()
            self._producer.join(timeout=60)
            if self._producer.error is not None:
                raise RuntimeError(f"streaming data producer failed: {self._producer.error!r}")
            elapsed = time.time() - self._stream_t0
            M = self._ring.M
            n_seed = self._producer.n_seed
            n_chunks = self._producer.n_chunks
            total = self._producer.samples_generated
            stream_msg = (
                f"[Round {round_idx}][streaming] effective sims: {total} "
                f"({n_seed} seed buffers + {n_chunks} refresh chunks x {M} samples); "
                f"round wall-time {elapsed:.1f}s "
                f"({elapsed/60:.1f} min); gen rate ~{total/max(elapsed,1):.0f} samples/s"
            )
            print(stream_msg)
            _slog = os.path.join(DATA_ROOT_DIR, TIME_OF_EXECUTION, "round_summary.log")
            os.makedirs(os.path.dirname(_slog), exist_ok=True)
            with open(_slog, "a") as _lf:
                _lf.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  {stream_msg}\n")

        trunc_path = os.path.join(logger.log_dir, "truncation.ckpt")
        os.makedirs(os.path.dirname(trunc_path), exist_ok=True)
        trainer.save_checkpoint(trunc_path)
        print(f"Saved final checkpoint to {trunc_path}")
        # ---- Log early stopping reason and final LRs --------------------
        stop_reason = "max_epochs"
        for cb in callbacks_list:
            if isinstance(cb, (VolumeRatioEarlyStopping, DifferentialEntropyEarlyStopping)):
                if cb.stop_reason:
                    stop_reason = cb.stop_reason

        # Round-final signals for the campaign-level convergence monitor.
        self._last_stopped_via = ""
        for cb in callbacks_list:
            if isinstance(cb, VolumeRatioEarlyStopping):
                self._last_stopped_via = cb.stopped_via
        self._last_marginal_entropies = _round_marginal_entropies(
            plot_posterior_callback,
            keys=utils.ordered_prior_keys(
                self.datagen_conf.get("spin_param_basis", "chi1chi2")))
        self._last_median_tau = _round_median_tau(locals().get("lt_h5_path"))
        self._last_volume_ratios = _round_volume_ratios(plot_posterior_callback)
        # Append this round's final entropy per marginal to the cross-round
        # history that drives entropy-plateau (cold-restart) reinit.
        for k, h in _round_entropies(plot_posterior_callback).items():
            self._entropy_history.setdefault(k, []).append(float(h))
        # Round-end empirical coverage of each marginal's truncation box over the
        # (up to) 1000-sample val pool — drives the truncation veto in run().
        self._last_coverage = self._compute_round_coverage(round_idx)
        self._persist_round_state(round_idx)
        opt = self.model.optimizers()
        if isinstance(opt, list):
            opt = opt[0]
        if len(opt.param_groups) >= 2:
            lr_info = (f"lr_ae={opt.param_groups[0]['lr']:.2e}, "
                       f"lr_nre={opt.param_groups[1]['lr']:.2e}")
        else:
            lr_info = f"lr={opt.param_groups[0]['lr']:.2e}"
        summary = (f"[Round {round_idx}] Stopped after epoch {trainer.current_epoch}. "
                   f"Reason: {stop_reason}. Final {lr_info}.")
        print(summary)

        # Append to a persistent logfile so the reason is preserved across rounds
        logfile = os.path.join(DATA_ROOT_DIR, TIME_OF_EXECUTION, "round_summary.log")
        os.makedirs(os.path.dirname(logfile), exist_ok=True)
        with open(logfile, "a") as _lf:
            _lf.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  {summary}\n")

        if hasattr(self.model, "widest_boxes"):
            print(f"Widest boxes after training: {self.model.widest_boxes}")

        # Copy YAML sidecar to log directory
        shutil.copy(self.data_fname_yaml, logger.log_dir)

        # Update the persisted autoencoder reference for next round
        self._autoencoder = self.model.autoencoder

        # Free cached data (HDF5 path only; streaming has no full_dataset)
        if hasattr(self.data_module, "full_dataset"):
            self.data_module.full_dataset.clear_cache()

    # -----------------------------------------------------------------
    # Round  (data generation + joint training)
    # -----------------------------------------------------------------

    def round(self, idx, sampler_init_kwargs):
        self._generate_data(round_idx=idx, sampler_init_kwargs=sampler_init_kwargs)
        self._train_joint(round_idx=idx)

    # -----------------------------------------------------------------
    # Run  (multi-round loop, same prior-update logic as tmnre.py)
    # -----------------------------------------------------------------

    def run(self, n_rounds=1, start_round=1):
        for i in range(start_round, start_round + n_rounds):
            print(f"Running round {i}...")
            # Resolve scheduled marginals for this round
            active_marginals = resolve_marginals_for_round(self.train_conf, i)
            self.train_conf["marginals"] = active_marginals
            validate_marginals(active_marginals)
            print(f"Active marginals for round {i}: {active_marginals}")

            dist_uniform_in_volume = self.datagen_conf.get("prior_dist_volumetric", True)
            spin_param_basis = self.datagen_conf.get("spin_param_basis", "chi1chi2")
            if i == 1 and self.fisher_prior_bounds is not None:
                self.datagen_conf["prior"].update(copy.deepcopy(self.fisher_prior_bounds))
                sampler_kwargs = {"prior_bounds": self.fisher_prior_bounds,
                                  "dist_uniform_in_volume": dist_uniform_in_volume,
                                  "spin_param_basis": spin_param_basis}
                print("[Fisher] Using Fisher-based prior for round 1.")
                import yaml as _yaml
                _out = os.path.join(DATA_ROOT_DIR, TIME_OF_EXECUTION, "fisher_prior_round_1.yaml")
                os.makedirs(os.path.dirname(_out), exist_ok=True)
                with open(_out, "w") as _f:
                    _yaml.safe_dump({"fisher_prior_bounds": self.fisher_prior_bounds}, _f)
                print(f"[Fisher] Saved Fisher prior bounds to {_out}")
            else:
                sampler_kwargs = {"prior_bounds": self.datagen_conf["prior"],
                                  "dist_uniform_in_volume": dist_uniform_in_volume,
                                  "spin_param_basis": spin_param_basis}

            self.round(idx=i, sampler_init_kwargs=sampler_kwargs)

            # ---- Update prior from posterior contours -------------------
            if self.train_conf["device"] == "cuda":
                torch.cuda.empty_cache()
            # NOTE: the per-marginal PP plot (utils.pp_plot) used to live here
            # but is now produced by the PPKS callback as an overlay every
            # ``run_every_n_epochs`` cumulative epochs.
            out_idx = 0
            # Parameter names follow the run's spin basis (slots 2,3). The prior
            # dict is keyed with these names, so truncation must write the same.
            prior_keys = utils.ordered_prior_keys(
                self.datagen_conf.get("spin_param_basis", "chi1chi2"))
            # Coverage-gated truncation veto: skip narrowing a parameter whose
            # round-end box coverage over the val pool is below threshold (the
            # network is overconfident, so its box would risk excluding the truth).
            veto_conf = self.train_conf.get("truncation_veto", {})
            veto_enabled = veto_conf.get("enabled", False)
            min_cov = float(veto_conf.get("min_coverage", 0.95))
            vetoed_this_round = []   # marginals whose truncation was skipped
            for key, marginal_list in self.train_conf["marginals"].items():
                for marginal in marginal_list:
                    marginal_key = tuple(marginal)

                    cov_entry = self._last_coverage.get(marginal_key)
                    if (veto_enabled and cov_entry is not None
                            and cov_entry[0] < min_cov):
                        name = "-".join(prior_keys[j] for j in marginal_key)
                        print(f"[TruncVeto] round {i}: keeping prior for {name} "
                              f"(coverage {cov_entry[0]:.3f} < {min_cov}); "
                              f"truncation skipped.", flush=True)
                        vetoed_this_round.append({
                            "name": name,
                            "marginal": list(marginal_key),
                            "coverage": float(cov_entry[0]),
                            "n_inside": int(cov_entry[1]),
                            "n_total": int(cov_entry[2]),
                        })
                        out_idx += 1
                        continue

                    if len(marginal) == 1:
                        param_name = prior_keys[marginal[0]]
                        if hasattr(self.model, "widest_boxes") and marginal_key in self.model.widest_boxes:
                            widest_interval = self.model.widest_boxes[marginal_key]
                            tmp = copy.deepcopy(self.datagen_conf["prior"])
                            tmp[param_name] = [widest_interval[0], widest_interval[1]]
                            self.datagen_conf["prior"] = tmp
                        else:
                            print(f"Warning: No widest_interval for 1D marginal {marginal_key} ({param_name})")

                    elif len(marginal) == 2:
                        inj1, inj2 = marginal
                        if marginal_key == (7, 8):  # sky marginal  
                            print("truncating sky prior using pembhb.sky_truncation.truncate_sky_prior() ...")
                            from pembhb.sky_truncation import truncate_sky_prior
                            _, sky_info = truncate_sky_prior(
                                self.model, self.dataloader_obs, out_param_idx=out_idx,
                                datagen_conf=self.datagen_conf,
                                mode="rectangle",
                                credible_level=0.999, dilation_factor=1.1,
                            )
                        elif hasattr(self.model, "widest_boxes") and marginal_key in self.model.widest_boxes: 
                            print(f"Updating prior for 2D marginal {marginal_key} using widest box from model ...")
                            widest_box = self.model.widest_boxes[marginal_key]
                            tmp = copy.deepcopy(self.datagen_conf["prior"])
                            tmp[prior_keys[inj1]] = [widest_box[0], widest_box[1]]
                            tmp[prior_keys[inj2]] = [widest_box[2], widest_box[3]]
                            self.datagen_conf["prior"] = tmp
                        else:
                            print(f"Warning: No widest_box for 2D marginal {marginal_key}")

                    out_idx += 1

            print(f"Updated prior after round {i}: {self.datagen_conf['prior']}")
            self._plot_updated_prior_bounds(self.datagen_conf["prior"])

            # Log which marginals had their truncation vetoed this round (empty
            # list when the veto is off or nothing was vetoed) for review.
            if veto_enabled:
                names = [v["name"] for v in vetoed_this_round]
                print(f"[TruncVeto] round {i}: vetoed {len(vetoed_this_round)} "
                      f"marginal(s): {names}", flush=True)
                import yaml as _yaml
                veto_save_path = os.path.join(
                    DATA_ROOT_DIR, TIME_OF_EXECUTION,
                    f"vetoed_params_after_round_{i}.yaml")
                os.makedirs(os.path.dirname(veto_save_path), exist_ok=True)
                with open(veto_save_path, "w") as _f:
                    _yaml.safe_dump({
                        "round": int(i),
                        "min_coverage": min_cov,
                        "vetoed": vetoed_this_round,
                    }, _f)

            # Persist the updated prior so a resumed run can start from
            # the correct (narrowed) prior rather than the round's data prior.
            import yaml as _yaml
            prior_save_path = os.path.join(DATA_ROOT_DIR, TIME_OF_EXECUTION, f"prior_after_round_{i}.yaml")
            os.makedirs(os.path.dirname(prior_save_path), exist_ok=True)
            #this line exist because numpy stuff can't be dumped into yaml
            prior_plain = {k: [float(v[0]), float(v[1])] for k, v in self.datagen_conf["prior"].items()}
            with open(prior_save_path, "w") as _f:
                _yaml.safe_dump({"prior": prior_plain}, _f)

            # ---- Sanity check: does the new window still contain the truth? --
            # The proposal for round i+1 is self.datagen_conf["prior"].  If the
            # observation's true value for a *truncated* parameter has fallen
            # outside its new [lo, hi], every subsequent round is sampling a
            # region that excludes the truth and inference is doomed.  Check
            # only the parameters that were actually narrowed this round (those
            # appearing in the active marginals); untouched params keep the full
            # prior and cannot miss.
            truncated_idxs = set()
            for marginal_list in self.train_conf["marginals"].values():
                for marginal in marginal_list:
                    truncated_idxs.update(marginal)

            true_params = self.dataset_observation[0]["params"]
            # Names follow the obs dataset's own recorded spin basis, so slots
            # 2,3 truth values are interpreted in the same coordinates as the
            # prior dict (which is keyed by the run's basis).
            obs_keys = utils.ordered_prior_keys(
                getattr(self.dataset_observation.dataset, "spin_param_basis", "chi1chi2"))
            violations = []
            for idx in sorted(truncated_idxs):
                param_name = obs_keys[idx]
                lo, hi = self.datagen_conf["prior"][param_name]
                true_val = float(true_params[idx])
                if not (lo <= true_val <= hi):
                    violations.append((param_name, true_val, float(lo), float(hi)))

            if violations:
                detail = "\n".join(
                    f"    {name}: true={true_val:.6g} outside [{lo:.6g}, {hi:.6g}]"
                    for name, true_val, lo, hi in violations
                )
                raise ValueError(
                    f"[Truncation] Round {i} proposal window misses the true value "
                    f"for {len(violations)} parameter(s):\n{detail}\n"
                    f"The truncated prior (saved to {prior_save_path}) excludes the "
                    "observation's true parameters; subsequent rounds cannot recover them."
                )

            if self.train_conf["device"] == "cuda":
                torch.cuda.empty_cache()

            # ---- Campaign-level convergence: stop the whole chain? ----------
            if self.chain_monitor is not None and self._last_marginal_entropies:
                stop_chain = self.chain_monitor.update(
                    i, self._last_marginal_entropies,
                    stopped_via=self._last_stopped_via,
                    median_tau=self._last_median_tau,
                )
                n_conv = len(self.chain_monitor.converged_at)
                n_tot = len(self._last_marginal_entropies)
                print(f"[ChainConv] round {i}: stopped_via='{self._last_stopped_via}', "
                      f"median_tau={self._last_median_tau}, "
                      f"converged {n_conv}/{n_tot} marginals")
                if stop_chain:
                    print(f"[ChainConv] STOP chain after round {i}: "
                          f"{self.chain_monitor.stop_reason}")
                    print(self.chain_monitor.summary())
                    break

        # Per-marginal convergence record at end of the campaign (converged or
        # simply out of rounds).
        if self.chain_monitor is not None:
            print(self.chain_monitor.summary())


# -------------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-config", default="train_config.yaml",
                        help="Train config filename inside configs/")
    parser.add_argument("--datagen-config", default="datagen_config.yaml",
                        help="Datagen config filename inside configs/ (default: datagen_config.yaml)")
    parser.add_argument("--resume", default=None,
                        help="Resume a previous run: pass the exact TIME_OF_EXECUTION string "
                             "(e.g. 2026/03/31/autoencoder_joint_v1). The last completed round "
                             "is detected automatically.")
    parser.add_argument("--n_rounds", type=int, default=10,
                        help="Number of rounds to run (default: 10)")
    parser.add_argument("--obs-path", required=True,
                        help="Path to the observation HDF5 file. The file should contain "
                             "a stored 'noise_fd' dataset for reproducible posteriors "
                             "(see scripts/add_noise_to_obs.py).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Master RNG seed for reproducibility (seeds torch, numpy, "
                             "python random via Lightning seed_everything). Default: 42.")
    parser.add_argument("name", help="Unique name for this run (appended to timestamps in logs and plots). "
                                     "Ignored when --resume is used.")

    args = parser.parse_args()

    train_config_filename = args.train_config
    run_name = args.name
    datagen_config_filename = args.datagen_config

    train_config = utils.read_config(os.path.join(ROOT_DIR, "configs", train_config_filename))
    datagen_config = utils.read_config(os.path.join(ROOT_DIR, "configs", datagen_config_filename))

    # Activate the configured precision (default: float32)
    set_precision(train_config.get("precision", "float32"))

    # Fix all RNG seeds for reproducibility (torch, numpy, python random, CUDA)
    seed_everything(args.seed, workers=True)

    ds_type = train_config["architecture"]["data_summary"]["type"].lower()

    if args.resume is None:
        # Fresh run: build a new timestamped name.
        TIME_OF_EXECUTION = get_timestamp() + f"/{ds_type}_{run_name}"
        trainer = SequentialTrainerJoint(
            train_conf=train_config,
            datagen_conf=datagen_config,
            dataset_obs_path=args.obs_path,
            seed=args.seed,
        )
        trainer.run(n_rounds=args.n_rounds)
    else:
        # Resume: keep the original run name so logs go to the same directory.
        TIME_OF_EXECUTION = args.resume
        last_round = _discover_last_round(TIME_OF_EXECUTION, DATA_ROOT_DIR)
        if last_round == 0:
            raise RuntimeError(
                f"No completed rounds found for run '{TIME_OF_EXECUTION}'. "
                f"Check that the log directory exists under {DATA_ROOT_DIR}/logs/."
            )
        print(f"[Resume] Run '{TIME_OF_EXECUTION}': last completed round = {last_round}. "
              f"Continuing for {args.n_rounds} more round(s).")
        trainer = SequentialTrainerJoint(
            train_conf=train_config,
            datagen_conf=datagen_config,
            dataset_obs_path=args.obs_path,
            resume=True,
            seed=args.seed,
        )
        trainer._restore_from_checkpoint(TIME_OF_EXECUTION, last_round)
        trainer.run(n_rounds=args.n_rounds, start_round=last_round + 1)
