"""Shared helpers for plot_posterior.py and the visualise_*.py scripts.

Pure helpers: filesystem discovery of round directories, checkpoint loading
(with optional CLI override), prior-box reading, parameter-marginal listing, the observation loader.
No plotting code.
"""

import os
import re
from glob import glob

import numpy as np
import yaml

from torch.utils.data import DataLoader, Subset

from pembhb.data import MBHBDataset
from pembhb.model import InferenceNetwork, load_inference_network
from pembhb.utils import _ORDERED_PRIOR_KEYS, eval_posterior_2d, ordered_prior_keys, mbhb_collate_fn


from pembhb import DATA_ROOT_DIR

BASE_LOG_DIR = os.path.join(DATA_ROOT_DIR, "logs")


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

def _same_observation(a: str, b: str) -> bool:
    """Same file, or a copy holding identical signal and stored noise."""
    if os.path.realpath(a) == os.path.realpath(b):
        return True
    import h5py
    with h5py.File(a, "r") as fa, h5py.File(b, "r") as fb:
        for k in ("wave_fd", "noise_fd"):
            if (k in fa) != (k in fb):
                return False
            if k in fa and not np.array_equal(fa[k][()], fb[k][()]):
                return False
    return True


def resolve_obs_path(name: str, obs_path: str | None = None) -> str:
    """The observation the run was trained on (``observation_used.yaml``).

    A user-supplied ``obs_path`` is accepted only if it is that observation;
    a different one raises, since the posterior would not belong to it.
    """
    record = os.path.join(DATA_ROOT_DIR, name, "observation_used.yaml")
    used = None
    if os.path.exists(record):
        with open(record) as f:
            used = (yaml.safe_load(f) or {}).get("obs_path")
    if used is None:
        if obs_path is None:
            raise FileNotFoundError(
                f"{record} not found (older run?): pass the observation explicitly.")
        print(f"[obs] no observation_used.yaml for '{name}'; trusting {obs_path}")
        return obs_path
    if obs_path is None:
        if not os.path.exists(used):
            raise FileNotFoundError(f"observation {used} recorded for '{name}' no longer exists; pass it explicitly.")
        return used
    if os.path.exists(used) and not _same_observation(obs_path, used):
        raise ValueError(
            f"{obs_path} is not the observation run '{name}' was trained on ({used}).")
    return obs_path


def build_obs_dataloader(data_path: str) -> DataLoader:
    """Single-observation loader using the stored noise realisation."""
    ds = MBHBDataset(data_path, cache_in_memory=False)
    if not ds.has_stored_noise:
        print(f"[warn] {data_path} has no stored 'noise_fd'; posteriors will use "
              f"freshly-drawn noise on every call.")
    return DataLoader(
        Subset(ds, indices=[0]),
        batch_size=1, shuffle=False,
        collate_fn=lambda b: mbhb_collate_fn(
            b, ds.noise_scale, noise_factor=1.0, noise_shuffling=False,
        ),
    )


# ---------------------------------------------------------------------------
# Round directory discovery
# ---------------------------------------------------------------------------

def _run_name_from_round_dir(round_dir: str, base_log_dir: str = BASE_LOG_DIR) -> str:
    """Recover the run name (TIME_OF_EXECUTION) from a version directory path.

    Supports both nested (``{base}/{name}/round_<N>/version_<M>``) and legacy
    flat (``{base}/{name}_round_<N>/version_<M>``) layouts.
    """
    parent = os.path.dirname(round_dir.rstrip("/"))
    rel = os.path.relpath(parent, base_log_dir)
    nested = re.match(r"^(.*)/round_\d+$", rel)
    if nested:
        return nested.group(1)
    flat = re.match(r"^(.*)_round_\d+$", rel)
    if flat:
        return flat.group(1)
    raise ValueError(f"Cannot recover run name from round_dir={round_dir!r}")


def _sidecar_yaml_path(round_dir: str, round_number: int) -> str:
    name = _run_name_from_round_dir(round_dir)
    return os.path.join(DATA_ROOT_DIR, name, f"simulation_round_{round_number}.yaml")


def find_round_dirs(name: str, base_log_dir: str = BASE_LOG_DIR) -> list:
    """Return sorted list of version dirs (one per round) for a given run name.

    Tries the nested layout first, then the legacy flat layout. Requires both
    a checkpoint (or override) in the version dir AND a sidecar YAML in the
    data dir for each round to be included.
    """
    flat = sorted(glob(os.path.join(base_log_dir, f"{name}_round_*")))
    nested = sorted(glob(os.path.join(base_log_dir, name, "round_*")))
    candidates = flat or nested
    out = []
    for d in candidates:
        m = re.search(r"[/_]round_(\d+)$", d)
        if not m:
            continue
        rn = int(m.group(1))
        versions = sorted(
            (int(mv.group(1))
             for entry in os.listdir(d)
             for mv in [re.match(r"^version_(\d+)$", entry)] if mv),
            reverse=True,
        )
        if not versions:
            continue
        sidecar_ok = os.path.isfile(
            os.path.join(DATA_ROOT_DIR, name, f"simulation_round_{rn}.yaml")
        )
        chosen = None
        for v in versions:
            vdir = os.path.join(d, f"version_{v}")
            if sidecar_ok and glob(os.path.join(vdir, "checkpoints", "*.ckpt")):
                chosen = vdir
                break
        if chosen:
            out.append((rn, chosen))
        else:
            print(f"  WARNING: no usable version in {d} — skipping.")
    out.sort(key=lambda x: x[0])
    return [v for _, v in out]


def round_dirs_upto(name: str, last_round: int | None = None) -> list:
    """Version dirs for rounds 1..last_round (all rounds if None); raises on gaps."""
    dirs = find_round_dirs(name)
    if not dirs:
        raise RuntimeError(f"No round directories found for run '{name}' under {BASE_LOG_DIR}.")
    if last_round is not None:
        if not 1 <= last_round <= len(dirs):
            raise ValueError(f"round {last_round} out of range [1, {len(dirs)}] for '{name}'.")
        dirs = dirs[:last_round]
    return dirs


def load_duration_weeks(round_dir: str, round_number: int) -> float:
    with open(_sidecar_yaml_path(round_dir, round_number)) as f:
        conf = yaml.safe_load(f)
    wf = conf.get("waveform_params") or conf["conf"]["waveform_params"]
    return float(wf["duration"])


def load_prior_box(round_dir: str, round_number: int) -> dict:
    """Return {param: (low, high)} from the round's sidecar YAML."""
    with open(_sidecar_yaml_path(round_dir, round_number)) as f:
        conf = yaml.safe_load(f)
    prior = conf.get("prior") or conf["conf"]["prior"]
    return {k: tuple(v) for k, v in prior.items()}


# ---------------------------------------------------------------------------
# Model loading with optional final-round override
# ---------------------------------------------------------------------------

# CLI-driven: maps a realpath(ckpt_dir) → an explicit .ckpt file.
_CKPT_OVERRIDES: dict[str, str] = {}


def load_model(ckpt_dir: str) -> InferenceNetwork:
    """Load checkpoint for the given ckpt_dir.

    Resolution order:
      1. _CKPT_OVERRIDES[realpath(ckpt_dir)]
      2. {ckpt_dir}/../truncation.ckpt  (end-of-round model)
      3. First *.ckpt in ckpt_dir.
    """
    norm = os.path.realpath(ckpt_dir)
    override = _CKPT_OVERRIDES.get(norm)
    if override and os.path.exists(override):
        print(f"[load_model] override: loading {override}")
        return load_inference_network(override)
    trunc = os.path.join(ckpt_dir, "../truncation.ckpt")
    if os.path.exists(trunc):
        return load_inference_network(trunc)
    print(f"WARNING: truncation.ckpt not found in {os.path.dirname(ckpt_dir)}; "
          f"falling back to first .ckpt in {ckpt_dir}.")
    ckpts = glob(os.path.join(ckpt_dir, "*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No .ckpt in {ckpt_dir}")
    model = load_inference_network(ckpts[0])
    try:
        model.data_summary.autoencoder.architecture = "conv"
    except AttributeError:
        pass
    return model


def register_ckpt_override(round_dir: str, ckpt_path: str) -> None:
    """Register a final-round checkpoint override at the realpath of its ckpt dir."""
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"ckpt override path does not exist: {ckpt_path}")
    norm = os.path.realpath(os.path.join(round_dir, "checkpoints"))
    _CKPT_OVERRIDES[norm] = ckpt_path
    print(f"[ckpt-override] {norm} ← {ckpt_path}")


# ---------------------------------------------------------------------------
# Model introspection
# ---------------------------------------------------------------------------

def detect_basis(box: dict) -> str:
    """Return ``"chieff_chidiff"`` if the prior/box dict carries those keys,
    else ``"chi1chi2"`` (default/legacy)."""
    return "chieff_chidiff" if ("chi_eff" in box or "chi_diff" in box) else "chi1chi2"


def keys_for_model(model: InferenceNetwork) -> list:
    """Ordered 11 parameter names for the basis the model was trained on.

    The model's stored ``bounds_trained`` carries the prior keys used at
    training; pick the spin pair from those.
    """
    return ordered_prior_keys(detect_basis(getattr(model, "bounds_trained", {}) or {}))


def get_all_marginals(model: InferenceNetwork) -> list:
    """Return [(label, ndim, in_param_idx, out_param_idx), ...] for every head.

    ``in_param_idx`` is an int for 1-D, a tuple for 2-D. Labels reflect the
    model's basis (e.g. chi_eff / chi_diff at slots 2,3 when applicable).
    """
    keys = keys_for_model(model)
    out = []
    for out_idx, marginal in enumerate(model.marginals_list):
        ndim = len(marginal)
        if ndim == 1:
            out.append((keys[marginal[0]], 1, marginal[0], out_idx))
        elif ndim == 2:
            label = f"{keys[marginal[0]]} vs {keys[marginal[1]]}"
            out.append((label, 2, tuple(marginal), out_idx))
    return out


def find_out_param_idx(model: InferenceNetwork, in_param_idx):
    """Return output column for *in_param_idx*, or None if not in this model."""
    for out_idx, marginal in enumerate(model.marginals_list):
        ndim = len(marginal)
        if (marginal[0] if ndim == 1 else tuple(marginal)) == in_param_idx:
            return out_idx
    return None


# ---------------------------------------------------------------------------
# 2-D posterior evaluation
# ---------------------------------------------------------------------------
# Thin wrapper kept for callers that prefer (norm2d_with_batch, inj, gx, gy).
# The underlying logic lives in ``pembhb.utils.eval_posterior_2d``.

def compute_normalised_posterior(
    dataloader, model, in_param_idx, out_param_idx, bounds_0, bounds_1, ngrid_points=100,
):
    """Evaluate and normalise the 2-D posterior on a regular grid.

    Returns ``(norm2d, inj_params, gx, gy)`` with ``norm2d`` shape
    ``(batch, ngrid, ngrid)`` integrating to 1.
    """
    return eval_posterior_2d(
        model, dataloader, in_param_idx, out_param_idx,
        ngrid_points=ngrid_points,
        bounds_0=bounds_0, bounds_1=bounds_1,
        keep_batch_dim=True,
    )


# ---------------------------------------------------------------------------
# Paper styling (shared by the entropy / truncation figures)
# ---------------------------------------------------------------------------

PT_PER_INCH = 72.27
TEXTWIDTH_PT = 2 * 246.0   # two-column width of the paper style

# Paul Tol's "muted" qualitative scheme (colour-blind safe), + black as a 10th.
# https://personal.sron.nl/~pault/  — distinguishable under deuteranopia,
# protanopia and tritanopia. Markers add a second, colour-independent cue.
TOL_MUTED = ["#332288", "#88CCEE", "#44AA99", "#117733", "#999933",
             "#DDCC77", "#CC6677", "#882255", "#AA4499", "#000000"]
MARKERS = ["o", "s", "^", "v", "D", "P", "X", "*", "<", ">"]

# Parameter labels in maths mode. ``inc`` is sampled as cos(iota) and ``beta``
# as sin(beta) — the labels name the sampled quantity, not the angle.
LABELS = {
    "logMchirp": r"$\log_{10}(\mathcal{M}_c/M_\odot)$",
    "q":         r"$q$",
    "chi1":      r"$\chi_1$",
    "chi2":      r"$\chi_2$",
    "chi_eff":   r"$\chi_\mathrm{eff}$",
    "chi_diff":  r"$\chi_\mathrm{diff}$",
    "dist":      r"$d_L$",
    "phi":       r"$\phi$",
    "cosinc":       r"$\cos\iota$",
    "lambda":    r"$\lambda$",
    "sinbeta":      r"$\sin\beta$",
    "psi":       r"$\psi$",
    "Deltat":    r"$\Delta t$",
}


def latex_label(label: str) -> str:
    """Map a marginal label ('dist', 'lambda vs beta') to maths mode."""
    if " vs " in label:
        a, b = label.split(" vs ")
        return f"{latex_label(a)} $\\times$ {latex_label(b)}"
    return LABELS.get(label, label.replace("_", r"\_"))


def apply_paper_style(fontsize: float, uniform: bool = False) -> None:
    """Set the shared paper rcParams.

    By default tick/legend labels are shrunk a bit relative to ``fontsize``
    (historical look for the multi-panel truncation/ratio figures). Pass
    ``uniform=True`` to keep ticks, legend and axis labels all at exactly
    ``fontsize`` (e.g. for a journal single-column figure spec).
    """
    import matplotlib.pyplot as plt

    tick_size = fontsize if uniform else fontsize - 3
    legend_size = fontsize if uniform else fontsize - 2
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": fontsize,
        "axes.titlesize": fontsize,
        "axes.labelsize": fontsize,
        "xtick.labelsize": tick_size,
        "ytick.labelsize": tick_size,
        "legend.fontsize": legend_size,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "lines.linewidth": 1.0,
    })


def cumulative_train_hours(round_dirs: list) -> np.ndarray:
    """Cumulative wall-clock training time [h] at the end of each round.

    Per round the duration is (last − first) scalar-event wall time in that
    version dir, so the gaps between rounds (data generation, plotting,
    queueing) are excluded.
    """
    from tensorboard.backend.event_processing import event_accumulator

    durations = []
    for rd in round_dirs:
        ea = event_accumulator.EventAccumulator(
            rd, size_guidance={event_accumulator.SCALARS: 0},
        )
        ea.Reload()
        times = [t for tag in ea.Tags()["scalars"]
                 for ev in (ea.Scalars(tag),) for t in (ev[0].wall_time,
                                                        ev[-1].wall_time)]
        durations.append(max(times) - min(times) if times else np.nan)
    return np.nancumsum(np.asarray(durations, dtype=float)) / 3600.0


def add_time_axis(ax, rounds, cum_h, label, nbins: int = 3,
                  min_sep_frac: float = 0.15, highlight_final: bool = False,
                  final_min_sep_frac: float = 0.12):
    """Top secondary x-axis showing cumulative training time.

    Rounds get slower as the run goes on, so equally-spaced *time* ticks bunch
    up at the right of the axis; ticks closer than ``min_sep_frac`` of the axis
    span to their predecessor are dropped.

    ``highlight_final=True`` adds an explicit bold tick at the final round's
    cumulative training time (dropping any auto tick within
    ``final_min_sep_frac`` of it), colored ``C0`` to match a matching dotted
    guide line drawn on the primary axis at the final round.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    rounds = np.asarray(rounds, dtype=float)
    cum_h = np.asarray(cum_h, dtype=float)
    r2t = lambda r: np.interp(r, rounds, cum_h)
    t2r = lambda t: np.interp(t, cum_h, rounds)
    sec = ax.secondary_xaxis("top", functions=(r2t, t2r))

    span = rounds[-1] - rounds[0]
    kept = []
    for t in MaxNLocator(nbins=nbins, integer=True).tick_values(
            cum_h[0], cum_h[-1]):
        if not cum_h[0] - 1e-9 <= t <= cum_h[-1] + 1e-9:
            continue
        x = float(t2r(t))
        if kept and abs(x - kept[-1][1]) < min_sep_frac * span:
            continue
        kept.append((t, x))
    sec.set_xticks([t for t, _ in kept])

    if highlight_final:
        t_last = float(cum_h[-1])
        tspan = float(cum_h[-1] - cum_h[0])
        auto = [t for t in sec.get_xticks() if abs(t - t_last) > final_min_sep_frac * tspan]
        sec.set_xticks(auto + [t_last])
        sec.set_xticklabels([f"{t:.0f}" for t in auto]
                            + [rf"$\mathbf{{{t_last:.1f}}}$"])
        for lbl in sec.get_xticklabels()[len(auto):]:
            lbl.set_color("C0")
        ax.axvline(rounds[-1], color="C0", linestyle=":", linewidth=0.7, alpha=0.6)

    sec.tick_params(labelsize=plt.rcParams["xtick.labelsize"])
    if label:
        sec.set_xlabel(label, labelpad=2)
    return sec


def save_figure(fig, outdir: str, stem: str) -> str:
    """Save PDF + PNG at the exact declared figure size (no bbox trimming, so
    the width really is the requested column width)."""
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, f"{stem}.pdf")
    fig.savefig(out_path)
    fig.savefig(out_path.replace(".pdf", ".png"), dpi=300)
    plt.close(fig)
    print(f"saved {out_path} (+ .png)")
    return out_path
