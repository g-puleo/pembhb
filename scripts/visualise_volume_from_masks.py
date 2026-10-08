"""Prior-volume evolution computed from the *stored truncation masks*.

This is the a-posteriori correction to ``visualise_volume_ratio_evolution.py``.
That script multiplies the TensorBoard ``volume_ratio`` scalar, whose numerator
is the raw posterior credible region — NOT clipped to the previous accepted set
— so it can exceed 1 (the posterior reclaiming mass outside the sampled mask via
the multimode re-evaluation) and the cumulative product is non-monotone.

Here we ignore that scalar entirely and read the truncation output itself:

  * 1-D accepted sets (possibly gapped intervals) from
    ``prior_after_round_{i}.yaml`` -> ``intervals_1d`` (keyed by param index);
  * 2-D accepted regions from ``truncation_round_{i}.npz`` (both npz formats;
    with ``truncation.refine: true`` each mode lives on its own subgrid).

Volume of a marginal in round *i* is the measure of its accepted set (1-D: total
interval width; 2-D: ``Σ mask·dV`` over every mode's own cells — never a
rasterisation onto a common grid, which moves refined edges by half a cell). The 11 physical parameters
are partitioned into 9 singletons + the ``(lambda, beta)`` pair, so the product
over marginals is a genuine 11-D volume.

The monotone truncation invariant ``A_N ⊆ A_{N-1}`` is re-enforced here by
intersecting each round's accepted set with every earlier one (evaluated on
the current round's cells). When the run already stored monotone masks (the
``clip_*_to_prev`` step was active) this is a no-op; it only bites if a future
run stores a mask that grew.

Usage
-----
    python scripts/visualise_volume_from_masks.py NAME [--last-round N]
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, MaxNLocator, NullLocator

from pembhb import ROOT_DIR, PLOTS_ROOT_DIR
from pembhb.mask_truncation import load_pair_region

from _visualise_common import (
    DATA_ROOT_DIR,
    find_round_dirs,
    load_model,
    keys_for_model,
    PT_PER_INCH,
    latex_label,
    apply_paper_style,
    save_figure as _save,
    cumulative_train_hours,
    add_time_axis as _add_time_axis,
    TOL_MUTED,
    MARKERS,
)


# --------------------------------------------------------------------------- #
# accepted-set measures
# --------------------------------------------------------------------------- #
def _intersect_intervals(a, b):
    """Intersection of two lists of ``[lo, hi]`` intervals."""
    out = []
    for a0, a1 in a:
        for b0, b1 in b:
            lo, hi = max(a0, b0), min(a1, b1)
            if hi > lo:
                out.append([lo, hi])
    return out


def _measure_1d(ivs):
    return float(sum(hi - lo for lo, hi in ivs))


def _intersected_area(region, previous):
    """Exact measure of ``region ∩ previous[0] ∩ previous[1] ...``.

    ``Σ_cells mask · Π_k prev_k.contains(centre) · dV`` over every part, on
    each part's own cells, so refined modes are measured at their native pitch.
    """
    total = 0.0
    for part in getattr(region, "parts", (region,)):
        m = part.mask.copy()
        for prev in previous:
            m &= prev.contains_grid(part.grids)
        total += part.volume(m)
    return total


# --------------------------------------------------------------------------- #
# per-run collection
# --------------------------------------------------------------------------- #
def _round_indices(name):
    i, out = 1, []
    while os.path.exists(os.path.join(DATA_ROOT_DIR, name,
                                      f"prior_after_round_{i}.yaml")):
        out.append(i)
        i += 1
    return out


def _initial_box(name):
    """Round-1 datagen prior box {name: (lo, hi)} — the reference volume."""
    with open(os.path.join(DATA_ROOT_DIR, name, "simulation_round_1.yaml")) as f:
        conf = yaml.safe_load(f)
    prior = conf.get("prior") or conf["conf"]["prior"]
    return {k: (float(v[0]), float(v[1])) for k, v in prior.items()}


def collect_mask_volumes(name, keys, last_round=None):
    """Return (labels, V[n_rounds, n_marginals]) of intersected accepted volumes.

    Column order: the 1-D marginals (in stored index order) then the 2-D pair.
    Volumes are absolute measures (rad, rad^2, ...); the caller normalises.
    """
    rounds = _round_indices(name)
    if last_round:
        rounds = [r for r in rounds if r <= last_round]
    if not rounds:
        raise RuntimeError(f"No prior_after_round_*.yaml for '{name}'.")

    # discover the marginal layout from the first round
    first = yaml.safe_load(open(
        os.path.join(DATA_ROOT_DIR, name, f"prior_after_round_{rounds[0]}.yaml")))
    idx_1d = sorted(int(k) for k in first["intervals_1d"])
    pairs_2d = [tuple(p) for p in first.get("pairs_2d", [])]
    labels = [keys[i] for i in idx_1d] + \
             [f"{keys[i]} vs {keys[j]}" for i, j in pairs_2d]

    run_1d = {i: None for i in idx_1d}          # running intersected intervals
    run_2d = {p: [] for p in pairs_2d}           # every earlier round's region
    rows = []
    for r in rounds:
        yml = yaml.safe_load(open(
            os.path.join(DATA_ROOT_DIR, name, f"prior_after_round_{r}.yaml")))
        row = []
        for i in idx_1d:
            ivs = [[float(lo), float(hi)] for lo, hi in yml["intervals_1d"][i]]
            run_1d[i] = ivs if run_1d[i] is None else _intersect_intervals(run_1d[i], ivs)
            row.append(_measure_1d(run_1d[i]))

        if pairs_2d:
            npz_path = os.path.join(DATA_ROOT_DIR, name,
                                    f"truncation_round_{r}.npz")
            for (i, j) in pairs_2d:
                # handles both npz formats; format 2 keeps each mode on its own subgrid
                region = load_pair_region(npz_path, i, j)
                row.append(_intersected_area(region, run_2d[(i, j)]))
                run_2d[(i, j)].append(region)
        rows.append(row)

    return rounds, labels, idx_1d, pairs_2d, np.asarray(rows, dtype=float)


def reference_volumes(name, keys, idx_1d, pairs_2d):
    """Initial-prior measure of each marginal, same column order as the data."""
    box = _initial_box(name)
    ref = [box[keys[i]][1] - box[keys[i]][0] for i in idx_1d]
    for (i, j) in pairs_2d:
        ref.append((box[keys[i]][1] - box[keys[i]][0]) *
                   (box[keys[j]][1] - box[keys[j]][0]))
    return np.asarray(ref, dtype=float)


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def _log_axis(ax, nticks=6):
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=nticks))
    ax.yaxis.set_minor_locator(NullLocator())


def plot_global(rounds, ratio_global, cum_h, outdir, width_pt, fontsize):
    """Product over marginals: the intersected 11-D volume vs the initial prior."""
    rounds = np.asarray(rounds)
    width_in = width_pt / PT_PER_INCH
    fig, ax = plt.subplots(figsize=(width_in, 0.62 * width_in),
                           constrained_layout=True)
    ax.plot(rounds, ratio_global, "-", color="C0", marker="o", markersize=2.5,
           linewidth=3.0)
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.7)
    ax.set_ylabel(r"$V_\mathrm{mask}/V_\mathrm{prior,\,1}$")
    ax.set_xlabel("round")
    ax.grid(True, linestyle=":", alpha=0.4, linewidth=0.5)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    _log_axis(ax)

    vmin = float(ratio_global[-1])
    k_lo = int(np.floor(np.log10(vmin)))
    stride = max(1, int(np.ceil(abs(k_lo) / 5)))
    decade_ticks = [10.0 ** k for k in range(0, k_lo - 1, -stride)]
    # Explicit final-point ytick (minimum volume, off-decade) alongside the
    # decade ticks, with a faint guide from the last marker to the left spine.
    ax.set_yticks(decade_ticks + [vmin])
    mant, exp = f"{vmin:.1e}".split("e")
    ax.set_yticklabels(
        [rf"$10^{{{int(round(np.log10(v)))}}}$" for v in decade_ticks]
        + [rf"$\mathbf{{{mant}\times10^{{{int(exp)}}}}}$"])
    ax.get_yticklabels()[-1].set_color("C0")
    ax.axhline(vmin, color="C0", linestyle=":", linewidth=0.7, alpha=0.6)
    ax.set_ylim(10.0 ** (k_lo - 1), 5.0)

    if cum_h is not None:
        cum_h = np.asarray(cum_h, dtype=float)
        _add_time_axis(ax, rounds, cum_h, label=r"training time [h]",
                       highlight_final=True)

    print(f"[global] final intersected volume {ratio_global[-1]:.3e} of the "
          f"initial prior (= 1 / {1.0 / ratio_global[-1]:.3e})")
    return _save(fig, outdir, "volume_from_masks_global")


def plot_per_marginal(rounds, labels, ratio, cum_h, outdir, width_pt, fontsize,
                      cols=2):
    """One panel per marginal, log y, of its accepted-set / initial ratio."""
    rounds = np.asarray(rounds)
    n = len(labels)
    nrows = int(np.ceil(n / cols))
    width_in = width_pt / PT_PER_INCH
    fig, axes = plt.subplots(nrows, cols, figsize=(width_in, 1.05 * nrows + 0.9),
                             squeeze=False, sharex=True, constrained_layout=True)
    for idx, label in enumerate(labels):
        ax = axes[idx // cols, idx % cols]
        ax.plot(rounds, ratio[:, idx], "-", color="darkorange")
        ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.7)
        ax.grid(True, linestyle=":", alpha=0.4, linewidth=0.5)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
        _log_axis(ax, nticks=4)
        if idx // cols == 0 and cum_h is not None:
            _add_time_axis(ax, rounds, cum_h,
                           label=r"$t_\mathrm{train}\,[\mathrm{h}]$")
            ax.set_title(latex_label(label), pad=fontsize * 2.6)
        else:
            ax.set_title(latex_label(label), pad=3)
    for k in range(n, nrows * cols):
        axes[k // cols, k % cols].set_visible(False)
    fig.supxlabel("round", fontsize=fontsize)
    fig.supylabel(r"$V_\mathrm{mask}/V_\mathrm{prior,\,1}$", fontsize=fontsize)
    return _save(fig, outdir, "volume_from_masks_per_marginal")


def plot_collapsed(rounds, labels, ratio, cum_h, outdir, width_pt, fontsize):
    """All marginals on one axis (per-marginal accepted / initial ratio)."""
    rounds = np.asarray(rounds)
    width_in = width_pt / PT_PER_INCH
    fig, ax = plt.subplots(figsize=(width_in, 0.45 * width_in),
                           constrained_layout=True)
    for idx, label in enumerate(labels):
        ax.plot(rounds, ratio[:, idx], "-", color=TOL_MUTED[idx % len(TOL_MUTED)],
                label=latex_label(label),
                marker=MARKERS[idx % len(MARKERS)], markersize=3.0,
                markevery=(idx % 5, 5))
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.6)
    ax.set_xlabel("round")
    ax.set_ylabel(r"$V_\mathrm{mask}/V_\mathrm{prior,\,1}$")
    ax.grid(True, linestyle=":", alpha=0.4, linewidth=0.5)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    _log_axis(ax, nticks=6)
    if cum_h is not None:
        _add_time_axis(ax, rounds, cum_h, label=r"training time [h]")
    handles, lab = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], color="grey", linestyle="--", linewidth=0.6))
    lab.append("initial prior")
    ax.legend(handles, lab, loc="center left", bbox_to_anchor=(1.02, 0.5),
              frameon=False, handlelength=1.4, labelspacing=0.35,
              borderaxespad=0.0)
    return _save(fig, outdir, "volume_from_masks_collapsed")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("name", help="Run name / TIME_OF_EXECUTION.")
    p.add_argument("--last-round", type=int, default=None,
                   help="Stop at this round (1-indexed, inclusive).")
    p.add_argument("--width-pt", type=float, default=246.0,
                   help="Grid / global figure width in points (246 = 1 column).")
    p.add_argument("--collapsed-width-pt", type=float, default=None,
                   help="Width of the collapsed figure (default: 2x--width-pt).")
    p.add_argument("--fontsize", type=float, default=12.0)
    p.add_argument("--no-time-axis", action="store_true",
                   help="Drop the cumulative-training-time secondary axis.")
    args = p.parse_args()

    round_dirs = find_round_dirs(args.name)
    if not round_dirs:
        raise RuntimeError(f"No round directories for '{args.name}'.")
    if args.last_round:
        round_dirs = round_dirs[: args.last_round]
    keys = keys_for_model(load_model(os.path.join(round_dirs[-1], "checkpoints")))

    rounds, labels, idx_1d, pairs_2d, V = collect_mask_volumes(
        args.name, keys, last_round=args.last_round)
    ref = reference_volumes(args.name, keys, idx_1d, pairs_2d)
    ratio = V / ref[None, :]
    ratio_global = np.prod(ratio, axis=1)

    outdir = os.path.join(
        PLOTS_ROOT_DIR,
        args.name + (f"_upto_round_{args.last_round}" if args.last_round else ""))
    os.makedirs(outdir, exist_ok=True)
    np.savez(os.path.join(outdir, "volume_from_masks.npz"),
             labels=np.array(labels), rounds=np.array(rounds),
             V=V, ref=ref, ratio=ratio, ratio_global=ratio_global)

    dv = np.diff(ratio_global)
    if np.any(dv > ratio_global[:-1] * 1e-9):
        bad = [int(rounds[i + 1]) for i in np.where(dv > ratio_global[:-1] * 1e-9)[0]]
        print(f"  NOTE: global volume increased at rounds {bad} even after "
              f"intersection (a stored mask grew — not a no-op here).")
    else:
        print("  global volume is monotone non-increasing (intersection was a no-op).")

    cum_h = None if args.no_time_axis else cumulative_train_hours(round_dirs)
    apply_paper_style(args.fontsize, uniform=True)
    plot_global(rounds, ratio_global, cum_h, outdir, args.width_pt, args.fontsize)
    plot_per_marginal(rounds, labels, ratio, cum_h, outdir, args.width_pt,
                      args.fontsize)
    plot_collapsed(rounds, labels, ratio, cum_h, outdir,
                   args.collapsed_width_pt or 2 * args.width_pt, args.fontsize)


if __name__ == "__main__":
    main()
