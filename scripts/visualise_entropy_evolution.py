"""Plot differential-entropy evolution of NRE posteriors across TMNRE rounds.

For each marginal (1-D or 2-D head), evaluate the NRE posterior on the
round's prior box, compute its differential entropy in nats, and plot the
sequence vs round index. One subplot per marginal.

A horizontal dashed reference line shows the entropy of the round-1 *flat*
prior on the same box (``log(b - a)`` for 1-D, ``log((b₀−a₀)(b₁−a₁))`` for
2-D).

Two figures are written, both paper-styled:
  * ``entropy_evolution``           — one panel per marginal (5×2 grid,
    single-column width), with a top axis giving cumulative training time
    read off the TensorBoard event files;
  * ``entropy_evolution_collapsed`` — all marginals on one axis at
    \\textwidth, colour-blind-safe palette + markers, legend on the right.

Entropies are cached to ``entropy_evolution.npz`` in the output directory,
so restyling the figures does not re-evaluate the NRE (pass ``--recompute``
to force).

Usage
-----
    python scripts/visualise_entropy_evolution.py NAME \\
        --data-path obs.h5 [--ngrid 60] [--last-round N]
    python scripts/visualise_entropy_evolution.py NAME   # replot from cache
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from torch.utils.data import DataLoader, Subset

from pembhb import ROOT_DIR, PLOTS_ROOT_DIR
from pembhb.utils import _ORDERED_PRIOR_KEYS

from _visualise_common import (
    resolve_obs_path,
    build_obs_dataloader,
    find_round_dirs,
    load_model,
    load_prior_box,
    get_all_marginals,
    find_out_param_idx,
    compute_normalised_posterior,
    PT_PER_INCH,
    latex_label,
    apply_paper_style,
    save_figure as _save,
    cumulative_train_hours,
    add_time_axis as _add_time_axis,
    TOL_MUTED,
    MARKERS,
)
from viz_helpers import (
    eval_nre_1d,
    differential_entropy_1d,
    differential_entropy_2d,
)


def _round_entropies(model, dataloader, prior_box, ngrid):
    """Return {label: H (nats)} for every marginal head in *model*.

    Skips heads that aren't present in this model (None from find_out_param_idx).
    """
    out = {}
    for label, ndim, in_idx, _ in get_all_marginals(model):
        out_idx = find_out_param_idx(model, in_idx)
        if out_idx is None:
            continue
        if ndim == 1:
            low, high = prior_box[label]
            grid, norm1d, _ = eval_nre_1d(
                model, dataloader, in_idx, out_idx, low, high, ngrid,
            )
            dp = float(grid[1] - grid[0])
            out[label] = differential_entropy_1d(norm1d, dp)
        else:
            p0, p1 = in_idx
            b0 = prior_box[_ORDERED_PRIOR_KEYS[p0]]
            b1 = prior_box[_ORDERED_PRIOR_KEYS[p1]]
            norm2d, _, gx, gy = compute_normalised_posterior(
                dataloader, model, in_idx, out_idx, b0, b1, ngrid_points=ngrid,
            )
            dp0 = float(gx[0, 1] - gx[0, 0])
            dp1 = float(gy[1, 0] - gy[0, 0])
            out[label] = differential_entropy_2d(norm2d[0], dp0, dp1)
    return out


def _flat_prior_entropy(label: str, ndim: int, prior_box: dict) -> float:
    """log-volume of the round-1 prior box for that marginal."""
    if ndim == 1:
        low, high = prior_box[label]
        return float(np.log(high - low))
    # label here is "p0 vs p1"; parse from get_all_marginals listing.
    raise ValueError("flat_prior_entropy for 2-D needs the param tuple, not a label.")


def compute_entropies(round_dirs: list, dataloader: DataLoader, ngrid: int):
    """Return (labels, H[n_rounds, n_marginals], flat_H[n_marginals])."""
    last_model = load_model(os.path.join(round_dirs[-1], "checkpoints"))
    marginals = get_all_marginals(last_model)
    if not marginals:
        raise RuntimeError("No marginals on the last-round model.")

    H_per_round: list[dict] = []
    for i, rd in enumerate(round_dirs, start=1):
        model = load_model(os.path.join(rd, "checkpoints"))
        prior_box = load_prior_box(rd, i)
        H_per_round.append(_round_entropies(model, dataloader, prior_box, ngrid))
        print(f"[round {i}] computed entropy for "
              f"{len(H_per_round[-1])}/{len(marginals)} marginals.")

    # Round-1 flat-prior baseline per marginal (1-D: log(b−a); 2-D: log((b0−a0)(b1−a1))).
    box1 = load_prior_box(round_dirs[0], 1)
    flat_H = []
    for label, ndim, in_idx, _ in marginals:
        if ndim == 1:
            low, high = box1[label]
            flat_H.append(float(np.log(high - low)))
        else:
            p0, p1 = in_idx
            a0, b0 = box1[_ORDERED_PRIOR_KEYS[p0]]
            a1, b1 = box1[_ORDERED_PRIOR_KEYS[p1]]
            flat_H.append(float(np.log((b0 - a0) * (b1 - a1))))

    labels = [m[0] for m in marginals]
    H = np.array([[d.get(lab, np.nan) for lab in labels] for d in H_per_round],
                 dtype=float)
    return labels, H, np.asarray(flat_H, dtype=float)


def plot_entropy_grid(labels, H, flat_H, cum_h, outdir, width_pt, fontsize,
                       height_in=None, cols=2):
    """One panel per marginal, ``cols``-wide grid, single-column figure width."""
    n = len(labels)
    rows = int(np.ceil(n / cols))
    width_in = width_pt / PT_PER_INCH
    height_in = height_in or 1.05 * rows + 0.9
    rounds = np.arange(1, H.shape[0] + 1)

    fig, axes = plt.subplots(rows, cols, figsize=(width_in, height_in),
                             squeeze=False, sharex=True,
                             constrained_layout=True)
    for idx, label in enumerate(labels):
        ax = axes[idx // cols, idx % cols]
        ax.plot(rounds, H[:, idx], "-", color="steelblue")
        ax.axhline(flat_H[idx], color="grey", linestyle="--", linewidth=0.7)
        ax.grid(True, linestyle=":", alpha=0.4, linewidth=0.5)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        # Top row: the secondary time axis occupies the space a title would
        # take, so push the title above it.
        if idx // cols == 0 and cum_h is not None:
            _add_time_axis(ax, rounds, cum_h,
                           label=r"$t_\mathrm{train}\,[\mathrm{h}]$")
            ax.set_title(latex_label(label), pad=fontsize * 2.6)
        else:
            ax.set_title(latex_label(label), pad=3)

    for k in range(n, rows * cols):
        axes[k // cols, k % cols].set_visible(False)

    fig.supxlabel("round", fontsize=fontsize)
    fig.supylabel(r"$H$ [nats]", fontsize=fontsize)

    return _save(fig, outdir, "entropy_evolution")


def plot_entropy_collapsed(labels, H, flat_H, cum_h, outdir, width_pt, fontsize,
                           legend_frac=0.32):
    """All marginals on one axis, colour-coded, 2-column legend below the axis.

    Each line's flat-prior entropy is prepended as its round-0 point, rather
    than drawn as a separate dashed reference line.
    """
    rounds = np.arange(0, H.shape[0] + 1)
    width_in = width_pt / PT_PER_INCH
    fig, ax = plt.subplots(figsize=(width_in, width_in))

    for idx, label in enumerate(labels):
        c = TOL_MUTED[idx % len(TOL_MUTED)]
        y = np.concatenate([[flat_H[idx]], H[:, idx]])
        ax.plot(rounds, y, "-", color=c, label=latex_label(label),
                marker=MARKERS[idx % len(MARKERS)], markersize=3.0,
                markevery=(idx % 5, 5))

    ax.set_xlabel("round")
    ax.set_ylabel(r"$H$ [nats]")
    ax.grid(True, linestyle=":", alpha=0.4, linewidth=0.5)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    if cum_h is not None:
        _add_time_axis(ax, rounds[1:], cum_h, label=r"training time [h]",
                       highlight_final=True)

    handles, lab = ax.get_legend_handles_labels()
    fig.tight_layout(rect=(0, legend_frac, 1, 1), pad=0.15)
    # Anchor the legend's *top* edge to the reserved boundary (rather than its
    # bottom edge to the figure bottom) so it sits flush under the axis,
    # instead of floating with a gap above it and slack below.
    fig.legend(handles, lab, loc="upper center", bbox_to_anchor=(0.5, legend_frac),
              ncol=2, frameon=False, handlelength=1.4, labelspacing=0.4,
              columnspacing=1.0, borderaxespad=0.1)

    return _save(fig, outdir, "entropy_evolution_collapsed")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("name", help="Run name / TIME_OF_EXECUTION.")
    p.add_argument("--data-path", default=None,
                    help="Observation HDF5 (preferably with stored noise_fd). "
                         "Not needed when replotting from cache.")
    p.add_argument("--ngrid", type=int, default=60,
                    help="Grid resolution per axis for posterior evaluation.")
    p.add_argument("--last-round", type=int, default=None,
                    help="Stop at this round (1-indexed, inclusive).")
    p.add_argument("--recompute", action="store_true",
                    help="Ignore the cached entropies and re-evaluate the NRE.")
    p.add_argument("--width-pt", type=float, default=246.0,
                    help="Figure width in points (246 = one A&A/PRD column).")
    p.add_argument("--collapsed-width-pt", type=float, default=None,
                    help="Width of the collapsed figure (default: 2×--width-pt, "
                         "i.e. full \\textwidth).")
    p.add_argument("--fontsize", type=float, default=12.0)
    p.add_argument("--height-in", type=float, default=None,
                    help="Grid-figure height in inches (default: auto).")
    args = p.parse_args()

    round_dirs = find_round_dirs(args.name)
    if not round_dirs:
        raise RuntimeError(f"No round directories for '{args.name}'.")
    if args.last_round:
        if not 1 <= args.last_round <= len(round_dirs):
            raise ValueError(
                f"--last-round={args.last_round} out of range [1, {len(round_dirs)}]"
            )
        round_dirs = round_dirs[: args.last_round]

    outdir = os.path.join(
        PLOTS_ROOT_DIR,
        args.name + (f"_upto_round_{args.last_round}" if args.last_round else ""),
    )
    cache = os.path.join(outdir, "entropy_evolution.npz")

    if os.path.exists(cache) and not args.recompute:
        z = np.load(cache, allow_pickle=False)
        labels, H, flat_H = list(z["labels"]), z["H"], z["flat_H"]
        if H.shape[0] != len(round_dirs):
            raise RuntimeError(
                f"cache has {H.shape[0]} rounds, {len(round_dirs)} on disk; "
                f"pass --recompute (or --last-round {H.shape[0]}).")
        print(f"loaded cached entropies from {cache}")
    else:
        labels, H, flat_H = compute_entropies(
            round_dirs, build_obs_dataloader(resolve_obs_path(args.name, args.data_path)), args.ngrid,
        )
        os.makedirs(outdir, exist_ok=True)
        np.savez(cache, labels=np.array(labels), H=H, flat_H=flat_H)
        print(f"cached entropies → {cache}")

    cum_h = cumulative_train_hours(round_dirs)
    apply_paper_style(args.fontsize, uniform=True)
    plot_entropy_grid(labels, H, flat_H, cum_h, outdir, args.width_pt,
                      args.fontsize, height_in=args.height_in)
    plot_entropy_collapsed(labels, H, flat_H, cum_h, outdir,
                           args.collapsed_width_pt or 2 * args.width_pt,
                           args.fontsize)


if __name__ == "__main__":
    main()
