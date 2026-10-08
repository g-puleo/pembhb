"""Visualise 1-D posterior evolution across TMNRE truncation rounds.

Produces a grid of subplots, one per parameter, with the round index on the
x-axis and parameter value on the y-axis. Each round contributes nested
equal-tailed credible bands (50/90/99%) of the NRE 1-D marginal at the obs,
held constant from round r to r+1. Each panel is an offset from the truth on a
symlog scale, so the prior-wide early rounds and the narrow final ones are both
legible. MCMC, if provided, is only compared against in the last-round figure.

Single-round figures (and the shared evaluation code) live in ``plot_posterior.py``;
sky evolution in ``visualise_sky_truncation.py``.

Figures are paper-styled: a ``--rows`` x ``--cols`` grid (default 2x6) at
``--width-pt`` (default \\textwidth = 2 x 246 pt) and ``--fontsize`` (10),
written as both PDF and PNG.

Usage
-----
    python scripts/visualise_truncation_rounds.py NAME \\
        --data-path obs.h5 [--mcmc-file mcmc.h5] [--ngrid-1d 200] \\
        [--last-round N] [--ckpt-final-round PATH] [--reason TAG] \\
        [--rows 2 --cols 6 --width-pt 492 --fontsize 10]
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from pembhb import ROOT_DIR, PLOTS_ROOT_DIR
from pembhb.utils import compute_fisher_prior_bounds

from _visualise_common import (
    DATA_ROOT_DIR,
    resolve_obs_path,
    build_obs_dataloader,
    find_round_dirs,
    load_duration_weeks,
    register_ckpt_override,
    PT_PER_INCH,
    TEXTWIDTH_PT,
    latex_label,
    apply_paper_style,
    save_figure,
)
from viz_helpers import eval_mcmc_kde_1d
from plot_posterior import (
    DEFAULT_ROWS,
    DEFAULT_COLS,
    _panel_title,
    _grid_layout,
    _place_legend,
    _axis_transforms_for,
    evaluate_round,
    load_mcmc,
    plot_1d_posteriors,
)
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, NullLocator, SymmetricalLogLocator


# ---------------------------------------------------------------------------
# Credible-band helpers
# ---------------------------------------------------------------------------

# Drawn outermost-first so the narrower bands sit on top.
BAND_LEVELS = (0.99, 0.90, 0.50)
BAND_ALPHA = {0.99: 0.20, 0.90: 0.38, 0.50: 0.70}

# Half-width of the symlog linear region, in units of the last round's 99% band.
SYMLOG_LINTHRESH_FACTOR = 5.0

def _symlog_ticks(linthresh, y_lo, y_hi, max_per_side=2):
    """Sparse tick list for a symlog axis: 0 plus a couple of decades a side.

    The default ``SymmetricalLogLocator`` also emits decades *inside* the
    linear region, which at these panel widths overprint the zero label.
    """
    ticks = [0.0]
    for sign, lim in ((1.0, y_hi), (-1.0, y_lo)):
        if sign * lim <= linthresh:
            continue
        k_min = int(np.ceil(np.log10(linthresh)))
        k_max = int(np.floor(np.log10(sign * lim)))
        decades = list(range(k_min, k_max + 1))
        if not decades:
            continue
        if len(decades) > max_per_side:
            idx = np.linspace(0, len(decades) - 1, max_per_side).round()
            decades = [decades[int(i)] for i in idx]
        ticks += [sign * 10.0 ** k for k in decades]
    return sorted(ticks)


def _equal_tailed_interval(grid, density, level):
    """Two-tailed ``level`` credible interval of a 1-D density on *grid*.

    Equal-tailed (percentile) rather than highest-density: the excluded
    probability is split evenly between the two tails.
    """
    cdf = np.cumsum(np.asarray(density, dtype=float))
    if cdf[-1] <= 0:
        return np.nan, np.nan
    cdf = cdf / cdf[-1]
    tail = 0.5 * (1.0 - level)
    grid = np.asarray(grid, dtype=float)
    return (float(np.interp(tail, cdf, grid)),
            float(np.interp(1.0 - tail, cdf, grid)))


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------

def _fisher_sigmas(round_dirs, obs_path, labels, basis, datagen_config=None):
    """Cramer-Rao 1-sigma per parameter, as a ``{label: sigma}`` dict.

    Obtained by asking :func:`compute_fisher_prior_bounds` for a **1-sigma**
    box and halving its width, so the Gaussian overlay uses exactly the same
    Fisher matrix the ``fisher_prior`` run option would build.

    ``datagen_config`` defaults to the run's own round-1 sidecar
    (``simulation_round_1.yaml``), so the waveform settings match the run
    rather than whatever is currently in ``configs/``.
    """
    import yaml as _yaml

    if datagen_config is None:
        side = os.path.join(round_dirs[0], "simulation_round_1.yaml")
        if not os.path.exists(side):
            raise FileNotFoundError(
                f"No datagen config given and {side} is missing; pass "
                f"--fisher-datagen-config explicitly.")
        with open(side, encoding="utf-8") as fh:
            conf = (_yaml.safe_load(fh) or {}).get("conf")
        if conf is None:
            raise ValueError(f"{side} has no 'conf' block.")
    else:
        with open(datagen_config, encoding="utf-8") as fh:
            conf = _yaml.safe_load(fh)
        conf = conf.get("conf", conf)

    bounds = compute_fisher_prior_bounds(
        datagen_config=conf,
        observation_file=obs_path,
        event_idx=0,
        varying_params=list(labels),
        fixed_params=[],
        n_sigma=1.0,
        param_n_sigma=None,
        spin_param_basis=basis,
    )
    return {k: 0.5 * (float(v[1]) - float(v[0])) for k, v in bounds.items()}


def _resolve_param_subset(all_params, requested):
    """Filter ``all_params`` (the ``data['params']`` tuples) down to *requested*.

    Matching is case-insensitive and underscore-optional, so ``deltat`` resolves
    to ``Deltat`` and ``chieff`` to ``chi_eff``. The user's requested order is
    preserved. Raises ``ValueError`` (listing the available labels) if any name
    is unknown.
    """
    lookup = {}
    for tup in all_params:
        label = tup[0]
        lookup.setdefault(label.lower(), tup)
        lookup.setdefault(label.lower().replace("_", ""), tup)
    subset = []
    for name in requested:
        key = name.lower()
        tup = lookup.get(key) or lookup.get(key.replace("_", ""))
        if tup is None:
            available = ", ".join(t[0] for t in all_params)
            raise ValueError(
                f"--params: unknown parameter '{name}'. Available: {available}."
            )
        subset.append(tup)
    return subset


def _compute_round_data(round_dirs, dataloader, ngrid_1d, mcmc_samples_path):
    """Load each round's model exactly once and evaluate every 1-D marginal.

    Returns a dict consumed by the render functions so no model is loaded (nor
    posterior re-evaluated) more than once across the normal figure, the zoom
    figure, and the last-round-vs-MCMC figure:

    ``params`` (last round's marginal list — the canonical/superset order),
    ``per_round_densities`` (list of ``{label: (grid, density, inj) or None}``,
    keyed by label; a label absent in a round is simply missing), and
    ``per_round_priors``, ``duration_weeks``, ``mcmc_samples``,
    ``mcmc_param_names``, ``n_rounds``.
    """
    n_rounds = len(round_dirs)
    if n_rounds == 0:
        raise ValueError("No round directories provided.")

    per_round_densities: list[dict] = []
    per_round_priors: list[dict] = []
    params = None
    nre_basis = "chi1chi2"
    for r_idx, rd in enumerate(round_dirs, start=1):
        _, round_params, densities, prior_box, nre_basis = evaluate_round(
            rd, r_idx, dataloader, ngrid_1d)
        per_round_densities.append(densities)
        per_round_priors.append(prior_box)
        # Later rounds can introduce marginals, so the last round's list is the
        # canonical superset used for panel layout.
        params = round_params

    if not params:
        raise RuntimeError("Model has no 1-D-recoverable marginals.")

    duration_weeks = load_duration_weeks(round_dirs[0], 1)
    mcmc_samples, mcmc_param_names = load_mcmc(mcmc_samples_path, nre_basis)

    return {
        "params": params,
        "per_round_densities": per_round_densities,
        "per_round_priors": per_round_priors,
        "duration_weeks": duration_weeks,
        "mcmc_samples": mcmc_samples,
        "mcmc_param_names": mcmc_param_names,
        "n_rounds": n_rounds,
        "round_numbers": list(range(1, n_rounds + 1)),
        "nre_basis": nre_basis,
    }


def plot_interval_evolution(
    data: dict,
    mcmc_samples_path: str | None,
    outdir: str,
    ngrid_1d: int = 200,
    reason: str = "truncation",
    y_range_per_label: dict | None = None,
    zoom_sigmas: float | None = None,
    filename_suffix: str = "",
    width_pt: float = TEXTWIDTH_PT,
    height_in: float | None = None,
    rows: int = DEFAULT_ROWS,
    cols: int = DEFAULT_COLS,
):
    """Render the 1-D credible-interval evolution figure from precomputed
    *data* (see :func:`_compute_round_data`).

    One subplot per parameter. Each round contributes nested equal-tailed
    credible bands (:data:`BAND_LEVELS`) drawn piecewise-constant from ``x=r``
    to ``x=r+1``, so the whole panel reads as a staircase rather than a row of
    violins. The truncated prior bounds are the black staircase and the true
    (injection) value the red dashed line.

    Every panel is plotted as an **offset from the truth** on a symlog y-axis
    whose linear region is ``SYMLOG_LINTHRESH_FACTOR`` times the width of the
    last round's 99% band. Early rounds (prior-wide) and the final rounds
    (posterior-narrow) are then both legible in one panel — on a linear axis
    the last ~30 rounds collapse onto the truth line.

    MCMC is not drawn here (see :func:`plot_posterior.plot_1d_posteriors`); when
    ``zoom_sigmas`` is given the MCMC spread is still used to size the window.

    ``zoom_sigmas`` (float) clips each panel's y-axis to a window centered on
    the **ground-truth injection** with half-width ``zoom_sigmas * sigma_mcmc``.
    The center is the true injection (``res[2]``, never the prior midpoint or
    the MCMC median); window and bands share the same transform so they stay
    aligned by construction. Used for the "zoom on MCMC" companion figure.

    ``y_range_per_label`` (dict mapping label → ``(y_lo, y_hi)``) is an explicit
    manual override of the y-axis window; it wins over ``zoom_sigmas``.
    """
    os.makedirs(outdir, exist_ok=True)
    params = data["params"]
    per_round_densities = data["per_round_densities"]
    per_round_priors = data["per_round_priors"]
    duration_weeks = data["duration_weeks"]
    mcmc_samples = data["mcmc_samples"]
    mcmc_param_names = data["mcmc_param_names"]
    n_rounds = data["n_rounds"]

    rows, cols = _grid_layout(len(params), rows, cols)
    width_in = width_pt / PT_PER_INCH
    fig, axes = plt.subplots(
        rows, cols, squeeze=False, constrained_layout=True,
        figsize=(width_in, height_in or 1.45 * rows + 0.55),
    )

    # Round r occupies [r, r+1].
    x_edges = np.arange(1, n_rounds + 2, dtype=float)
    for idx, (label, source_dim, in_idx, _, _) in enumerate(params):
        ax = axes[idx // cols, idx % cols]

        # Look up the NRE injection value first (it parameterises the
        # Deltat axis transform). It's constant across rounds.
        inj_for_transform = None
        for densities in per_round_densities:
            res = densities.get(label)
            if res is not None:
                inj_for_transform = res[2]
                break
        nre_to_y, mcmc_to_y, y_label = _axis_transforms_for(
            label, inj_for_transform, duration_weeks, mcmc_samples_path,
        )
        # Everything is drawn as an offset from the truth so the symlog scale
        # can be centred on it. For Deltat the transform already does this.
        truth_disp = (
            float(nre_to_y(np.array([inj_for_transform]))[0])
            if inj_for_transform is not None else 0.0
        )
        to_c = lambda v: nre_to_y(v) - truth_disp

        # Per-round, per-level intervals in truth-centred display coords.
        bands = {}
        for level in BAND_LEVELS:
            lo = np.full(n_rounds + 1, np.nan)
            hi = np.full(n_rounds + 1, np.nan)
            for r, densities in enumerate(per_round_densities):
                res = densities.get(label)
                if res is None:
                    continue
                grid, density, _ = res
                lo[r], hi[r] = _equal_tailed_interval(to_c(grid), density, level)
            lo[-1], hi[-1] = lo[-2], hi[-2]     # step="post" needs a last value
            bands[level] = (lo, hi)

        widest = max(BAND_LEVELS)
        y_min = np.nanmin(bands[widest][0])
        y_max = np.nanmax(bands[widest][1])
        for prior_box in per_round_priors:
            if label in prior_box:
                b_lo, b_hi = prior_box[label]
                y_min = min(y_min, float(to_c(np.array([b_lo]))[0]))
                y_max = max(y_max, float(to_c(np.array([b_hi]))[0]))
        if not np.isfinite(y_min) or not np.isfinite(y_max):
            ax.set_visible(False)
            continue
        y_pad = 0.05 * (y_max - y_min)
        y_lo, y_hi = y_min - y_pad, y_max + y_pad

        # Linear region of the symlog scale: a few times the final resolution.
        lo99, hi99 = bands[widest][0][n_rounds - 1], bands[widest][1][n_rounds - 1]
        last_width = hi99 - lo99 if np.isfinite(hi99 - lo99) else np.nan
        if not np.isfinite(last_width) or last_width <= 0:
            last_width = (y_max - y_min) / 100.0
        linthresh = SYMLOG_LINTHRESH_FACTOR * last_width

        # Zoom window: centered on the truth (now 0), half-width N * MCMC sigma.
        # MCMC is not drawn here, only used to size the window.
        if zoom_sigmas is not None and mcmc_samples is not None \
                and label in mcmc_param_names:
            col = mcmc_to_y(mcmc_samples[:, mcmc_param_names.index(label)])
            sigma_disp = float(np.std(col))
            y_lo, y_hi = -zoom_sigmas * sigma_disp, zoom_sigmas * sigma_disp
        if y_range_per_label and label in y_range_per_label:
            y_lo, y_hi = y_range_per_label[label]

        # Nested NRE credible bands, piecewise-constant over each round.
        for level in BAND_LEVELS:
            lo, hi = bands[level]
            ax.fill_between(x_edges, lo, hi, step="post", color="C0",
                            alpha=BAND_ALPHA[level], linewidth=0, zorder=1)

        # Truncated prior bounds as a staircase.
        plo = np.full(n_rounds + 1, np.nan)
        phi = np.full(n_rounds + 1, np.nan)
        for r, prior_box in enumerate(per_round_priors):
            if label in prior_box:
                b_lo, b_hi = prior_box[label]
                plo[r] = float(to_c(np.array([b_lo]))[0])
                phi[r] = float(to_c(np.array([b_hi]))[0])
        plo[-1], phi[-1] = plo[-2], phi[-2]
        ax.step(x_edges, plo, where="post", color="k", linewidth=0.6, zorder=2)
        ax.step(x_edges, phi, where="post", color="k", linewidth=0.6, zorder=2)

        # Truth line — at 0 by construction.
        ax.axhline(0.0, color="red", linestyle="--", linewidth=0.9, zorder=3)

        # Cosmetics.
        ax.set_xlim(1.0, n_rounds + 1)
        ax.set_yscale("symlog", linthresh=linthresh)
        ax.set_ylim(y_lo, y_hi)
        ax.set_yticks(_symlog_ticks(linthresh, y_lo, y_hi))
        ax.yaxis.set_minor_locator(NullLocator())
        # Thin the round ticks: at this width one label per round is illegible.
        # Ticks sit at the centre of each round's band.
        step = max(1, int(np.ceil(n_rounds / 3)))
        rounds_ticked = list(range(1, n_rounds + 1, step))
        if n_rounds - rounds_ticked[-1] > 0.5 * step:
            rounds_ticked.append(n_rounds)
        xticks = [r + 0.5 for r in rounds_ticked]
        xticklabels = [str(r) for r in rounds_ticked]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        # Panels are truth-centred, so the transformed Deltat label would be
        # redundant — only its unit is worth keeping.
        ax.set_title(_panel_title(label), pad=3)

    _finish_grid(axes, len(params), rows, cols, xlabel="round")
    fig.supylabel(r"$\theta - \theta_\mathrm{true}$")

    handles = [
        Patch(facecolor="C0", alpha=BAND_ALPHA[lv], linewidth=0,
              label=f"NRE {int(100 * lv)}%")
        for lv in BAND_LEVELS
    ]
    handles += [
        Line2D([0], [0], color="k", lw=0.6, label="prior bounds"),
        Line2D([0], [0], color="red", ls="--", lw=0.9, label="ground truth"),
    ]
    _place_legend(fig, axes, handles, len(params), rows, cols)

    return save_figure(
        fig, outdir, f"interval_evolution_{reason}{filename_suffix}",
    )


def _finish_grid(axes, n_panels: int, rows: int, cols: int, xlabel: str) -> None:
    """Hide the unused cells and label the x-axis of every panel."""
    for k in range(n_panels, rows * cols):
        axes[k // cols, k % cols].set_visible(False)
    for k in range(n_panels):
        axes[k // cols, k % cols].set_xlabel(xlabel)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Plot 1-D posterior violin evolution across TMNRE rounds.",
    )
    p.add_argument("name", help="Run name / TIME_OF_EXECUTION.")
    p.add_argument("--data-path", default=None,
                   help="Observation HDF5 (default: the run's observation_used.yaml; "
                        "a different observation raises).")
    p.add_argument("--mcmc-file", default=None,
                   help="Optional flat MCMC samples HDF5 for reference violin "
                        "and ±1σ band.")
    p.add_argument("--ngrid-1d", type=int, default=200,
                   help="Grid resolution for 1-D posterior evaluation.")
    p.add_argument("--last-round", type=int, default=None,
                   help="Stop at this round (1-indexed, inclusive).")
    p.add_argument("--ckpt-final-round", default=None,
                   help="Path to a checkpoint overriding the final round's "
                        "truncation.ckpt (e.g. a PP-KS trigger ckpt).")
    p.add_argument("--reason", default="auto",
                   help="Free-form tag for filename + suptitle. 'auto' = "
                        "'trigger' if --ckpt-final-round is set, else 'truncation'.")
    p.add_argument("--zoom-mcmc-sigmas", type=float, default=5.0,
                   help="When --mcmc-file is given, also save a zoomed-in "
                        "companion figure with each subplot's y-axis "
                        "restricted to ±N·σ_MCMC around the MCMC median. "
                        "0 disables the zoom output. Default: 5.")
    p.add_argument("--width-pt", type=float, default=TEXTWIDTH_PT,
                   help="Figure width in points (default: \\textwidth = 492).")
    p.add_argument("--height-in", type=float, default=None,
                   help="Figure height in inches (default: from the row count).")
    p.add_argument("--mcmc-height-in", type=float, default=None,
                   help="Figure height in inches for the last-round-vs-MCMC "
                        "figure (default: same as --height-in).")
    p.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    p.add_argument("--cols", type=int, default=DEFAULT_COLS)
    p.add_argument("--fontsize", type=float, default=10.0)
    p.add_argument("--params", nargs="+", default=None,
                   help="Also emit an extra evolution figure for just these "
                        "parameters, in the given order (grid shape from "
                        "--params-rows/--params-cols). Case-insensitive, "
                        "underscores optional (e.g. 'logmchirp deltat q', 'chieff').")
    p.add_argument("--params-rows", type=int, default=1,
                   help="Row count for the --params figure (default: 1, i.e. a "
                        "single row).")
    p.add_argument("--params-cols", type=int, default=None,
                   help="Column count for the --params figure (default: "
                        "len(--params)+1, one free cell reserved for the legend "
                        "at the end of the single default row).")
    p.add_argument("--fisher", action="store_true",
                   help="Overlay the Fisher (Cramer-Rao) Gaussian N(truth, "
                        "sigma_FIM) on the last-round 1-D posteriors. Uses the "
                        "same FIM that the fisher_prior run option builds.")
    p.add_argument("--fisher-datagen-config", default=None,
                   help="Datagen config for the FIM (default: the run's own "
                        "round-1 sidecar simulation_round_1.yaml, so the "
                        "waveform settings match the run).")
    p.add_argument("--params-width-pt", type=float, default=None,
                   help="Figure width in points for the --params figure "
                        "(default: same as --width-pt).")
    args = p.parse_args()
    if args.reason == "auto":
        args.reason = "trigger" if args.ckpt_final_round else "truncation"

    round_dirs = find_round_dirs(args.name)
    if not round_dirs:
        raise RuntimeError(f"No round directories found for name='{args.name}'.")
    print(f"Detected {len(round_dirs)} round(s) for '{args.name}'.")

    if args.last_round is not None:
        if not 1 <= args.last_round <= len(round_dirs):
            raise ValueError(f"--last-round={args.last_round} out of range "
                             f"[1, {len(round_dirs)}]")
        round_dirs = round_dirs[: args.last_round]
        print(f"Truncated to first {args.last_round} round(s).")

    if args.ckpt_final_round:
        register_ckpt_override(round_dirs[-1], args.ckpt_final_round)

    args.data_path = resolve_obs_path(args.name, args.data_path)
    dataloader = build_obs_dataloader(args.data_path)

    outdir = os.path.join(
        PLOTS_ROOT_DIR,
        args.name + (f"_upto_round_{args.last_round}" if args.last_round else ""),
    )

    # Load each round's model once and evaluate every marginal a single time;
    # all figures below render from this precomputed data.
    data = _compute_round_data(
        round_dirs, dataloader, args.ngrid_1d, args.mcmc_file,
    )

    fisher_sigmas = None
    if args.fisher:
        basis = data["nre_basis"]          # derived from the model's trained bounds
        labels = [pi[0] for pi in data["params"]]
        print(f"[fisher] computing the Fisher matrix for {len(labels)} parameter(s) ...")
        fisher_sigmas = _fisher_sigmas(
            round_dirs, args.data_path, labels, basis,
            datagen_config=args.fisher_datagen_config,
        )
        for k in labels:
            sg = fisher_sigmas.get(k)
            print(f"[fisher]   sigma({k}) = {sg:.6g}" if sg else f"[fisher]   sigma({k}) = n/a")

    apply_paper_style(args.fontsize)
    style = dict(width_pt=args.width_pt, height_in=args.height_in,
                 rows=args.rows, cols=args.cols)

    plot_interval_evolution(
        data=data,
        mcmc_samples_path=args.mcmc_file,
        outdir=outdir,
        ngrid_1d=args.ngrid_1d,
        reason=args.reason,
        **style,
    )

    # Optional MCMC-zoom companion figure. The window is centered on the
    # ground-truth injection (from the NRE eval), with half-width
    # zoom_sigmas * MCMC sigma. Re-renders the same precomputed data.
    if args.mcmc_file and args.zoom_mcmc_sigmas > 0:
        plot_interval_evolution(
            data=data,
            mcmc_samples_path=args.mcmc_file,
            outdir=outdir,
            ngrid_1d=args.ngrid_1d,
            reason=args.reason,
            zoom_sigmas=args.zoom_mcmc_sigmas,
            filename_suffix=f"_zoom{int(args.zoom_mcmc_sigmas)}sigma",
            **style,
        )

    # Last-round posterior vs MCMC, one overlaid panel per marginal.
    plot_1d_posteriors(
        data=data,
        mcmc_samples_path=args.mcmc_file,
        outdir=outdir,
        stem=f"round_{data['n_rounds']}_last_posterior_vs_mcmc_{args.reason}",
        fisher_sigmas=fisher_sigmas,
        **{**style, "height_in": args.mcmc_height_in or args.height_in},
    )

    # Extra 1xN evolution figure for a hand-picked subset of parameters.
    # Built after the full-grid figures so the original data["params"] is left
    # untouched for them; only the panel list is swapped here.
    if args.params:
        subset = _resolve_param_subset(data["params"], args.params)
        subset_data = {**data, "params": subset}
        labels = "_".join(lbl for lbl, *_ in subset)
        print(f"[params] extra 1x{len(subset)} figure for: {labels}")
        plot_interval_evolution(
            data=subset_data,
            mcmc_samples_path=args.mcmc_file,
            outdir=outdir,
            ngrid_1d=args.ngrid_1d,
            reason=args.reason,
            width_pt=args.params_width_pt or args.width_pt,
            height_in=args.height_in,
            rows=args.params_rows,
            # Default: one extra (blank) column reserves a free cell for
            # _place_legend's shrink-to-fit branch -- the "grid exactly
            # filled" branch it otherwise falls into places the legend at a
            # fixed fontsize with no shrink loop, which clips at large
            # --fontsize values. Pass --params-rows/--params-cols explicitly
            # (e.g. 2x2 for 3 params) for a non-single-row layout; the same
            # free-cell legend placement applies as long as rows*cols > len(params).
            cols=args.params_cols or len(subset) + 1,
            filename_suffix=f"_params_{labels}",
        )


if __name__ == "__main__":
    main()
