#!/usr/bin/env python
"""Trained posterior of one TMNRE round, evaluated on the run's observation.

    ./scripts/plot_posterior.py --run-name 2026/10/08/channelizedmlp_my_run \
        [--round N] [--mcmc-file mcmc.h5] [--proposal] [--obs-path obs.h5]

Figures (``$PEMBHB_PLOTS_DIR/<run>/posterior_round_<N>/``):

* ``posterior_1d``  — every 1-D marginal (native heads, or marginalised from a
  2-D head), optionally against MCMC;
* ``sky_zoom.pdf`` / ``sky_zoom_lonlat.pdf`` — the (lambda, sin beta) posterior
  re-evaluated on a fine grid over its credible region (as in
  ``visualise_sky_truncation.py``), optionally against MCMC;
* ``modes_<param>.pdf`` — when the round's truncation found several modes and
  ``truncation.refine`` was on, each mode on its own refined grid.

``--proposal`` overlays the truncation this posterior produced, i.e. the prior
of round N+1: interval bounds in 1-D, the accepted-region contour in 2-D.

The observation defaults to the one recorded in ``observation_used.yaml``;
passing a different one raises. The same evaluation/plotting code builds the
round-by-round figures of ``visualise_truncation_rounds.py``.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from pembhb import PLOTS_ROOT_DIR
from pembhb.mask_truncation import load_truncation, region_of_pair
from pembhb.sampler import chi12_to_chieff_chidiff
from pembhb.utils import eval_posterior_2d

from _visualise_common import (
    DATA_ROOT_DIR,
    round_dirs_upto,
    load_model,
    load_prior_box,
    load_duration_weeks,
    get_all_marginals,
    find_out_param_idx,
    register_ckpt_override,
    compute_normalised_posterior,
    keys_for_model,
    detect_basis,
    resolve_obs_path,
    build_obs_dataloader,
    PT_PER_INCH,
    TEXTWIDTH_PT,
    latex_label,
    apply_paper_style,
    save_figure,
)
from viz_helpers import (
    load_mcmc_samples,
    eval_nre_1d,
    marginalise_2d_to_1d,
    deltat_axis_transforms,
    eval_mcmc_kde_1d,
)


# Curve/ground-truth linewidth in plot_last_round_vs_mcmc's panels + legend.
LAST_ROUND_LW = 1.5

# Panel titles for the parameters that carry a unit. Everything else falls back
# to ``latex_label`` (dimensionless: q, spins, cos(iota), sin(beta), logMchirp).
UNIT_LABELS = {
    "dist":   r"$d_L\,[\mathrm{Gpc}]$",
    "phi":    r"$\phi\,[\mathrm{rad}]$",
    "lambda": r"$\lambda\,[\mathrm{rad}]$",
    "psi":    r"$\psi\,[\mathrm{rad}]$",
    "Deltat": r"$\Delta t\,[\mathrm{s}]$",
}


def _panel_title(label: str) -> str:
    return UNIT_LABELS.get(label, latex_label(label))



DEFAULT_ROWS, DEFAULT_COLS = 2, 6



def _maybe_remap_mcmc_to_basis(samples, names, basis: str):
    """If *basis* is ``"chieff_chidiff"`` and MCMC carries ``(chi1, chi2, q)``,
    derive ``(chi_eff, chi_diff)`` and replace the chi1/chi2 columns in place.

    No-op otherwise. Returns ``(samples, names)``.
    """
    if basis != "chieff_chidiff":
        return samples, names
    need = {"chi1", "chi2", "q"}
    if not need.issubset(names):
        return samples, names
    i1, i2, iq = names.index("chi1"), names.index("chi2"), names.index("q")
    q = samples[:, iq]; c1 = samples[:, i1]; c2 = samples[:, i2]
    chi_eff, chi_diff = chi12_to_chieff_chidiff(q, c1, c2)
    samples = samples.copy()
    samples[:, i1] = chi_eff
    samples[:, i2] = chi_diff
    names = list(names); names[i1] = "chi_eff"; names[i2] = "chi_diff"
    print(f"[mcmc] remapped chi1,chi2 → chi_eff,chi_diff to match NRE basis")
    return samples, names


# ---------------------------------------------------------------------------
# Parameter listing — every dim the model can produce a 1-D marginal for
# ---------------------------------------------------------------------------

def _iter_param_marginals(model):
    """Yield ``(param_label, source_dim, in_idx, out_idx, axis_to_keep)``
    for every parameter the model has at least one head for.

    Preference: native 1-D head over 2-D-derived. When a parameter is only
    inside a 2-D head, yield it with source_dim=2 and axis_to_keep set to
    the axis (0 or 1) of that head whose marginalisation produces it.
    """
    keys = keys_for_model(model)
    one_d = {}
    two_d_only = {}
    for label, ndim, in_idx, out_idx in get_all_marginals(model):
        if ndim == 1:
            one_d[in_idx] = (label, 1, in_idx, out_idx, None)
        else:
            for axis, p in enumerate(in_idx):
                if p in one_d:
                    continue
                two_d_only.setdefault(
                    p, (keys[p], 2, in_idx, out_idx, axis)
                )
    # Emit in canonical parameter order so subplots are in a predictable order.
    for p_idx in range(len(keys)):
        if p_idx in one_d:
            yield one_d[p_idx]
        elif p_idx in two_d_only:
            yield two_d_only[p_idx]


def _eval_1d_marginal(model, dataloader, param_info, prior_box, ngrid):
    """Evaluate the 1-D marginal density for one parameter.

    Returns (grid_1d, norm1d, inj_value) or None if the marginal is not
    present in this model (e.g. introduced only in a later round).
    """
    label, source_dim, in_idx, _, axis_to_keep = param_info
    out_idx = find_out_param_idx(model, in_idx)
    if out_idx is None:
        return None
    if source_dim == 1:
        low, high = prior_box[label]
        return eval_nre_1d(model, dataloader, in_idx, out_idx, low, high, ngrid)
    # 2-D head → marginalise.
    p0, p1 = in_idx
    keys = keys_for_model(model)
    bounds_0 = prior_box[keys[p0]]
    bounds_1 = prior_box[keys[p1]]
    norm2d, inj_params, gx, gy = compute_normalised_posterior(
        dataloader, model, in_idx, out_idx, bounds_0, bounds_1,
        ngrid_points=ngrid,
    )
    return marginalise_2d_to_1d(norm2d[0], gx, gy, axis_to_keep, inj_params[0])


# ---------------------------------------------------------------------------
# Evaluation of one round (shared with visualise_truncation_rounds.py)
# ---------------------------------------------------------------------------

def evaluate_round(round_dir, round_number, dataloader, ngrid_1d):
    """Load one round's model and evaluate every 1-D marginal on its prior box.

    Returns ``(model, params, densities, prior_box, basis)``; ``densities`` is
    ``{label: (grid, density, inj) or None}``.
    """
    model = load_model(os.path.join(round_dir, "checkpoints"))
    prior_box = load_prior_box(round_dir, round_number)
    params = list(_iter_param_marginals(model))
    densities = {pi[0]: _eval_1d_marginal(model, dataloader, pi, prior_box, ngrid_1d)
                 for pi in params}
    basis = detect_basis(getattr(model, "bounds_trained", {}) or {})
    print(f"[round {round_number}] "
          f"{sum(v is not None for v in densities.values())}/{len(params)} marginals evaluated.")
    return model, params, densities, prior_box, basis


def load_mcmc(mcmc_samples_path, basis):
    """``(samples, names)`` in the NRE spin basis, or ``(None, None)``."""
    if not mcmc_samples_path:
        return None, None
    samples, names = load_mcmc_samples(mcmc_samples_path)
    return _maybe_remap_mcmc_to_basis(samples, names, basis)


def _grid_layout(n_panels: int, rows: int = DEFAULT_ROWS,
                 cols: int = DEFAULT_COLS) -> tuple:
    """Requested (rows, cols), growing rows if the run has more marginals."""
    if n_panels > rows * cols:
        rows = int(np.ceil(n_panels / cols))
        print(f"[layout] {n_panels} panels exceed the requested grid; "
              f"using {rows}x{cols}.")
    return rows, cols


def _axis_transforms_for(label: str, inj_val: float | None,
                          duration_weeks: float | None,
                          mcmc_samples_path: str | None):
    """Return (nre_to_y, mcmc_to_y, y_label) for one parameter.

    For ``Deltat``, both sides are mapped to "seconds offset from true merger"
    (see :func:`viz_helpers.deltat_axis_transforms`). For other parameters,
    identity transforms are used.
    """
    if label == "Deltat" and inj_val is not None and duration_weeks is not None:
        nre_to_y, mcmc_to_y, y_label, _ = deltat_axis_transforms(
            inj_val, duration_weeks, mcmc_samples_path,
        )
        return nre_to_y, mcmc_to_y, y_label
    identity = lambda v: np.asarray(v, dtype=float)
    return identity, identity, label


def _place_legend(fig, axes, handles, n_panels: int, rows: int,
                  cols: int) -> None:
    """Put the legend in the blank cells of a ragged last row, else in a single
    row underneath the figure.

    The layout is frozen first (``draw`` then a null layout engine) so the
    legend can be anchored in figure coordinates without constrained_layout
    reserving space for it — reserving space would stretch the column it sits
    in and break the uniform panel grid.
    """
    n_free = rows * cols - n_panels
    if n_free < 1:
        fig.legend(handles=handles, loc="outside lower center",
                   ncol=len(handles), frameon=False, handlelength=1.6,
                   columnspacing=1.4, borderaxespad=0.0)
        return

    r0, c0 = n_panels // cols, n_panels % cols
    first = axes[r0, c0]
    last = axes[rows - 1, cols - 1]
    fig.canvas.draw()
    p0, p1 = first.get_position(), last.get_position()
    fig.set_layout_engine("none")
    # The hidden cell draws no tick labels, so its left gutter is free space —
    # claim it, otherwise the legend is squeezed into ~60% of the column.
    x0 = p0.x0
    if c0 > 0:
        x0 = axes[r0, c0 - 1].get_position().x1 + 0.006
    rect = (x0, p1.y0, p1.x1 - x0, p0.y1 - p1.y0)

    # Shrink until the legend fits inside the free cells: at these panel widths
    # the default size spills over the neighbouring axes.
    fontsize = plt.rcParams["axes.labelsize"]   # match the axis labels
    for _ in range(4):
        leg = fig.legend(handles=handles, loc="center", frameon=False,
                         handlelength=1.0, handletextpad=0.4, borderpad=0.1,
                         labelspacing=0.5, borderaxespad=0.0,
                         fontsize=fontsize, bbox_to_anchor=rect,
                         bbox_transform=fig.transFigure)
        fig.canvas.draw()
        bb = leg.get_window_extent().transformed(fig.transFigure.inverted())
        scale = min(rect[2] / bb.width, rect[3] / bb.height)
        if scale >= 0.99 or fontsize <= 4.0:
            break
        leg.remove()
        fontsize = max(4.0, fontsize * 0.98 * scale)
    print(f"[legend] fontsize {fontsize:.1f} pt "
          f"(axis labels: {plt.rcParams['axes.labelsize']:.1f} pt)")


def plot_1d_posteriors(
    data: dict,
    mcmc_samples_path: str | None,
    outdir: str,
    stem: str,
    round_idx: int = -1,
    width_pt: float = TEXTWIDTH_PT,
    height_in: float | None = None,
    rows: int = DEFAULT_ROWS,
    cols: int = DEFAULT_COLS,
    fisher_sigmas: dict | None = None,
    proposal_1d: dict | None = None,
):
    """One panel per marginal: round ``round_idx``'s NRE 1-D posterior (default:
    the last round in *data*) and the corresponding MCMC posterior overlaid on
    the same axis. Consumes precomputed *data* (see :func:`evaluate_round`) —
    no model is loaded here.

    ``proposal_1d`` (``{label: [[lo, hi], ...]}``) draws the truncation this
    posterior proposes for the next round as vertical bounds.

    All curves are drawn as densities over the *display* coordinate produced by
    :func:`_axis_transforms_for` (identity for most parameters; "seconds offset
    from true merger" for ``Deltat``) and each is renormalised to unit area over
    the shared window so their shapes are directly comparable. The true
    (injection) value is a red dashed line.
    """
    os.makedirs(outdir, exist_ok=True)
    params = data["params"]
    if len(params) > 1:
        # Panel [0,0] has no left neighbour to lend its title overhang room
        # to, so a wide title there (e.g. "log10(Mc/Msun)", typically first
        # in the canonical order) clips against the figure's outer edge at
        # large --fontsize. [0,1] has neighbours on both sides -- swap the
        # first two panels' content so the wide title lands there instead.
        params = [params[1], params[0]] + list(params[2:])
    last_densities = data["per_round_densities"][round_idx]
    duration_weeks = data["duration_weeks"]
    mcmc_samples = data["mcmc_samples"]
    mcmc_param_names = data["mcmc_param_names"]
    n_rounds = data["round_numbers"][round_idx]

    if mcmc_samples is None:
        print("[warn] no MCMC samples; drawing NRE last-round marginals only.")

    rows, cols = _grid_layout(len(params), rows, cols)
    width_in = width_pt / PT_PER_INCH
    fig, axes = plt.subplots(
        rows, cols, squeeze=False, constrained_layout=True,
        figsize=(width_in, height_in or 1.45 * rows + 0.55),
    )

    for idx, pi in enumerate(params):
        label = pi[0]
        ax = axes[idx // cols, idx % cols]
        res = last_densities.get(label)
        if res is None:
            ax.set_visible(False)
            continue
        grid_1d, norm1d, inj = res
        nre_to_x, mcmc_to_x, x_label = _axis_transforms_for(
            label, inj, duration_weeks, mcmc_samples_path,
        )
        x_grid = nre_to_x(grid_1d)
        dx = abs(float(x_grid[1] - x_grid[0]))  # uniform (linear transform)

        y_nre = norm1d / max(np.sum(norm1d) * dx, 1e-300)
        ax.plot(x_grid, y_nre, color="C0", lw=LAST_ROUND_LW)
        ax.fill_between(x_grid, y_nre, alpha=0.2, color="C0")

        if mcmc_samples is not None and label in mcmc_param_names:
            mvals = eval_mcmc_kde_1d(
                mcmc_samples, mcmc_param_names, label, x_grid,
                sample_transform=mcmc_to_x,
            )
            if mvals is not None:
                mvals = mvals / max(np.sum(mvals) * dx, 1e-300)
                ax.plot(x_grid, mvals, color="grey", lw=LAST_ROUND_LW)
                ax.fill_between(x_grid, mvals, alpha=0.15, color="grey")

        # Fisher (Cramer-Rao) Gaussian: N(inj, sigma_FIM) built on the NATIVE
        # grid then renormalised over the display axis. The display transform is
        # linear, so the shape carries over unchanged and no Jacobian is needed.
        sig = (fisher_sigmas or {}).get(label)
        if sig is not None and np.isfinite(sig) and sig > 0:
            z = (np.asarray(grid_1d, dtype=float) - float(inj)) / float(sig)
            fvals = np.exp(-0.5 * z ** 2)
            fvals = fvals / max(np.sum(fvals) * dx, 1e-300)
            ax.plot(x_grid, fvals, color="C1", ls="-.", lw=LAST_ROUND_LW)

        for lo, hi in (proposal_1d or {}).get(label, []):
            for v in (lo, hi):
                ax.axvline(float(nre_to_x(np.array([v]))[0]), color="C2", ls=":", lw=1.0)

        mu_disp = float(nre_to_x(np.array([inj]))[0])
        ax.axvline(mu_disp, color="red", ls="--", lw=LAST_ROUND_LW)
        title = _panel_title(label)
        ax.set_title(title, pad=3)
        ax.set_yticks([])
        ax.xaxis.set_major_locator(MaxNLocator(nbins=2))

    for k in range(len(params), rows * cols):
        axes[k // cols, k % cols].set_visible(False)

    handles = [
        Line2D([0], [0], color="C0", lw=LAST_ROUND_LW, label=f"NRE\n(round {n_rounds})"),
        Line2D([0], [0], color="red", ls="--", lw=LAST_ROUND_LW, label="ground truth"),
    ]
    if mcmc_samples is not None:
        handles.insert(1, Line2D([0], [0], color="grey", lw=LAST_ROUND_LW, label="MCMC"))
    if fisher_sigmas:
        handles.append(Line2D([0], [0], color="C1", ls="-.", lw=LAST_ROUND_LW,
                              label="Fisher (CRLB)"))
    if proposal_1d:
        handles.append(Line2D([0], [0], color="C2", ls=":", lw=1.0,
                              label="next-round\nproposal"))
    _place_legend(fig, axes, handles, len(params), rows, cols)
    return save_figure(fig, outdir, stem)


# ---------------------------------------------------------------------------
# Truncation proposal and per-mode views
# ---------------------------------------------------------------------------

def load_proposal(name, round_number, keys):
    """The truncation round ``round_number`` produced (= prior of the next round).

    Returns ``(intervals_1d, regions_2d)``: ``{label: [[lo, hi], ...]}`` and
    ``{(i, j): MultiRegion}``. Empty dicts if the round was not truncated.
    """
    yml = os.path.join(DATA_ROOT_DIR, name, f"prior_after_round_{round_number}.yaml")
    npz = os.path.join(DATA_ROOT_DIR, name, f"truncation_round_{round_number}.npz")
    if not os.path.exists(yml):
        print(f"[proposal] {yml} not found; nothing to overlay.")
        return {}, {}
    t = load_truncation(yml, npz)
    intervals = {keys[i]: ivs for i, ivs in t["intervals_1d"].items()}
    regions = {tuple(int(v) for v in m["idx"]): region_of_pair(m) for m in t["masks_2d"]}
    return intervals, regions


def refine_enabled(round_dir):
    """``truncation.refine`` as recorded in the round's hparams.yaml."""
    path = os.path.join(round_dir, "hparams.yaml")
    if not os.path.exists(path):
        return False
    with open(path) as f:
        hp = yaml.safe_load(f) or {}
    trunc = (hp.get("train_conf") or hp).get("truncation") or {}
    return bool(trunc.get("refine", False))


def plot_modes_1d(model, dataloader, label, in_idx, intervals, ngrid, coarse,
                  outdir, round_number):
    """One panel per mode: the posterior on that mode's own interval."""
    out_idx = find_out_param_idx(model, in_idx)
    grid_c, dens_c, inj = coarse
    dp = grid_c[1] - grid_c[0]
    fig, axes = plt.subplots(1, len(intervals), squeeze=False,
                             figsize=(2.6 * len(intervals), 2.2), constrained_layout=True)
    for k, (lo, hi) in enumerate(intervals):
        ax = axes[0, k]
        grid, dens, _ = eval_nre_1d(model, dataloader, in_idx, out_idx, lo, hi, ngrid)
        mass = float(np.sum(dens_c[(grid_c >= lo) & (grid_c <= hi)]) * dp)
        ax.plot(grid, dens, color="C0", lw=1.5)
        ax.fill_between(grid, dens, alpha=0.2, color="C0")
        if lo <= inj <= hi:
            ax.axvline(inj, color="red", ls="--", lw=1.5)
        ax.set_title(f"mode {k + 1}: mass {mass:.3f}", pad=3)
        ax.set_xlabel(_panel_title(label))
        ax.set_yticks([])
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
    fig.suptitle(f"round {round_number}: {len(intervals)} modes of {latex_label(label)}")
    return save_figure(fig, outdir, f"modes_{label}")


def plot_modes_2d(model, dataloader, pair, region, keys, ngrid, inj, outdir,
                  round_number, show_proposal):
    """One panel per mode: the posterior on that mode's own refined subgrid."""
    i, j = pair
    out_idx = find_out_param_idx(model, (i, j))
    swap = out_idx is None
    if swap:
        out_idx = find_out_param_idx(model, (j, i))
    parts = region.parts
    fig, axes = plt.subplots(1, len(parts), squeeze=False,
                             figsize=(3.0 * len(parts), 2.8), constrained_layout=True)
    for k, part in enumerate(parts):
        ax = axes[0, k]
        bx = (float(part.grids[0].min()), float(part.grids[0].max()))
        by = (float(part.grids[1].min()), float(part.grids[1].max()))
        idx, b0, b1 = ((j, i), by, bx) if swap else ((i, j), bx, by)
        norm2d, _, gx, gy = eval_posterior_2d(model, dataloader, idx, out_idx,
                                              ngrid_points=ngrid, bounds_0=b0,
                                              bounds_1=b1, keep_batch_dim=True)
        norm = norm2d[0]
        if swap:
            gx, gy, norm = gy.T, gx.T, norm.T
        ax.pcolormesh(gx, gy, norm, shading="auto", cmap="Blues")
        if show_proposal:
            ax.contour(part.grids[0], part.grids[1], part.mask.astype(float),
                       levels=[0.5], colors="C2", linewidths=1.2)
        if inj is not None and bx[0] <= inj[0] <= bx[1] and by[0] <= inj[1] <= by[1]:
            ax.plot(inj[0], inj[1], "r+", markersize=9, markeredgewidth=1.5)
        ax.set_title(f"mode {k + 1}", pad=3)
        ax.set_xlabel(latex_label(keys[i]))
        ax.set_ylabel(latex_label(keys[j]))
    fig.suptitle(f"round {round_number}: {len(parts)} modes of "
                 f"({latex_label(keys[i])}, {latex_label(keys[j])})")
    return save_figure(fig, outdir, f"modes_{keys[i]}_{keys[j]}")


# ---------------------------------------------------------------------------
# Sky
# ---------------------------------------------------------------------------

def plot_sky(round_dir, round_number, dataloader, mcmc_file, outdir, ngrid,
             ngrid_zoom, cr_area, proposal_region=None, fontsize=10.0):
    """Zoomed (lambda, sin beta) posterior, reusing visualise_sky_truncation."""
    import visualise_sky_truncation as vst

    rec, model, ctx, inj = vst._evaluate_one_round(
        round_dir, round_number, dataloader, ngrid, cr_area, need_area=True)
    if ctx is None:
        print(f"[sky] round {round_number} has no sky head; skipping.")
        return None
    print(f"[sky] {cr_area:.0%} credible area: {rec['area']:.3g} sq deg")
    mcmc_kde = None
    if mcmc_file:
        _, mcmc_kde = vst._mcmc_sky_density(mcmc_file, ngrid, cr_area)
    include = None
    if proposal_region is not None:
        (x_lo, x_hi), (y_lo, y_hi) = proposal_region.bounds()
        include = ((x_lo, x_hi), (y_lo, y_hi))
    computed = vst._compute_final_zoom(rec, mcmc_kde, cr_area, model, dataloader,
                                       ctx, ngrid_zoom, include=include)
    style = dict(fontsize=fontsize, figsize=(4.8, 3.4), lw=1.5)
    for coords, fname in (("native", "sky_zoom.pdf"), ("lonlat", "sky_zoom_lonlat.pdf")):
        vst._plot_final_zoom(computed, inj, cr_area, os.path.join(outdir, fname),
                             coords=coords, region=proposal_region, **style)
    return inj


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-name", required=True, help="run tag (TIME_OF_EXECUTION)")
    p.add_argument("--round", type=int, default=None, help="round to plot (default: last)")
    p.add_argument("--obs-path", default=None,
                   help="observation HDF5 (default: the run's observation_used.yaml); "
                        "a different observation raises")
    p.add_argument("--mcmc-file", default=None, help="flat MCMC samples HDF5 to compare with")
    p.add_argument("--proposal", action="store_true",
                   help="overlay the truncation this posterior proposes for the next round")
    p.add_argument("--ckpt", default=None, help="checkpoint overriding the round's truncation.ckpt")
    p.add_argument("--ngrid-1d", type=int, default=200)
    p.add_argument("--ngrid-2d", type=int, default=200, help="coarse sky grid")
    p.add_argument("--ngrid-zoom", type=int, default=300, help="fine sky grid over the credible region")
    p.add_argument("--ngrid-mode", type=int, default=128, help="grid per mode in the per-mode figures")
    p.add_argument("--cr-area", type=float, default=0.90)
    p.add_argument("--no-sky", action="store_true", help="skip the sky figures")
    p.add_argument("--fontsize", type=float, default=10.0)
    p.add_argument("--width-pt", type=float, default=TEXTWIDTH_PT)
    p.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    p.add_argument("--cols", type=int, default=DEFAULT_COLS)
    args = p.parse_args()

    name = args.run_name
    round_dirs = round_dirs_upto(name, args.round)
    r = len(round_dirs)
    rd = round_dirs[-1]
    if args.ckpt:
        register_ckpt_override(rd, args.ckpt)

    obs_path = resolve_obs_path(name, args.obs_path)
    print(f"[obs] {obs_path}")
    dataloader = build_obs_dataloader(obs_path)
    outdir = os.path.join(PLOTS_ROOT_DIR, name, f"posterior_round_{r}")
    os.makedirs(outdir, exist_ok=True)
    apply_paper_style(args.fontsize)

    model, params, densities, _, basis = evaluate_round(rd, r, dataloader, args.ngrid_1d)
    keys = keys_for_model(model)
    mcmc_samples, mcmc_names = load_mcmc(args.mcmc_file, basis)
    intervals, regions = load_proposal(name, r, keys)

    data = {
        "params": params,
        "per_round_densities": [densities],
        "round_numbers": [r],
        "duration_weeks": load_duration_weeks(round_dirs[0], 1),
        "mcmc_samples": mcmc_samples,
        "mcmc_param_names": mcmc_names,
    }
    plot_1d_posteriors(data, args.mcmc_file, outdir, "posterior_1d",
                       width_pt=args.width_pt, rows=args.rows, cols=args.cols,
                       proposal_1d=intervals if args.proposal else None)

    sky_region = regions.get((7, 8))
    inj_sky = None
    if not args.no_sky:
        inj_sky = plot_sky(rd, r, dataloader, args.mcmc_file, outdir, args.ngrid_2d,
                           args.ngrid_zoom, args.cr_area,
                           proposal_region=sky_region if args.proposal else None,
                           fontsize=args.fontsize)

    multi_1d = {lbl: ivs for lbl, ivs in intervals.items() if len(ivs) > 1}
    multi_2d = {pair: reg for pair, reg in regions.items() if len(reg.parts) > 1}
    if (multi_1d or multi_2d) and not refine_enabled(rd):
        print("[modes] several modes found but truncation.refine was off: "
              "no refined per-mode grids to show.")
    elif multi_1d or multi_2d:
        by_label = {pi[0]: pi for pi in params}
        for label, ivs in multi_1d.items():
            pi = by_label.get(label)
            if pi is None or pi[1] != 1 or densities.get(label) is None:
                continue
            plot_modes_1d(model, dataloader, label, pi[2], ivs, args.ngrid_mode,
                          densities[label], outdir, r)
        for pair, reg in multi_2d.items():
            inj = inj_sky if pair == (7, 8) else None
            plot_modes_2d(model, dataloader, pair, reg, keys, args.ngrid_mode,
                          inj, outdir, r, args.proposal)
    print(f"[done] figures in {outdir}")


if __name__ == "__main__":
    main()
