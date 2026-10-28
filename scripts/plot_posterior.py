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
* ``modes_<param>.pdf`` — when the region round N was trained on (round N-1's
  accepted set) has several modes: the posterior on each mode's own grid, with
  its mass normalised over that region (the masses sum to 1). With
  ``--mcmc-file`` the MCMC is overlaid on the mode nearest its median.

``--proposal`` overlays the truncation this posterior produced, i.e. the prior
of round N+1: interval bounds in 1-D, the accepted-region contour in 2-D, and
in the per-mode figures the fraction of the posterior it keeps.

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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from pembhb import PLOTS_ROOT_DIR
from pembhb.mask_truncation import load_truncation, region_of_pair
from pembhb.sampler import chi12_to_chieff_chidiff
from pembhb.utils import get_logratios_grid, get_logratios_grid_2d

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
    support_1d: dict | None = None,
):
    """One panel per marginal: round ``round_idx``'s NRE 1-D posterior (default:
    the last round in *data*) and the corresponding MCMC posterior overlaid on
    the same axis. Consumes precomputed *data* (see :func:`evaluate_round`) —
    no model is loaded here.

    ``proposal_1d`` (``{label: [[lo, hi], ...]}``) draws the truncation this
    posterior proposes for the next round as vertical bounds. ``support_1d``
    (same form) is the region the round was trained on; the rest of the panel,
    where the network extrapolates and the pipeline zeroes the density, is greyed.

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

        support = (support_1d or {}).get(label)
        if support:
            edges = [float(grid_1d[0])] + [v for iv in sorted(support) for v in iv] + [float(grid_1d[-1])]
            for g0, g1 in zip(edges[0::2], edges[1::2]):
                if g1 > g0:
                    ax.axvspan(*nre_to_x(np.array([g0, g1])), color="0.85", lw=0, zorder=0)
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
    if support_1d:
        handles.append(Patch(facecolor="0.85", label="not trained on\n(zeroed)"))
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


MASS_NGRID_1D = 2000   # points per interval
MASS_NGRID_2D = 256    # points per axis and window

# The posterior of round r lives on round r-1's accepted set (its domain: the
# network was trained there, and the pipeline zeroes the density outside it
# before truncating). The per-mode figures show the modes of that domain, with
# their posterior masses normalised over it, so they sum to 1. Round r's own
# accepted set (the proposal, i.e. round r+1's prior) is only overlaid, together
# with the fraction of the posterior it keeps. Integrals are of the unnormalised
# ratio exp(log r), so densities evaluated on different grids share one
# normalisation.

def _log_integral(lr, cell, weight=None):
    lr = np.asarray(lr, dtype=float)
    w = np.ones_like(lr) if weight is None else np.asarray(weight, dtype=float)
    if not w.any():
        return -np.inf
    m = lr[w > 0].max()
    return m + np.log(np.sum(np.exp(lr - m) * w) * cell)


def _intersect(a, b):
    out = []
    for a0, a1 in a:
        for b0, b1 in b:
            lo, hi = max(a0, b0), min(a1, b1)
            if hi > lo:
                out.append((lo, hi))
    return out


def _log_integral_1d(model, dataloader, in_idx, out_idx, intervals, n):
    """log ∫ r dθ over a union of disjoint intervals."""
    terms = []
    for lo, hi in intervals:
        lr, _, _ = get_logratios_grid(dataloader, model, n, in_param_idx=in_idx,
                                      out_param_idx=out_idx, low=lo, high=hi)
        # analytic spacing: in float32 a narrow grid's points can coincide
        terms.append(_log_integral(np.asarray(lr)[0], (hi - lo) / (n - 1)))
    return np.logaddexp.reduce(terms) if terms else -np.inf


def domain_masses_1d(model, dataloader, in_idx, domain, n, proposal=None):
    """Posterior mass of each interval of ``domain`` (they sum to 1), and the
    mass the ``proposal`` intervals keep (None without a proposal)."""
    out_idx = find_out_param_idx(model, in_idx)
    log_m = [_log_integral_1d(model, dataloader, in_idx, out_idx, [iv], n) for iv in domain]
    log_z = np.logaddexp.reduce(log_m)
    masses = [float(np.exp(v - log_z)) for v in log_m]
    kept = None
    if proposal:
        kept = float(np.exp(_log_integral_1d(model, dataloader, in_idx, out_idx,
                                             _intersect(proposal, domain), n) - log_z))
    return masses, kept


def _eval_window_2d(model, dataloader, pair, bx, by, n):
    """``(log r, gx, gy, cell)`` on an n×n grid, rows = 2nd param, cols = 1st param."""
    i, j = pair
    out_idx = find_out_param_idx(model, (i, j))
    swap = out_idx is None
    if swap:
        out_idx = find_out_param_idx(model, (j, i))
    idx, b0, b1 = ((j, i), by, bx) if swap else ((i, j), bx, by)
    lr, _, gx, gy = get_logratios_grid_2d(dataloader, model, n, out_param_idx=out_idx,
                                          in_param_idx=idx, bounds_0=b0, bounds_1=b1)
    lr, gx, gy = np.asarray(lr)[0], np.asarray(gx), np.asarray(gy)
    if swap:
        lr, gx, gy = lr.T, gy.T, gx.T
    cell = (bx[1] - bx[0]) / (n - 1) * (by[1] - by[0]) / (n - 1)
    return lr, gx[0, :], gy[:, 0], cell


def _part_window(part):
    return ((float(part.grids[0].min()), float(part.grids[0].max())),
            (float(part.grids[1].min()), float(part.grids[1].max())))


def domain_masses_2d(model, dataloader, pair, domain, n, proposal=None):
    """Posterior mass of each part of the ``domain`` region (they sum to 1), and
    the mass the ``proposal`` region keeps (None without a proposal)."""
    log_m, log_k = [], []
    for part in domain.parts:
        lr, gx, gy, cell = _eval_window_2d(model, dataloader, pair, *_part_window(part), n)
        w = part.contains_grid((gx, gy))
        log_m.append(_log_integral(lr, cell, w))
        if proposal is not None:
            log_k.append(_log_integral(lr, cell, w & proposal.contains_grid((gx, gy))))
    log_z = np.logaddexp.reduce(log_m)
    masses = [float(np.exp(v - log_z)) for v in log_m]
    kept = float(np.exp(np.logaddexp.reduce(log_k) - log_z)) if log_k else None
    return masses, kept


def _kept_note(kept):
    return "" if kept is None else f"\nnext-round proposal (green) keeps {kept:.4f} of the posterior"


def _nearest_box(point, boxes):
    """Index of the box (``[(lo, hi), ...]`` per axis) nearest to ``point``;
    distance 0 inside a box."""
    d = [np.hypot.reduce([max(lo - p, 0.0, p - hi) for p, (lo, hi) in zip(point, box)])
         for box in boxes]
    return int(np.argmin(d))


def _mcmc_note(k):
    return "" if k is None else f"\nMCMC (grey) on mode {k + 1}, the nearest to its median"


HPD_LEVELS = (0.68, 0.95)


def _hpd_levels(h, levels=HPD_LEVELS):
    """Density values enclosing ``levels`` of the mass of ``h`` (nan = outside),
    increasing; None if ``h`` carries no mass."""
    flat = np.sort(h[np.isfinite(h)].ravel())[::-1]
    if flat.sum() <= 0:
        return None
    cum = np.cumsum(flat) / flat.sum()
    return sorted({float(flat[min(np.searchsorted(cum, lev), flat.size - 1)]) for lev in levels})


def _identity_2d(x, y):
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def _sky_to_deg(lam, sinbeta):
    """(lambda, sin beta) -> (ecliptic longitude, latitude) in degrees."""
    return np.degrees(np.asarray(lam, dtype=float)), np.degrees(np.arcsin(np.clip(sinbeta, -1, 1)))


def _mcmc_contours(ax, xy, bx, by, to_disp=_identity_2d, nbins=60):
    """HPD contours of the samples falling in the window ``bx`` x ``by``, drawn
    in display coordinates."""
    from scipy.ndimage import gaussian_filter
    h, ex, ey = np.histogram2d(xy[:, 0], xy[:, 1], bins=nbins, range=(bx, by))
    h = gaussian_filter(h.T, 1.5)
    lv = _hpd_levels(h)
    if lv is None:
        return
    cx, cy = to_disp(0.5 * (ex[1:] + ex[:-1]), 0.5 * (ey[1:] + ey[:-1]))
    ax.contour(cx, cy, h, levels=lv, colors="0.3", linewidths=1.2)


def plot_modes_1d(model, dataloader, label, in_idx, domain, masses, kept, proposal,
                  ngrid, inj, outdir, round_number, mcmc_vals=None, transforms=None):
    """One panel per interval of the domain: the posterior on that interval.

    ``mcmc_vals`` (raw MCMC samples of this parameter) is drawn on the interval
    nearest its median; ``transforms`` = ``(nre_to_x, mcmc_to_x, x_label)`` maps
    both to the display axis (see :func:`_axis_transforms_for`).
    """
    from scipy.stats import gaussian_kde
    nre_to_x, mcmc_to_x, x_label = transforms or _axis_transforms_for(label, None, None, None)
    out_idx = find_out_param_idx(model, in_idx)
    best = mcmc_x = None
    if mcmc_vals is not None:
        mcmc_x = mcmc_to_x(mcmc_vals)
        best = _nearest_box([np.median(mcmc_x)], [[tuple(nre_to_x(np.array(iv)))] for iv in domain])
    fig, axes = plt.subplots(1, len(domain), squeeze=False,
                             figsize=(2.6 * len(domain), 2.2), constrained_layout=True)
    for k, ((lo, hi), mass) in enumerate(zip(domain, masses)):
        ax = axes[0, k]
        grid, dens, _ = eval_nre_1d(model, dataloader, in_idx, out_idx, lo, hi, ngrid)
        x = nre_to_x(grid)
        dx = abs(float(x[1] - x[0]))
        y = dens / max(np.sum(dens) * dx, 1e-300)
        ax.plot(x, y, color="C0", lw=1.5)
        ax.fill_between(x, y, alpha=0.2, color="C0")
        if k == best:
            m = gaussian_kde(mcmc_x)(x)
            m = m / max(np.sum(m) * dx, 1e-300)
            ax.plot(x, m, color="grey", lw=1.5)
            ax.fill_between(x, m, alpha=0.15, color="grey")
        for p_lo, p_hi in _intersect(proposal or [], [(lo, hi)]):
            for v in nre_to_x(np.array([p_lo, p_hi])):
                ax.axvline(v, color="C2", ls=":", lw=1.2)
        if lo <= inj <= hi:
            ax.axvline(float(nre_to_x(np.array([inj]))[0]), color="red", ls="--", lw=1.5)
        ax.set_title(f"mode {k + 1}: mass {mass:.3f}", pad=3)
        ax.set_xlabel(x_label if label == "Deltat" else _panel_title(label))
        ax.set_yticks([])
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
    fig.suptitle(f"round {round_number}: {latex_label(label)} on the {len(domain)} "
                 f"modes it was trained on{_kept_note(kept)}{_mcmc_note(best)}")
    return save_figure(fig, outdir, f"modes_{label}")


def plot_modes_2d(model, dataloader, pair, domain, masses, kept, proposal, keys, ngrid,
                  inj, outdir, round_number, mcmc_xy=None):
    """One panel per part of the domain: 68/95% HPD contours of the posterior on
    that part's subgrid. The sky pair is drawn in degrees (longitude, latitude);
    HPD levels are taken on the native (lambda, sin beta) grid, whose cells are
    equal solid angle.

    ``mcmc_xy`` (raw MCMC samples, N x 2) is drawn as 68/95% contours on the
    part nearest its median."""
    i, j = pair
    parts = domain.parts
    sky = (keys[i], keys[j]) == ("lambda", "sinbeta")
    to_disp = _sky_to_deg if sky else _identity_2d
    xlabel, ylabel = ((r"$\lambda$ [deg]", r"$\beta$ [deg]") if sky
                      else (latex_label(keys[i]), latex_label(keys[j])))
    best = None
    if mcmc_xy is not None:
        best = _nearest_box(np.median(mcmc_xy, axis=0), [_part_window(p) for p in parts])
    fig, axes = plt.subplots(1, len(parts), squeeze=False,
                             figsize=(3.0 * len(parts), 2.8), constrained_layout=True)
    for k, (part, mass) in enumerate(zip(parts, masses)):
        ax = axes[0, k]
        bx, by = _part_window(part)
        lr, gx, gy, _ = _eval_window_2d(model, dataloader, pair, bx, by, ngrid)
        inside = part.contains_grid((gx, gy))
        acc = proposal.contains_grid((gx, gy)) if proposal is not None else None
        # a mode across lambda = 0/2pi has a full-period window: recentre on 0
        wrap = sky and bx[1] - bx[0] > 0.99 * 2 * np.pi
        shift = (lambda v: np.where(v > np.pi, v - 2 * np.pi, v)) if wrap else (lambda v: v)
        if wrap:
            order = np.argsort(shift(gx))
            gx, lr, inside = shift(gx)[order], lr[:, order], inside[:, order]
            acc = acc[:, order] if acc is not None else None
        dens = np.where(inside, np.exp(lr - lr[inside].max()), np.nan)
        dx, dy = to_disp(gx, gy)
        lv = _hpd_levels(dens)
        if lv is not None:
            ax.contour(dx, dy, np.nan_to_num(dens), levels=lv, colors="C0", linewidths=1.5)
        ax.contour(dx, dy, inside.astype(float), levels=[0.5], colors="0.4",
                   linewidths=1.0, linestyles="dashed")
        if acc is not None and acc.any():
            ax.contour(dx, dy, acc.astype(float), levels=[0.5], colors="C2", linewidths=1.2)
        if k == best:
            xy = np.column_stack([shift(mcmc_xy[:, 0]), mcmc_xy[:, 1]])
            _mcmc_contours(ax, xy, (float(gx[0]), float(gx[-1])), by, to_disp)
        if inj is not None and bx[0] <= inj[0] <= bx[1] and by[0] <= inj[1] <= by[1]:
            ix, iy = to_disp(shift(inj[0]), inj[1])
            ax.plot(ix, iy, "r+", markersize=9, markeredgewidth=1.5)
        cols = np.flatnonzero(inside.any(axis=0))
        if wrap and cols.size:
            pad = 2 * abs(float(dx[1] - dx[0]))
            ax.set_xlim(dx[cols[0]] - pad, dx[cols[-1]] + pad)
        ax.set_title(f"mode {k + 1}: mass {mass:.3f}", pad=3)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    name = "sky" if sky else f"({latex_label(keys[i])}, {latex_label(keys[j])})"
    fig.suptitle(f"round {round_number}: {name}, {len(parts)} trained modes, NRE 68/95% (blue)"
                 f"{_kept_note(kept)}"
                 + ("" if best is None else f"\nMCMC 68/95% (grey) on mode {best + 1}, nearest its median"))
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
    # the domain round r was trained on = round r-1's accepted set
    prev_intervals, prev_regions = load_proposal(name, r - 1, keys) if r > 1 else ({}, {})

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
                       proposal_1d=intervals if args.proposal else None,
                       support_1d=prev_intervals or None)

    sky_region = regions.get((7, 8))
    inj_sky = None
    if not args.no_sky:
        inj_sky = plot_sky(rd, r, dataloader, args.mcmc_file, outdir, args.ngrid_2d,
                           args.ngrid_zoom, args.cr_area,
                           proposal_region=sky_region if args.proposal else None,
                           fontsize=args.fontsize)

    # Per-mode figures: modes of the domain (round r-1's accepted set), when it
    # has several.
    multi_1d = {lbl: ivs for lbl, ivs in prev_intervals.items() if len(ivs) > 1}
    multi_2d = {pair: reg for pair, reg in prev_regions.items() if len(reg.parts) > 1}

    def mcmc_col(label):
        if mcmc_samples is None or label not in mcmc_names:
            return None
        return mcmc_samples[:, mcmc_names.index(label)]

    if multi_1d or multi_2d:
        by_label = {pi[0]: pi for pi in params}
        for label, domain in multi_1d.items():
            pi = by_label.get(label)
            if pi is None or pi[1] != 1 or densities.get(label) is None:
                continue
            proposal = intervals.get(label) if args.proposal else None
            masses, kept = domain_masses_1d(model, dataloader, pi[2], domain,
                                            MASS_NGRID_1D, proposal)
            print(f"[modes] {label}: masses {np.round(masses, 4).tolist()}"
                  + ("" if kept is None else f", proposal keeps {kept:.4f}"))
            inj = densities[label][2]
            plot_modes_1d(model, dataloader, label, pi[2], domain, masses, kept, proposal,
                          args.ngrid_mode, inj, outdir, r, mcmc_vals=mcmc_col(label),
                          transforms=_axis_transforms_for(label, inj, data["duration_weeks"],
                                                          args.mcmc_file))
        for pair, domain in multi_2d.items():
            i, j = pair
            proposal = regions.get(pair) if args.proposal else None
            masses, kept = domain_masses_2d(model, dataloader, pair, domain,
                                            MASS_NGRID_2D, proposal)
            print(f"[modes] {keys[i]}-{keys[j]}: masses {np.round(masses, 4).tolist()}"
                  + ("" if kept is None else f", proposal keeps {kept:.4f}"))
            inj = inj_sky if pair == (7, 8) else None
            cols = [mcmc_col(keys[i]), mcmc_col(keys[j])]
            # Deltat is stored in different units by NRE and MCMC: no 2-D overlay
            mcmc_xy = (np.column_stack(cols) if all(c is not None for c in cols)
                       and "Deltat" not in (keys[i], keys[j]) else None)
            plot_modes_2d(model, dataloader, pair, domain, masses, kept, proposal, keys,
                          args.ngrid_mode, inj, outdir, r, mcmc_xy=mcmc_xy)
    print(f"[done] figures in {outdir}")


if __name__ == "__main__":
    main()
