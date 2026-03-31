"""
Visualise the evolution of posterior contours across successive truncation rounds.

The top-level entry point is ``plot_all_marginals``, which iterates over every
marginal output of the trained model and produces:

* **2-D marginals** – one figure with N columns (one per round), each showing
  posterior contours (no colorscale), a dashed rectangle for the next round's
  prior window, and connecting lines to the next panel's frame.  The final
  round additionally displays inset 1-D marginal distributions (top and
  right) with optional MCMC overlay.

* **1-D marginals** – one figure showing how the truncated prior window
  (lower and upper bound) evolves across rounds (x-axis = round index),
  plus a final-round 1-D posterior density with optional MCMC overlay.

Usage
-----
Edit ``round_dirs``, ``data_path``, and optionally ``mcmc_samples_dir`` at the
bottom of this file, then run::

    python scripts/visualise_truncation_rounds.py
"""

import os
from glob import glob

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from pembhb import ROOT_DIR
from pembhb.model import InferenceNetwork
from pembhb.data import MBHBDataset
from scipy.stats import gaussian_kde
from matplotlib.lines import Line2D

from pembhb.utils import (
    _ORDERED_PRIOR_KEYS,
    get_logratios_grid,
    get_logratios_grid_2d,
    contour_levels,
    posterior_contours_2d,
    mbhb_collate_fn,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_prior_box(round_dir: str, round_number: int) -> dict:
    """Read the prior bounds stored in ``simulation_round_<round_number>.yaml``.

    Supports both top-level ``prior:`` and nested ``conf.prior:`` layouts.

    Returns a dict mapping parameter name → ``(low, high)`` tuple.
    """
    round_dir_basename = os.path.basename(round_dir)
    round_path_data = f"/data/gpuleo/mbhb/{round_dir_basename}"
    yaml_path = os.path.join(round_path_data, f"simulation_round_{round_number}.yaml")
    with open(yaml_path, "r") as f:
        conf = yaml.safe_load(f)
    prior = conf.get("prior") or conf["conf"]["prior"]
    return {k: tuple(v) for k, v in prior.items()}


def load_model(ckpt_dir: str) -> InferenceNetwork:
    """Load the first ``*.ckpt`` checkpoint found inside *ckpt_dir*."""
    ckpt_files = glob(os.path.join(ckpt_dir, "*.ckpt"))
    if not ckpt_files:
        raise FileNotFoundError(f"No .ckpt file found in {ckpt_dir}")
    model = InferenceNetwork.load_from_checkpoint(ckpt_files[0])
    model.eval()
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    # workaround for older checkpoints that lack the architecture attribute
    try:
        model.data_summary.autoencoder.architecture = "conv"
    except AttributeError:
        pass
    return model


def get_all_marginals(model: InferenceNetwork) -> list:
    """Return all marginals as ``[(label, ndim, in_param_idx, out_param_idx), ...]``.

    ``label`` is a human-readable string.
    ``ndim`` is 1 or 2.
    ``in_param_idx`` is an ``int`` for 1-D, a ``tuple`` for 2-D.
    ``out_param_idx`` is the column index in the model's output tensor.
    """
    result = []
    for out_idx, marginal in enumerate(model.marginals_list):
        ndim = len(marginal)
        if ndim == 1:
            label = _ORDERED_PRIOR_KEYS[marginal[0]]
            in_idx = marginal[0]
        elif ndim == 2:
            label = f"{_ORDERED_PRIOR_KEYS[marginal[0]]} vs {_ORDERED_PRIOR_KEYS[marginal[1]]}"
            in_idx = tuple(marginal)
        else:
            continue  # skip higher-dimensional marginals
        result.append((label, ndim, in_idx, out_idx))
    return result


def find_out_param_idx(model: InferenceNetwork, in_param_idx):
    """Return the output column index for *in_param_idx* in *model.marginals_list*.

    Returns ``None`` if the marginal is not present in this model (e.g. it was
    only introduced in a later round).
    """
    for out_idx, marginal in enumerate(model.marginals_list):
        ndim = len(marginal)
        in_idx = marginal[0] if ndim == 1 else tuple(marginal)
        if in_idx == in_param_idx:
            return out_idx
    return None


def compute_normalised_posterior(
    dataloader, model, in_param_idx, out_param_idx, bounds_0, bounds_1, ngrid_points=100
):
    """Evaluate and normalise the 2-D posterior on a regular grid.

    Returns ``(norm2d, inj_params, gx, gy)`` where ``norm2d`` has shape
    ``(batch_size, ngrid_points, ngrid_points)``.
    """
    logratios, inj_params, gx, gy = get_logratios_grid_2d(
        dataloader,
        model,
        ngrid_points=ngrid_points,
        in_param_idx=in_param_idx,
        out_param_idx=out_param_idx,
        bounds_0=bounds_0,
        bounds_1=bounds_1,
    )
    ratios = np.exp(logratios)
    dp0 = gx[0, 1] - gx[0, 0]   # spacing along x (varies along columns)
    dp1 = gy[1, 0] - gy[0, 0]   # spacing along y (varies along rows)
    norm2d = ratios / np.sum(ratios * dp0 * dp1, axis=(1, 2), keepdims=True)
    return norm2d, inj_params, gx, gy


def connect_box_to_next_axes(
    fig,
    rect_patch,
    ax_rect,
    ax_next,
    color="red",
    linewidth=1.0,
    alpha=0.6,
    linestyle="-",
):
    """Draw four lines connecting the corners of *rect_patch* (drawn on *ax_rect*
    in its data coordinates) to the corresponding corners of *ax_next*'s axis frame.

    Matching: bottom-left → bottom-left, bottom-right → bottom-right,
              top-right → top-right, top-left → top-left.

    ``fig.canvas.draw()`` must have been called beforehand so that all
    transforms are fully initialised.
    """
    # Rectangle corners in ax_rect's data coordinates
    x0 = rect_patch.get_x()
    y0 = rect_patch.get_y()
    x1 = x0 + rect_patch.get_width()
    y1 = y0 + rect_patch.get_height()
    rect_corners_data = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])

    # ax_next's axis-frame corners in its own data coordinates
    xl = ax_next.get_xlim()
    yl = ax_next.get_ylim()
    next_corners_data = np.array(
        [[xl[0], yl[0]], [xl[1], yl[0]], [xl[1], yl[1]], [xl[0], yl[1]]]
    )

    # Transform: data → display (pixels) → figure (0–1)
    inv_fig = fig.transFigure.inverted()
    rect_corners_fig = inv_fig.transform(ax_rect.transData.transform(rect_corners_data))
    next_corners_fig = inv_fig.transform(ax_next.transData.transform(next_corners_data))

    lines = []
    for (rx, ry), (nx, ny) in zip(rect_corners_fig, next_corners_fig):
        line = plt.Line2D(
            [rx, nx],
            [ry, ny],
            transform=fig.transFigure,
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            linestyle=linestyle,
            zorder=1000,
            clip_on=False,
        )
        fig.add_artist(line)
        lines.append(line)

    return lines


# ---------------------------------------------------------------------------
# 1-D prior-window evolution
# ---------------------------------------------------------------------------

def _hpd_interval_1d(norm1d: np.ndarray, grid: np.ndarray, level: float):
    """Return the highest-posterior-density interval at credibility *level*.

    Uses the same cumulative-sum approach as ``get_widest_interval_1d`` in
    tmnre.py.  The tail probability outside the interval is ``1 - level``.

    Parameters
    ----------
    norm1d : 1-D array
        Normalised posterior density evaluated at *grid* points.
    grid : 1-D array
        Parameter values corresponding to *norm1d*.
    level : float
        Target credibility (e.g. 0.50, 0.90, 0.9999).

    Returns
    -------
    (low, high) : float, float
    """
    dp = grid[1] - grid[0]
    eps = 1.0 - level
    cumsum = np.cumsum(norm1d * dp)
    idx_low  = np.searchsorted(cumsum, eps / 2)
    idx_high = np.searchsorted(cumsum, 1.0 - eps / 2)
    idx_high = min(idx_high, len(grid) - 1)
    return float(grid[idx_low]), float(grid[idx_high])


def plot_1d_prior_evolution(
    round_dirs: list,
    param_key: str,
    in_param_idx: int,
    out_param_idx: int,
    dataloader,
    figsize: tuple = (7, 4),
    hpd_levels: tuple = (0.50, 0.90, 0.9999),
    band_colors: tuple = ("#4393c3", "#92c5de", "#d1e5f0"),
    ngrid_points: int = 1000,
    mcmc_samples_dir: str = None,
    ngrid_points_1d: int = 500,
):
    """Plot how the 1-D posterior HPD intervals evolve across rounds.

    For each round the model is loaded, the 1-D posterior is evaluated on a
    grid spanning the round's prior window, and HPD intervals at the levels
    defined by *hpd_levels* are shown as filled bands.  The prior window
    itself (99.99 % HPD bound, matching the truncation logic in tmnre.py)
    is shown as dashed lines.

    Additionally, the final-round 1-D posterior density is plotted on a
    secondary axes (right-hand panel) with an optional MCMC KDE overlay
    when *mcmc_samples_dir* is provided.

    Reuses ``get_logratios_grid`` from pembhb.utils (the same function called
    by ``get_widest_interval_1d`` in tmnre.py).

    Parameters
    ----------
    round_dirs : list of str
        One path per round.
    param_key : str
        Parameter name (used for axis labels).
    in_param_idx : int
        Index of the parameter in the model input.
    out_param_idx : int
        Column index in the model output tensor.
    dataloader : DataLoader
        Observation data loader.
    figsize : tuple
        Figure size in inches for the HPD evolution panel.
    hpd_levels : tuple of float
        Credibility levels to show as bands, from narrowest (innermost) to
        widest (outermost).  Default: 50 %, 90 %, 99.99 %.
    band_colors : tuple of str
        Fill colours for each band, matched by position to *hpd_levels*.
    ngrid_points : int
        Grid resolution for posterior evaluation (HPD panel).
    mcmc_samples_dir : str, optional
        Directory containing ``flat_samples.npy`` and
        ``varying_params.txt``.  When provided, the MCMC marginal KDE is
        overlaid on the final-round density panel.
    ngrid_points_1d : int
        Grid resolution for the final-round 1-D density panel.

    Returns
    -------
    fig, axes : tuple
        ``fig`` is the figure; ``axes`` is a dict with keys ``"hpd"`` and
        ``"density"`` pointing to the two axes.
    """
    n_rounds = len(round_dirs)
    round_indices = np.arange(1, n_rounds + 1)

    # prior window bounds from YAML (for dashed reference lines)
    prior_lows, prior_highs = [], []
    for i, d in enumerate(round_dirs):
        box = load_prior_box(d, i + 1)
        lo, hi = box[param_key]
        prior_lows.append(lo)
        prior_highs.append(hi)

    # HPD intervals from model evaluation: shape (n_rounds, n_levels, 2)
    hpd_lows  = {lvl: [] for lvl in hpd_levels}
    hpd_highs = {lvl: [] for lvl in hpd_levels}

    # Store final-round density for the secondary panel
    final_norm1d = None
    final_grid_1d = None
    final_inj = None

    for i, round_dir in enumerate(round_dirs):
        print(f"  [1-D {param_key}] Round {i + 1}: evaluating posterior...")
        model = load_model(os.path.join(round_dir, "checkpoints"))

        # Look up the output column for this marginal in this round's model.
        # It may be absent if the marginal was only introduced in a later round.
        _out_param_idx = find_out_param_idx(model, in_param_idx)
        if _out_param_idx is None:
            print(f"  [1-D {param_key}] Round {i + 1}: marginal not in model — skipping.")
            del model
            for lvl in hpd_levels:
                hpd_lows[lvl].append(np.nan)
                hpd_highs[lvl].append(np.nan)
            continue

        # Use higher resolution for the final round if requested
        _ngrid = ngrid_points_1d if (i == n_rounds - 1) else ngrid_points
        if in_param_idx == 10:
            low = -2.5-3e-5
            high = -2.5+3e-5
        else:
            low = None
            high = None

        logratios, inj_params, grid = get_logratios_grid(
            dataloader,
            model,
            ngrid_points=_ngrid,
            in_param_idx=in_param_idx,
            out_param_idx=_out_param_idx,
            low=low,
            high=high
        )
        del model
        ratios = np.exp(logratios[0])
        dp = grid[1, 0] - grid[0, 0]
        norm1d = ratios / np.sum(ratios * dp)
        grid_1d = grid[:, 0]

        # Store injection value from the first available round (same obs every round)
        if final_inj is None:
            final_inj = float(inj_params[0])

        # HPD intervals (use the ngrid_points resolution for consistency
        # when this is not the final round; for the final round the higher
        # resolution only helps)
        for lvl in hpd_levels:
            lo, hi = _hpd_interval_1d(norm1d, grid_1d, lvl)
            hpd_lows[lvl].append(lo)
            hpd_highs[lvl].append(hi)

        # Keep final available round's data for the density panel
        final_norm1d = norm1d
        final_grid_1d = grid_1d

    # ---- Create figure with two panels: HPD evolution + final-round density ----
    fig, (ax_hpd, ax_dens) = plt.subplots(
        1, 2, figsize=(figsize[0] * 2 + 1, figsize[1]),
        gridspec_kw={"width_ratios": [1, 1], "wspace": 0.35},
    )

    # --- Left panel: HPD band evolution (unchanged logic) ---
    # Filled HPD bands (widest first so narrower ones paint on top)
    for lvl, color in zip(reversed(hpd_levels), reversed(band_colors)):
        lows_arr  = np.array(hpd_lows[lvl])
        highs_arr = np.array(hpd_highs[lvl])
        pct = int(round(lvl * 100))
        ax_hpd.fill_between(
            round_indices, lows_arr, highs_arr,
            color=color, alpha=0.85,
            label=f"{pct} % HPD",
        )
        # Draw the bounding lines explicitly
        ax_hpd.plot(round_indices, lows_arr,  color=color, linewidth=1.5)
        ax_hpd.plot(round_indices, highs_arr, color=color, linewidth=1.5)

    # Prior window as dashed reference
    ax_hpd.plot(round_indices, prior_lows,  color="black",
                linestyle="--", linewidth=1.2, label="prior window")
    ax_hpd.plot(round_indices, prior_highs, color="black",
                linestyle="--", linewidth=1.2)

    # True parameter value as a horizontal reference line
    if final_inj is not None:
        ax_hpd.axhline(y=final_inj, color="r", linestyle="--",
                       linewidth=1.5, label="True value")

    ax_hpd.set_xlabel("Round")
    ax_hpd.set_ylabel(param_key)
    ax_hpd.set_title(f"1-D posterior HPD evolution: {param_key}")
    ax_hpd.set_xticks(round_indices)
    ax_hpd.legend(loc="best", fontsize=9)
    ax_hpd.grid(True, linestyle="--", alpha=0.4)

    # --- Right panel: final-round 1-D posterior density ---
    if final_norm1d is None:
        ax_dens.text(0.5, 0.5, "marginal not trained\nin any round",
                     ha="center", va="center", transform=ax_dens.transAxes,
                     fontsize=11, color="gray")
        axes_dict = {"hpd": ax_hpd, "density": ax_dens}
        return fig, axes_dict

    ax_dens.plot(final_grid_1d, final_norm1d, "b-", linewidth=2, label="NRE")
    ax_dens.fill_between(final_grid_1d, 0, final_norm1d, alpha=0.3, color="b")
    ax_dens.axvline(x=final_inj, color="r", linestyle="--", linewidth=2, label="Injection")

    # MCMC overlay
    if mcmc_samples_dir is not None:
        flat_samples, mcmc_param_names = load_mcmc_samples(mcmc_samples_dir)
        if param_key in mcmc_param_names:
            idx_mc = mcmc_param_names.index(param_key)
            mcmc_samp = flat_samples[:, idx_mc]
            kde = gaussian_kde(mcmc_samp)
            mcmc_marginal = kde(final_grid_1d)
            dp = final_grid_1d[1] - final_grid_1d[0]
            mcmc_marginal = mcmc_marginal / np.sum(mcmc_marginal * dp)
            ax_dens.plot(final_grid_1d, mcmc_marginal, color="orange",
                         linewidth=2, label="MCMC")
            ax_dens.fill_between(final_grid_1d, 0, mcmc_marginal,
                                 alpha=0.3, color="orange")

    ax_dens.set_xlabel(param_key)
    ax_dens.set_ylabel("Posterior Density")
    ax_dens.set_title(f"Final-round 1-D posterior: {param_key}")
    ax_dens.legend(loc="best", fontsize=9)
    ax_dens.grid(True, linestyle="--", alpha=0.4)

    axes_dict = {"hpd": ax_hpd, "density": ax_dens}
    return fig, axes_dict


# ---------------------------------------------------------------------------
# MCMC overlay helpers
# ---------------------------------------------------------------------------

def load_mcmc_samples(samples_dir: str):
    """Load flat MCMC samples and their parameter names.

    Expects:
      - ``flat_samples.npy`` : array of shape (N_samples, N_params)
      - ``varying_params.txt`` : one parameter name per line

    Returns ``(flat_samples, param_names)``.
    """
    flat_samples = np.load(os.path.join(samples_dir, "flat_samples.npy"))
    with open(os.path.join(samples_dir, "varying_params.txt"), "r") as f:
        param_names = [line.strip() for line in f if line.strip()]
    return flat_samples, param_names


def overlay_mcmc_contours(
    ax,
    flat_samples: np.ndarray,
    mcmc_param_names: list,
    p0_key: str,
    p1_key: str,
    ngrid_points: int = 100,
    color: str = "cyan",
    linewidths: float = 1.5,
    linestyles: str = "-",
    alpha: float = 0.9,
    label: str = "MCMC",
    is_sky: bool = False,
    use_mollweide: bool = True,
):
    """Overlay Gaussian-KDE credibility contours from MCMC flat samples on *ax*.

    The two parameter columns are selected by name from *mcmc_param_names*.
    Contour levels match the standard 1-/2-/3-/4-sigma credibility targets
    (68.27 %, 95.45 %, 99.73 %, 99.99 %) computed via ``contour_levels``.

    When *is_sky* is ``True`` the parameter pair is assumed to be
    ``(lambda, beta)`` and the samples are reprojected to
    (lon, lat) coordinates before KDE evaluation.  If *use_mollweide* is
    ``True`` the grid spans the full sky in radians; otherwise it spans
    the current axis limits in degrees.
    """
    idx0 = mcmc_param_names.index(p0_key)
    idx1 = mcmc_param_names.index(p1_key)
    samples_x = flat_samples[:, idx0]
    samples_y = flat_samples[:, idx1]

    if is_sky:
        # Transform MCMC samples to (lon, lat)
        lam_col = samples_x if p0_key == "lambda" else samples_y
        bet_col = samples_y if p1_key == "beta" else samples_x
        lon_rad = lam_col - np.pi
        lat_rad = np.arcsin(np.clip(bet_col, -1.0, 1.0))

        if use_mollweide:
            # Full-sky in radians
            samples_x = lon_rad
            samples_y = lat_rad
            gx_1d = np.linspace(-np.pi, np.pi, ngrid_points)
            gy_1d = np.linspace(-np.pi / 2.0, np.pi / 2.0, ngrid_points)
        else:
            # Restricted sky in degrees — grid matches current axes
            samples_x = np.degrees(lon_rad)
            samples_y = np.degrees(lat_rad)
            xl = ax.get_xlim()
            yl = ax.get_ylim()
            # Clip samples to the axis range so outliers don't inflate the
            # KDE bandwidth and push all density off the evaluation grid.
            mask = (
                (samples_x >= xl[0]) & (samples_x <= xl[1])
                & (samples_y >= yl[0]) & (samples_y <= yl[1])
            )
            samples_x = samples_x[mask]
            samples_y = samples_y[mask]
            print(f"    MCMC samples clipped away: {np.sum(~mask)} / {len(mask)}")
            if len(samples_x) < 10:
                print("    WARNING: fewer than 10 MCMC samples inside axes "
                      "— skipping MCMC contour overlay.")
                return None
            gx_1d = np.linspace(xl[0], xl[1], ngrid_points)
            gy_1d = np.linspace(yl[0], yl[1], ngrid_points)
    else:
        # Evaluate on a regular grid matching the current axis limits
        xl = ax.get_xlim()
        yl = ax.get_ylim()
        gx_1d = np.linspace(xl[0], xl[1], ngrid_points)
        gy_1d = np.linspace(yl[0], yl[1], ngrid_points)

    # Gaussian KDE
    kde = gaussian_kde(np.vstack([samples_x, samples_y]))

    gx, gy = np.meshgrid(gx_1d, gy_1d)
    kde_vals = kde(np.vstack([gx.ravel(), gy.ravel()])).reshape(ngrid_points, ngrid_points)

    # Normalise (matches the convention used by contour_levels)
    dp0 = gx_1d[1] - gx_1d[0]
    dp1 = gy_1d[1] - gy_1d[0]
    kde_sum = np.sum(kde_vals)
    if kde_sum == 0:
        print("    WARNING: KDE evaluated to zero on grid — skipping MCMC overlay.")
        return None
    kde_norm = kde_vals / (kde_sum * dp0 * dp1)

    # Compute credibility-level thresholds on the KDE density
    levels, level_labels = contour_levels(kde_norm, targets=[0.6827, 0.9545, 0.9973])
    cs = ax.contour(
        gx,
        gy,
        kde_norm,
        levels=levels,
        colors=color,
        linewidths=linewidths,
        linestyles=linestyles,
        alpha=alpha,
        zorder=5,
    )
    fmt = {lev: f"{p:.3f}" for lev, p in zip(levels, level_labels)}
    ax.clabel(cs, fmt=fmt, fontsize=7)

    # Add a legend entry for the MCMC contours
    proxy = Line2D([0], [0], color=color, linewidth=linewidths,
                   linestyle=linestyles, label=label)
    ax.legend(handles=[proxy], loc="upper right", fontsize=8)

    return cs


# ---------------------------------------------------------------------------
# 1-D marginal insets for the final-round 2-D panel
# ---------------------------------------------------------------------------

def _add_1d_marginal_insets(
    ax,
    norm2d,
    gx,
    gy,
    inj_params,
    p0_key: str,
    p1_key: str,
    mcmc_samples_dir: str = None,
    _sky_deg: bool = False,
    _p0_key_raw: str = None,
    _p1_key_raw: str = None,
):
    """Add top and right inset axes to *ax* showing 1-D marginals.

    The 1-D marginals are obtained by numerically integrating the 2-D
    posterior over the complementary axis, matching the convention in
    ``plot_corner_with_marginals`` from the interactive notebook:

    * ``marginal_0 = sum(norm2d * dp1, axis=0)``  →  p(param_0)
    * ``marginal_1 = sum(norm2d * dp0, axis=1)``  →  p(param_1)

    When *mcmc_samples_dir* is provided, a Gaussian-KDE marginal from the
    MCMC flat samples is overlaid in orange.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The 2-D contour axes (last-round panel).
    norm2d : ndarray, shape (ngrid, ngrid)
        Normalised 2-D posterior (single observation, no batch dim).
    gx, gy : ndarray, shape (ngrid, ngrid)
        Meshgrids (xy indexing).  May be in degree coordinates when
        ``_sky_deg`` is True.
    inj_params : 1-D array of length 2
        True parameter values ``[p0_true, p1_true]``.
    p0_key, p1_key : str
        Parameter names for labelling.
    mcmc_samples_dir : str, optional
        Path to MCMC samples directory.
    _sky_deg : bool
        If True, grids are in (lon, lat) degrees; MCMC samples must be
        converted from native (lambda, beta) to degrees before KDE.
    _p0_key_raw, _p1_key_raw : str, optional
        Original parameter keys ("lambda"/"beta") used to look up MCMC
        columns when ``_sky_deg`` is True.

    Returns
    -------
    ax_top, ax_right : Axes
        The two inset axes.
    """
    # Grid vectors and spacings
    grid_0 = gx[0, :]   # param_0 values (columns)
    grid_1 = gy[:, 0]   # param_1 values (rows)
    dp0 = grid_0[1] - grid_0[0]
    dp1 = grid_1[1] - grid_1[0]

    if _sky_deg:
        # norm2d is a density in native (λ, β) space but the grids are in
        # degrees.  Apply the inverse Jacobian so that the marginals
        # integrate correctly in degree-space:
        #   p(lon°, lat°) = p(λ, β) · (π/180)² · cos(lat)
        # where cos(lat) = sqrt(1 − β²).  grid_1 is lat_deg (rows).
        lat_rad = np.radians(grid_1)                   # shape (ngrid,)
        cos_lat = np.cos(lat_rad)                       # = sqrt(1 - β²)
        jac_inv = (np.pi / 180.0) ** 2 * cos_lat        # shape (ngrid,)
        norm2d = norm2d * jac_inv[:, np.newaxis]         # broadcast over columns

    # Marginalise: norm2d[i, j] → axis 0 = param_1, axis 1 = param_0
    marginal_0 = np.sum(norm2d * dp1, axis=0)  # p(param_0), shape (ngrid,)
    marginal_1 = np.sum(norm2d * dp0, axis=1)  # p(param_1), shape (ngrid,)

    # --- Top inset: marginal for param_0 (x-axis of the 2-D plot) ---
    ax_top = ax.inset_axes([0.0, 1.02, 1.0, 0.25])  # [x0, y0, width, height] in axes fraction
    ax_top.plot(grid_0, marginal_0, "b-", linewidth=1.5, label="NRE")
    ax_top.fill_between(grid_0, 0, marginal_0, alpha=0.3, color="b")
    ax_top.axvline(x=float(inj_params[0]), color="r", linestyle="--", linewidth=1.5)
    ax_top.set_xlim(ax.get_xlim())
    ax_top.tick_params(labelbottom=False, labelleft=False, left=False, bottom=True)
    ax_top.set_ylabel("p", fontsize=8)
    ax_top.grid(alpha=0.3)

    # --- Right inset: marginal for param_1 (y-axis of the 2-D plot) ---
    ax_right = ax.inset_axes([1.02, 0.0, 0.25, 1.0])
    ax_right.plot(marginal_1, grid_1, "b-", linewidth=1.5, label="NRE")
    ax_right.fill_betweenx(grid_1, 0, marginal_1, alpha=0.3, color="b")
    ax_right.axhline(y=float(inj_params[1]), color="r", linestyle="--", linewidth=1.5)
    ax_right.set_ylim(ax.get_ylim())
    ax_right.tick_params(labelbottom=False, labelleft=False, left=False, bottom=True)
    ax_right.set_xlabel("p", fontsize=8)
    ax_right.grid(alpha=0.3)

    # --- MCMC KDE overlay ---
    if mcmc_samples_dir is not None:
        flat_samples, mcmc_param_names = load_mcmc_samples(mcmc_samples_dir)

        if _sky_deg:
            # Axes are in (lon_deg, lat_deg); MCMC columns are (lambda, beta)
            _lk = _p0_key_raw if _p0_key_raw == "lambda" else _p1_key_raw
            _bk = _p0_key_raw if _p0_key_raw == "beta"   else _p1_key_raw
            _has_lon = _lk in mcmc_param_names
            _has_lat = _bk in mcmc_param_names
            if _has_lon:
                lam_samp = flat_samples[:, mcmc_param_names.index(_lk)]
                lon_deg_samp = np.degrees(lam_samp - np.pi)
                kde_0 = gaussian_kde(lon_deg_samp)
                mcmc_m0 = kde_0(grid_0)
                mcmc_m0 = mcmc_m0 / np.sum(mcmc_m0 * dp0)
                ax_top.plot(grid_0, mcmc_m0, color="orange", linewidth=1.5, label="MCMC")
                ax_top.fill_between(grid_0, 0, mcmc_m0, alpha=0.3, color="orange")
            if _has_lat:
                bet_samp = flat_samples[:, mcmc_param_names.index(_bk)]
                lat_deg_samp = np.degrees(np.arcsin(np.clip(bet_samp, -1, 1)))
                kde_1 = gaussian_kde(lat_deg_samp)
                mcmc_m1 = kde_1(grid_1)
                mcmc_m1 = mcmc_m1 / np.sum(mcmc_m1 * dp1)
                ax_right.plot(mcmc_m1, grid_1, color="orange", linewidth=1.5, label="MCMC")
                ax_right.fill_betweenx(grid_1, 0, mcmc_m1, alpha=0.3, color="orange")
        else:
            _has_p0 = p0_key in mcmc_param_names
            _has_p1 = p1_key in mcmc_param_names

            if _has_p0:
                mcmc_samp_0 = flat_samples[:, mcmc_param_names.index(p0_key)]
                kde_0 = gaussian_kde(mcmc_samp_0)
                mcmc_m0 = kde_0(grid_0)
                mcmc_m0 = mcmc_m0 / np.sum(mcmc_m0 * dp0)
                ax_top.plot(grid_0, mcmc_m0, color="orange", linewidth=1.5, label="MCMC")
                ax_top.fill_between(grid_0, 0, mcmc_m0, alpha=0.3, color="orange")

            if _has_p1:
                mcmc_samp_1 = flat_samples[:, mcmc_param_names.index(p1_key)]
                kde_1 = gaussian_kde(mcmc_samp_1)
                mcmc_m1 = kde_1(grid_1)
                mcmc_m1 = mcmc_m1 / np.sum(mcmc_m1 * dp1)
                ax_right.plot(mcmc_m1, grid_1, color="orange", linewidth=1.5, label="MCMC")
                ax_right.fill_betweenx(grid_1, 0, mcmc_m1, alpha=0.3, color="orange")

    # Compact legend on the top inset only
    ax_top.legend(loc="upper right", fontsize=7, framealpha=0.8)

    return ax_top, ax_right


# ---------------------------------------------------------------------------
# Sky-localisation helpers
# ---------------------------------------------------------------------------

# Full-sky bounds in the model's native parametrisation
_FULL_SKY_LAMBDA = (0.0, 2.0 * np.pi)   # ecliptic longitude
_FULL_SKY_BETA   = (-1.0, 1.0)          # sin(ecliptic latitude)
_SR_TO_SQDEG     = (180.0 / np.pi) ** 2  # steradians → square degrees


def _is_full_sky(bounds_lambda: tuple, bounds_beta: tuple, rtol: float = 0.02) -> bool:
    """Return ``True`` if the prior window covers (nearly) the full sky.

    Compares the lambda and beta bounds against the canonical full-sky
    ranges with relative tolerance *rtol*.
    """
    lam_range = bounds_lambda[1] - bounds_lambda[0]
    bet_range = bounds_beta[1] - bounds_beta[0]
    full_lam  = _FULL_SKY_LAMBDA[1] - _FULL_SKY_LAMBDA[0]
    full_bet  = _FULL_SKY_BETA[1]  - _FULL_SKY_BETA[0]
    return (abs(lam_range - full_lam) / full_lam < rtol and
            abs(bet_range - full_bet) / full_bet < rtol)


def _compute_sky_area(density_2d: np.ndarray, threshold: float,
                      dp_lambda: float, dp_beta: float) -> float:
    """Compute the sky area enclosed by the iso-density contour at *threshold*.

    The grid lives in (lambda, beta) space where beta = sin(ecliptic
    latitude).  The solid-angle element is
    ``dΩ = dλ × d(sin lat) = dp_lambda × dp_beta``, so the area in
    steradians is simply the number of cells above the threshold
    multiplied by the cell area.

    Returns the area in **square degrees**.
    """
    n_cells = np.sum(density_2d >= threshold)
    area_sr = float(n_cells) * dp_lambda * dp_beta
    return area_sr * _SR_TO_SQDEG


def _to_mollweide_coords(gx: np.ndarray, gy: np.ndarray, p0_key: str, p1_key: str):
    """Convert a model-space meshgrid to Mollweide (lon, lat) coordinates.

    For the sky-localisation marginal (lambda, beta):

    * ``lambda`` ∈ [0, 2π] is the ecliptic longitude  → longitude in [-π, π]
    * ``beta`` = sin(ecliptic latitude)               → latitude  in [-π/2, π/2]

    Matplotlib's Mollweide projection expects longitude in [-π, π] and
    latitude in [-π/2, π/2], both in **radians**.

    Parameters
    ----------
    gx, gy : ndarray of shape (ngrid, ngrid)
        Meshgrids in model parameter space (the quantities returned by
        ``get_logratios_grid_2d``).  ``gx`` corresponds to ``p0_key`` and
        ``gy`` to ``p1_key``.
    p0_key, p1_key : str
        Parameter names; must be ``"lambda"`` and ``"beta"`` in some order.

    Returns
    -------
    lon, lat : ndarray of shape (ngrid, ngrid)
        Meshgrids in Mollweide (longitude, latitude) coordinates.
    """
    if p0_key == "lambda":
        # gx is lambda [0, 2π], gy is beta = sin(lat)
        lon = gx - np.pi
        lat = np.arcsin(np.clip(gy, -1.0, 1.0))
    else:
        # p0_key == "beta", p1_key == "lambda"
        # gx is beta = sin(lat), gy is lambda [0, 2π]
        lon = gy - np.pi
        lat = np.arcsin(np.clip(gx, -1.0, 1.0))
    return lon, lat


def _to_lonlat_deg(gx: np.ndarray, gy: np.ndarray, p0_key: str, p1_key: str):
    """Convert a model-space meshgrid to (longitude, latitude) in **degrees**.

    Uses the same mapping as ``_to_mollweide_coords`` but outputs degrees
    instead of radians, suitable for a Cartesian axes.
    """
    lon_rad, lat_rad = _to_mollweide_coords(gx, gy, p0_key, p1_key)
    return np.degrees(lon_rad), np.degrees(lat_rad)


# ---------------------------------------------------------------------------
# Main plotting function
# ---------------------------------------------------------------------------

def plot_truncation_rounds(
    round_dirs: list,
    dataloader,
    marginal_pair_idx: int = 0,
    ngrid_points: int = 100,
    figsize_per_panel: tuple = (5, 5),
    connect_boxes: bool = True,
    rect_color: str = "red",
    rect_lw: float = 2.0,
    wspace: float = 0.35,
    mcmc_samples_dir: str = None,
    in_param_idx_override=None,
):
    """Plot posterior-contour evolution across successive truncation rounds.

    The final-round panel additionally includes inset axes showing
    1-D marginal distributions (top: param_0, right: param_1) obtained
    by integrating the 2-D posterior over the complementary axis.  When
    *mcmc_samples_dir* is given, MCMC KDE marginals are overlaid.

    Parameters
    ----------
    round_dirs : list of str
        Paths to each round's log directory.  Each must contain a
        ``checkpoints/`` sub-directory and a
        ``simulation_round_<n>.yaml`` file (1-indexed).
    dataloader : DataLoader
        Data loader providing the observation(s) to condition on.
    marginal_pair_idx : int
        Index into the list of available 2-D marginals (derived from the
        first round's model).  Change this value to plot a different
        parameter pair.
    ngrid_points : int
        Grid resolution for posterior evaluation.
    figsize_per_panel : tuple
        ``(width, height)`` per subplot panel in inches.
    connect_boxes : bool
        If ``True``, draw connecting lines from each next-round prior
        rectangle to the next panel's axis frame.
    rect_color : str
        Colour used for the next-round bounding rectangles and connecting lines.
    rect_lw : float
        Line width of the next-round bounding rectangles.
    wspace : float
        Horizontal spacing between subplots, leaving room for the
        connecting lines.
    mcmc_samples_dir : str, optional
        Directory containing ``flat_samples.npy`` and
        ``varying_params.txt`` from a previous MCMC run.  When provided,
        Gaussian-KDE credibility contours from those samples are
        superimposed on the **last** round's panel, and 1-D MCMC
        marginals are overlaid on the inset axes.

    Returns
    -------
    fig, axes, sky_areas
        ``sky_areas`` is a dict mapping round index → ``{"nre": area, "mcmc": area}``
        in square degrees (only populated for sky-localisation marginals).
    """
    n_rounds = len(round_dirs)

    # Load prior boxes for every round (yaml files are 1-indexed)
    prior_boxes = [load_prior_box(d, i + 1) for i, d in enumerate(round_dirs)]

    if in_param_idx_override is not None:
        # Caller supplied in_param_idx directly; derive label and skip discovery.
        in_param_idx_fixed = in_param_idx_override
        p0 = _ORDERED_PRIOR_KEYS[in_param_idx_fixed[0]]
        p1 = _ORDERED_PRIOR_KEYS[in_param_idx_fixed[1]]
        marginal_label = f"{p0} vs {p1}"
    else:
        # Discover 2-D marginals across ALL rounds (union), so marginals that
        # appear only in later rounds are included.
        marginals_2d_seen = {}  # in_idx -> (label, out_idx) from first occurrence
        for _rd in round_dirs:
            _m = load_model(os.path.join(_rd, "checkpoints"))
            for label, ndim, in_idx, out_idx in get_all_marginals(_m):
                if ndim == 2 and in_idx not in marginals_2d_seen:
                    marginals_2d_seen[in_idx] = (label, out_idx)
            del _m
        marginals_2d = [(lbl, in_idx, oidx)
                        for in_idx, (lbl, oidx) in marginals_2d_seen.items()]
        if not marginals_2d:
            raise ValueError("No 2-D marginals found in any model output.")
        print("    Available 2-D marginals (across all rounds):")
        for j, (lbl, _, _) in enumerate(marginals_2d):
            print(f"      [{j}] {lbl}")
        if marginal_pair_idx >= len(marginals_2d):
            raise IndexError(
                f"marginal_pair_idx={marginal_pair_idx} out of range "
                f"({len(marginals_2d)} 2-D marginals available)."
            )
        marginal_label, in_param_idx_fixed, _ = marginals_2d[marginal_pair_idx]

    # Detect sky-localisation marginal (lambda vs beta)
    p0_key_fixed = _ORDERED_PRIOR_KEYS[in_param_idx_fixed[0]]
    p1_key_fixed = _ORDERED_PRIOR_KEYS[in_param_idx_fixed[1]]
    is_sky = {p0_key_fixed, p1_key_fixed} == {"lambda", "beta"}

    # For sky marginals decide Mollweide vs Cartesian *per round*.
    # Determine the projection for each round once; the first round that
    # has a restricted prior triggers Cartesian for that round onward.
    if is_sky:
        sky_use_mollweide = []
        for i_r in range(n_rounds):
            box_r = prior_boxes[i_r]
            lam_key = "lambda"
            bet_key = "beta"
            sky_use_mollweide.append(
                _is_full_sky(box_r[lam_key], box_r[bet_key])
            )
        # Create figure with per-panel projection
        fig = plt.figure(figsize=(figsize_per_panel[0] * n_rounds, figsize_per_panel[1]))
        axes = []
        for k in range(n_rounds):
            proj = "mollweide" if sky_use_mollweide[k] else None
            axes.append(fig.add_subplot(1, n_rounds, k + 1, projection=proj))
    else:
        sky_use_mollweide = [False] * n_rounds   # unused, keeps indexing safe
        fig, axes = plt.subplots(
            1,
            n_rounds,
            figsize=(figsize_per_panel[0] * n_rounds, figsize_per_panel[1]),
            sharex=False,
            sharey=False,
        )
        if n_rounds == 1:
            axes = [axes]

    # Collect sky-area measurements (populated in the sky branch)
    sky_areas = {}  # round_idx → {"nre": area_sqdeg, "mcmc": area_sqdeg}

    # Collect (rect_patch, ax) for each round that has a following round
    rectangles = []

    for i, (round_dir, ax) in enumerate(zip(round_dirs, axes)):
        print(f"\n--- Round {i + 1} ---")
        print(f"    Loading model from: {round_dir}")
        model = load_model(os.path.join(round_dir, "checkpoints"))

        in_param_idx  = in_param_idx_fixed
        # Look up the correct output column for this marginal in this round's model.
        out_param_idx = find_out_param_idx(model, in_param_idx)
        if out_param_idx is None:
            print(f"    Marginal '{marginal_label}' not in round {i + 1} model — skipping panel.")
            ax.text(0.5, 0.5, "not trained\nthis round",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=11, color="gray")
            ax.set_title(f"Round {i + 1}")
            rectangles.append(None)
            del model
            continue

        p0_key = _ORDERED_PRIOR_KEYS[in_param_idx[0]]
        p1_key = _ORDERED_PRIOR_KEYS[in_param_idx[1]]

        box = prior_boxes[i]
        bounds_0 = box[p0_key]
        bounds_1 = box[p1_key]

        print(f"    Marginal : {marginal_label}")
        print(f"    Bounds   : {p0_key}={bounds_0}, {p1_key}={bounds_1}")

        # Evaluate normalised posterior on the grid
        norm2d, inj_params, gx, gy = compute_normalised_posterior(
            dataloader,
            model,
            in_param_idx,
            out_param_idx,
            bounds_0=bounds_0,
            bounds_1=bounds_1,
            ngrid_points=ngrid_points,
        )
        levels, level_labels = contour_levels(norm2d, targets=[0.6827, 0.9545, 0.9973])

        if is_sky:
            # ------------------------------------------------------------------
            # Sky path: Mollweide for full-sky priors, Cartesian for
            # restricted priors.  Both use (lon, lat) coordinates.
            # ------------------------------------------------------------------
            use_mollweide = sky_use_mollweide[i]

            # Injection point in (lon, lat)
            lam_inj = float(inj_params[0, 0]) if p0_key == "lambda" else float(inj_params[0, 1])
            bet_inj = float(inj_params[0, 1]) if p1_key == "beta"   else float(inj_params[0, 0])
            lon_inj_rad = lam_inj - np.pi
            lat_inj_rad = np.arcsin(np.clip(bet_inj, -1.0, 1.0))

            if use_mollweide:
                # --- Full-sky Mollweide (radians) ---
                lon_grid, lat_grid = _to_mollweide_coords(gx, gy, p0_key, p1_key)
                cs = ax.contour(
                    lon_grid, lat_grid, norm2d[0], levels=levels,
                    colors=["blue"] * len(levels), linewidths=1.5, zorder=4,
                )
                fmt = {lev: f"{p:.3f}" for lev, p in zip(levels, level_labels)}
                ax.clabel(cs, fmt=fmt, fontsize=8)
                ax.plot(lon_inj_rad, lat_inj_rad, "r+", markersize=10,
                        markeredgewidth=2, zorder=5)
            else:
                # --- Restricted-sky Cartesian (degrees) ---
                lon_deg, lat_deg = _to_lonlat_deg(gx, gy, p0_key, p1_key)
                lon_inj_deg = np.degrees(lon_inj_rad)
                lat_inj_deg = np.degrees(lat_inj_rad)

                cs = ax.contour(
                    lon_deg, lat_deg, norm2d[0], levels=levels,
                    colors=["blue"] * len(levels), linewidths=1.5, zorder=4,
                )
                fmt = {lev: f"{p:.3f}" for lev, p in zip(levels, level_labels)}
                ax.clabel(cs, fmt=fmt, fontsize=8)
                ax.plot(lon_inj_deg, lat_inj_deg, "r+", markersize=10,
                        markeredgewidth=2, zorder=5)
                ax.set_xlabel("Longitude [deg]")
                ax.set_ylabel("Latitude [deg]")
                ax.set_xlim(lon_deg.min(), lon_deg.max())
                ax.set_ylim(lat_deg.min(), lat_deg.max())
                ax.grid(True, linestyle="--", alpha=0.4)
                ax.set_aspect("equal", adjustable="box")

            # --- NRE sky area (widest contour: levels[0] = lowest threshold) ---
            dp_lam = gx[0, 1] - gx[0, 0]  if p0_key == "lambda" else gy[0, 1] - gy[0, 0]
            dp_bet = gy[1, 0] - gy[0, 0]  if p1_key == "beta"   else gx[1, 0] - gx[0, 0]
            nre_area = _compute_sky_area(norm2d[0], levels[0], dp_lam, dp_bet)
            sky_areas[i] = {"nre": nre_area}
            print(f"    NRE sky area ({level_labels[0]*100:.1f}% CR): "
                  f"{nre_area:.2f} sq deg")

            ax.set_title(f"Round {i + 1}  ({nre_area:.1f} sq deg, {level_labels[0]*100:.0f}% CR)")

            # --- Overlay MCMC contours + area on last panel ---
            if mcmc_samples_dir is not None and i == n_rounds - 1:
                print("    Overlaying MCMC contours...")
                flat_samples, mcmc_param_names = load_mcmc_samples(mcmc_samples_dir)
                mcmc_cs = overlay_mcmc_contours(
                    ax, flat_samples, mcmc_param_names, p0_key, p1_key,
                    is_sky=True, use_mollweide=use_mollweide,
                )
                # Compute MCMC sky area from the KDE on the *model* grid
                idx0 = mcmc_param_names.index(p0_key)
                idx1 = mcmc_param_names.index(p1_key)
                _lam_samp = flat_samples[:, idx0] if p0_key == "lambda" else flat_samples[:, idx1]
                _bet_samp = flat_samples[:, idx1] if p1_key == "beta"   else flat_samples[:, idx0]
                # Evaluate on the same (lambda, beta) grid as NRE
                if p0_key == "lambda":
                    lam_1d, bet_1d = gx[0, :], gy[:, 0]
                else:
                    lam_1d, bet_1d = gy[0, :], gx[:, 0]
                # Clip MCMC samples to the grid range to avoid KDE bandwidth
                # inflation from outliers that fall far outside the prior window.
                lam_lo, lam_hi = float(lam_1d.min()), float(lam_1d.max())
                bet_lo, bet_hi = float(bet_1d.min()), float(bet_1d.max())
                mask = (
                    (_lam_samp >= lam_lo) & (_lam_samp <= lam_hi)
                    & (_bet_samp >= bet_lo) & (_bet_samp <= bet_hi)
                )
                _lam_clip = _lam_samp[mask]
                _bet_clip = _bet_samp[mask]
                if len(_lam_clip) < 10:
                    print("    WARNING: fewer than 10 MCMC samples fall inside "
                          "the grid — skipping MCMC sky area computation.")
                else:
                    kde_sky = gaussian_kde(np.vstack([_lam_clip, _bet_clip]))
                    LG, BG = np.meshgrid(lam_1d, bet_1d)
                    kde_vals = kde_sky(np.vstack([LG.ravel(), BG.ravel()])).reshape(LG.shape)
                    kde_sum = np.sum(kde_vals)
                    if kde_sum > 0:
                        kde_norm = kde_vals / (kde_sum * dp_lam * dp_bet)
                    else:
                        print("    WARNING: KDE evaluated to zero on the grid "
                              "— skipping MCMC sky area.")
                        kde_norm = None
                    if kde_norm is not None:
                        mcmc_levels, mcmc_labels = contour_levels(
                            kde_norm, targets=[0.6827, 0.9545, 0.9973]
                        )
                        mcmc_area = _compute_sky_area(kde_norm, mcmc_levels[0],
                                                      dp_lam, dp_bet)
                        sky_areas[i]["mcmc"] = mcmc_area
                        print(f"    MCMC sky area ({mcmc_labels[0]*100:.1f}% CR): "
                              f"{mcmc_area:.2f} sq deg")

            # --- 1-D marginal insets on the final-round panel ---
            if i == n_rounds - 1 and not use_mollweide:
                print("    Adding 1-D marginal insets (sky, Cartesian)...")
                # Grids need to be in the same coordinates as the axes (degrees)
                lon_deg_grid, lat_deg_grid = _to_lonlat_deg(gx, gy, p0_key, p1_key)
                _add_1d_marginal_insets(
                    ax, norm2d[0], lon_deg_grid, lat_deg_grid,
                    np.array([np.degrees(lon_inj_rad), np.degrees(lat_inj_rad)]),
                    "lon [deg]", "lat [deg]",
                    mcmc_samples_dir=mcmc_samples_dir,
                    _sky_deg=True, _p0_key_raw=p0_key, _p1_key_raw=p1_key,
                )

            # Prior-window rectangle for restricted-sky Cartesian panels
            if not use_mollweide and i + 1 < n_rounds:
                nb = prior_boxes[i + 1]
                # Convert next-round bounds to (lon, lat) degrees
                nb_lam = nb["lambda"]
                nb_bet = nb["beta"]
                nb_lon0 = np.degrees(nb_lam[0] - np.pi)
                nb_lon1 = np.degrees(nb_lam[1] - np.pi)
                nb_lat0 = np.degrees(np.arcsin(np.clip(nb_bet[0], -1, 1)))
                nb_lat1 = np.degrees(np.arcsin(np.clip(nb_bet[1], -1, 1)))
                rect = Rectangle(
                    (nb_lon0, nb_lat0),
                    nb_lon1 - nb_lon0,
                    nb_lat1 - nb_lat0,
                    linewidth=rect_lw,
                    edgecolor=rect_color,
                    facecolor="none",
                    linestyle="--",
                    zorder=10,
                )
                ax.add_patch(rect)
                rectangles.append((rect, ax))
            else:
                rectangles.append(None)

        else:
            # ------------------------------------------------------------------
            # Standard path: posterior_contours_2d on a regular Cartesian axes.
            # ------------------------------------------------------------------
            posterior_contours_2d(
                gx,
                gy,
                norm2d[0],
                inj_params[0],
                ax_buffer=ax,
                parameter_names=[p0_key, p1_key],
                title=f"Round {i + 1}",
                levels=levels,
                levels_labels=level_labels,
                do_plot=True,
                show_colormap=False,
            )

            # Force axis limits to this round's prior box
            ax.set_xlim(bounds_0)
            ax.set_ylim(bounds_1)

            # Overlay MCMC KDE contours on the last panel
            if mcmc_samples_dir is not None and i == n_rounds - 1:
                print("    Overlaying MCMC contours...")
                flat_samples, mcmc_param_names = load_mcmc_samples(mcmc_samples_dir)
                overlay_mcmc_contours(
                    ax,
                    flat_samples,
                    mcmc_param_names,
                    p0_key,
                    p1_key,
                )

            # Add 1-D marginal insets on the final-round panel
            if i == n_rounds - 1:
                print("    Adding 1-D marginal insets...")
                _add_1d_marginal_insets(
                    ax,
                    norm2d[0],
                    gx,
                    gy,
                    inj_params[0],
                    p0_key,
                    p1_key,
                    mcmc_samples_dir=mcmc_samples_dir,
                )

            # Draw a dashed rectangle marking the *next* round's prior bounds
            if i + 1 < n_rounds:
                nb = prior_boxes[i + 1]
                nb_x0, nb_x1 = nb[p0_key]
                nb_y0, nb_y1 = nb[p1_key]
                rect = Rectangle(
                    (nb_x0, nb_y0),
                    nb_x1 - nb_x0,
                    nb_y1 - nb_y0,
                    linewidth=rect_lw,
                    edgecolor=rect_color,
                    facecolor="none",
                    linestyle="--",
                    zorder=10,
                )
                ax.add_patch(rect)
                rectangles.append((rect, ax))
            else:
                rectangles.append(None)

        del model  # release GPU memory between rounds

    # Finalise layout before computing figure-space transforms
    plt.subplots_adjust(wspace=wspace)
    fig.canvas.draw()

    # Draw connecting lines: corners of rect in axes[i] → corners of axes[i+1] frame
    if connect_boxes:
        for i, rect_info in enumerate(rectangles):
            if rect_info is None:
                continue
            rect_patch, ax_rect = rect_info
            ax_next = axes[i + 1]
            connect_box_to_next_axes(
                fig,
                rect_patch,
                ax_rect,
                ax_next,
                color=rect_color,
                linewidth=1.0,
                alpha=0.6,
                linestyle="-",
            )

    return fig, axes, sky_areas


# ---------------------------------------------------------------------------
# Top-level dispatcher: all marginals
# ---------------------------------------------------------------------------

def plot_all_marginals(
    round_dirs: list,
    dataloader,
    ngrid_points: int = 100,
    ngrid_points_1d: int = 500,
    figsize_per_panel: tuple = (5, 5),
    figsize_1d: tuple = (7, 4),
    connect_boxes: bool = True,
    rect_color: str = "red",
    rect_lw: float = 2.0,
    wspace: float = 0.35,
    mcmc_samples_dir: str = None,
):
    """Produce one figure per model marginal, dispatching by dimensionality.

    Iterates over all entries in ``model.marginals_list`` in order and
    produces:

    * **2-D marginals** → ``plot_truncation_rounds``: posterior-contour
      evolution across rounds, one panel per round.  The final-round panel
      includes inset 1-D marginal distributions with optional MCMC overlay.
    * **1-D marginals** → ``plot_1d_prior_evolution``: HPD interval bands
      (50 %, 90 %, 99.99 %) as a function of round index, plus a
      final-round density panel with optional MCMC overlay.

    Parameters
    ----------
    round_dirs : list of str
        Paths to each round's log directory.
    dataloader : DataLoader
        Observation data loader.
    ngrid_points : int
        Grid resolution for 2-D posterior evaluation.
    ngrid_points_1d : int
        Grid resolution for standalone 1-D posterior evaluation
        (final-round density panel).  Default 500.
    figsize_per_panel : tuple
        ``(width, height)`` per 2-D subplot panel.
    figsize_1d : tuple
        Figure size for 1-D HPD + density plots.
    connect_boxes, rect_color, rect_lw, wspace, mcmc_samples_dir
        Forwarded to ``plot_truncation_rounds`` for 2-D marginals, and
        *mcmc_samples_dir* also to ``plot_1d_prior_evolution``.

    Returns
    -------
    dict mapping ``label → (fig, axes_or_ax)``
    """
    # Discover all marginals across ALL rounds (union), so that marginals
    # introduced in later rounds (e.g. round 2+) are not missed.
    all_marginals_seen = {}  # in_param_idx -> (label, ndim, out_param_idx)
    for _rd in round_dirs:
        _m = load_model(os.path.join(_rd, "checkpoints"))
        for label, ndim, in_idx, out_idx in get_all_marginals(_m):
            if in_idx not in all_marginals_seen:
                all_marginals_seen[in_idx] = (label, ndim, out_idx)
        del _m
    all_marginals = [(label, ndim, in_idx, out_idx)
                     for in_idx, (label, ndim, out_idx) in all_marginals_seen.items()]

    n1d = sum(1 for _, ndim, _, _ in all_marginals if ndim == 1)
    n2d = sum(1 for _, ndim, _, _ in all_marginals if ndim == 2)
    print(f"Found {n2d} 2-D marginal(s) and {n1d} 1-D marginal(s) across all rounds.")
    for label, ndim, in_idx, out_idx in all_marginals:
        print(f"  {ndim}-D: {label}  (in={in_idx}, out={out_idx})")

    results = {}
    pair_idx = 0  # running counter for 2-D marginals

    for label, ndim, in_param_idx, out_param_idx in all_marginals:
        if ndim == 2:
            print(f"\n=== 2-D marginal [{pair_idx}]: {label} ===")
            fig, axes, sky_areas = plot_truncation_rounds(
                round_dirs=round_dirs,
                dataloader=dataloader,
                marginal_pair_idx=pair_idx,
                ngrid_points=ngrid_points,
                figsize_per_panel=figsize_per_panel,
                connect_boxes=connect_boxes,
                rect_color=rect_color,
                rect_lw=rect_lw,
                wspace=wspace,
                mcmc_samples_dir=mcmc_samples_dir,
                in_param_idx_override=in_param_idx,
            )
            results[label] = (fig, axes, sky_areas)
            pair_idx += 1

        elif ndim == 1:
            print(f"\n=== 1-D marginal: {label} ===")
            fig, axes_dict = plot_1d_prior_evolution(
                round_dirs=round_dirs,
                param_key=label,
                in_param_idx=in_param_idx,
                out_param_idx=out_param_idx,
                dataloader=dataloader,
                figsize=figsize_1d,
                ngrid_points=ngrid_points,
                mcmc_samples_dir=mcmc_samples_dir,
                ngrid_points_1d=ngrid_points_1d,
            )
            results[label] = (fig, axes_dict, {})

    return results


# ---------------------------------------------------------------------------
# Configuration — edit these to match your setup
# ---------------------------------------------------------------------------
name = "202603267dim_inc_phi"
round_dirs = [
    f"/data/gpuleo/mbhb/logs/{name}_round_1/version_0",
    f"/data/gpuleo/mbhb/logs/{name}_round_2/version_0",
    f"/data/gpuleo/mbhb/logs/{name}_round_3/version_0",
    # f"/data/gpuleo/mbhb/logs/{name}_round_4/version_0",
    # f"/data/gpuleo/mbhb/logs/{name}_round_5/version_0",
    # f"/data/gpuleo/mbhb/logs/{name}_round_6/version_0",
    # f"/data/gpuleo/mbhb/logs/{name}_round_7/version_0",
]

data_path = "/data/gpuleo/mbhb/obs_logspace_freqonly_q3.h5"

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    dataset_observation = MBHBDataset(data_path, cache_in_memory=False)
    dataset_subset = Subset(dataset_observation, indices=[0])
    dataloader_obs = DataLoader(
        dataset_subset,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda b: mbhb_collate_fn(
            b, dataset_subset, noise_shuffling=False, noise_factor=1.0
        ),
    )

    outdir = os.path.join(ROOT_DIR, f"plots/{name}")
    os.makedirs(outdir, exist_ok=True)

    figures = plot_all_marginals(
        round_dirs=round_dirs,
        dataloader=dataloader_obs,
        ngrid_points=100,
        ngrid_points_1d=100,
        mcmc_samples_dir=None
    )

    for label, (fig, _, sky_areas) in figures.items():
        safe_label = label.replace(" ", "_").replace("/", "_")
        out_path = os.path.join(outdir, f"{safe_label}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {out_path}")
        if sky_areas:
            for rnd, areas in sorted(sky_areas.items()):
                parts = [f"NRE={areas['nre']:.2f} sq deg"]
                if "mcmc" in areas:
                    parts.append(f"MCMC={areas['mcmc']:.2f} sq deg")
                print(f"    Round {rnd + 1} sky area: {', '.join(parts)}")

    plt.show()