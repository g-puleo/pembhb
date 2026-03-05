"""
Visualise the evolution of posterior contours across successive truncation rounds.

The top-level entry point is ``plot_all_marginals``, which iterates over every
marginal output of the trained model and produces:

* **2-D marginals** – one figure with N columns (one per round), each showing
  posterior contours (no colorscale), a dashed rectangle for the next round's
  prior window, and connecting lines to the next panel's frame.

* **1-D marginals** – one figure showing how the truncated prior window
  (lower and upper bound) evolves across rounds (x-axis = round index).

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
    yaml_path = os.path.join(round_dir, f"simulation_round_{round_number}.yaml")
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
):
    """Plot how the 1-D posterior HPD intervals evolve across rounds.

    For each round the model is loaded, the 1-D posterior is evaluated on a
    grid spanning the round's prior window, and HPD intervals at the levels
    defined by *hpd_levels* are shown as filled bands.  The prior window
    itself (99.99 % HPD bound, matching the truncation logic in tmnre.py)
    is shown as dashed lines.

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
        Figure size in inches.
    hpd_levels : tuple of float
        Credibility levels to show as bands, from narrowest (innermost) to
        widest (outermost).  Default: 50 %, 90 %, 99.99 %.
    band_colors : tuple of str
        Fill colours for each band, matched by position to *hpd_levels*.
    ngrid_points : int
        Grid resolution for posterior evaluation.

    Returns
    -------
    fig, ax
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

    for i, round_dir in enumerate(round_dirs):
        print(f"  [1-D {param_key}] Round {i + 1}: evaluating posterior...")
        model = load_model(os.path.join(round_dir, "checkpoints"))
        logratios, inj_params, grid = get_logratios_grid(
            dataloader,
            model,
            ngrid_points=ngrid_points,
            in_param_idx=in_param_idx,
            out_param_idx=out_param_idx,
        )
        del model
        ratios = np.exp(logratios[0])
        dp = grid[1, 0] - grid[0, 0]
        norm1d = ratios / np.sum(ratios * dp)
        grid_1d = grid[:, 0]
        for lvl in hpd_levels:
            lo, hi = _hpd_interval_1d(norm1d, grid_1d, lvl)
            hpd_lows[lvl].append(lo)
            hpd_highs[lvl].append(hi)

    fig, ax = plt.subplots(figsize=figsize)

    # Filled HPD bands (widest first so narrower ones paint on top)
    for lvl, color in zip(reversed(hpd_levels), reversed(band_colors)):
        lows_arr  = np.array(hpd_lows[lvl])
        highs_arr = np.array(hpd_highs[lvl])
        pct = int(round(lvl * 100))
        ax.fill_between(
            round_indices, lows_arr, highs_arr,
            color=color, alpha=0.85,
            label=f"{pct} % HPD",
        )
        # Draw the bounding lines explicitly
        ax.plot(round_indices, lows_arr,  color=color, linewidth=1.5)
        ax.plot(round_indices, highs_arr, color=color, linewidth=1.5)

    # Prior window as dashed reference
    ax.plot(round_indices, prior_lows,  color="black",
            linestyle="--", linewidth=1.2, label="prior window")
    ax.plot(round_indices, prior_highs, color="black",
            linestyle="--", linewidth=1.2)

    ax.set_xlabel("Round")
    ax.set_ylabel(param_key)
    ax.set_title(f"1-D posterior HPD evolution: {param_key}")
    ax.set_xticks(round_indices)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    return fig, ax


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
):
    """Overlay Gaussian-KDE credibility contours from MCMC flat samples on *ax*.

    The two parameter columns are selected by name from *mcmc_param_names*.
    Contour levels match the standard 1-/2-/3-/4-sigma credibility targets
    (68.27 %, 95.45 %, 99.73 %, 99.99 %) computed via ``contour_levels``.

    When *is_sky* is ``True`` the parameter pair is assumed to be
    ``(lambda, beta)`` and the samples are reprojected to Mollweide
    (lon, lat) coordinates before KDE evaluation.  The evaluation grid
    spans the full sky ([-π, π] × [-π/2, π/2]) rather than the current
    axis limits.
    """
    idx0 = mcmc_param_names.index(p0_key)
    idx1 = mcmc_param_names.index(p1_key)
    samples_x = flat_samples[:, idx0]
    samples_y = flat_samples[:, idx1]

    if is_sky:
        # Transform MCMC samples to Mollweide (lon, lat) — same reparametrisation
        # as the posterior grid: lon = lambda - pi, lat = arcsin(beta).
        lam_col = samples_x if p0_key == "lambda" else samples_y
        bet_col = samples_y if p1_key == "beta" else samples_x
        samples_x = lam_col - np.pi                          # lon in [-π, π]
        samples_y = np.arcsin(np.clip(bet_col, -1.0, 1.0))  # lat in [-π/2, π/2]
        gx_1d = np.linspace(-np.pi, np.pi, ngrid_points)
        gy_1d = np.linspace(-np.pi / 2.0, np.pi / 2.0, ngrid_points)
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
    kde_norm = kde_vals / (np.sum(kde_vals) * dp0 * dp1)

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
# Sky-localisation coordinate transform
# ---------------------------------------------------------------------------

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
):
    """Plot posterior-contour evolution across successive truncation rounds.

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
        superimposed on the **last** round's panel.  The two parameter
        columns are selected automatically based on the marginal pair
        being plotted.

    Returns
    -------
    fig, axes
    """
    n_rounds = len(round_dirs)

    # Load prior boxes for every round (yaml files are 1-indexed)
    prior_boxes = [load_prior_box(d, i + 1) for i, d in enumerate(round_dirs)]

    # Discover 2-D marginals from the first round's model
    model_first = load_model(os.path.join(round_dirs[0], "checkpoints"))
    marginals_2d = [
        (label, in_idx, out_idx)
        for label, ndim, in_idx, out_idx in get_all_marginals(model_first)
        if ndim == 2
    ]
    del model_first
    if not marginals_2d:
        raise ValueError("No 2-D marginals found in the model output.")
    print("    Available 2-D marginals:")
    for j, (lbl, _, _) in enumerate(marginals_2d):
        print(f"      [{j}] {lbl}")
    if marginal_pair_idx >= len(marginals_2d):
        raise IndexError(
            f"marginal_pair_idx={marginal_pair_idx} out of range "
            f"({len(marginals_2d)} 2-D marginals available)."
        )
    marginal_label, in_param_idx_fixed, out_param_idx_fixed = marginals_2d[marginal_pair_idx]

    # Detect sky-localisation marginal (lambda vs beta) → Mollweide projection
    p0_key_fixed = _ORDERED_PRIOR_KEYS[in_param_idx_fixed[0]]
    p1_key_fixed = _ORDERED_PRIOR_KEYS[in_param_idx_fixed[1]]
    is_sky = {p0_key_fixed, p1_key_fixed} == {"lambda", "beta"}

    # Create figure — Mollweide axes for the sky-location pair, standard otherwise
    if is_sky:
        fig = plt.figure(figsize=(figsize_per_panel[0] * n_rounds, figsize_per_panel[1]))
        axes = [
            fig.add_subplot(1, n_rounds, k + 1, projection="mollweide")
            for k in range(n_rounds)
        ]
    else:
        fig, axes = plt.subplots(
            1,
            n_rounds,
            figsize=(figsize_per_panel[0] * n_rounds, figsize_per_panel[1]),
            sharex=False,
            sharey=False,
        )
        if n_rounds == 1:
            axes = [axes]

    # Collect (rect_patch, ax) for each round that has a following round
    rectangles = []

    for i, (round_dir, ax) in enumerate(zip(round_dirs, axes)):
        print(f"\n--- Round {i + 1} ---")
        print(f"    Loading model from: {round_dir}")
        model = load_model(os.path.join(round_dir, "checkpoints"))

        in_param_idx  = in_param_idx_fixed
        out_param_idx = out_param_idx_fixed
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
            # Mollweide path: reparametrise (lambda, beta) → (lon, lat) and
            # plot directly on the Mollweide-projected axes.
            # ------------------------------------------------------------------
            lon_grid, lat_grid = _to_mollweide_coords(gx, gy, p0_key, p1_key)

            # Injection point in Mollweide coordinates
            lam_inj = float(inj_params[0, 0]) if p0_key == "lambda" else float(inj_params[0, 1])
            bet_inj = float(inj_params[0, 1]) if p1_key == "beta"   else float(inj_params[0, 0])
            lon_inj = lam_inj - np.pi
            lat_inj = np.arcsin(np.clip(bet_inj, -1.0, 1.0))

            ax.contourf(lon_grid, lat_grid, norm2d[0], levels=20, cmap="viridis", alpha=0.85)
            cs = ax.contour(
                lon_grid, lat_grid, norm2d[0],
                levels=levels,
                colors=["white"] * len(levels),
                linewidths=1.5,
                zorder=4,
            )
            fmt = {lev: f"{p:.3f}" for lev, p in zip(levels, level_labels)}
            ax.clabel(cs, fmt=fmt, fontsize=8)
            ax.plot(lon_inj, lat_inj, "r+", markersize=10, markeredgewidth=2, zorder=5)
            ax.set_title(f"Round {i + 1}")

            # Overlay MCMC contours on last panel (Mollweide-aware)
            if mcmc_samples_dir is not None and i == n_rounds - 1:
                print("    Overlaying MCMC contours (Mollweide)...")
                flat_samples, mcmc_param_names = load_mcmc_samples(mcmc_samples_dir)
                overlay_mcmc_contours(
                    ax, flat_samples, mcmc_param_names, p0_key, p1_key, is_sky=True
                )

            rectangles.append(None)  # no prior rectangles or connecting lines

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

    return fig, axes


# ---------------------------------------------------------------------------
# Top-level dispatcher: all marginals
# ---------------------------------------------------------------------------

def plot_all_marginals(
    round_dirs: list,
    dataloader,
    ngrid_points: int = 100,
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
      evolution across rounds, one panel per round.
    * **1-D marginals** → ``plot_1d_prior_evolution``: HPD interval bands
      (50 %, 90 %, 99.99 %) as a function of round index.

    Parameters
    ----------
    round_dirs : list of str
        Paths to each round's log directory.
    dataloader : DataLoader
        Observation data loader.
    ngrid_points : int
        Grid resolution for posterior evaluation (both 1-D and 2-D).
    figsize_per_panel : tuple
        ``(width, height)`` per 2-D subplot panel.
    figsize_1d : tuple
        Figure size for 1-D HPD plots.
    connect_boxes, rect_color, rect_lw, wspace, mcmc_samples_dir
        Forwarded to ``plot_truncation_rounds`` for 2-D marginals.

    Returns
    -------
    dict mapping ``label → (fig, axes_or_ax)``
    """
    # Discover all marginals once from the first round's model
    model_0 = load_model(os.path.join(round_dirs[0], "checkpoints"))
    all_marginals = get_all_marginals(model_0)
    del model_0

    n1d = sum(1 for _, ndim, _, _ in all_marginals if ndim == 1)
    n2d = sum(1 for _, ndim, _, _ in all_marginals if ndim == 2)
    print(f"Found {n2d} 2-D marginal(s) and {n1d} 1-D marginal(s).")
    for label, ndim, in_idx, out_idx in all_marginals:
        print(f"  {ndim}-D: {label}  (in={in_idx}, out={out_idx})")

    results = {}
    pair_idx = 0  # running counter for 2-D marginals

    for label, ndim, in_param_idx, out_param_idx in all_marginals:
        if ndim == 2:
            print(f"\n=== 2-D marginal [{pair_idx}]: {label} ===")
            fig, axes = plot_truncation_rounds(
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
            )
            results[label] = (fig, axes)
            pair_idx += 1

        elif ndim == 1:
            print(f"\n=== 1-D marginal: {label} ===")
            fig, ax = plot_1d_prior_evolution(
                round_dirs=round_dirs,
                param_key=label,
                in_param_idx=in_param_idx,
                out_param_idx=out_param_idx,
                dataloader=dataloader,
                figsize=figsize_1d,
                ngrid_points=ngrid_points,
            )
            results[label] = (fig, ax)

    return results


# ---------------------------------------------------------------------------
# Configuration — edit these to match your setup
# ---------------------------------------------------------------------------
name = "20260304autoenc_5d_fixdim_v2"
round_dirs = [
    f"/data/gpuleo/mbhb/logs/{name}_round_1/version_0",
    f"/data/gpuleo/mbhb/logs/{name}_round_2/version_0",
    f"/data/gpuleo/mbhb/logs/{name}_round_3/version_0",
    #f"/data/gpuleo/mbhb/logs/{name}_round_4/version_0",
]

data_path = "/data/gpuleo/mbhb/observation_skyloc_tc_mass.h5"

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
        mcmc_samples_dir=None
    )

    for label, (fig, _) in figures.items():
        safe_label = label.replace(" ", "_").replace("/", "_")
        out_path = os.path.join(outdir, f"{safe_label}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {out_path}")

    plt.show()