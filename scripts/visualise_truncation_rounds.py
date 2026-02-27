"""
Visualise the evolution of posterior contours across successive truncation rounds.

For each round i one subplot is drawn showing:
  - posterior contours from the round's trained model (no colorscale)
  - a dashed rectangle marking the prior box of the *next* round
  - lines connecting that rectangle's corners to the corresponding corners
    of the next subplot's axis frame

The routine is reusable: call ``plot_truncation_rounds`` with any
``marginal_pair_idx`` to select a different 2-D parameter pair.

Usage
-----
Edit the ``round_dirs``, ``data_path``, and ``MARGINAL_PAIR_IDX`` constants
at the bottom, then run::

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


def get_available_marginals(model: InferenceNetwork) -> list:
    """Return ``[(label, (in_param_idx, out_param_idx)), ...]`` for every 2-D marginal."""
    available = []
    for out_idx, marginal in enumerate(model.marginals_list):
        if len(marginal) == 2:
            label = (
                f"{_ORDERED_PRIOR_KEYS[marginal[0]]} vs {_ORDERED_PRIOR_KEYS[marginal[1]]}"
            )
            available.append((label, (tuple(marginal), out_idx)))
    return available


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
):
    """Overlay Gaussian-KDE credibility contours from MCMC flat samples on *ax*.

    The two parameter columns are selected by name from *mcmc_param_names*.
    Contour levels match the standard 1-/2-/3-/4-sigma credibility targets
    (68.27 %, 95.45 %, 99.73 %, 99.99 %) computed via ``contour_levels``.

    The evaluation grid spans the current axis limits so that the contours
    are automatically clipped to the visible region.
    """
    idx0 = mcmc_param_names.index(p0_key)
    idx1 = mcmc_param_names.index(p1_key)
    samples_x = flat_samples[:, idx0]
    samples_y = flat_samples[:, idx1]

    # Gaussian KDE
    kde = gaussian_kde(np.vstack([samples_x, samples_y]))

    # Evaluate on a regular grid matching the current axis limits
    xl = ax.get_xlim()
    yl = ax.get_ylim()
    gx_1d = np.linspace(xl[0], xl[1], ngrid_points)
    gy_1d = np.linspace(yl[0], yl[1], ngrid_points)
    gx, gy = np.meshgrid(gx_1d, gy_1d)
    kde_vals = kde(np.vstack([gx.ravel(), gy.ravel()])).reshape(ngrid_points, ngrid_points)

    # Normalise (matches the convention used by contour_levels)
    dp0 = gx_1d[1] - gx_1d[0]
    dp1 = gy_1d[1] - gy_1d[0]
    kde_norm = kde_vals / (np.sum(kde_vals) * dp0 * dp1)

    # Compute credibility-level thresholds on the KDE density
    levels, level_labels = contour_levels(kde_norm, targets=[0.6827, 0.9545, 0.9973])
    print(levels)
    breakpoint()
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
    fig, axes = plt.subplots(
        1,
        n_rounds,
        figsize=(figsize_per_panel[0] * n_rounds, figsize_per_panel[1]),
        sharex=False,
        sharey=False,
    )
    if n_rounds == 1:
        axes = [axes]

    # Load prior boxes for every round (yaml files are 1-indexed)
    prior_boxes = [load_prior_box(d, i + 1) for i, d in enumerate(round_dirs)]

    # Discover available 2-D marginals — populated on the first loop iteration
    available_marginals = None

    # Collect (rect_patch, ax) for each round that has a following round
    rectangles = []

    for i, (round_dir, ax) in enumerate(zip(round_dirs, axes)):
        print(f"\n--- Round {i + 1} ---")
        print(f"    Loading model from: {round_dir}")
        model = load_model(os.path.join(round_dir, "checkpoints"))

        # Discover marginals once from the first loaded model
        if available_marginals is None:
            available_marginals = get_available_marginals(model)
            if not available_marginals:
                raise ValueError("No 2-D marginals found in the model output.")
            print("    Available 2-D marginals:")
            for j, (lbl, _) in enumerate(available_marginals):
                print(f"      [{j}] {lbl}")
            if marginal_pair_idx >= len(available_marginals):
                raise IndexError(
                    f"marginal_pair_idx={marginal_pair_idx} out of range "
                    f"({len(available_marginals)} marginals available)."
                )

        marginal_label, (in_param_idx, out_param_idx) = available_marginals[marginal_pair_idx]
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

        # Plot contours — no colorscale
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

        del model  # release GPU memory between rounds

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
# Configuration — edit these to match your setup
# ---------------------------------------------------------------------------
name = "20260226_narrowprior_autoenc"
round_dirs = [
    f"/data/gpuleo/mbhb/logs/{name}_round_1/version_0",
    f"/data/gpuleo/mbhb/logs/{name}_round_2/version_0",
    f"/data/gpuleo/mbhb/logs/{name}_round_3/version_0",
    f"/data/gpuleo/mbhb/logs/{name}_round_4/version_0",
]

data_path = "/u/g/gpuleo/pembhb/data/testes_newdata_fixall_notmcq.h5"

# Index into the list of available 2-D marginals printed at runtime.
# 0 = first pair found in the model output.  Change to plot a different pair.
MARGINAL_PAIR_IDX = 0

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    dataset_observation = MBHBDataset(data_path, cache_in_memory=False)
    dataset_subset = Subset(dataset_observation, indices=[2])
    dataloader_obs = DataLoader(
        dataset_subset,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda b: mbhb_collate_fn(
            b, dataset_subset, noise_shuffling=False, noise_factor=1.0
        ),
    )

    fig, axes = plot_truncation_rounds(
        round_dirs=round_dirs,
        dataloader=dataloader_obs,
        marginal_pair_idx=MARGINAL_PAIR_IDX,
        ngrid_points=100,
        mcmc_samples_dir=os.path.join(ROOT_DIR, "mc_results_emcee_vec/2d_masses"),
    )

    out_path = os.path.join(ROOT_DIR, f"plots/{name}", "truncation_rounds.png") 
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved figure to {out_path}")
    plt.show()