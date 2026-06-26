"""Visualise 1-D posterior evolution across TMNRE truncation rounds.

Produces a single figure: a grid of subplots, one per parameter, with the
round index on the x-axis and parameter value on the y-axis. Each round
contributes one symmetric violin (NRE 1-D marginal at the obs); MCMC, if
provided, contributes the rightmost violin and a faint horizontal ±1σ band.

2-D contour evolution lives in ``visualise_2d_truncation.py``.

Usage
-----
    python scripts/visualise_truncation_rounds.py NAME \\
        --data-path obs.h5 [--mcmc-file mcmc.h5] [--ngrid-1d 200] \\
        [--last-round N] [--ckpt-final-round PATH] [--reason TAG]
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Subset
from scipy.stats import gaussian_kde

from pembhb import ROOT_DIR
from pembhb.data import MBHBDataset
from pembhb.utils import mbhb_collate_fn

from _visualise_common import (
    DATA_ROOT_DIR,
    find_round_dirs,
    load_model,
    load_prior_box,
    load_duration_weeks,
    get_all_marginals,
    find_out_param_idx,
    register_ckpt_override,
    compute_normalised_posterior,
    keys_for_model,
    detect_basis,
)
from viz_helpers import (
    load_mcmc_samples,
    eval_nre_1d,
    marginalise_2d_to_1d,
    deltat_axis_transforms,
)
from pembhb.sampler import chi12_to_chieff_chidiff
from pembhb.utils import compute_fisher_sigmas_for_testset
import h5py
import yaml as _yaml


# ---------------------------------------------------------------------------
# Violin geometry helpers
# ---------------------------------------------------------------------------

def _density_to_violin(grid, density, x_center, max_half_width=0.4):
    """Return (y, x_left, x_right) for a mirrored fill_betweenx violin.

    The density is scaled so its max maps to ``max_half_width``. The mirror
    is symmetric about ``x_center``.
    """
    d = np.asarray(density, dtype=float)
    if d.max() > 0:
        half = d / d.max() * max_half_width
    else:
        half = np.zeros_like(d)
    return np.asarray(grid, dtype=float), x_center - half, x_center + half


def _gaussian_density_on_grid(grid, mu, sigma):
    """N(mu, sigma) density evaluated on *grid*."""
    return np.exp(-0.5 * ((grid - mu) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))


def _compute_fisher_sigmas(round_dirs, obs_path: str) -> dict:
    """Cramer-Rao 1-sigma per parameter at the obs injection, in the run's basis.

    Uses the round-1 sidecar ``conf`` block as datagen_config (carries
    ``spin_param_basis`` so chieff/chidiff runs are handled), reads the obs
    truth row, and calls ``compute_fisher_sigmas_for_testset`` on every
    non-degenerate prior param. Returns ``{param_name: sigma_fisher}``.
    """
    # round-1 sidecar lives next to the data, not the log dir.
    from _visualise_common import _sidecar_yaml_path
    with open(_sidecar_yaml_path(round_dirs[0], 1)) as f:
        sc = _yaml.safe_load(f)
    datagen_conf = sc["conf"]
    with h5py.File(obs_path, "r") as f:
        true_params = np.asarray(f["source_parameters"][0:1])  # (1, 11)
    varying = [k for k, v in datagen_conf["prior"].items() if v[0] != v[1]]
    if not varying:
        return {}
    sigmas, vp = compute_fisher_sigmas_for_testset(
        datagen_conf, true_params, varying, backend="cpu",
    )
    return {name: float(s) for name, s in zip(vp, sigmas[0])}


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


def _samples_to_violin(samples, x_center, ngrid=200, max_half_width=0.4,
                       y_range=None):
    """KDE → mirrored fill_betweenx violin.

    Restricts the KDE evaluation grid to *y_range* if given so different-round
    violins line up vertically with the same axes.
    """
    samples = np.asarray(samples, dtype=float).ravel()
    if samples.size < 2:
        return None
    kde = gaussian_kde(samples)
    if y_range is None:
        lo, hi = np.percentile(samples, [0.5, 99.5])
    else:
        lo, hi = y_range
    grid = np.linspace(lo, hi, ngrid)
    return _density_to_violin(grid, kde(grid), x_center, max_half_width)


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
# Main figure
# ---------------------------------------------------------------------------

def _grid_layout(n_panels: int) -> tuple:
    """Pick a (rows, cols) grid that's roughly 16:9 with at most 4 columns."""
    cols = min(4, n_panels)
    rows = int(np.ceil(n_panels / cols))
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


def plot_violin_evolution(
    round_dirs,
    dataloader,
    mcmc_samples_path: str | None,
    outdir: str,
    ngrid_1d: int = 200,
    reason: str = "truncation",
    y_range_per_label: dict | None = None,
    filename_suffix: str = "",
):
    """Build the unified 1-D violin-evolution figure.

    One subplot per parameter. x-axis: round 1..R (NRE) + an extra slot for
    MCMC (if provided). y-axis: parameter value. Faint horizontal ±1σ MCMC
    band + median, and a red dotted line at the true (injection) value.

    ``y_range_per_label`` (dict mapping label → ``(y_lo, y_hi)``) overrides
    the auto-computed full-range y-axis. When set, every violin still draws
    its full density but the axis clips outside the override window — useful
    for a "zoom on MCMC" companion figure.
    """
    os.makedirs(outdir, exist_ok=True)
    n_rounds = len(round_dirs)
    if n_rounds == 0:
        raise ValueError("No round directories provided.")

    # Use the last round's model to enumerate which params we'll plot
    # (later-round models can introduce marginals; an earlier round simply
    # contributes None for those slots).
    last_model = load_model(os.path.join(round_dirs[-1], "checkpoints"))
    params = list(_iter_param_marginals(last_model))
    if not params:
        raise RuntimeError("Model has no 1-D-recoverable marginals.")

    # Duration in weeks — read from round 1's sidecar (used only by the
    # Deltat axis transform).
    duration_weeks = load_duration_weeks(round_dirs[0], 1)

    # Optional MCMC samples. Remap chi1/chi2 → chi_eff/chi_diff if the NRE
    # was trained on the chieff_chidiff basis (read off the last-round model).
    mcmc_samples = mcmc_param_names = None
    fisher_sigmas_by_name = {}
    if mcmc_samples_path:
        mcmc_samples, mcmc_param_names = load_mcmc_samples(mcmc_samples_path)
        nre_basis = detect_basis(getattr(last_model, "bounds_trained", {}) or {})
        mcmc_samples, mcmc_param_names = _maybe_remap_mcmc_to_basis(
            mcmc_samples, mcmc_param_names, nre_basis,
        )
        # Fisher CRLB at the obs injection, in the same basis the run uses.
        try:
            obs_path = dataloader.dataset.dataset.filename
            fisher_sigmas_by_name = _compute_fisher_sigmas(round_dirs, obs_path)
            print(f"[fisher] CRLB sigmas: " + ", ".join(
                f"{k}={v:.3e}" for k, v in fisher_sigmas_by_name.items()))
        except Exception as e:
            print(f"[fisher] WARN: could not compute Fisher sigmas: {e}")

    # Pre-compute every (round, param) → (grid, density, inj) so we know the
    # y-axis range per parameter before drawing. Also stash the per-round
    # prior box so whiskers can read its boundaries during drawing.
    per_round_densities: list[dict] = []
    per_round_priors: list[dict] = []
    for r_idx, rd in enumerate(round_dirs, start=1):
        model = load_model(os.path.join(rd, "checkpoints"))
        prior_box = load_prior_box(rd, r_idx)
        densities = {}
        for pi in params:
            res = _eval_1d_marginal(model, dataloader, pi, prior_box, ngrid_1d)
            densities[pi[0]] = res  # keyed by param label
        per_round_densities.append(densities)
        per_round_priors.append(prior_box)
        print(f"[round {r_idx}] {sum(v is not None for v in densities.values())}"
              f"/{len(params)} marginals evaluated.")

    rows, cols = _grid_layout(len(params))
    fig, axes = plt.subplots(
        rows, cols, figsize=(3.2 * cols, 2.6 * rows), squeeze=False,
    )

    half_width = 0.4
    x_mcmc = n_rounds + 1
    x_fisher = n_rounds + 2  # only used if fisher_sigmas_by_name is available
    for idx, (label, source_dim, in_idx, _, _) in enumerate(params):
        ax = axes[idx // cols, idx % cols]

        # Look up the NRE injection value first (it parameterises the
        # Deltat axis transform). It's constant across rounds.
        inj_for_transform = None
        for densities in per_round_densities:
            res = densities[label]
            if res is not None:
                inj_for_transform = res[2]
                break
        nre_to_y, mcmc_to_y, y_label = _axis_transforms_for(
            label, inj_for_transform, duration_weeks, mcmc_samples_path,
        )

        # Determine y-range from all non-None densities (NRE) and MCMC, in
        # the display coordinates produced by the transforms.
        y_min, y_max = +np.inf, -np.inf
        for densities in per_round_densities:
            res = densities[label]
            if res is None:
                continue
            grid_disp = nre_to_y(res[0])
            y_min = min(y_min, float(grid_disp.min()))
            y_max = max(y_max, float(grid_disp.max()))
        mcmc_col_disp = None
        if mcmc_samples is not None and label in mcmc_param_names:
            mcmc_col_disp = mcmc_to_y(mcmc_samples[:, mcmc_param_names.index(label)])
            y_min = min(y_min, float(np.percentile(mcmc_col_disp, 0.5)))
            y_max = max(y_max, float(np.percentile(mcmc_col_disp, 99.5)))
        if not np.isfinite(y_min) or not np.isfinite(y_max):
            ax.set_visible(False)
            continue
        y_pad = 0.05 * (y_max - y_min)
        y_lo = y_min - y_pad
        y_hi = y_max + y_pad
        if y_range_per_label and label in y_range_per_label:
            y_lo, y_hi = y_range_per_label[label]

        # MCMC ±1σ band + median.
        truth_val = None
        if mcmc_col_disp is not None:
            mlo, mmed, mhi = np.percentile(mcmc_col_disp, [15.865, 50.0, 84.135])
            ax.axhspan(mlo, mhi, color="grey", alpha=0.15, zorder=0)
            ax.axhline(mmed, color="grey", linestyle="--", linewidth=0.7, zorder=0)

        # Per-round NRE violins (in display coords) + prior-box whiskers.
        whisker_half = 0.5 * half_width
        for r_idx, (densities, prior_box) in enumerate(
            zip(per_round_densities, per_round_priors), start=1,
        ):
            res = densities[label]
            if res is None:
                continue
            grid, density, _ = res
            grid_disp = nre_to_y(grid)
            y, xl, xr = _density_to_violin(grid_disp, density, r_idx, half_width)
            ax.fill_betweenx(y, xl, xr, color="C0", alpha=0.55,
                              edgecolor="C0", linewidth=0.5)
            if label in prior_box:
                lo, hi = prior_box[label]
                lo_disp = float(nre_to_y(np.array([lo]))[0])
                hi_disp = float(nre_to_y(np.array([hi]))[0])
                # Horizontal whisker caps at lo and hi.
                ax.hlines([lo_disp, hi_disp],
                          r_idx - whisker_half, r_idx + whisker_half,
                          colors="k", linewidth=0.9, zorder=2)
                # Thin vertical connector through the violin (clipped to box).
                ax.vlines(r_idx, lo_disp, hi_disp,
                          colors="k", linewidth=0.5, linestyles=":",
                          alpha=0.6, zorder=2)
        truth_val = (
            float(nre_to_y(np.array([inj_for_transform]))[0])
            if inj_for_transform is not None else None
        )

        # MCMC violin in the last column (already in display coords).
        if mcmc_col_disp is not None:
            mres = _samples_to_violin(
                mcmc_col_disp, x_mcmc, ngrid=ngrid_1d,
                max_half_width=half_width, y_range=(y_lo, y_hi),
            )
            if mres is not None:
                y, xl, xr = mres
                ax.fill_betweenx(y, xl, xr, color="grey", alpha=0.5,
                                  edgecolor="grey", linewidth=0.5)

        # Fisher Gaussian violin: N(MCMC_median, sigma_fisher) drawn as a violin
        # in the column just past MCMC. Shows what the linear-Gaussian (Cramer-
        # Rao) approximation predicts for the same param.
        has_fisher = (
            mcmc_col_disp is not None
            and label in fisher_sigmas_by_name
            and np.isfinite(fisher_sigmas_by_name[label])
        )
        if has_fisher:
            sigma_f = fisher_sigmas_by_name[label]
            # Convert sigma to display units by mapping a unit interval at the
            # MCMC median through the transform (handles Deltat: days→seconds).
            mu_disp = float(np.median(mcmc_col_disp))
            # nre_to_y is applied to NRE-native (= param-native) values. mcmc_to_y
            # is applied to MCMC-native values. For Deltat the sigma is in the
            # NRE-native unit (days), so scale via nre_to_y derivative.
            if label == "Deltat":
                # Scale factor = mcmc_to_y(median+1) - mcmc_to_y(median), but
                # sigma_fisher is in NRE-native (days). Use NRE-native sigma
                # then transform centered at zero: dx_disp ≈ |nre_to_y(s) - nre_to_y(0)|.
                scale = abs(float(nre_to_y(np.array([sigma_f]))[0])
                            - float(nre_to_y(np.array([0.0]))[0]))
                sigma_disp = scale
            else:
                sigma_disp = sigma_f
            grid_g = np.linspace(y_lo, y_hi, ngrid_1d)
            dens_g = _gaussian_density_on_grid(grid_g, mu_disp, sigma_disp)
            y, xl, xr = _density_to_violin(grid_g, dens_g, x_fisher, half_width)
            ax.fill_betweenx(y, xl, xr, color="tab:green", alpha=0.45,
                              edgecolor="tab:green", linewidth=0.5)

        # Truth line.
        if truth_val is not None:
            ax.axhline(truth_val, color="red", linestyle=":", linewidth=0.9,
                       zorder=3)

        # Cosmetics.
        rightmost = (x_fisher if has_fisher
                     else (x_mcmc if mcmc_col_disp is not None else n_rounds))
        ax.set_xlim(0.4, rightmost + 0.6)
        ax.set_ylim(y_lo, y_hi)
        xticks = list(range(1, n_rounds + 1))
        xticklabels = [str(i) for i in xticks]
        if mcmc_col_disp is not None:
            xticks.append(x_mcmc)
            xticklabels.append("MCMC")
        if has_fisher:
            xticks.append(x_fisher)
            xticklabels.append("Fisher")
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, fontsize=8)
        suffix = "" if source_dim == 1 else "  (from 2-D)"
        ax.set_title(f"{y_label}{suffix}", fontsize=10)
        ax.tick_params(axis="y", labelsize=8)
        if idx // cols == rows - 1:
            ax.set_xlabel("round", fontsize=9)

    # Hide unused panels.
    for k in range(len(params), rows * cols):
        axes[k // cols, k % cols].set_visible(False)

    zoom_tag = f" — zoom on MCMC" if filename_suffix else ""
    fig.suptitle(
        f"1-D posterior evolution — NRE per round, MCMC reference — "
        f"reason: {reason}{zoom_tag}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path = os.path.join(
        outdir, f"violin_evolution_{reason}{filename_suffix}.png",
    )
    fig.savefig(out_path, dpi=140)
    print(f"saved {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_dataloader(data_path: str) -> DataLoader:
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


def main():
    p = argparse.ArgumentParser(
        description="Plot 1-D posterior violin evolution across TMNRE rounds.",
    )
    p.add_argument("name", help="Run name / TIME_OF_EXECUTION.")
    p.add_argument("--data-path", required=True,
                   help="Observation HDF5 (preferably with stored noise_fd).")
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

    dataloader = _build_dataloader(args.data_path)

    outdir = os.path.join(
        ROOT_DIR,
        "plots",
        args.name + (f"_upto_round_{args.last_round}" if args.last_round else ""),
    )
    plot_violin_evolution(
        round_dirs=round_dirs,
        dataloader=dataloader,
        mcmc_samples_path=args.mcmc_file,
        outdir=outdir,
        ngrid_1d=args.ngrid_1d,
        reason=args.reason,
    )

    # Optional MCMC-zoom companion figure.
    if args.mcmc_file and args.zoom_mcmc_sigmas > 0:
        flat_samples, mcmc_param_names = load_mcmc_samples(args.mcmc_file)
        last_model = load_model(os.path.join(round_dirs[-1], "checkpoints"))
        nre_basis = detect_basis(getattr(last_model, "bounds_trained", {}) or {})
        flat_samples, mcmc_param_names = _maybe_remap_mcmc_to_basis(
            flat_samples, mcmc_param_names, nre_basis,
        )
        params = list(_iter_param_marginals(last_model))
        duration_weeks = load_duration_weeks(round_dirs[0], 1)
        first_round_prior = load_prior_box(round_dirs[0], 1)
        # Single-point obs prior → midpoint is the injection value (Deltat
        # axis transform needs it).
        deltat_inj = None
        if "Deltat" in first_round_prior:
            lo, hi = first_round_prior["Deltat"]
            deltat_inj = 0.5 * (lo + hi)

        y_range_per_label = {}
        N = args.zoom_mcmc_sigmas
        for label, _, _, _, _ in params:
            if label not in mcmc_param_names:
                continue
            col = flat_samples[:, mcmc_param_names.index(label)]
            # Apply transform if Deltat.
            _, mcmc_to_y, _ = _axis_transforms_for(
                label, deltat_inj, duration_weeks, args.mcmc_file,
            )
            col_disp = mcmc_to_y(col)
            med = float(np.median(col_disp))
            std = float(np.std(col_disp))
            y_range_per_label[label] = (med - N * std, med + N * std)

        plot_violin_evolution(
            round_dirs=round_dirs,
            dataloader=dataloader,
            mcmc_samples_path=args.mcmc_file,
            outdir=outdir,
            ngrid_1d=args.ngrid_1d,
            reason=args.reason,
            y_range_per_label=y_range_per_label,
            filename_suffix=f"_zoom{int(N)}sigma",
        )


if __name__ == "__main__":
    main()
