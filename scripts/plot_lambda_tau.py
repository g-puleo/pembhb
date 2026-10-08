#!/usr/bin/env python
"""λ–τ plane of the test-pool posteriors, per marginal and round.

During training ``CalibrationMonitor`` stores, per test-pool simulation i
and parameter, the posterior mean μ_i and std σ_i, the true value and the
Fisher σ, in ``$PEMBHB_DATA_DIR/<run>/lambda_tau_stats_round_<r>.h5``. From those:

    λ_i = (truth_i − μ_i) / σ_i        pull; calibrated posteriors give N(0, 1)
    τ_i = σ_i / σ_Fisher,i             width relative to the Cramér–Rao bound; ≈ 1

The figure shows, per parameter, 50%/90% KDE contours of the test pool in the
(λ, log10 τ) plane at the last evaluation of each round, coloured by round.
The target is the origin: unbiased and as narrow as the Fisher bound.

    python scripts/plot_lambda_tau.py RUN_TAG [--lambda-range 5]

Writes ``$PEMBHB_PLOTS_DIR/<run>/lambda_tau_plane.png``.
"""
import argparse
import os
import re
from glob import glob

import h5py
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pembhb import DATA_ROOT_DIR, PLOTS_ROOT_DIR


def _round_of(path):
    m = re.search(r"lambda_tau_stats_round_(\d+)\.h5$", path)
    return int(m.group(1)) if m else -1


def load_stats(run_tag, data_root=DATA_ROOT_DIR):
    """``{(label, param): {"cum_ep", "round", "lambda", "tau"}}`` over all rounds.

    ``lambda``/``tau`` have shape ``(n_eval_total, n_test)``, sorted by
    cumulative epoch; ``round`` is the parallel per-eval round index.
    """
    files = sorted(glob(os.path.join(data_root, run_tag,
                                     "lambda_tau_stats_round_*.h5")),
                   key=_round_of)
    if not files:
        raise FileNotFoundError(
            f"no lambda_tau_stats_round_*.h5 under {os.path.join(data_root, run_tag)} "
            f"(is calibration_monitor.lambda_tau.enabled on?)")

    acc = {}
    for fp in files:
        rnd = _round_of(fp)
        with h5py.File(fp, "r") as f:
            cum_ep = f["cum_ep"][:]
            for label in f:
                if label == "cum_ep":
                    continue
                for param, pg in f[label].items():
                    mean = pg["posterior_mean"][:]          # (n_eval, n_test)
                    std = pg["posterior_std"][:]
                    lam = (pg["ground_truth"][:][None, :] - mean) / std
                    tau = np.abs(std / pg["fisher_sigma"][:][None, :])
                    d = acc.setdefault((label, param), {"cum_ep": [], "round": [],
                                                        "lambda": [], "tau": []})
                    d["cum_ep"].append(cum_ep)
                    d["round"].append(np.full(cum_ep.shape, rnd))
                    d["lambda"].append(lam)
                    d["tau"].append(tau)

    out = {}
    for key, d in acc.items():
        cum = np.concatenate(d["cum_ep"])
        order = np.argsort(cum, kind="stable")
        out[key] = {
            "cum_ep": cum[order],
            "round": np.concatenate(d["round"])[order],
            "lambda": np.concatenate(d["lambda"], axis=0)[order],
            "tau": np.concatenate(d["tau"], axis=0)[order],
        }
    return out


def _kde_credible_levels(kde, points, fractions=(0.9, 0.5)):
    """Density thresholds enclosing the given probability fractions.

    The level for fraction f is the (1−f) quantile of the KDE evaluated at the
    samples — the standard sample-density trick for HPD contours.
    """
    d = kde(points)
    levels = sorted({float(np.quantile(d, 1.0 - f)) for f in fractions})
    return [lv for lv in levels if lv > 0]


def plot_lambda_tau_plane(stats, out_path, lambda_range=5.0):
    """Per marginal head: KDE contours of the test set in the (λ, log10 τ) plane,
    one contour set per round (last eval), coloured early→late. Two credible
    contours (50%, 90%) per round. The ideal is the origin (λ=0, log10 τ=0):
    unbiased and at the Cramér-Rao floor — a converging chain drifts to the star.
    """
    import matplotlib as mpl
    from scipy.stats import gaussian_kde

    keys = sorted(stats.keys())
    n = len(keys)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4 * nrow), squeeze=False)

    all_rounds = sorted({int(r) for k in keys for r in stats[k]["round"]})
    norm = mpl.colors.Normalize(vmin=all_rounds[0], vmax=all_rounds[-1])
    cmap = mpl.cm.brg
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    for ax, key in zip(axes.ravel(), keys):
        label, param = key
        rounds = np.asarray(stats[key]["round"])
        lam, tau = stats[key]["lambda"], stats[key]["tau"]

        # Per-round last-eval point clouds in (λ, log10 τ).
        clouds = []
        for r in sorted(set(rounds.tolist())):
            last = np.where(rounds == r)[0][-1]
            x = np.clip(lam[last], -lambda_range, lambda_range)
            z = np.log10(tau[last])
            m = np.isfinite(x) & np.isfinite(z)
            if m.sum() >= 5:
                clouds.append((r, x[m], z[m]))
        if not clouds:
            ax.axis("off")
            continue

        ax_x = np.concatenate([c[1] for c in clouds])
        ax_z = np.concatenate([c[2] for c in clouds])
        xpad = 0.05 * (np.ptp(ax_x) or 1.0)
        zpad = 0.05 * (np.ptp(ax_z) or 1.0)
        XX, YY = np.meshgrid(
            np.linspace(ax_x.min() - xpad, ax_x.max() + xpad, 120),
            np.linspace(ax_z.min() - zpad, ax_z.max() + zpad, 120),
        )
        grid = np.vstack([XX.ravel(), YY.ravel()])

        for r, x, z in clouds:
            color = cmap(norm(r))
            pts = np.vstack([x, z])
            if x.std() == 0 or z.std() == 0:
                ax.scatter(x, z, s=8, color=color, alpha=0.5, edgecolors="none")
                continue
            try:
                kde = gaussian_kde(pts)
            except np.linalg.LinAlgError:
                ax.scatter(x, z, s=8, color=color, alpha=0.5, edgecolors="none")
                continue
            levels = _kde_credible_levels(kde, pts)
            if not levels:
                continue
            ax.contour(XX, YY, kde(grid).reshape(XX.shape),
                       levels=levels, colors=[color], linewidths=1.2)

        ax.axvline(0.0, color="k", ls="--", lw=0.8, alpha=0.6)
        ax.axhline(0.0, color="k", ls="--", lw=0.8, alpha=0.6)
        ax.plot(0.0, 0.0, "r*", ms=13)
        ax.set_xlabel(r"$\lambda_i$")
        ax.set_ylabel(r"$\log_{10}\tau_i$")
        ax.set_title(f"{label} : {param}" if label != param else param, fontsize=9)
    for ax in axes.ravel()[len(keys):]:
        ax.axis("off")
    fig.colorbar(sm, ax=axes, shrink=0.8, label="round")
    fig.suptitle(r"$(\lambda,\log_{10}\tau)$ 50%/90% KDE contours — last eval per round",
                 fontsize=12)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_tag", help="run tag (TIME_OF_EXECUTION)")
    ap.add_argument("--lambda-range", type=float, default=5.0,
                    help="λ is clipped to ±this (default 5)")
    args = ap.parse_args()

    stats = load_stats(args.run_tag)
    out_dir = os.path.join(PLOTS_ROOT_DIR, args.run_tag)
    os.makedirs(out_dir, exist_ok=True)
    plot_lambda_tau_plane(stats, os.path.join(out_dir, "lambda_tau_plane.png"),
                          args.lambda_range)


if __name__ == "__main__":
    main()
