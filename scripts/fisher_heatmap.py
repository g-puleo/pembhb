"""Heatmap + eigenvalues of the Fisher matrix for an MCMC run.

Reuses ``fisher_vs_mcmc_corner.build_fisher_setup`` so the Fisher is computed
with exactly the same observation / grid / frequency-mask / expansion point as
the overlay (and as the MCMC itself). Prints the eigenvalues (ascending, via the
symmetric solver) and condition number, draws an annotated heatmap with the
parameter names on the ticks, and saves the matrix to ``fisher_matrix.npz``.

The colour encodes the *normalised* Fisher  F_ij / sqrt(F_ii F_jj)  (range
[-1, 1], so the diagonal is 1 and off-diagonal correlations are visible), while
the printed number in each cell is the actual F_ij value — a raw-value colour
scale would be useless because the entries span many orders of magnitude.

Usage
-----
    /data/gpuleo/envs/lisa_pip/bin/python scripts/fisher_heatmap.py \
        [--mcmc-config configs/mcmc_config.yaml] \
        [--datagen-config configs/datagen_config.yaml] \
        [--outdir mc_results_emcee_vec/<name>]
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))
sys.path.insert(0, os.path.join(ROOT_DIR, "scripts"))

from pembhb.utils import read_config, compute_fisher_matrix_waveform_deriv  # noqa: E402
from fisher_vs_mcmc_corner import build_fisher_setup  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mcmc-config", default=os.path.join(ROOT_DIR, "configs", "mcmc_config.yaml"))
    ap.add_argument("--datagen-config", default=os.path.join(ROOT_DIR, "configs", "datagen_config.yaml"))
    ap.add_argument("--outdir", default=None,
                    help="MCMC run directory (default: mc_results_emcee_vec/<output_name>).")
    ap.add_argument("--out", default=None, help="Output PNG path.")
    args = ap.parse_args()

    mcmc_conf = read_config(args.mcmc_config)
    datagen_config = read_config(args.datagen_config)
    outdir = args.outdir or os.path.join(ROOT_DIR, "mc_results_emcee_vec", mcmc_conf["output_name"])
    with open(os.path.join(outdir, "varying_params.txt")) as f:
        varying_params = [ln.strip() for ln in f if ln.strip()]
    n = len(varying_params)

    simulator, true_tmnre_params, freq_mask = build_fisher_setup(mcmc_conf, datagen_config)
    fisher, _ = compute_fisher_matrix_waveform_deriv(
        simulator, true_tmnre_params, varying_params, freq_mask=freq_mask,
    )
    fisher = 0.5 * (fisher + fisher.T)                       # enforce symmetry

    evals = np.linalg.eigvalsh(fisher)                      # ascending
    cond = np.abs(evals[-1]) / np.abs(evals[0]) if evals[0] != 0 else np.inf
    print(f"[heatmap] run: {outdir}  ({n}D: {varying_params})")
    print("[heatmap] Fisher eigenvalues (ascending):")
    for k, ev in enumerate(evals):
        print(f"  λ[{k}] = {ev: .6e}")
    print(f"[heatmap] cond(F) = |λ_max|/|λ_min| = {cond:.3e}")

    # Normalised matrix for the colour scale; actual F_ij for the annotations.
    d = np.sqrt(np.diag(fisher))
    norm = fisher / np.outer(d, d)

    fig, ax = plt.subplots(figsize=(1.15 * n + 2.5, 1.15 * n + 1.5))
    im = ax.imshow(norm, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(n)); ax.set_xticklabels(varying_params, rotation=45, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(varying_params)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{fisher[i, j]:.1e}", ha="center", va="center",
                    fontsize=7, color="black" if abs(norm[i, j]) < 0.6 else "white")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$F_{ij}/\sqrt{F_{ii}F_{jj}}$  (colour);  cell text = $F_{ij}$")
    ax.set_title(f"Fisher matrix — {os.path.basename(outdir)}\n"
                 f"cond = {cond:.2e},  λ ∈ [{evals[0]:.2e}, {evals[-1]:.2e}]", fontsize=11)
    fig.tight_layout()

    out = args.out or os.path.join(outdir, "fisher_matrix_heatmap.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    np.savez(os.path.join(outdir, "fisher_matrix.npz"),
             fisher=fisher, eigenvalues=evals, varying_params=np.array(varying_params))
    print(f"[heatmap] saved -> {out}")
    print(f"[heatmap] matrix saved -> {os.path.join(outdir, 'fisher_matrix.npz')}")


if __name__ == "__main__":
    main()
