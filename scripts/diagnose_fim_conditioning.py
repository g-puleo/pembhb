"""Diagnose the conditioning of the waveform-derivative Fisher matrix.

Builds a simulator consistent with how the observation was generated (same
frequency grid / modes / waveform_kwargs, via ``utils.compute_fisher_prior_bounds``
conventions), then computes the Fisher matrix
``F_ij = <∂_i h | ∂_j h>`` (see :func:`pembhb.utils.compute_fisher_matrix_waveform_deriv`)
**once**, using the per-parameter recommended finite-difference steps and, by
default, the ``O(dx⁴)`` Richardson-extrapolated derivative (pass
``--central-difference`` for the plain ``O(dx²)`` central difference).

The matrix is real-symmetric and positive semi-definite by construction, so we
diagonalise it with ``numpy.linalg.eigh`` (the symmetric solver: orthonormal
eigenvectors, real ascending eigenvalues). We report the condition number, the
eigenvalues, the worst- and best-constrained eigen-directions (which parameter
combinations they are), and per-parameter sigmas computed two ways — full
inverse vs diagonal-only. A large condition number / full-vs-diagonal σ ratio
flags a genuine parameter degeneracy (e.g. LISA orientation) rather than
finite-difference noise.
"""

import argparse
import copy
import os
import sys

import h5py
import numpy as np
import yaml

# Make src/ importable
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))

from pembhb.simulator import MBHBSimulatorFD  # noqa: E402
from pembhb.utils import (  # noqa: E402
    _ORDERED_PRIOR_KEYS,
    compute_fisher_matrix_waveform_deriv,
)


def build_simulator(datagen_config: dict, obs_path: str, event_idx: int = 0):
    """Build a CPU simulator consistent with the observation and return it
    together with the full true-parameter vector for *event_idx*.

    Mirrors the grid-consistency handling of
    :func:`pembhb.utils.compute_fisher_prior_bounds`.
    """
    with h5py.File(obs_path, "r") as f:
        freqs_obs = f["frequencies"][:]
        true_params_arr = f["source_parameters"][event_idx]

    fisher_config = copy.deepcopy(datagen_config)
    fisher_config["backend"] = "cpu"
    wp = fisher_config["waveform_params"]

    # Dummy prior so the UniformSampler initialises without complaint.
    dummy_prior = {k: [float(true_params_arr[i]), float(true_params_arr[i])]
                   for i, k in enumerate(_ORDERED_PRIOR_KEYS)}

    simulator = MBHBSimulatorFD(
        fisher_config,
        sampler_init_kwargs={"prior_bounds": dummy_prior},
        seed=42,
        n_freq_bins=wp.get("n_freq_bins", 4096),
        freq_spacing=wp.get("freq_spacing", "linear"),
    )
    assert np.allclose(freqs_obs, simulator.freqs), \
        "Frequency mismatch between obs file and simulator!"

    return simulator, np.asarray(true_params_arr, dtype=np.float64)


def _print_eigvec(param_names, eigval, vec):
    """Show one eigen-direction as its dominant parameter combination."""
    order = np.argsort(-np.abs(vec))
    terms = "  ".join(f"{vec[i]:+.2f}·{param_names[i]}" for i in order[:4])
    sigma = 1.0 / np.sqrt(eigval) if eigval > 0 else float("nan")
    print(f"    λ={eigval:>11.3e}  (σ~λ^-1/2={sigma:.2e}):  {terms}")


def diagnose(fim: np.ndarray, param_names):
    """Diagonalise the symmetric FIM and print conditioning diagnostics."""
    n = len(param_names)
    fim_sym = 0.5 * (fim + fim.T)              # enforce exact symmetry
    print(f"\n  FIM diagonal: {np.diag(fim_sym)}")

    # Symmetric diagonalisation: eigh exploits the symmetry, returning real
    # ascending eigenvalues and an orthonormal eigenvector matrix V (columns).
    # F = V Λ Vᵀ.
    eigvals, eigvecs = np.linalg.eigh(fim_sym)   # ascending
    print(f"  Eigenvalues (ascending): {eigvals}")
    cond = np.abs(eigvals[-1]) / np.abs(eigvals[0]) if eigvals[0] != 0 else np.inf
    print(f"  cond(FIM) = |λ_max|/|λ_min| = {cond:.3e}")
    if eigvals[0] < 0:
        print(f"  WARNING: negative eigenvalue {eigvals[0]:.3e} — should not occur "
              f"for a PSD waveform-derivative Fisher (numerical noise?).")

    # Each column V[:, k] is a principal axis in parameter space. The smallest
    # eigenvalues are the worst-constrained directions (near-degeneracies); the
    # largest are the best-constrained.
    n_show = min(3, n)
    print("  Worst-constrained directions (smallest λ):")
    for k in range(n_show):
        _print_eigvec(param_names, eigvals[k], eigvecs[:, k])
    print("  Best-constrained directions (largest λ):")
    for k in range(n - 1, n - 1 - n_show, -1):
        _print_eigvec(param_names, eigvals[k], eigvecs[:, k])

    # Sigmas from the full inverse. Reuse the eigendecomposition rather than
    # np.linalg.inv: F⁻¹ = V Λ⁻¹ Vᵀ, stable for a symmetric PSD matrix (no
    # general LU solve; just reciprocate the eigenvalues).
    if eigvals[0] > 0:
        fim_inv = (eigvecs / eigvals) @ eigvecs.T     # V Λ⁻¹ Vᵀ
        diag_inv = np.diag(fim_inv)
        sigma_full = np.where(diag_inv > 0, np.sqrt(diag_inv), np.nan)
    else:
        sigma_full = np.full(n, np.nan)               # not invertible / not PSD

    # Sigmas from diagonal only (treats each param as if isolated).
    diag_fim = np.diag(fim_sym)
    sigma_diag = np.where(diag_fim > 0, 1.0 / np.sqrt(diag_fim), np.nan)

    print(f"  {'param':<12} {'σ_full_inv':>14} {'σ_diag_only':>14}  ratio (full/diag)")
    for name, sf, sd in zip(param_names, sigma_full, sigma_diag):
        ratio = sf / sd if np.isfinite(sf) and np.isfinite(sd) and sd > 0 else float("nan")
        print(f"  {name:<12} {sf:>14.3e} {sd:>14.3e}  {ratio:>14.3e}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datagen-config", default=os.path.join(ROOT_DIR, "configs/datagen_config.yaml"))
    p.add_argument("--obs-path", default="/data/gpuleo/mbhb/obs_fmin1e-4_fmax2e-2_newdf_withnoise.h5")
    p.add_argument("--varying", nargs="+",
                   default=_ORDERED_PRIOR_KEYS,
                   help="Parameter names to include in the FIM (default: full 11D).")
    p.add_argument("--central-difference", action="store_true",
                   help="Use the plain O(dx²) central difference instead of the "
                        "default O(dx⁴) Richardson-extrapolated derivative.")
    args = p.parse_args()

    print(f"[diag] Loading datagen config: {args.datagen_config}")
    with open(args.datagen_config) as f:
        datagen_config = yaml.safe_load(f)
    print(f"[diag] Loading observation: {args.obs_path}")

    simulator, true_params_arr = build_simulator(datagen_config, args.obs_path)
    true_full = {k: float(true_params_arr[i]) for i, k in enumerate(_ORDERED_PRIOR_KEYS)}
    print(f"[diag] True params: {true_full}")
    print(f"[diag] Varying ({len(args.varying)}D): {args.varying}")
    deriv = "central-difference O(dx²)" if args.central_difference else "Richardson O(dx⁴)"
    print(f"[diag] Derivative: {deriv}; steps from FISHER_ABSOLUTE_STEP_DEFAULTS.")

    print(f"\n{'='*70}\n[diag] Fisher matrix\n{'='*70}")
    fim, _ = compute_fisher_matrix_waveform_deriv(
        simulator, true_params_arr, args.varying,
        use_richardson=not args.central_difference,
    )
    diagnose(fim, args.varying)


if __name__ == "__main__":
    main()
