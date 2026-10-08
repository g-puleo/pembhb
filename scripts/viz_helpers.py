"""Pure helpers shared by :mod:`visualise_truncation_rounds`.

Keeping these in a separate module trims the main visualisation script.
Everything here is self-contained: no plotting, no model loading.
"""

import os
import numpy as np
import h5py
from scipy.stats import gaussian_kde

from pembhb.utils import get_logratios_grid


# ---------------------------------------------------------------------------
# MCMC samples
# ---------------------------------------------------------------------------

# Rename MCMC-file keys to the pembhb internal convention on load.
_MCMC_KEY_TO_INTERNAL = {
    "tref":     "Deltat",
    "dist_Gpc": "dist",
    "cosinc":   "inc",
    "sinbeta":  "beta",
}

# Constants used by the Deltat-axis transform (seconds offset from the
# true merger time).
SECONDS_PER_DAY = 86400.0


def load_mcmc_samples(samples_path: str):
    """Load flat MCMC samples from an HDF5 file (one dataset per parameter).

    Keys are renamed to the pembhb internal convention on load
    (``tref → Deltat``, ``dist_Gpc → dist``, ``cosinc → inc``,
    ``sinbeta → beta``).  ``Deltat`` is left in raw seconds-from-start;
    the caller is responsible for the time-coordinate transform.
    """
    with h5py.File(samples_path, "r") as f:
        file_keys = list(f.keys())
        param_names = [_MCMC_KEY_TO_INTERNAL.get(k, k) for k in file_keys]
        columns = [np.asarray(f[k][()]).ravel() for k in file_keys]
    return np.column_stack(columns), param_names


def deltat_axis_transforms(inj_days: float, duration_weeks: float,
                           mcmc_samples_path: str = None):
    """Return (nre_to_x, mcmc_to_x, x_label, nre_density_scale) for ``Deltat``.

    Both NRE and MCMC are mapped to ``seconds offset from the true merger``.
    The brutal patch for ``5D_linear_freq*.h5`` files (which already store
    Deltat in pembhb's native days-from-end convention) is preserved.
    """
    duration_sec = duration_weeks * 7.0 * SECONDS_PER_DAY
    tref_true_sec = duration_sec + inj_days * SECONDS_PER_DAY
    nre_to_x  = lambda v: (np.asarray(v, dtype=float) - inj_days) * SECONDS_PER_DAY
    mcmc_to_x = lambda v: np.asarray(v, dtype=float) - tref_true_sec
    if (
        mcmc_samples_path is not None
        and "5D_linear_freq" in os.path.basename(mcmc_samples_path)
    ):
        mcmc_to_x = nre_to_x
    return nre_to_x, mcmc_to_x, r"$\Delta t - \Delta t_{\rm true}$ [s]", 1.0 / SECONDS_PER_DAY


# ---------------------------------------------------------------------------
# Posterior summaries (1-D and 2-D)
# ---------------------------------------------------------------------------

def hpd_interval_1d(norm1d: np.ndarray, grid: np.ndarray, level: float):
    """Highest-posterior-density interval at credibility *level*.

    Uses the same cumulative-sum approach as ``get_widest_interval_1d`` in
    tmnre.py; the tail probability outside the interval is ``1 - level``.
    """
    dp = grid[1] - grid[0]
    eps = 1.0 - level
    cumsum = np.cumsum(norm1d * dp)
    idx_low  = np.searchsorted(cumsum, eps / 2)
    idx_high = np.searchsorted(cumsum, 1.0 - eps / 2)
    idx_high = min(idx_high, len(grid) - 1)
    return float(grid[idx_low]), float(grid[idx_high])


def differential_entropy_1d(norm1d: np.ndarray, dp: float) -> float:
    """Differential entropy H = -∫ p log p dx via Riemann sum (nats)."""
    with np.errstate(divide="ignore"):
        log_p = np.where(norm1d > 0, np.log(norm1d), 0.0)
    return -float(np.sum(norm1d * log_p * dp))


def differential_entropy_2d(norm2d: np.ndarray, dp0: float, dp1: float) -> float:
    """Joint differential entropy H = -∫∫ p log p dx dy via Riemann sum (nats)."""
    with np.errstate(divide="ignore"):
        log_p = np.where(norm2d > 0, np.log(norm2d), 0.0)
    return -float(np.sum(norm2d * log_p * dp0 * dp1))


# ---------------------------------------------------------------------------
# NRE / MCMC marginal evaluation
# ---------------------------------------------------------------------------

def eval_nre_1d(model, dataloader, in_param_idx, out_param_idx, low, high, ngrid):
    """Evaluate a 1-D NRE posterior on a regular grid in ``[low, high]``.

    Returns ``(grid_1d, norm1d, inj)`` with ``sum(norm1d * dp) == 1``.
    """
    logratios, inj_params, grid = get_logratios_grid(
        dataloader, model, ngrid_points=ngrid,
        in_param_idx=in_param_idx, out_param_idx=out_param_idx,
        low=low, high=high,
    )
    grid_1d = grid[:, 0]
    dp = grid_1d[1] - grid_1d[0]
    ratios = np.exp(logratios[0])
    norm1d = ratios / np.sum(ratios * dp)
    return grid_1d, norm1d, float(inj_params[0])


def marginalise_2d_to_1d(norm2d_single, gx, gy, axis_to_keep, inj_params):
    """Integrate a normalised 2-D posterior over one axis.

    *norm2d_single* has shape ``(ngrid, ngrid)`` (no batch dim).
    *axis_to_keep* is 0 (keep param_0, the x/columns axis) or 1 (param_1).
    Returns ``(grid_1d, norm1d, inj)`` consistent with :func:`eval_nre_1d`.
    """
    grid_0 = gx[0, :]; dp0 = grid_0[1] - grid_0[0]
    grid_1 = gy[:, 0]; dp1 = grid_1[1] - grid_1[0]
    if axis_to_keep == 0:
        return grid_0, np.sum(norm2d_single * dp1, axis=0), float(inj_params[0])
    return grid_1, np.sum(norm2d_single * dp0, axis=1), float(inj_params[1])


def eval_mcmc_kde_1d(flat_samples, mcmc_param_names, param_key, grid_1d,
                     sample_transform=None):
    """Evaluate the MCMC marginal KDE for *param_key* on *grid_1d*.

    Returns the (unnormalised) KDE values or ``None`` if the parameter is
    absent from *mcmc_param_names*.
    """
    if param_key not in mcmc_param_names:
        return None
    samp = flat_samples[:, mcmc_param_names.index(param_key)]
    if sample_transform is not None:
        samp = sample_transform(samp)
    return gaussian_kde(samp)(grid_1d)


# ---------------------------------------------------------------------------
# Sky (lambda, sin beta) coordinate + area helpers
# ---------------------------------------------------------------------------
# The sky marginal lives in the native parametrisation (lambda, beta) where
# lambda in [0, 2pi] is ecliptic longitude and beta = sin(ecliptic latitude)
# in [-1, 1].  Ecliptic latitude itself is arcsin(beta).  The solid-angle
# element is dOmega = dlambda * d(sin lat) = dp_lambda * dp_beta, so sky area
# is a plain cell count in the native grid.  These helpers are pure (no
# plotting) and were resurrected from the pre-refactor visualiser.

_FULL_SKY_LAMBDA = (0.0, 2.0 * np.pi)   # ecliptic longitude
_FULL_SKY_BETA   = (-1.0, 1.0)          # sin(ecliptic latitude)
_SR_TO_SQDEG     = (180.0 / np.pi) ** 2  # steradians -> square degrees


def is_full_sky(bounds_lambda, bounds_beta, rtol: float = 0.02) -> bool:
    """Return True if the (lambda, beta) window covers (nearly) the full sky."""
    lam_range = bounds_lambda[1] - bounds_lambda[0]
    bet_range = bounds_beta[1] - bounds_beta[0]
    full_lam  = _FULL_SKY_LAMBDA[1] - _FULL_SKY_LAMBDA[0]
    full_bet  = _FULL_SKY_BETA[1]  - _FULL_SKY_BETA[0]
    return (abs(lam_range - full_lam) / full_lam < rtol and
            abs(bet_range - full_bet) / full_bet < rtol)


def compute_sky_area(density_2d, threshold, dp_lambda, dp_beta) -> float:
    """Sky area (square degrees) enclosed by the iso-density contour at *threshold*.

    The grid lives in (lambda, beta = sin lat) space, so dOmega = dp_lambda *
    dp_beta and the area is a cell count times the cell area.
    """
    n_cells = np.sum(density_2d >= threshold)
    area_sr = float(n_cells) * dp_lambda * dp_beta
    return area_sr * _SR_TO_SQDEG


def to_mollweide_coords(gx, gy, p0_key, p1_key):
    """Map a native-space meshgrid (gx=p0_key, gy=p1_key) to Mollweide (lon, lat).

    lambda in [0, 2pi] -> lon = lambda - pi in [-pi, pi];
    beta = sin(lat)    -> lat = arcsin(beta) in [-pi/2, pi/2].  Radians, as
    matplotlib's ``projection="mollweide"`` expects.
    """
    if p0_key == "lambda":
        lon = gx - np.pi
        lat = np.arcsin(np.clip(gy, -1.0, 1.0))
    else:
        lon = gy - np.pi
        lat = np.arcsin(np.clip(gx, -1.0, 1.0))
    return lon, lat