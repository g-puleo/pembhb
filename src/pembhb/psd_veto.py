"""PSD-null veto for the TDI transfer-function nulls.

The LISA response vanishes at ``f_n = n*c/(4L)`` (29.979, 59.958, 89.938 mHz),
so the PSD collapses by ~8 decades and ``1/PSD`` diverges. bbhx splines the
response from ~1024 nodes (``simulator.py:527-537``), far too coarse to resolve
a notch ~5e-5 Hz wide, so the numerator does *not* vanish with it and the
noise-weighted integrand spikes ~4 decades.

Bands are found once on a fixed reference grid and returned in **Hz**, never as
indices: they are then re-resolved against any grid, including a subsampled one
too coarse to detect them on.
"""

import numpy as np
from scipy.constants import c
from scipy.ndimage import maximum_filter1d
from bbhx.utils.constants import L_SI

# Grid on which bands are detected: 1 week, [1e-4, 1e-1] Hz. Pinned, *not*
# derived from the run's duration: ``env_bins`` is a bin count, so detected
# widths scale with df (the n=2 null measures 4.2e-4 Hz wide at 1 week but
# 5.3e-5 at 8). Pinning makes the bands a property of the noise model alone.
REF_FMIN, REF_FMAX, REF_DF = 1e-4, 1e-1, 1.0 / 604800.0


def psd_null_bands(freqs, psd, K=10.0, env_bins=201):
    """[fmin, fmax] of each band where the PSD collapses, so 1/PSD diverges.

    ``env`` is a running maximum over ``env_bins`` samples, so ``psd < env/K``
    is a *depth-below-local-envelope* test: scale-free, sensitive only to the
    notch being a hole relative to its own neighbourhood. A bin is flagged if
    any channel dips there. Note ``K`` applies to the PSD, not the ASD.

    :param freqs: frequency grid, shape (n_freqs,)
    :param psd: PSD, shape (n_channels, n_freqs) or (n_freqs,)
    :return: band edges in Hz, shape (n_bands, 2)
    """
    psd = np.atleast_2d(np.asarray(psd, dtype=float))
    env = maximum_filter1d(psd, size=env_bins, axis=-1)
    deep = np.any(psd < env / K, axis=0)
    edges = np.flatnonzero(np.diff(np.r_[0, deep.astype(np.int8), 0])).reshape(-1, 2)
    return np.array([[freqs[a], freqs[b - 1]] for a, b in edges]).reshape(-1, 2)


def bin_widths(f):
    """Midpoint-rule per-bin width. Bridges gaps left by deleted bins.

    Endpoints get the *full* adjacent gap, not half: each bin represents a
    df-wide chunk, as in the standard FFT/GW convention. The integrand
    vanishes at both ends of the band, so the choice is numerically moot
    (1e-15 on SNR) — but it is the convention every stored dataset uses.
    """
    d = np.diff(f)
    w = np.empty(f.size)
    w[0], w[-1] = d[0], d[-1]
    w[1:-1] = 0.5 * (d[:-1] + d[1:])
    return w


def tdi_null_freqs(fmax, armlength=L_SI):
    """TDI transfer-function nulls in (0, fmax]:  f_n = n*c/(4L)."""
    f1 = c / (4.0 * armlength)
    return f1 * np.arange(1, int(np.floor(fmax / f1)) + 1)


def reference_veto_bands(channels, noise_model, K=10.0, env_bins=201):
    """PSD-null bands in Hz, detected on the fixed reference grid."""
    from pembhb.simulator import build_asd  # lazy: circular import

    freqs = np.arange(REF_FMIN, REF_FMAX, REF_DF)
    asd = build_asd(freqs, channels, noise_model)
    return psd_null_bands(freqs, asd ** 2, K=K, env_bins=env_bins)


def bands_to_mask(freqs, bands):
    """Boolean mask over *freqs* of the bins falling inside any of *bands*."""
    m = np.zeros(np.size(freqs), dtype=bool)
    if bands is None or np.size(bands) == 0:
        return m
    for lo, hi in np.atleast_2d(bands):
        m |= (freqs >= lo) & (freqs <= hi)
    return m
