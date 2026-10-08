"""§2 — evaluate the network only on the region it was trained on.

Covers the pure re-weighting helper (pembhb.regions.apply_prev_mask) and, as the
motivating case, reproduces the observed cos-ι non-monotonicity from the live run
(round 1 excludes the edge-on band; round 2 hands it back) and shows hard-zero
keeps the exclusion while 'off' lets the gap reconnect.
"""
import numpy as np
import pytest

from pembhb.regions import Region, apply_prev_mask, prev_keep_mask
from pembhb.mask_truncation import analyse_posterior_1d

TWO_PI = 2 * np.pi


def _norm1d(d, g):
    return d / np.sum(d * (g[1] - g[0]))


# ----------------------------------------------------------------------
# apply_prev_mask policies
# ----------------------------------------------------------------------
def test_apply_prev_mask_off_and_noprev_are_noops():
    g = np.linspace(-1, 1, 100)
    d = np.ones_like(g)
    out, status = apply_prev_mask(d, None, (g,), policy="hard")
    assert status == "noprev" and out is d
    prev = {"kind": "1d", "intervals": [[-1.0, 0.0]]}
    out, status = apply_prev_mask(d, prev, (g,), policy="off")
    assert status == "off" and out is d


def test_apply_prev_mask_hard_zeros_outside_1d():
    g = np.linspace(-1, 1, 201)
    d = np.ones_like(g)
    prev = {"kind": "1d", "intervals": [[-0.5, 0.5]]}
    out, status = apply_prev_mask(d, prev, (g,), policy="hard")
    assert status == "applied"
    keep = (g >= -0.5) & (g <= 0.5)
    assert np.all(out[~keep] == 0.0)
    assert np.all(out[keep] > 0.0)
    assert np.isclose(np.sum(out), 1.0)                 # renormalised


def test_apply_prev_mask_hysteresis_scales_outside_1d():
    g = np.linspace(-1, 1, 201)
    d = np.ones_like(g)
    prev = {"kind": "1d", "intervals": [[-0.5, 0.5]]}
    out, status = apply_prev_mask(d, prev, (g,), policy="hysteresis",
                                  hysteresis_weight=0.1)
    assert status == "applied"
    keep = (g >= -0.5) & (g <= 0.5)
    # inside : outside weight ratio is 1 : 0.1 (before the shared renormalise)
    assert np.allclose(out[keep].mean() / out[~keep].mean(), 10.0, rtol=1e-6)
    assert np.all(out[~keep] > 0.0)                     # not zeroed


def test_apply_prev_mask_degenerate_no_overlap():
    g = np.linspace(5.0, 6.0, 100)
    d = np.ones_like(g)
    prev = {"kind": "1d", "intervals": [[0.0, 1.0]]}    # disjoint from the grid
    out, status = apply_prev_mask(d, prev, (g,), policy="hard")
    assert status == "degenerate" and out is d          # left untouched


def test_apply_prev_mask_invalid_policy_raises():
    g = np.linspace(-1, 1, 50)
    prev = {"kind": "1d", "intervals": [[-0.5, 0.5]]}
    with pytest.raises(ValueError):
        apply_prev_mask(np.ones_like(g), prev, (g,), policy="bogus")


def test_prev_keep_mask_2d_resamples_region():
    g0 = np.linspace(0, 1, 60); g1 = np.linspace(0, 1, 40)
    gx, gy = np.meshgrid(g0, g1, indexing="xy")
    mask = np.zeros((40, 60), bool)
    xin = (g0 >= 0.3) & (g0 <= 0.6); yin = (g1 >= 0.2) & (g1 <= 0.5)
    mask[np.ix_(yin, xin)] = True
    prev = {"kind": "2d", "region": Region(mask, (g0, g1))}
    # a finer, shifted grid
    ng0 = np.linspace(0.2, 0.7, 50); ng1 = np.linspace(0.1, 0.6, 44)
    keep = prev_keep_mask(prev, (ng0, ng1))
    assert keep.shape == (44, 50)
    ix = np.argmin(np.abs(ng0 - 0.45)); iy = np.argmin(np.abs(ng1 - 0.35))
    assert keep[iy, ix]
    ox = np.argmin(np.abs(ng0 - 0.68)); oy = np.argmin(np.abs(ng1 - 0.15))
    assert not keep[oy, ox]


def test_apply_prev_mask_2d_hard_zeros_outside():
    g0 = np.linspace(0, 1, 60); g1 = np.linspace(0, 1, 40)
    gx, gy = np.meshgrid(g0, g1, indexing="xy")
    mask = np.zeros((40, 60), bool)
    mask[np.ix_((g1 >= 0.2) & (g1 <= 0.5), (g0 >= 0.3) & (g0 <= 0.6))] = True
    prev = {"kind": "2d", "region": Region(mask, (g0, g1))}
    dens = np.ones((40, 60))
    out, status = apply_prev_mask(dens, prev, (g0, g1), policy="hard")
    assert status == "applied"
    keep = prev_keep_mask(prev, (g0, g1))
    assert np.all(out[~keep] == 0.0) and np.all(out[keep] > 0.0)


# ----------------------------------------------------------------------
# the motivating case: cos-ι sign degeneracy must NOT reconnect
# ----------------------------------------------------------------------
def _round1_inc_intervals():
    """Round 1 correctly excluded the edge-on band: [-1, -0.0909] ∪ [0.0909, 1]."""
    return {"kind": "1d", "intervals": [[-1.0, -0.0909], [0.0909, 1.0]]}


def _round2_reconnecting_density(g):
    """A round-2 posterior that, on the FULL envelope grid, puts enough mass in
    the excluded gap that the HPD level set covers the gap centre (cos-ι = 0)."""
    peaks = np.exp(-0.5 * ((g + 0.5) / 0.20) ** 2) + np.exp(-0.5 * ((g - 0.5) / 0.20) ** 2)
    gap_leak = 0.5 * np.exp(-0.5 * (g / 0.25) ** 2)     # spurious extrapolation in the gap
    return _norm1d(peaks + gap_leak, g)


def _accepts_gap_centre(density, g, cl=0.99):
    res = analyse_posterior_1d(g.reshape(-1, 1), density, credible_level=cl,
                               dilation_factor=1.0, period=None)
    region = Region(res["mask"], (g,))
    return bool(region.contains(np.array([0.0]))[0]), res


def test_cos_iota_gap_reconnects_without_zeroing():
    g = np.linspace(-1.0, 1.0, 200)
    d = _round2_reconnecting_density(g)
    # 'off' -> the network's gap extrapolation survives; cos-ι = 0 is accepted
    d_off, _ = apply_prev_mask(d.copy(), _round1_inc_intervals(), (g,), policy="off")
    accepts, _ = _accepts_gap_centre(d_off, g)
    assert accepts        # the bug: the previously-excluded edge-on band is back


def test_cos_iota_gap_stays_closed_with_hard_zero():
    g = np.linspace(-1.0, 1.0, 200)
    d = _round2_reconnecting_density(g)
    d_hard, status = apply_prev_mask(d.copy(), _round1_inc_intervals(), (g,), policy="hard")
    assert status == "applied"
    # the gap centre was zeroed, so it can never re-enter the HPD region
    accepts, res = _accepts_gap_centre(d_hard, g)
    assert not accepts        # exclusion preserved
    # and every accepted pixel is inside round 1's set up to the 1-px dilation
    # halo (monotone non-increasing by construction, modulo rasterisation)
    from scipy import ndimage
    keep = prev_keep_mask(_round1_inc_intervals(), (g,))
    grown = ndimage.binary_dilation(keep, iterations=1)
    assert np.all(res["mask"][~grown] == False)


# ----------------------------------------------------------------------
# the trainer wrapper delegates to apply_prev_mask (plumbing check)
# ----------------------------------------------------------------------
def test_trainer_wrapper_delegates(monkeypatch):
    import types
    import importlib.util
    import os
    spec = importlib.util.spec_from_file_location(
        "tmnre_joint_under_test",
        os.path.join(os.path.dirname(__file__), "..", "scripts", "tmnre_joint.py"))
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:                              # heavy optional deps
        pytest.skip(f"tmnre_joint import unavailable: {exc}")

    g = np.linspace(-1, 1, 101)
    obj = types.SimpleNamespace(
        _prev_accepted={(6,): {"kind": "1d", "intervals": [[-0.5, 0.5]]}})
    out = mod.SequentialTrainerJoint._zero_outside_prev(
        obj, (6,), np.ones_like(g), (g,), {"zero_outside_prev_mask": "hard"})
    keep = (g >= -0.5) & (g <= 0.5)
    assert np.all(out[~keep] == 0.0) and np.all(out[keep] > 0.0)
