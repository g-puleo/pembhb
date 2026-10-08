"""§2 monotonicity: A_N := A_N ∩ A_{N-1}.

The zeroing keeps the HPD *level set* inside the previous region, but
``analyse_posterior_*`` then dilates each mode, which can grow it back past the
previous boundary. ``clip_intervals_to_prev`` / ``clip_labels_to_prev`` intersect
the dilated result with the previous accepted set so exclusions are permanent.
"""
import numpy as np

from pembhb.regions import (
    clip_intervals_to_prev, clip_labels_to_prev, Region, prev_keep_mask,
)
from pembhb.mask_truncation import analyse_posterior_1d, analyse_posterior_2d


def _envelope(intervals):
    return [min(lo for lo, _ in intervals), max(hi for _, hi in intervals)]


# ----------------------------------------------------------------------
# 1D: dilation escapes the previous interval; the clip pulls it back
# ----------------------------------------------------------------------
def test_1d_dilation_escapes_then_clip_contains():
    g = np.linspace(-1.0, 1.0, 201)
    grid = g[:, None]
    # a mode filling most of [-0.5, 0.5]
    density = np.exp(-0.5 * (g / 0.25) ** 2)
    density /= density.sum() * (g[1] - g[0])
    prev = {"kind": "1d", "intervals": [[-0.5, 0.5]]}

    res = analyse_posterior_1d(grid, density, credible_level=0.997,
                               dilation_factor=1.5, period=None)
    # dilation pushed the mode past the previous boundary (the failure mode)
    env = _envelope(res["intervals"])
    assert env[0] < -0.5 or env[1] > 0.5

    clipped = clip_intervals_to_prev(res["mask"], g, prev, period=None)
    cenv = _envelope(clipped)
    # Containment is exact on the mask; reported as intervals it can overshoot
    # by up to half a cell, because prev_keep_mask keeps a cell whose *centre*
    # is inside prev while the interval reports that cell's outer *edge*. The
    # slack is half the current grid's pitch and shrinks with refinement.
    half = 0.5 * float(g[1] - g[0])
    assert cenv[0] >= -0.5 - half - 1e-9 and cenv[1] <= 0.5 + half + 1e-9


def test_1d_multimodal_gap_restored():
    # Previous round excluded the centre band (two modes). A reconnecting fresh
    # mask must be re-split into two intervals — the cos-ι scenario.
    g = np.linspace(-1.0, 1.0, 201)
    prev = {"kind": "1d", "intervals": [[-0.8, -0.2], [0.2, 0.8]]}
    fresh_mask = (g >= -0.8) & (g <= 0.8)          # gap filled
    out = clip_intervals_to_prev(fresh_mask, g, prev, period=None)
    assert len(out) == 2
    # neither interval crosses the excluded gap (-0.2, 0.2)
    for lo, hi in out:
        assert hi <= -0.2 + 1e-9 or lo >= 0.2 - 1e-9


def test_1d_no_overlap_keeps_fresh():
    g = np.linspace(-1.0, 1.0, 101)
    prev = {"kind": "1d", "intervals": [[0.8, 0.95]]}
    fresh_mask = (g >= -0.6) & (g <= 0.4)
    out = clip_intervals_to_prev(fresh_mask, g, prev, period=None)
    assert _envelope(out)[0] < -0.5      # not emptied; fresh retained


def test_1d_prev_none_is_identity():
    g = np.linspace(-1.0, 1.0, 101)
    fresh_mask = (g >= -0.6) & (g <= 0.4)
    out = clip_intervals_to_prev(fresh_mask, g, None, period=None)
    assert _envelope(out)[0] < -0.5 and _envelope(out)[1] > 0.35


# ----------------------------------------------------------------------
# 2D: intersection is exactly fresh ∩ prev, on a non-square grid
# ----------------------------------------------------------------------
def test_2d_clip_is_fresh_and_prev_nonsquare():
    grid_x = np.linspace(0.0, 1.0, 51)
    grid_y = np.linspace(0.0, 1.0, 41)          # non-square catches transposition
    MX, MY = np.meshgrid(grid_x, grid_y, indexing="xy")   # (ny, nx)
    prev_mask = (MX >= 0.3) & (MX <= 0.6) & (MY >= 0.4) & (MY <= 0.7)
    prev = {"kind": "2d", "region": Region(prev_mask, (grid_x, grid_y))}
    fresh = ((MX >= 0.2) & (MX <= 0.7) & (MY >= 0.3) & (MY <= 0.8)).astype(int)

    labels, comps = clip_labels_to_prev(fresh, grid_x, grid_y, prev,
                                        period=(None, None))
    got = labels > 0
    assert got.shape == (grid_y.size, grid_x.size)
    assert np.array_equal(got, (fresh > 0) & prev_mask)
    assert np.all(got <= prev_mask)             # subset of prev
    assert set(comps.keys()) == {1}             # single component survives


def test_2d_dilation_escape_contained_end_to_end():
    grid_x = np.linspace(0.0, 1.0, 61)
    grid_y = np.linspace(0.0, 1.0, 61)
    MX, MY = np.meshgrid(grid_x, grid_y, indexing="xy")
    prev_mask = (MX >= 0.35) & (MX <= 0.65) & (MY >= 0.35) & (MY <= 0.65)
    prev = {"kind": "2d", "region": Region(prev_mask, (grid_x, grid_y))}

    # a broad blob whose HPD + dilation spills outside prev
    density = np.exp(-0.5 * (((MX - 0.5) / 0.15) ** 2 + ((MY - 0.5) / 0.15) ** 2))
    density /= density.sum()
    labels, comps, gx, gy = analyse_posterior_2d(
        MX, MY, density, period=(None, None),
        credible_level=0.997, dilation_factor=1.5)
    assert np.any((labels > 0) & ~prev_mask)    # escaped (the failure mode)

    labels_c, _ = clip_labels_to_prev(labels, gx, gy, prev, period=(None, None))
    assert np.all((labels_c > 0) <= prev_mask)  # contained after clip


def test_2d_prev_none_is_identity():
    grid_x = np.linspace(0.0, 1.0, 31)
    grid_y = np.linspace(0.0, 1.0, 31)
    MX, MY = np.meshgrid(grid_x, grid_y, indexing="xy")
    fresh = ((MX >= 0.2) & (MX <= 0.7)).astype(int)
    labels, _ = clip_labels_to_prev(fresh, grid_x, grid_y, None)
    assert np.array_equal(labels > 0, fresh > 0)


# ----------------------------------------------------------------------
# trainer flag semantics: default off; only active under the 'hard' policy
# ----------------------------------------------------------------------
def _load_trainer():
    import importlib.util
    import os
    import pytest
    spec = importlib.util.spec_from_file_location(
        "tmnre_joint_under_test",
        os.path.join(os.path.dirname(__file__), "..", "scripts", "tmnre_joint.py"))
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:                              # heavy optional deps
        pytest.skip(f"tmnre_joint import unavailable: {exc}")
    return mod


def test_monotonic_flag_semantics():
    import types
    mod = _load_trainer()
    fn = mod.SequentialTrainerJoint._monotonic_enabled
    obj = types.SimpleNamespace(_monotonic_warned=False)

    # default: off
    assert fn(obj, {}) is False
    # on + hard -> active
    assert fn(obj, {"enforce_monotonic_regions": True,
                    "zero_outside_prev_mask": "hard"}) is True
    # on + non-hard -> ignored (no-op) and warns once
    assert fn(obj, {"enforce_monotonic_regions": True,
                    "zero_outside_prev_mask": "hysteresis"}) is False
    assert obj._monotonic_warned is True
