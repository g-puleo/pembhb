"""Mask-truncation contract tests.

Ported from the session scratchpad (see MASK_TRUNCATION_OPEN_ISSUES.md — "port
them to test/").  Each block preserves the invariants of the original script:
HPD level-set analysis (1D/2D), the periodicity axis convention (the (row,col)
vs (x,y) transposition guard), interval / mask sampling, truth violations,
save/load persistence, and the chieff_chidiff post-overwrite spin rejection.
"""
import os
import shutil
import tempfile

import numpy as np
import pytest

import pembhb
from pembhb.mask_truncation import (
    analyse_posterior_1d, analyse_posterior_2d,
    _periodic_labelling_2d, _dilate_mode_2d,
    _sample_from_intervals, _sample_2d_from_components,
    save_truncation, load_truncation, components_from_labels,
    truth_violations, format_violations, MaskRejectSampler,
)
from pembhb import utils

TWO_PI = 2 * np.pi


@pytest.fixture
def float64_precision():
    """Sampler tests run in float64 (matching the scratchpad originals)."""
    prev = pembhb.get_precision()
    pembhb.set_precision("float64")
    yield
    pembhb.set_precision(prev)


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _norm1d(d, g):
    return d / np.sum(d * (g[1] - g[0]))


def _make_grids(x_lo, x_hi, y_lo, y_hi, nx=100, ny=100):
    g0 = np.linspace(x_lo, x_hi, nx)
    g1 = np.linspace(y_lo, y_hi, ny)
    gx, gy = np.meshgrid(g0, g1, indexing="xy")
    return gx, gy


def _norm2d(gx, gy, d):
    return d / np.sum(d * (gx[0, 1] - gx[0, 0]) * (gy[1, 0] - gy[0, 0]))


# ======================================================================
# analyse_posterior_1d
# ======================================================================
def test_analyse_1d_unimodal():
    g = np.linspace(0.0, 1.0, 100)
    d = np.exp(-0.5 * ((g - 0.5) / 0.05) ** 2)
    res = analyse_posterior_1d(g.reshape(-1, 1), _norm1d(d, g), period=None,
                               credible_level=0.99)
    assert len(res["intervals"]) == 1


def test_analyse_1d_bimodal():
    g = np.linspace(0.0, 1.0, 100)
    d = (np.exp(-0.5 * ((g - 0.3) / 0.03) ** 2)
         + np.exp(-0.5 * ((g - 0.7) / 0.03) ** 2))
    res = analyse_posterior_1d(g.reshape(-1, 1), _norm1d(d, g), period=None,
                               credible_level=0.99)
    assert len(res["intervals"]) == 2


def test_analyse_1d_wrapped_splits():
    g = np.linspace(0.0, TWO_PI, 100)
    d = np.exp(4.0 * np.cos(g))          # peak at 0 == 2pi
    res = analyse_posterior_1d(g.reshape(-1, 1), _norm1d(d, g), period=TWO_PI,
                               credible_level=0.99)
    iv = res["intervals"]
    assert len(iv) == 2
    assert np.isclose(iv[0][0], 0.0) and np.isclose(iv[-1][1], TWO_PI)


def test_analyse_1d_wrapped_declared_nonperiodic():
    g = np.linspace(0.0, TWO_PI, 100)
    d = np.exp(4.0 * np.cos(g))
    res = analyse_posterior_1d(g.reshape(-1, 1), _norm1d(d, g), period=None,
                               credible_level=0.99)
    assert len(res["intervals"]) == 2   # two separate end modes, never merged


def test_analyse_1d_effective_periodicity_subwindow():
    # period 2pi declared, but the grid is an already-truncated [1, 2] window:
    # the seam is not physical any more and must not be stitched.
    g = np.linspace(1.0, 2.0, 100)
    d = (np.exp(-0.5 * ((g - 1.0) / 0.02) ** 2)
         + np.exp(-0.5 * ((g - 2.0) / 0.02) ** 2))
    res = analyse_posterior_1d(g.reshape(-1, 1), _norm1d(d, g), period=TWO_PI,
                               credible_level=0.99)
    assert len(res["intervals"]) == 2


# ======================================================================
# analyse_posterior_2d
# ======================================================================
def test_analyse_2d_single_blob():
    gx, gy = _make_grids(0, 1, 0, 1)
    d = np.exp(-0.5 * (((gx - 0.5) / 0.05) ** 2 + ((gy - 0.4) / 0.05) ** 2))
    _, comps, _, _ = analyse_posterior_2d(gx, gy, _norm2d(gx, gy, d), period=(None, None))
    assert len(comps) == 1


def test_analyse_2d_two_blobs():
    gx, gy = _make_grids(0, 1, 0, 1)
    d = (np.exp(-0.5 * (((gx - 0.25) / 0.03) ** 2 + ((gy - 0.25) / 0.03) ** 2))
         + np.exp(-0.5 * (((gx - 0.75) / 0.03) ** 2 + ((gy - 0.75) / 0.03) ** 2)))
    _, comps, _, _ = analyse_posterior_2d(gx, gy, _norm2d(gx, gy, d), period=(None, None))
    assert len(comps) == 2


def test_analyse_2d_x_wrapped():
    gx, gy = _make_grids(0, TWO_PI, -1, 1)
    d = np.exp(4.0 * np.cos(gx)) * np.exp(-0.5 * (gy / 0.2) ** 2)
    _, comps, _, _ = analyse_posterior_2d(gx, gy, _norm2d(gx, gy, d), period=(TWO_PI, None))
    assert len(comps) == 1
    c = comps[next(iter(comps))]
    assert len(c["x_intervals"]) == 2       # x wraps -> two sub-intervals
    assert len(c["y_intervals"]) == 1


def test_analyse_2d_y_wrapped():
    gx, gy = _make_grids(-1, 1, 0, TWO_PI)
    d = np.exp(-0.5 * (gx / 0.2) ** 2) * np.exp(4.0 * np.cos(gy))
    _, comps, _, _ = analyse_posterior_2d(gx, gy, _norm2d(gx, gy, d), period=(None, TWO_PI))
    assert len(comps) == 1
    c = comps[next(iter(comps))]
    assert len(c["y_intervals"]) == 2
    assert len(c["x_intervals"]) == 1


# ======================================================================
# periodicity axis convention: (row, col) = (y, x)  [transposition guard]
# ======================================================================
def test_periodic_labelling_unit_convention():
    # split across the COLUMN seam
    m = np.zeros((5, 7), bool); m[2, 0] = m[2, -1] = True
    assert _periodic_labelling_2d(m, (False, True))[1] == 1
    assert _periodic_labelling_2d(m, (True, False))[1] == 2
    # split across the ROW seam
    m = np.zeros((5, 7), bool); m[0, 3] = m[-1, 3] = True
    assert _periodic_labelling_2d(m, (True, False))[1] == 1
    assert _periodic_labelling_2d(m, (False, True))[1] == 2


def test_dilate_wraps_correct_axis():
    d = np.zeros((5, 7), bool); d[0, 3] = True
    assert bool(_dilate_mode_2d(d, iterations=1, periodic=(True, False))[-1, 3])
    assert not bool(_dilate_mode_2d(d, iterations=1, periodic=(False, True))[-1, 3])


def test_label_2d_last_row_seam():
    m = np.zeros((3, 3), int); m[2, 0] = m[2, 2] = 1
    assert _periodic_labelling_2d(m.astype(bool), (False, True))[1] == 1
    assert _periodic_labelling_2d(m.astype(bool), (False, False))[1] == 2


def test_label_2d_torus_corner():
    m = np.zeros((5, 5), int); m[0, 0] = m[4, 4] = 1
    assert _periodic_labelling_2d(m.astype(bool), (True, True))[1] == 1
    assert _periodic_labelling_2d(m.astype(bool), (True, False))[1] == 2
    assert _periodic_labelling_2d(m.astype(bool), (False, False))[1] == 2


def test_analyse_2d_nonsquare_no_transposition():
    nx, ny = 80, 50
    gx, gy = _make_grids(0, 1, 0, 1, nx=nx, ny=ny)
    d = np.exp(-0.5 * (((gx - 0.5) / 0.05) ** 2 + ((gy - 0.4) / 0.05) ** 2))
    labels, comps, grid_x, grid_y = analyse_posterior_2d(
        gx, gy, _norm2d(gx, gy, d), period=(None, None))
    assert labels.shape == (ny, nx)
    assert grid_x.shape == (nx,) and grid_y.shape == (ny,)
    assert len(comps) == 1


def test_analyse_2d_torus_corner_one_component():
    gx, gy = _make_grids(0, TWO_PI, 0, TWO_PI, nx=80, ny=50)
    d = np.exp(12.0 * np.cos(gx)) * np.exp(12.0 * np.cos(gy))
    _, comps, _, _ = analyse_posterior_2d(
        gx, gy, _norm2d(gx, gy, d), period=(TWO_PI, TWO_PI), credible_level=0.99)
    assert len(comps) == 1
    c = comps[next(iter(comps))]
    assert len(c["x_intervals"]) == 2 and len(c["y_intervals"]) == 2


def test_analyse_2d_effective_periodicity_subwindow():
    gx, gy = _make_grids(1.0, 2.0, -1, 1, nx=80, ny=50)
    d = (np.exp(-0.5 * (((gx - 1.0) / 0.03) ** 2 + (gy / 0.25) ** 2))
         + np.exp(-0.5 * (((gx - 2.0) / 0.03) ** 2 + (gy / 0.25) ** 2)))
    _, comps, _, _ = analyse_posterior_2d(
        gx, gy, _norm2d(gx, gy, d), period=(TWO_PI, None), credible_level=0.99)
    assert len(comps) == 2       # truncated window must NOT wrap

    gx, gy = _make_grids(0.0, TWO_PI, -1, 1, nx=80, ny=50)
    d = (np.exp(-0.5 * (((gx - 0.0) / 0.05) ** 2 + (gy / 0.25) ** 2))
         + np.exp(-0.5 * (((gx - TWO_PI) / 0.05) ** 2 + (gy / 0.25) ** 2)))
    _, comps, _, _ = analyse_posterior_2d(
        gx, gy, _norm2d(gx, gy, d), period=(TWO_PI, None), credible_level=0.99)
    assert len(comps) == 1       # full period -> merges


# ======================================================================
# _sample_from_intervals (1D)
# ======================================================================
def test_sample_intervals_width_weighting_and_flatness():
    rng = np.random.default_rng(0)
    N = 400_000
    iv = [[0.0, 1.0], [2.0, 5.0]]      # widths 1 and 3 -> 0.25 / 0.75
    s = _sample_from_intervals(iv, N, rng)
    f0 = np.mean((s >= 0) & (s <= 1))
    f1 = np.mean((s >= 2) & (s <= 5))
    assert f0 == pytest.approx(0.25, abs=0.01)
    assert f1 == pytest.approx(0.75, abs=0.01)
    assert np.mean(~(((s >= 0) & (s <= 1)) | ((s >= 2) & (s <= 5)))) == 0.0
    sub = s[(s >= 2) & (s <= 5)]
    h, _ = np.histogram(sub, bins=3, range=(2, 5))
    assert np.allclose(h / h.sum(), 1 / 3, atol=0.01)


def test_sample_intervals_cube_uniform_in_volume():
    rng = np.random.default_rng(0)
    N = 400_000
    s = _sample_from_intervals([[3.0, 30.0]], N, rng, cube=True)
    c = s ** 3                          # d^3 should be uniform on [27, 27000]
    h, _ = np.histogram(c, bins=5, range=(27, 27000))
    assert np.allclose(h / h.sum(), 0.2, atol=0.01)
    # volume-uniform mean differs from the linear-uniform mean (16.5)
    assert abs(s.mean() - (3 + 30) / 2) > 3.0


# ======================================================================
# _sample_2d_from_components
# ======================================================================
def test_sample_2d_all_inside_and_weighted():
    rng = np.random.default_rng(0)
    N = 300_000
    gx, gy = _make_grids(0, 1, 0, 1)
    d = (np.exp(-0.5 * (((gx - 0.25) / 0.03) ** 2 + ((gy - 0.25) / 0.03) ** 2))
         + np.exp(-0.5 * (((gx - 0.75) / 0.05) ** 2 + ((gy - 0.75) / 0.05) ** 2)))
    labels, comps, grid_x, grid_y = analyse_posterior_2d(
        gx, gy, _norm2d(gx, gy, d), period=(None, None))
    x, y, acc = _sample_2d_from_components(comps, labels, grid_x, grid_y, N, rng)
    assert len(x) == N
    # One membership rule: the sampler indexes from the cell vertex, so its
    # floor() agrees with Region.contains' nearest-centre round(). Use the
    # canonical helper rather than a local copy of the arithmetic.
    from pembhb.regions import Region as _Region, _nearest_index
    assert _Region(labels > 0, (grid_x, grid_y)).contains(np.stack([x, y])).all()
    col = np.clip(_nearest_index(x, grid_x), 0, labels.shape[1] - 1)
    row = np.clip(_nearest_index(y, grid_y), 0, labels.shape[0] - 1)
    lab = labels[row, col]
    assert np.mean(lab > 0) == 1.0
    # per-component sampled fraction tracks its pixel weight
    tot = (labels > 0).sum()
    for k in comps:
        assert np.mean(lab == k) == pytest.approx((labels == k).sum() / tot, abs=0.02)


def test_sample_2d_thin_ridge_high_fill_no_nan():
    # a thin diagonal ridge has a big bounding box but small area -> guards the
    # "expected_acc > 1 -> nbinom NaN" fix (fill fraction stays in (0, 1]).
    rng = np.random.default_rng(1)
    N = 200_000
    gx, gy = _make_grids(0, 1, 0, 1, nx=120, ny=120)
    blob = np.exp(-0.5 * (((gx - 0.25) / 0.05) ** 2 + ((gy - 0.78) / 0.05) ** 2))
    u = ((gx - 0.70) + (gy - 0.30)) / np.sqrt(2)
    v = ((gx - 0.70) - (gy - 0.30)) / np.sqrt(2)
    ridge = np.exp(-0.5 * ((u / 0.09) ** 2 + (v / 0.010) ** 2))
    labels, comps, grid_x, grid_y = analyse_posterior_2d(
        gx, gy, _norm2d(gx, gy, blob + ridge), period=(None, None))
    x, y, acc = _sample_2d_from_components(comps, labels, grid_x, grid_y, N, rng)
    assert np.isfinite(acc) and len(x) == N
    # One membership rule: the sampler indexes from the cell vertex so its
    # floor() agrees with Region.contains' nearest-centre round().
    from pembhb.regions import Region as _Region
    assert _Region(labels > 0, (grid_x, grid_y)).contains(np.stack([x, y])).all()


# ======================================================================
# truth_violations
# ======================================================================
_KEYS = ["logMchirp", "q", "chi1", "chi2", "dist", "phi", "inc",
         "lambda", "beta", "psi", "Deltat"]


@pytest.fixture
def tv_fixture():
    gx0 = np.linspace(0, 1, 80)      # x -> columns -> param 7
    gy0 = np.linspace(0, 1, 50)      # y -> rows    -> param 8
    gx, gy = np.meshgrid(gx0, gy0, indexing="xy")
    d = (np.exp(-0.5 * (((gx - 0.20) / 0.03) ** 2 + ((gy - 0.20) / 0.03) ** 2))
         + np.exp(-0.5 * (((gx - 1.00) / 0.03) ** 2 + ((gy - 0.80) / 0.03) ** 2)))
    d /= np.sum(d * (gx0[1] - gx0[0]) * (gy0[1] - gy0[0]))
    labels, comps, grid_x, grid_y = analyse_posterior_2d(
        gx, gy, d, period=(None, None), credible_level=0.99)
    assert len(comps) == 2
    masks_2d = [{"idx": (7, 8), "labels": labels, "grid_x": grid_x,
                 "grid_y": grid_y, "components": comps}]
    intervals_1d = {0: [[5.20, 5.30], [5.50, 5.60]]}
    prior_box = {k: [0.0, 1.0] for k in _KEYS}
    prior_box["logMchirp"] = [5.2, 5.6]
    prior_box["q"] = [0.1, 1.0]
    return labels, grid_x, grid_y, masks_2d, intervals_1d, prior_box


def _truth(**kw):
    t = np.zeros(11)
    t[0] = 5.25
    t[1] = 0.5
    t[7], t[8] = 0.20, 0.20
    for k, v in kw.items():
        t[_KEYS.index(k)] = v
    return t


def test_tv_clean(tv_fixture):
    _, _, _, masks_2d, intervals_1d, prior_box = tv_fixture
    assert truth_violations(_truth(), prior_box, intervals_1d, masks_2d, _KEYS,
                            check_idxs=[0, 1, 7, 8]) == []


def test_tv_hole_between_components(tv_fixture):
    _, _, _, masks_2d, intervals_1d, prior_box = tv_fixture
    v = truth_violations(_truth(**{"lambda": 0.5, "beta": 0.5}), prior_box,
                         intervals_1d, masks_2d, _KEYS, check_idxs=[0, 1, 7, 8])
    assert len(v) == 1
    assert v[0]["kind"] == "2d-mask"
    assert v[0]["marginal"] == [7, 8]
    assert v[0]["name"] == "lambda-beta"


def test_tv_offgrid_not_clipped(tv_fixture):
    labels, grid_x, grid_y, masks_2d, intervals_1d, prior_box = tv_fixture
    v = truth_violations(_truth(**{"lambda": 1.30, "beta": 0.80}), prior_box,
                         intervals_1d, masks_2d, _KEYS, check_idxs=[0, 1, 7, 8])
    assert len(v) == 1
    assert "off the posterior grid" in v[0]["detail"]
    # the clipping trap is real: the snapped-to pixel is inside the mask
    col_clipped = min(int(round((1.30 - grid_x[0]) / (grid_x[1] - grid_x[0]))),
                      labels.shape[1] - 1)
    row = int(round((0.80 - grid_y[0]) / (grid_y[1] - grid_y[0])))
    assert labels[row, col_clipped] > 0


def test_tv_1d_between_intervals(tv_fixture):
    _, _, _, masks_2d, intervals_1d, prior_box = tv_fixture
    v = truth_violations(_truth(logMchirp=5.40), prior_box, intervals_1d,
                         masks_2d, _KEYS, check_idxs=[0, 1, 7, 8])
    assert len(v) == 1 and v[0]["kind"] == "1d-mask"


def test_tv_box_untruncated(tv_fixture):
    _, _, _, masks_2d, intervals_1d, prior_box = tv_fixture
    v = truth_violations(_truth(q=1.5), prior_box, intervals_1d, masks_2d, _KEYS,
                         check_idxs=[0, 1, 7, 8])
    assert len(v) == 1 and v[0]["kind"] == "box"


def test_tv_masked_not_also_box_checked(tv_fixture):
    _, _, _, masks_2d, intervals_1d, prior_box = tv_fixture
    v = truth_violations(_truth(logMchirp=5.45), prior_box, intervals_1d,
                         masks_2d, _KEYS, check_idxs=[0, 1, 7, 8])
    assert len(v) == 1 and v[0]["kind"] == "1d-mask"


def test_tv_check_idxs_limits_box(tv_fixture):
    _, _, _, masks_2d, intervals_1d, prior_box = tv_fixture
    v = truth_violations(_truth(q=1.5), prior_box, intervals_1d, masks_2d, _KEYS,
                         check_idxs=[0, 7, 8])   # q omitted
    assert v == []


def test_tv_multiple_misses(tv_fixture):
    _, _, _, masks_2d, intervals_1d, prior_box = tv_fixture
    v = truth_violations(_truth(logMchirp=5.40, q=1.5, **{"lambda": 0.5, "beta": 0.5}),
                         prior_box, intervals_1d, masks_2d, _KEYS,
                         check_idxs=[0, 1, 7, 8])
    assert len(v) == 3
    assert isinstance(format_violations(v), str) and format_violations(v)


# ======================================================================
# save_truncation / load_truncation
# ======================================================================
def test_persistence_roundtrip():
    tmp = tempfile.mkdtemp()
    try:
        yaml_path = os.path.join(tmp, "prior_after_round_3.yaml")
        npz_path = os.path.join(tmp, "truncation_round_3.npz")

        g1 = np.linspace(0, 1, 200).reshape(-1, 1)
        d1 = (np.exp(-0.5 * ((g1[:, 0] - 0.25) / 0.02) ** 2)
              + np.exp(-0.5 * ((g1[:, 0] - 0.75) / 0.02) ** 2))
        d1 /= np.sum(d1 * (g1[1, 0] - g1[0, 0]))
        res1 = analyse_posterior_1d(g1, d1, credible_level=0.99, dilation_factor=1.1)
        assert len(res1["intervals"]) == 2

        gx0 = np.linspace(0, TWO_PI, 80)
        gy0 = np.linspace(-1, 1, 50)
        gx, gy = np.meshgrid(gx0, gy0, indexing="xy")
        d2 = np.exp(6.0 * np.cos(gx)) * np.exp(-0.5 * (gy / 0.25) ** 2)
        d2 /= np.sum(d2 * (gx0[1] - gx0[0]) * (gy0[1] - gy0[0]))
        labels, comps, grid_x, grid_y = analyse_posterior_2d(
            gx, gy, d2, period=(TWO_PI, None), credible_level=0.99)
        assert len(comps[next(iter(comps))]["x_intervals"]) == 2

        prior_box = {"logMchirp": [5.2, 5.6], "q": [0.1, 1.0], "lambda": [0.0, TWO_PI]}
        intervals_1d = {0: res1["intervals"]}
        from pembhb.regions import Region as _Region
        masks_2d = [{"idx": (7, 8),
                     "region": _Region(labels > 0, (grid_x, grid_y), (True, False)),
                     "labels": labels, "grid_x": grid_x,
                     "grid_y": grid_y, "components": comps}]

        save_truncation(yaml_path, npz_path, prior_box, intervals_1d, masks_2d, mode="mask")
        assert os.path.exists(yaml_path) and os.path.exists(npz_path)

        out = load_truncation(yaml_path, npz_path)
        assert out["mode"] == "mask"
        assert out["prior"] == {k: [float(a), float(b)] for k, (a, b) in prior_box.items()}
        assert np.allclose(np.array(out["intervals_1d"][0]), np.array(res1["intervals"]))
        assert type(next(iter(out["intervals_1d"]))) is int

        m = out["masks_2d"][0]
        assert m["idx"] == (7, 8)
        region = m["region"]
        assert len(region.parts) == 1
        assert region.parts[0].mask.shape == (50, 80)     # not transposed
        assert np.array_equal(region.parts[0].mask, labels > 0)
        assert np.allclose(region.parts[0].grids[0], grid_x)
        assert np.allclose(region.parts[0].grids[1], grid_y)

        # periodicity round-trips, so the wrapped lambda mode stays ONE
        # component with two x sub-intervals rather than splitting at the seam
        assert region.periodic == (True, False)
        assert len(region.components()) == 1
        assert len(region.intervals(0)) == 2
    finally:
        shutil.rmtree(tmp)


def test_persistence_missing_npz_raises():
    tmp = tempfile.mkdtemp()
    try:
        yaml_path = os.path.join(tmp, "p.yaml")
        npz_path = os.path.join(tmp, "t.npz")
        gx0 = np.linspace(0, 1, 40); gy0 = np.linspace(0, 1, 40)
        gx, gy = np.meshgrid(gx0, gy0, indexing="xy")
        d = np.exp(-0.5 * (((gx - 0.5) / 0.05) ** 2 + ((gy - 0.5) / 0.05) ** 2))
        d /= np.sum(d * (gx0[1] - gx0[0]) * (gy0[1] - gy0[0]))
        labels, comps, grid_x, grid_y = analyse_posterior_2d(gx, gy, d, period=(None, None))
        from pembhb.regions import Region as _Region
        masks_2d = [{"idx": (7, 8),
                     "region": _Region(labels > 0, (grid_x, grid_y), (True, False)),
                     "labels": labels, "grid_x": grid_x,
                     "grid_y": grid_y, "components": comps}]
        save_truncation(yaml_path, npz_path, {"logMchirp": [5.2, 5.6]}, {}, masks_2d, mode="mask")
        os.remove(npz_path)
        with pytest.raises(RuntimeError):
            load_truncation(yaml_path, npz_path)
    finally:
        shutil.rmtree(tmp)


def test_persistence_legacy_rectangle_yaml():
    tmp = tempfile.mkdtemp()
    try:
        import yaml as _yaml
        legacy = os.path.join(tmp, "legacy.yaml")
        prior_box = {"logMchirp": [5.2, 5.6], "q": [0.1, 1.0]}
        with open(legacy, "w") as f:
            _yaml.safe_dump({"prior": {k: [float(a), float(b)]
                                       for k, (a, b) in prior_box.items()}}, f)
        out = load_truncation(legacy, os.path.join(tmp, "nonexistent.npz"))
        assert out["mode"] == "rectangle"
        assert out["masks_2d"] == [] and out["intervals_1d"] == {}
        assert "logMchirp" in out["prior"]
    finally:
        shutil.rmtree(tmp)


# ======================================================================
# MaskRejectSampler — chieff_chidiff post-overwrite spin rejection
# ======================================================================
def test_chieff_no_unphysical_spins(float64_precision):
    from pembhb.sampler import chieff_chidiff_to_chi12
    keys = utils.ordered_prior_keys("chieff_chidiff")
    assert keys[2:4] == ["chi_eff", "chi_diff"]
    I_Q, I_CHIEFF, I_CHIDIFF = 1, 2, 3
    prior = {k: [0.0, 1.0] for k in keys}
    prior[keys[0]] = [5.0, 6.0]
    prior[keys[I_Q]] = [0.1, 1.0]
    prior[keys[I_CHIEFF]] = [-1.0, 1.0]
    prior[keys[I_CHIDIFF]] = [-1.0, 1.0]
    prior["dist"] = [5.0, 50.0]
    prior["lambda"] = [0.0, TWO_PI]
    prior["beta"] = [-1.0, 1.0]
    T_OBS = 365 * 24 * 3600.0
    N = 20000

    def make(intervals_1d, seed):
        return MaskRejectSampler(
            prior_bounds=prior, intervals_1d=intervals_1d, masks_2d=[],
            rng=np.random.default_rng([42, seed]), dist_uniform_in_volume=True,
            spin_param_basis="chieff_chidiff")

    # 1) intervals straddling the physical boundary
    intervals = {I_CHIEFF: [[0.35, 0.65]], I_CHIDIFF: [[0.30, 0.60]]}
    s = make(intervals, 1)
    _, tm = s.sample(N, T_OBS)
    assert tm.shape == (11, N)
    chi1, chi2 = chieff_chidiff_to_chi12(tm[I_Q], tm[I_CHIEFF], tm[I_CHIDIFF])
    assert not ((np.abs(chi1) > 1.0) | (np.abs(chi2) > 1.0)).any()
    assert 0.0 < s.last_acceptance_ratio < 1.0
    for idx in (I_CHIEFF, I_CHIDIFF):
        lo, hi = intervals[idx][0]
        assert ((tm[idx] >= lo) & (tm[idx] <= hi)).all()
    assert tm[0].min() < 5.05 and tm[0].max() > 5.95

    # control: the pre-rejection draw IS invalid (the bug is real)
    raw = s._draw_once(N, T_OBS)
    c1, c2 = chieff_chidiff_to_chi12(raw[I_Q], raw[I_CHIEFF], raw[I_CHIDIFF])
    assert ((np.abs(c1) > 1.0) | (np.abs(c2) > 1.0)).any()


def test_chieff_physical_intervals_full_acceptance(float64_precision):
    from pembhb.sampler import chieff_chidiff_to_chi12
    keys = utils.ordered_prior_keys("chieff_chidiff")
    prior = {k: [0.0, 1.0] for k in keys}
    prior[keys[0]] = [5.0, 6.0]; prior[keys[1]] = [0.1, 1.0]
    prior[keys[2]] = [-1.0, 1.0]; prior[keys[3]] = [-1.0, 1.0]
    prior["dist"] = [5.0, 50.0]; prior["lambda"] = [0.0, TWO_PI]; prior["beta"] = [-1.0, 1.0]
    s = MaskRejectSampler(prior_bounds=prior,
                          intervals_1d={2: [[-0.2, 0.2]], 3: [[-0.1, 0.1]]}, masks_2d=[],
                          rng=np.random.default_rng([42, 3]), dist_uniform_in_volume=True,
                          spin_param_basis="chieff_chidiff")
    _, tm = s.sample(20000, 365 * 24 * 3600.0)
    c1, c2 = chieff_chidiff_to_chi12(tm[1], tm[2], tm[3])
    assert not ((np.abs(c1) > 1) | (np.abs(c2) > 1)).any()
    assert s.last_acceptance_ratio == 1.0


def test_chieff_entirely_unphysical_raises(float64_precision):
    keys = utils.ordered_prior_keys("chieff_chidiff")
    prior = {k: [0.0, 1.0] for k in keys}
    prior[keys[0]] = [5.0, 6.0]; prior[keys[1]] = [0.1, 1.0]
    prior[keys[2]] = [-1.0, 1.0]; prior[keys[3]] = [-1.0, 1.0]
    prior["dist"] = [5.0, 50.0]; prior["lambda"] = [0.0, TWO_PI]; prior["beta"] = [-1.0, 1.0]
    s = MaskRejectSampler(prior_bounds=prior,
                          intervals_1d={2: [[0.95, 1.0]], 3: [[0.95, 1.0]]}, masks_2d=[],
                          rng=np.random.default_rng([42, 5]), dist_uniform_in_volume=True,
                          spin_param_basis="chieff_chidiff")
    with pytest.raises(RuntimeError):
        s.sample(2000, 365 * 24 * 3600.0)


# ======================================================================
# End-to-end wiring: analyse -> save -> sample -> truth check -> resume
# ======================================================================
def test_wiring_roundend_and_resume(float64_precision):
    prior_keys = utils.ordered_prior_keys("chi1chi2")
    IDX_MC = prior_keys.index("logMchirp")
    IDX_LAM, IDX_BETA = prior_keys.index("lambda"), prior_keys.index("beta")
    assert (IDX_LAM, IDX_BETA) == (7, 8)

    prior = {k: [0.0, 1.0] for k in prior_keys}
    prior["logMchirp"] = [5.0, 6.0]
    prior["lambda"] = [0.0, TWO_PI]
    prior["beta"] = [-1.0, 1.0]
    prior["dist"] = [5.0, 50.0]

    # 1D bimodal logMchirp
    g = np.linspace(*prior["logMchirp"], 100)
    d = np.exp(-0.5 * ((g - 5.3) / 0.02) ** 2) + np.exp(-0.5 * ((g - 5.7) / 0.02) ** 2)
    d /= np.sum(d * (g[1] - g[0]))
    res = analyse_posterior_1d(g.reshape(-1, 1), d, credible_level=0.997,
                               dilation_factor=1.2, period=None)
    assert len(res["intervals"]) == 2
    intervals_1d = {IDX_MC: res["intervals"]}

    # 2D sky marginal straddling the lambda seam
    gx0 = np.linspace(0.0, TWO_PI, 80); gy0 = np.linspace(-np.pi / 2, np.pi / 2, 80)
    gx, gy = np.meshgrid(gx0, gy0, indexing="xy")
    d2 = sum(np.exp(-0.5 * (((gx - cx) / 0.12) ** 2 + ((gy - cy) / 0.12) ** 2))
             for cx, cy in [(0.15, 0.3), (TWO_PI - 0.15, 0.3)])
    d2 /= np.sum(d2 * (gx0[1] - gx0[0]) * (gy0[1] - gy0[0]))
    labels, comps, grid_x, grid_y = analyse_posterior_2d(
        gx, gy, d2, period=(TWO_PI, None), credible_level=0.997, dilation_factor=1.2)
    assert len(comps) == 1
    masks_2d = [{"idx": (IDX_LAM, IDX_BETA), "labels": labels,
                 "grid_x": grid_x, "grid_y": grid_y, "components": comps}]

    def envelope(ivs):
        return [min(lo for lo, _ in ivs), max(hi for _, hi in ivs)]

    prior["logMchirp"] = envelope(res["intervals"])
    allx = [iv for c in comps.values() for iv in c["x_intervals"]]
    ally = [iv for c in comps.values() for iv in c["y_intervals"]]
    prior["lambda"], prior["beta"] = envelope(allx), envelope(ally)

    tmp = tempfile.mkdtemp()
    try:
        ypath = os.path.join(tmp, "prior_after_round_1.yaml")
        npath = os.path.join(tmp, "truncation_round_1.npz")
        save_truncation(ypath, npath, prior, intervals_1d, masks_2d, mode="mask")
        assert os.path.exists(ypath) and os.path.exists(npath)

        sampler = MaskRejectSampler(
            prior_bounds=prior, intervals_1d=intervals_1d, masks_2d=masks_2d,
            rng=np.random.default_rng([42, 1]),
            dist_uniform_in_volume=True, spin_param_basis="chi1chi2")
        N = 20000
        _, tm = sampler.sample(N, t_obs_end=365 * 24 * 3600.0)
        assert tm.shape == (11, N)

        inside_1d = np.zeros(N, bool)
        for lo, hi in intervals_1d[IDX_MC]:
            inside_1d |= (tm[IDX_MC] >= lo) & (tm[IDX_MC] <= hi)
        assert inside_1d.all()
        # One membership rule: the sampler indexes from the cell vertex so its
        # floor() agrees with Region.contains' nearest-centre round().
        from pembhb.regions import Region as _Region
        _reg = _Region(labels > 0, (grid_x, grid_y))
        assert _reg.contains(np.stack([tm[IDX_LAM], tm[IDX_BETA]])).all()
        assert tm[1].min() < 0.05 and tm[1].max() > 0.95
        assert abs(tm[4].mean() - 38.1) < 1.0     # dist volume-uniform

        truncated_idxs = {IDX_MC, IDX_LAM, IDX_BETA, 1}
        good = np.zeros(11)
        good[IDX_MC], good[IDX_LAM], good[IDX_BETA], good[1] = 5.30, 0.15, 0.30, 0.5
        good[4] = 20.0
        assert truth_violations(good, prior, intervals_1d, masks_2d, prior_keys,
                                check_idxs=truncated_idxs) == []
        bad = good.copy(); bad[IDX_MC] = 5.50; bad[IDX_LAM] = np.pi
        v = truth_violations(bad, prior, intervals_1d, masks_2d, prior_keys,
                             check_idxs=truncated_idxs)
        assert len(v) == 2
        assert sorted(x["kind"] for x in v) == ["1d-mask", "2d-mask"]

        # resume: load + rebuild reproduces the stream bit-for-bit
        loaded = load_truncation(ypath, npath)
        assert loaded["mode"] == "mask"
        s2 = MaskRejectSampler(
            prior_bounds=loaded["prior"], intervals_1d=loaded["intervals_1d"],
            masks_2d=loaded["masks_2d"], rng=np.random.default_rng([42, 1]),
            dist_uniform_in_volume=True, spin_param_basis="chi1chi2")
        _, tm2 = s2.sample(N, t_obs_end=365 * 24 * 3600.0)
        assert np.allclose(tm, tm2)

        os.remove(npath)
        with pytest.raises(RuntimeError):
            load_truncation(ypath, npath)
    finally:
        shutil.rmtree(tmp)


# ---------------------------------------------------------------------------
# _hpd_threshold with several grids of different resolution
# ---------------------------------------------------------------------------

def test_hpd_threshold_single_grid_forms_agree():
    from pembhb.mask_truncation import _hpd_threshold
    rng = np.random.default_rng(0)
    d = rng.random((40, 30))
    t = _hpd_threshold(d, 0.9)
    assert _hpd_threshold(d, 0.9, cell_volumes=0.7) == t
    assert _hpd_threshold([d], 0.9, cell_volumes=[0.7]) == t


def test_hpd_threshold_pools_mass_across_resolutions():
    # Gaussian at x=1 split at 0: coarse cells on the left, 4x finer on the right.
    # Off-centre so the two grids hold different mass (a centred split hides the bug).
    from pembhb.mask_truncation import _hpd_threshold
    dx_c, dx_f = 0.1, 0.025
    xc = np.arange(-6.0, 0.0, dx_c) + dx_c / 2
    xf = np.arange(0.0, 7.0, dx_f) + dx_f / 2
    pc, pf = np.exp(-0.5 * (xc - 1) ** 2), np.exp(-0.5 * (xf - 1) ** 2)

    t = _hpd_threshold([pc, pf], 0.99, cell_volumes=[dx_c, dx_f])
    left, right = 1 - xc[pc >= t].min(), xf[pf >= t].max() - 1
    assert abs(left - right) < dx_c                     # same cut on both sides
    assert abs(right - 2.576) < dx_c                    # 99% two-sided Gaussian

    t_unweighted = _hpd_threshold([pc, pf], 0.99)       # fine side over-counted 4x
    assert t_unweighted != t


def test_physical_bounds_hold_when_a_mode_saturates_the_box_edge():
    """cos(iota) and sin(beta) can never be proposed outside [-1, 1].

    Two changes could in principle have broken this: intervals now report cell
    EDGES (half a cell wider than the centres), and the 2D sampler now indexes
    from the cell vertex. Neither can: `_intervals_from_indices` clamps the edge
    padding to the grid ends, and the grid spans exactly the prior box, so the
    drawn range is bounded by the intervals. The vertex change only decides
    which draws are REJECTED, never the range they come from.
    """
    from pembhb.regions import Region, region_from_hpd
    rng = np.random.default_rng(0)

    # face-on/face-off: density piled against both cos(iota) edges, dilation on
    g = np.linspace(-1.0, 1.0, 100)
    d = np.exp(-0.5 * ((g - 1.0) / 0.05) ** 2) + np.exp(-0.5 * ((g + 1.0) / 0.05) ** 2)
    r = region_from_hpd(d / d.sum(), (g,), credible_level=0.9999, dilation_factor=1.5)
    ivs = r.intervals(0)
    assert min(lo for lo, _ in ivs) >= -1.0
    assert max(hi for _, hi in ivs) <= 1.0
    x = _sample_from_intervals(ivs, 100_000, rng)
    assert np.all(np.abs(x) <= 1.0)

    # a sky mode pressed against sin(beta) = +1
    gx, gy = np.linspace(0.0, TWO_PI, 80), np.linspace(-1.0, 1.0, 80)
    MX, MY = np.meshgrid(gx, gy, indexing="xy")
    d2 = np.exp(-0.5 * (((MX - 3.0) / 0.3) ** 2 + ((MY - 1.0) / 0.08) ** 2))
    r2 = region_from_hpd(d2 / d2.sum(), (gx, gy), credible_level=0.999,
                         dilation_factor=1.5, periods=(TWO_PI, None))
    lab = r2.labels()
    comps = components_from_labels(lab, grid_x=gx, grid_y=gy)
    X, Y, _acc = _sample_2d_from_components(comps, lab, gx, gy, 100_000, rng)
    assert np.all(np.abs(Y) <= 1.0)
    assert X.min() >= 0.0 and X.max() <= TWO_PI
    assert r2.contains(np.stack([X, Y])).all()


def test_dist_in_a_2d_pair_is_sampled_uniform_in_d(float64_precision):
    """A volumetric axis inside a 2D marginal loses p(d) ~ d^2 — by decision.

    `_sample_2d_from_components` draws both axes uniformly over the mask, so
    pairing `dist` with another parameter gives uniform-in-d, NOT the
    uniform-in-volume prior the 1D path applies via `cube=True`. That is what we
    want for the paired-marginal experiment; this test pins it so a future
    volumetric fix cannot change it silently.
    """
    from scipy.stats import kstest
    from pembhb.regions import Region

    lo, hi = 1.0, 100.0
    g_inc = np.linspace(-1.0, 1.0, 60)
    g_dist = np.linspace(lo, hi, 60)
    region = Region(np.ones((60, 60), dtype=bool), (g_inc, g_dist))

    rng = np.random.default_rng(0)
    x, d = region.draw(100_000, rng)

    assert region.contains(np.stack([x, d])).all()          # §2.5 invariant

    # Uniform in d, not in d**3. The support is exactly [lo, hi]: the mode's
    # interval is padded to cell edges but then CLAMPED to the grid ends
    # (_intervals_from_indices), so it cannot spill past the prior box.
    assert d.min() >= lo and d.max() <= hi
    assert kstest(d, "uniform", args=(lo, hi - lo)).pvalue > 0.01
    assert abs(d.mean() - 0.5 * (lo + hi)) < 1.0

    # and decisively NOT the volumetric prior the 1D path would give
    volumetric_mean = 0.75 * (hi**4 - lo**4) / (hi**3 - lo**3)
    assert abs(d.mean() - volumetric_mean) > 10.0


def test_1d_dist_still_gets_the_volumetric_prior(float64_precision):
    """The 1D path is unchanged: dist alone is still uniform in volume."""
    rng = np.random.default_rng(1)
    lo, hi = 1.0, 100.0
    d = _sample_from_intervals([[lo, hi]], 100_000, rng, cube=True)
    volumetric_mean = 0.75 * (hi**4 - lo**4) / (hi**3 - lo**3)
    assert abs(d.mean() - volumetric_mean) < 1.0
    assert d.mean() > 0.5 * (lo + hi) + 10.0          # skewed to large d
