"""Region class + builders (src/pembhb/regions.py).

Covers the primitive (`contains`), the §2 resampler (`contains_grid`), the
measures (`volume_fraction`, `bounds`, `intervals`, `components`), sampling and
persistence, and — crucially — equivalence with the scattered implementations
the refactor unifies:

  * region_from_hpd          reproduces analyse_posterior_1d/2d masks exactly
  * region_from_equal_tailed ⊇ get_widest_interval_1d, ≤ 1 px per edge
  * region_from_bounds       round-trips a box within 1 px
"""
import numpy as np
import pytest

from pembhb.regions import (
    Region, region_from_hpd, region_from_equal_tailed,
    region_from_main_mode, region_from_bounds,
)
from pembhb.mask_truncation import analyse_posterior_1d, analyse_posterior_2d

TWO_PI = 2 * np.pi


# ----------------------------------------------------------------------
# fixtures / helpers
# ----------------------------------------------------------------------
def _grid(lo, hi, n):
    return np.linspace(lo, hi, n)


def _norm1d(d, g):
    return d / np.sum(d * (g[1] - g[0]))


def _mesh(x_lo, x_hi, y_lo, y_hi, nx, ny):
    g0 = _grid(x_lo, x_hi, nx)
    g1 = _grid(y_lo, y_hi, ny)
    gx, gy = np.meshgrid(g0, g1, indexing="xy")   # (ny, nx)
    return g0, g1, gx, gy


def _norm2d(d, gx, gy):
    return d / np.sum(d * (gx[0, 1] - gx[0, 0]) * (gy[1, 0] - gy[0, 0]))


# ----------------------------------------------------------------------
# Region primitives
# ----------------------------------------------------------------------
def test_region_1d_shapes_and_contains():
    g = _grid(0, 1, 100)
    mask = (g >= 0.3) & (g <= 0.6)
    r = Region(mask, (g,))
    assert r.ndim == 1
    # inside vs outside
    assert r.contains(np.array([0.45])).tolist() == [True]
    assert r.contains(np.array([0.1])).tolist() == [False]
    # out of grid -> not contained (never clipped onto an edge)
    assert r.contains(np.array([2.0])).tolist() == [False]
    assert r.contains(np.array([-5.0])).tolist() == [False]


def test_region_2d_shape_validation():
    g0, g1, gx, gy = _mesh(0, 1, 0, 1, nx=80, ny=50)
    mask = np.zeros((50, 80), bool)
    mask[10:20, 30:40] = True
    r = Region(mask, (g0, g1))
    assert r.mask.shape == (50, 80)      # (n_y, n_x)
    # a transposed mask must be rejected
    with pytest.raises(ValueError):
        Region(mask.T, (g0, g1))


def test_region_2d_contains_and_offgrid():
    g0, g1, gx, gy = _mesh(0, 1, 0, 1, nx=80, ny=50)
    mask = np.zeros((50, 80), bool)
    # a rectangle x in [0.3,0.6], y in [0.2,0.4]
    xin = (g0 >= 0.3) & (g0 <= 0.6)
    yin = (g1 >= 0.2) & (g1 <= 0.4)
    mask[np.ix_(yin, xin)] = True
    r = Region(mask, (g0, g1))
    assert r.contains(np.array([[0.45], [0.3]]))[0]         # inside
    assert not r.contains(np.array([[0.45], [0.9]]))[0]     # outside in y
    assert not r.contains(np.array([[1.5], [0.3]]))[0]      # off grid in x


def test_volume_fraction():
    g0, g1, gx, gy = _mesh(0, 1, 0, 1, nx=100, ny=100)
    mask = np.zeros((100, 100), bool)
    mask[:50, :20] = True                # 50*20 / (100*100) = 0.10
    r = Region(mask, (g0, g1))
    assert r.volume_fraction() == pytest.approx(0.10)


def test_bounds_1d_and_2d():
    g = _grid(0, 1, 101)
    mask = (g >= 0.30) & (g <= 0.50)
    r = Region(mask, (g,))
    lo, hi = r.bounds()
    assert lo == pytest.approx(0.30, abs=0.011) and hi == pytest.approx(0.50, abs=0.011)

    g0, g1, gx, gy = _mesh(0, 1, 0, 1, nx=101, ny=101)
    mask2 = np.zeros((101, 101), bool)
    xin = (g0 >= 0.2) & (g0 <= 0.7)
    yin = (g1 >= 0.1) & (g1 <= 0.3)
    mask2[np.ix_(yin, xin)] = True
    (xlo, xhi), (ylo, yhi) = Region(mask2, (g0, g1)).bounds()
    assert xlo == pytest.approx(0.2, abs=0.011) and xhi == pytest.approx(0.7, abs=0.011)
    assert ylo == pytest.approx(0.1, abs=0.011) and yhi == pytest.approx(0.3, abs=0.011)


def test_components_and_intervals_1d():
    g = _grid(0, 1, 200)
    d = (np.exp(-0.5 * ((g - 0.25) / 0.02) ** 2)
         + np.exp(-0.5 * ((g - 0.75) / 0.02) ** 2))
    r = region_from_hpd(_norm1d(d, g), (g,), credible_level=0.99, dilation_factor=1.1)
    assert len(r.components()) == 2
    assert len(r.intervals(0)) == 2


def test_intervals_wrapped_1d():
    g = _grid(0, TWO_PI, 100)
    d = np.exp(4.0 * np.cos(g))          # peak at 0 == 2pi
    r = region_from_hpd(_norm1d(d, g), (g,), credible_level=0.99,
                        dilation_factor=1.0, periods=(TWO_PI,))
    ivs = r.intervals(0)
    assert len(ivs) == 2                 # wraps -> two sub-intervals
    assert ivs[0][0] == pytest.approx(0.0)
    assert ivs[-1][1] == pytest.approx(TWO_PI)


# ----------------------------------------------------------------------
# contains_grid — the §2 resampler
# ----------------------------------------------------------------------
def test_contains_grid_1d_resample():
    g_old = _grid(-1, 1, 100)
    mask = (g_old >= -0.5) & (g_old <= 0.5)
    r = Region(mask, (g_old,))
    g_new = _grid(-1, 1, 250)            # finer, different range partitioning
    keep = r.contains_grid((g_new,))
    assert keep.shape == (250,)
    # a point known inside / outside
    assert keep[np.argmin(np.abs(g_new - 0.0))]
    assert not keep[np.argmin(np.abs(g_new - 0.9))]


def test_contains_grid_2d_resample_shape():
    g0, g1, gx, gy = _mesh(0, 1, 0, 1, nx=80, ny=50)
    mask = np.zeros((50, 80), bool)
    xin = (g0 >= 0.3) & (g0 <= 0.6)
    yin = (g1 >= 0.2) & (g1 <= 0.5)
    mask[np.ix_(yin, xin)] = True
    r = Region(mask, (g0, g1))
    ng0 = _grid(0.2, 0.7, 40)
    ng1 = _grid(0.1, 0.6, 33)
    keep = r.contains_grid((ng0, ng1))
    assert keep.shape == (33, 40)        # (m_y, m_x)
    # centre of the accepted rectangle is kept
    ix = np.argmin(np.abs(ng0 - 0.45)); iy = np.argmin(np.abs(ng1 - 0.35))
    assert keep[iy, ix]


# ----------------------------------------------------------------------
# equivalence: region_from_hpd == analyse_posterior_*
# ----------------------------------------------------------------------
def test_hpd_1d_reproduces_analyse():
    g = _grid(0, 1, 120)
    d = _norm1d(np.exp(-0.5 * ((g - 0.5) / 0.05) ** 2), g)
    res = analyse_posterior_1d(g.reshape(-1, 1), d, credible_level=0.99,
                               dilation_factor=1.2, period=None)
    r = region_from_hpd(d, (g,), credible_level=0.99, dilation_factor=1.2,
                        periods=(None,))
    assert np.array_equal(r.mask, res["mask"])


def test_hpd_2d_reproduces_analyse():
    g0, g1, gx, gy = _mesh(0, TWO_PI, -1, 1, nx=80, ny=50)
    d = _norm2d(np.exp(4.0 * np.cos(gx)) * np.exp(-0.5 * (gy / 0.25) ** 2), gx, gy)
    labels, comps, grid_x, grid_y = analyse_posterior_2d(
        gx, gy, d, period=(TWO_PI, None), credible_level=0.99, dilation_factor=1.1)
    r = region_from_hpd(d, (g0, g1), credible_level=0.99, dilation_factor=1.1,
                        periods=(TWO_PI, None))
    assert np.array_equal(r.mask, labels > 0)
    # intervals reproduce the per-component projection (union over components)
    ref_x = [iv for c in comps.values() for iv in c["x_intervals"]]
    got_x = r.intervals(0)
    assert len(got_x) == len(ref_x)


# ----------------------------------------------------------------------
# equivalence: region_from_equal_tailed ⊇ get_widest_interval_1d, ≤1px
# ----------------------------------------------------------------------
def _widest_interval_reference(norm1d, grid, eps, dilation):
    """The exact interval maths of utils.get_widest_interval_1d, inlined so the
    equivalence can be checked without a trained model."""
    dp = grid[1] - grid[0]
    cumsum = np.cumsum(norm1d * dp)
    idx_low = int(np.searchsorted(cumsum, eps / 2))
    idx_high = min(int(np.searchsorted(cumsum, 1 - eps / 2)), len(grid) - 1)
    lo, hi = float(grid[idx_low]), float(grid[idx_high])
    if dilation != 1.0:
        c = 0.5 * (lo + hi); half = 0.5 * (hi - lo) * dilation
        lo = max(c - half, float(grid[0]))
        hi = min(c + half, float(grid[-1]))
    return lo, hi


@pytest.mark.parametrize("eps,dil", [(1e-4, 1.0), (1e-3, 1.0), (1e-4, 1.1), (1e-3, 1.2)])
def test_equal_tailed_superset_of_widest_interval(eps, dil):
    g = _grid(5.0, 6.0, 100)
    d = _norm1d(np.exp(-0.5 * ((g - 5.5) / 0.08) ** 2), g)
    lo_ref, hi_ref = _widest_interval_reference(d, g, eps, dil)
    r = region_from_equal_tailed(d, (g,), eps=eps, dilation=dil)
    lo_new, hi_new = r.bounds()
    dp = g[1] - g[0]
    # superset: the new region contains the old interval ...
    assert lo_new <= lo_ref + 1e-12
    assert hi_new >= hi_ref - 1e-12
    # ... by at most one pixel per edge
    assert (lo_ref - lo_new) <= dp + 1e-9
    assert (hi_new - hi_ref) <= dp + 1e-9


def test_equal_tailed_no_dilation_is_exact():
    g = _grid(5.0, 6.0, 100)
    d = _norm1d(np.exp(-0.5 * ((g - 5.5) / 0.08) ** 2), g)
    lo_ref, hi_ref = _widest_interval_reference(d, g, 1e-4, 1.0)
    lo_new, hi_new = region_from_equal_tailed(d, (g,), eps=1e-4, dilation=1.0).bounds()
    assert lo_new == pytest.approx(lo_ref)
    assert hi_new == pytest.approx(hi_ref)


# ----------------------------------------------------------------------
# region_from_bounds / region_from_main_mode
# ----------------------------------------------------------------------
def test_region_from_bounds_1d_and_2d():
    g = _grid(0, 1, 101)
    r = region_from_bounds([0.3, 0.6], (g,))
    lo, hi = r.bounds()
    assert lo <= 0.3 and hi >= 0.6                 # outward superset
    assert (0.3 - lo) <= (g[1] - g[0]) + 1e-9

    g0, g1, gx, gy = _mesh(0, 1, 0, 1, nx=101, ny=81)
    r2 = region_from_bounds([[0.2, 0.7], [0.1, 0.4]], (g0, g1))
    assert r2.mask.shape == (81, 101)
    (xlo, xhi), (ylo, yhi) = r2.bounds()
    assert xlo <= 0.2 and xhi >= 0.7 and ylo <= 0.1 and yhi >= 0.4


def test_main_mode_selects_by_mass_not_area():
    # Broad shallow blob (LARGE area, small mass) at 0.2 vs narrow tall blob
    # (small area, LARGE mass) at 0.8.  Area-based selection would pick 0.2;
    # mass-based selection must pick 0.8.
    g0, g1, gx, gy = _mesh(0, 1, 0, 1, nx=120, ny=120)
    broad = 1.0 * np.exp(-0.5 * (((gx - 0.2) / 0.10) ** 2 + ((gy - 0.2) / 0.10) ** 2))
    tall = 12.0 * np.exp(-0.5 * (((gx - 0.8) / 0.045) ** 2 + ((gy - 0.8) / 0.045) ** 2))
    d = _norm2d(broad + tall, gx, gy)
    full = region_from_hpd(d, (g0, g1), credible_level=0.99, dilation_factor=1.0)
    comps = full.components()
    assert len(comps) == 2

    def cx(r):
        (xlo, xhi), _ = r.bounds()
        return 0.5 * (xlo + xhi)

    near02 = min(comps, key=lambda r: abs(cx(r) - 0.2))
    near08 = min(comps, key=lambda r: abs(cx(r) - 0.8))
    # the broad blob genuinely has the larger AREA (so area-based would pick it)
    assert near02.volume_fraction() > near08.volume_fraction()

    main = region_from_main_mode(d, (g0, g1), credible_level=0.99, dilation_factor=1.0)
    assert len(main.components()) == 1
    assert cx(main) > 0.6           # mass-based -> the tall blob at 0.8

    # a plain area-based main_component picks the OTHER (broad) blob, proving
    # the selection criterion actually changed the answer
    assert cx(full.main_component()) < 0.4


# ----------------------------------------------------------------------
# persistence
# ----------------------------------------------------------------------
def test_to_from_arrays_roundtrip_1d():
    g = _grid(0, 1, 100)
    mask = (g >= 0.2) & (g <= 0.5)
    r = Region(mask, (g,), periodic=(False,))
    r2 = Region.from_arrays(r.to_arrays())
    assert np.array_equal(r.mask, r2.mask)
    assert np.allclose(r.grids[0], r2.grids[0])


def test_to_from_arrays_roundtrip_2d():
    g0, g1, gx, gy = _mesh(0, TWO_PI, -1, 1, nx=80, ny=50)
    d = _norm2d(np.exp(4.0 * np.cos(gx)) * np.exp(-0.5 * (gy / 0.25) ** 2), gx, gy)
    r = region_from_hpd(d, (g0, g1), credible_level=0.99, dilation_factor=1.1,
                        periods=(TWO_PI, None))
    r2 = Region.from_arrays(r.to_arrays())
    assert np.array_equal(r.mask, r2.mask)
    assert r2.mask.shape == (50, 80)             # not transposed
    assert np.allclose(r.grids[0], r2.grids[0]) and np.allclose(r.grids[1], r2.grids[1])


# ----------------------------------------------------------------------
# draw
# ----------------------------------------------------------------------
def test_draw_1d_lands_inside():
    g = _grid(0, 1, 200)
    d = (np.exp(-0.5 * ((g - 0.25) / 0.02) ** 2)
         + np.exp(-0.5 * ((g - 0.75) / 0.02) ** 2))
    r = region_from_hpd(_norm1d(d, g), (g,), credible_level=0.99, dilation_factor=1.1)
    rng = np.random.default_rng(0)
    s = r.draw(50_000, rng)
    assert r.contains(s).mean() > 0.999


# ----------------------------------------------------------------------
# veto inner-loop semantics (compute_truncation_coverage now builds these)
# ----------------------------------------------------------------------
def test_veto_2d_nonsky_uses_mask_not_bbox():
    # The migrated veto builds region_from_main_mode per 2D sample and tests
    # Region.contains. On a bimodal NON-periodic (non-sky) posterior whose modes
    # sit on a diagonal, the joint bounding box also covers the gap and the OTHER
    # mode; mask membership of the single main mode must count both as NOT
    # covered. (The old veto applied sky-only logic here and mishandled it.)
    g0, g1, gx, gy = _mesh(0, 1, 0, 1, nx=100, ny=100)
    big = 3.0 * np.exp(-0.5 * (((gx - 0.25) / 0.05) ** 2 + ((gy - 0.25) / 0.05) ** 2))
    small = 1.0 * np.exp(-0.5 * (((gx - 0.75) / 0.04) ** 2 + ((gy - 0.75) / 0.04) ** 2))
    d = _norm2d(big + small, gx, gy)
    region = region_from_main_mode(d, (g0, g1), credible_level=0.99,
                                   dilation_factor=1.0, periods=(None, None))
    assert len(region.components()) == 1                        # main mode only
    assert region.contains(np.array([[0.25], [0.25]]))[0]       # main-mode centre
    assert not region.contains(np.array([[0.5], [0.5]]))[0]     # diagonal gap
    assert not region.contains(np.array([[0.75], [0.75]]))[0]   # the OTHER mode


def test_veto_2d_main_mode_periodic_axis():
    # a non-sky-like marginal with a periodic x-axis: region_from_main_mode must
    # accept the (previously sky-only) periodic handling for any 2D pair.
    g0, g1, gx, gy = _mesh(0, TWO_PI, -1, 1, nx=80, ny=50)
    d = _norm2d(np.exp(4.0 * np.cos(gx)) * np.exp(-0.5 * (gy / 0.25) ** 2), gx, gy)
    region = region_from_main_mode(d, (g0, g1), credible_level=0.99,
                                   dilation_factor=1.0, periods=(TWO_PI, None))
    # a truth just past the 2π seam (x≈0.1) is in the wrapped main mode
    assert region.contains(np.array([[0.1], [0.0]]))[0]
    assert region.contains(np.array([[TWO_PI - 0.1], [0.0]]))[0]


def test_draw_2d_lands_inside():
    g0, g1, gx, gy = _mesh(0, 1, 0, 1, nx=100, ny=100)
    d = (np.exp(-0.5 * (((gx - 0.25) / 0.03) ** 2 + ((gy - 0.25) / 0.03) ** 2))
         + np.exp(-0.5 * (((gx - 0.72) / 0.05) ** 2 + ((gy - 0.72) / 0.05) ** 2)))
    r = region_from_hpd(_norm2d(d, gx, gy), (g0, g1), credible_level=0.99,
                        dilation_factor=1.0)
    rng = np.random.default_rng(0)
    x, y = r.draw(50_000, rng)
    # One membership rule for everything: the sampler indexes from the cell
    # vertex so its floor() agrees with Region.contains' round()-to-nearest-
    # centre. (This used to re-implement the sampler's old floor-from-centre
    # arithmetic here, which pinned the test to the half-cell bug it shared.)
    assert r.contains(np.stack([x, y])).all()
    # contains() (nearest-centre) agrees up to the known half-pixel convention gap
    assert r.contains(np.stack([x, y])).mean() > 0.95
