"""`MultiRegion`: one marginal's accepted set as a union of per-mode subgrids.

Each mode carries its own grid, so a mode spanning three pixels of the full-box
grid can be resolved at its own pitch. Everything the callers need is then a
reduction over the parts — with one exception: `volume_fraction` has no implicit
denominator any more (the parts cover only themselves, not the prior box), so
the reference set is passed in.
"""
import numpy as np
import pytest

from pembhb.regions import MultiRegion, Region


def _part_1d(lo, hi, n, mask=None, periodic=(False,)):
    g = np.linspace(lo, hi, n)
    m = np.ones(n, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    return Region(m, (g,), periodic)


def _part_2d(xlo, xhi, ylo, yhi, n, mask=None):
    gx = np.linspace(xlo, xhi, n)
    gy = np.linspace(ylo, yhi, n)
    m = np.ones((n, n), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    return Region(m, (gx, gy))


def _two_modes_1d():
    return MultiRegion([_part_1d(0.2, 0.3, 41), _part_1d(0.7, 0.8, 11)])


# ----------------------------------------------------------------------
# membership
# ----------------------------------------------------------------------
def test_contains_is_the_union_of_the_parts():
    mr = _two_modes_1d()
    inside = np.array([0.2, 0.25, 0.3, 0.7, 0.75, 0.8])
    outside = np.array([0.0, 0.15, 0.5, 0.65, 0.85, 1.0])
    assert mr.contains(inside).all()
    assert not mr.contains(outside).any()


def test_contains_grid_recovers_both_modes_on_a_foreign_grid():
    # the target grid is unrelated to either part's pitch or extent
    mr = _two_modes_1d()
    g = np.linspace(0.0, 1.0, 101)
    keep = mr.contains_grid((g,))
    assert keep.shape == g.shape
    # nearest-pixel lookup, so accept a one-cell fringe at each mode's edge
    np.testing.assert_array_equal(keep, ((g >= 0.2 - 5e-3) & (g <= 0.3 + 5e-3))
                                  | ((g >= 0.7 - 5e-3) & (g <= 0.8 + 5e-3)))


def test_gap_between_modes_is_not_bridged():
    mr = _two_modes_1d()
    g = np.linspace(0.0, 1.0, 1001)
    keep = mr.contains_grid((g,))
    gap = (g > 0.35) & (g < 0.65)
    assert not (keep & gap).any()


# ----------------------------------------------------------------------
# measures
# ----------------------------------------------------------------------
def test_volume_pools_measure_not_pixel_counts():
    # same physical width, 4x different pitch: pixel counts disagree, measures agree
    fine = _part_1d(0.2, 0.3, 41)
    coarse = _part_1d(0.7, 0.8, 11)
    mr = MultiRegion([fine, coarse])
    assert np.count_nonzero(fine.mask) / np.count_nonzero(coarse.mask) > 3.5
    assert fine.volume() == pytest.approx(coarse.volume(), rel=0.1)
    assert mr.volume() == pytest.approx(fine.volume() + coarse.volume())


def test_volume_fraction_against_a_previous_multiregion():
    prev = MultiRegion([_part_1d(0.2, 0.4, 81), _part_1d(0.7, 0.8, 41)])
    now = MultiRegion([_part_1d(0.2, 0.3, 41), _part_1d(0.7, 0.8, 41)])
    assert now.volume_fraction(prev) == pytest.approx(now.volume() / prev.volume())
    assert now.volume_fraction(prev) < 1.0          # nested -> a real ratio
    assert now.volume_fraction(prev.volume()) == pytest.approx(now.volume_fraction(prev))


def test_volume_fraction_needs_a_reference():
    with pytest.raises(TypeError):
        _two_modes_1d().volume_fraction()
    with pytest.raises(ValueError):
        _two_modes_1d().volume_fraction(0.0)


def test_region_volume_fraction_default_is_unchanged():
    g = np.linspace(0.0, 1.0, 100)
    r = Region((g >= 0.4) & (g < 0.5), (g,))
    assert r.volume_fraction() == pytest.approx(0.10)
    assert r.volume_fraction(1.0) == pytest.approx(r.volume())


def test_bounds_and_intervals_span_every_part():
    mr = _two_modes_1d()
    lo, hi = mr.bounds()
    assert lo == pytest.approx(0.2)
    assert hi == pytest.approx(0.8)
    ivs = sorted(mr.intervals(0))
    assert len(ivs) == 2
    assert ivs[0][0] == pytest.approx(0.2) and ivs[0][1] == pytest.approx(0.3)
    assert ivs[1][0] == pytest.approx(0.7) and ivs[1][1] == pytest.approx(0.8)


def test_components_splits_a_part_that_refinement_broke_in_two():
    g = np.linspace(0.0, 1.0, 101)
    split = (g < 0.3) | (g > 0.7)                    # one part, two components
    mr = MultiRegion([Region(split, (g,)), _part_1d(2.0, 2.1, 11)])
    assert len(mr) == 2
    assert len(mr.components()) == 3


def test_empty_part_does_not_break_bounds():
    empty = _part_1d(0.5, 0.6, 11, mask=np.zeros(11, dtype=bool))
    mr = MultiRegion([empty, _part_1d(0.2, 0.3, 41)])
    assert mr.bounds()[0] == pytest.approx(0.2)
    assert mr.volume() == pytest.approx(mr.parts[1].volume())


# ----------------------------------------------------------------------
# the coarse pass is a one-part MultiRegion
# ----------------------------------------------------------------------
def test_from_region_matches_the_single_grid_behaviour():
    g = np.linspace(0.0, 1.0, 101)
    r = Region((g >= 0.2) & (g <= 0.3), (g,))
    mr = MultiRegion.from_region(r)
    probe = np.linspace(0.0, 1.0, 57)
    np.testing.assert_array_equal(mr.contains(probe), r.contains(probe))
    assert mr.volume() == pytest.approx(r.volume())
    assert mr.volume_fraction(r.volume() / r.volume_fraction()) == pytest.approx(
        r.volume_fraction())
    assert mr.periodic == r.periodic


# ----------------------------------------------------------------------
# sampling
# ----------------------------------------------------------------------
def test_draw_1d_stays_inside_and_splits_by_width():
    rng = np.random.default_rng(0)
    mr = MultiRegion([_part_1d(0.2, 0.3, 41), _part_1d(0.7, 0.9, 21)])
    x = mr.draw(20000, rng)
    assert x.shape == (20000,)
    assert mr.contains(x).all()
    in_first = np.mean(x < 0.5)
    assert 0.30 < in_first < 0.37                    # widths 0.1 : 0.2


def test_draw_2d_weights_parts_by_volume():
    rng = np.random.default_rng(1)
    small = _part_2d(0.0, 0.1, 0.0, 0.1, 11)
    big = _part_2d(0.5, 0.7, 0.5, 0.7, 21)           # 4x the area
    mr = MultiRegion([small, big])
    x, y = mr.draw(20000, rng)
    assert mr.contains(np.stack([x, y])).all()
    frac_small = np.mean(x < 0.3)
    expected = small.volume() / (small.volume() + big.volume())
    assert abs(frac_small - expected) < 0.02


def test_draw_2d_respects_a_hole_in_one_part():
    # Regression for plan 2.5: the sampler indexes from the cell vertex, so
    # floor() there and round()-to-nearest-centre in contains() agree. Before
    # that they were half a cell apart and 1-3% of draws landed in the hole.
    rng = np.random.default_rng(2)
    n = 21
    mask = np.ones((n, n), dtype=bool)
    mask[5:15, 5:15] = False                         # punched-out centre
    part = _part_2d(0.0, 1.0, 0.0, 1.0, n, mask=mask)
    mr = MultiRegion([part])
    x, y = mr.draw(5000, rng)
    assert mr.contains(np.stack([x, y])).all()


def test_draw_from_an_empty_multiregion_raises():
    rng = np.random.default_rng(3)
    empty = _part_2d(0.0, 1.0, 0.0, 1.0, 11, mask=np.zeros((11, 11), dtype=bool))
    with pytest.raises(ValueError):
        MultiRegion([empty]).draw(10, rng)


# ----------------------------------------------------------------------
# construction guards
# ----------------------------------------------------------------------
def test_mixed_dimensionality_is_rejected():
    with pytest.raises(ValueError):
        MultiRegion([_part_1d(0.0, 1.0, 11), _part_2d(0.0, 1.0, 0.0, 1.0, 11)])


def test_no_parts_is_rejected():
    with pytest.raises(ValueError):
        MultiRegion([])


def test_periodic_is_inherited_from_the_parts():
    g_full = np.linspace(0.0, 2 * np.pi, 101)
    wrapped = Region(np.ones(101, dtype=bool), (g_full,), (True,))
    interior = _part_1d(1.0, 2.0, 11)                # a window is never periodic
    assert MultiRegion([wrapped, interior]).periodic == (True,)
    assert MultiRegion([interior]).periodic == (False,)
    assert MultiRegion([wrapped], periodic=(False,)).periodic == (False,)
