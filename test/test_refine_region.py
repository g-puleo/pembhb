"""Per-mode refinement: coarse pass fixes topology, subgrids fix resolution.

The evaluator here is an analytic density, so these test the refinement logic
itself rather than the network: what the coarse pass hands over, how the modes
are pooled into one threshold, and what survives the clip.
"""
import numpy as np
import pytest

from pembhb.regions import (
    MultiRegion, Region, mode_bounds, refine_region, region_from_hpd,
)


def _bimodal(x, centres=(0.25, 0.75), width=0.02, heights=(1.0, 1.0)):
    out = np.zeros_like(np.asarray(x, dtype=float))
    for c, h in zip(centres, heights):
        out = out + h * np.exp(-0.5 * ((np.asarray(x, dtype=float) - c) / width) ** 2)
    return out


def _evaluator_1d(fn):
    """`evaluate(bounds, n)` over an analytic density — raw, unnormalised."""
    def _evaluate(bounds, n):
        lo, hi = bounds
        g = np.linspace(float(lo), float(hi), int(n))
        return fn(g), (g,)
    return _evaluate


def _evaluator_2d(fn):
    def _evaluate(bounds, n):
        (xlo, xhi), (ylo, yhi) = bounds
        gx = np.linspace(float(xlo), float(xhi), int(n))
        gy = np.linspace(float(ylo), float(yhi), int(n))
        MX, MY = np.meshgrid(gx, gy, indexing="xy")
        return fn(MX, MY), (gx, gy)
    return _evaluate


# ----------------------------------------------------------------------
# mode_bounds: the subgrid extent
# ----------------------------------------------------------------------
def test_mode_bounds_pads_to_cell_edges():
    g = np.linspace(0.0, 1.0, 101)                      # dx = 0.01
    comp = Region((g >= 0.30) & (g <= 0.40), (g,))
    lo, hi = mode_bounds(comp)
    assert lo == pytest.approx(0.295)                   # centre - dx/2
    assert hi == pytest.approx(0.405)
    assert (hi - lo) == pytest.approx(np.count_nonzero(comp.mask) * 0.01)


def test_mode_bounds_clamps_at_the_prior_box_edge():
    g = np.linspace(-1.0, 1.0, 201)
    comp = Region(g <= -0.9, (g,))                      # touches the box edge
    lo, hi = mode_bounds(comp)
    assert lo == pytest.approx(-1.0)                    # not -1.005
    assert hi > -0.9


def test_mode_bounds_2d_is_per_axis():
    gx, gy = np.linspace(0.0, 1.0, 101), np.linspace(0.0, 2.0, 101)
    MX, MY = np.meshgrid(gx, gy, indexing="xy")
    comp = Region((MX >= 0.2) & (MX <= 0.3) & (MY >= 1.0) & (MY <= 1.2), (gx, gy))
    (xlo, xhi), (ylo, yhi) = mode_bounds(comp)
    assert xlo == pytest.approx(0.195) and xhi == pytest.approx(0.305)
    assert ylo == pytest.approx(0.99) and yhi == pytest.approx(1.21)


# ----------------------------------------------------------------------
# the point of the exercise: resolution
# ----------------------------------------------------------------------
def test_refined_modes_have_their_own_grids_at_the_refined_pitch():
    g = np.linspace(0.0, 1.0, 100)                      # ~4 px per mode
    coarse = region_from_hpd(_bimodal(g), (g,), 0.999, dilation_factor=1.0)
    assert len(coarse.components()) == 2

    refined, modes = refine_region(coarse, _evaluator_1d(_bimodal), ngrid=128,
                                   credible=0.999, dilation=1.0)
    assert len(refined.parts) == 2
    for part in refined.parts:
        assert part.grids[0].size == 128
        # each subgrid spans only its own mode, so its pitch is far finer
        assert part.grids[0][1] - part.grids[0][0] < 0.2 * (g[1] - g[0])
    assert len(modes) == 2
    assert all(m["threshold"] == modes[0]["threshold"] for m in modes)


def test_refinement_tightens_rather_than_widens():
    # Step 0's measured result: an HPD level set of an already-zeroed density
    # always lies inside the coarse mask it came from.
    g = np.linspace(0.0, 1.0, 100)
    coarse = region_from_hpd(_bimodal(g), (g,), 0.999, dilation_factor=1.0)
    refined, _ = refine_region(coarse, _evaluator_1d(_bimodal), ngrid=128,
                               credible=0.999, dilation=1.0)
    assert refined.volume() < coarse.volume()
    # and everything accepted is still inside the coarse set
    probe = np.linspace(0.0, 1.0, 5000)
    assert not (refined.contains(probe) & ~coarse.contains(probe)).any()


def test_one_threshold_is_shared_across_modes_of_unequal_mass():
    # A mode holding 1/10 the mass must not be cut at its own quantile: pooled,
    # it keeps a smaller share of its own volume than the dominant mode.
    g = np.linspace(0.0, 1.0, 200)
    dens = lambda x: _bimodal(x, heights=(1.0, 0.1))
    coarse = region_from_hpd(dens(g), (g,), 0.999, dilation_factor=1.0)
    assert len(coarse.components()) == 2
    _refined, modes = refine_region(coarse, _evaluator_1d(dens), ngrid=128,
                                    credible=0.99, dilation=1.0)
    thr = {m["threshold"] for m in modes}
    assert len(thr) == 1
    kept = [m["mask"].mean() for m in modes]
    assert min(kept) < max(kept)                        # not equal shares


def test_empty_mode_is_dropped():
    # a coarse mode with no density under it (a ghost) disappears
    g = np.linspace(0.0, 1.0, 101)
    coarse_mask = ((g >= 0.20) & (g <= 0.30)) | ((g >= 0.60) & (g <= 0.62))
    coarse = Region(coarse_mask, (g,))
    dens = lambda x: np.exp(-0.5 * ((np.asarray(x) - 0.25) / 0.02) ** 2) + 1e-300
    refined, modes = refine_region(coarse, _evaluator_1d(dens), ngrid=64,
                                   credible=0.999, dilation=1.0)
    assert len(modes) == 2                              # both were evaluated
    assert len(refined.parts) == 1                      # only one survived
    assert refined.bounds()[0] > 0.15 and refined.bounds()[1] < 0.35


def test_all_modes_empty_falls_back_to_the_coarse_region():
    g = np.linspace(0.0, 1.0, 101)
    coarse = Region((g >= 0.2) & (g <= 0.3), (g,))
    zero = lambda x: np.zeros_like(np.asarray(x, dtype=float))
    refined, _ = refine_region(coarse, _evaluator_1d(zero), ngrid=32,
                               credible=0.999, dilation=1.0)
    assert refined.volume() == pytest.approx(coarse.volume())


# ----------------------------------------------------------------------
# the §2 policy still applies on the subgrids
# ----------------------------------------------------------------------
def test_clip_keeps_the_refined_set_inside_prev():
    g = np.linspace(0.0, 1.0, 100)
    coarse = region_from_hpd(_bimodal(g), (g,), 0.999, dilation_factor=1.0)
    prev = {"kind": "1d", "intervals": [[0.22, 0.28], [0.72, 0.78]]}
    refined, _ = refine_region(coarse, _evaluator_1d(_bimodal), ngrid=128,
                               prev=prev, credible=0.9999, dilation=1.5,
                               clip=True)
    probe = np.linspace(0.0, 1.0, 5000)
    inside_prev = (((probe >= 0.22) & (probe <= 0.28))
                   | ((probe >= 0.72) & (probe <= 0.78)))
    # nearest-pixel membership, so allow one refined cell of slack
    slack = refined.parts[0].grids[0][1] - refined.parts[0].grids[0][0]
    escaped = probe[refined.contains(probe) & ~inside_prev]
    assert escaped.size == 0 or np.abs(
        np.concatenate([escaped - 0.28, 0.22 - escaped,
                        escaped - 0.78, 0.72 - escaped])).min() < slack


def test_modes_carry_what_a_plot_needs():
    g = np.linspace(0.0, 1.0, 100)
    coarse = region_from_hpd(_bimodal(g), (g,), 0.999, dilation_factor=1.0)
    _refined, modes = refine_region(coarse, _evaluator_1d(_bimodal), ngrid=64,
                                    credible=0.999, dilation=1.0)
    for m in modes:
        assert m["density"].shape == m["grids"][0].shape == (64,)
        assert m["mask"].shape == (64,)
        assert m["cell_volume"] > 0
        assert np.isfinite(m["threshold"])


# ----------------------------------------------------------------------
# 2D
# ----------------------------------------------------------------------
def _two_blobs(mx, my):
    a = np.exp(-0.5 * (((mx - 0.25) / 0.03) ** 2 + ((my - 0.25) / 0.03) ** 2))
    b = np.exp(-0.5 * (((mx - 0.75) / 0.03) ** 2 + ((my - 0.70) / 0.03) ** 2))
    return a + b


def test_2d_refinement_keeps_both_modes_and_shrinks_the_area():
    gx = gy = np.linspace(0.0, 1.0, 60)
    MX, MY = np.meshgrid(gx, gy, indexing="xy")
    coarse = region_from_hpd(_two_blobs(MX, MY), (gx, gy), 0.999,
                             dilation_factor=1.0)
    assert len(coarse.components()) == 2

    refined, modes = refine_region(coarse, _evaluator_2d(_two_blobs), ngrid=48,
                                   credible=0.999, dilation=1.0)
    assert len(refined.parts) == 2
    assert len(modes) == 2
    for part in refined.parts:
        assert part.mask.shape == (48, 48)
    assert refined.volume() < coarse.volume()
    # the two modes stay separated
    assert len(refined.components()) == 2


def test_2d_subgrid_is_finer_than_the_coarse_grid():
    gx = gy = np.linspace(0.0, 1.0, 60)
    MX, MY = np.meshgrid(gx, gy, indexing="xy")
    coarse = region_from_hpd(_two_blobs(MX, MY), (gx, gy), 0.999,
                             dilation_factor=1.0)
    refined, _ = refine_region(coarse, _evaluator_2d(_two_blobs), ngrid=48,
                               credible=0.999, dilation=1.0)
    coarse_dx = gx[1] - gx[0]
    for part in refined.parts:
        assert part.grids[0][1] - part.grids[0][0] < coarse_dx
        assert part.grids[1][1] - part.grids[1][0] < coarse_dx


def test_refined_result_round_trips_through_storage(tmp_path):
    from pembhb.mask_truncation import load_truncation, save_truncation
    gx = gy = np.linspace(0.0, 1.0, 60)
    MX, MY = np.meshgrid(gx, gy, indexing="xy")
    coarse = region_from_hpd(_two_blobs(MX, MY), (gx, gy), 0.999,
                             dilation_factor=1.0)
    refined, _ = refine_region(coarse, _evaluator_2d(_two_blobs), ngrid=48,
                               credible=0.999, dilation=1.0)
    y, n = str(tmp_path / "p.yaml"), str(tmp_path / "t.npz")
    save_truncation(y, n, {"a": [0.0, 1.0], "b": [0.0, 1.0]}, {},
                    [{"idx": (0, 1), "region": refined}])
    back = load_truncation(y, n)["masks_2d"][0]["region"]
    assert len(back.parts) == len(refined.parts)
    assert back.volume() == pytest.approx(refined.volume())


# ----------------------------------------------------------------------
# find_modes: the same mode-finder applied recursively -> a tree of modes
# ----------------------------------------------------------------------
def _late_split(bounds, n):
    """Structure that only becomes visible once the window has tightened.

    Wide window: one smooth blob. Narrow: it was really two all along. This is
    exactly the situation the recursion exists for -- and it is why "stop when
    nothing split" is the wrong rule, since the pass that tightens the window is
    the one that makes the split resolvable.
    """
    lo, hi = float(bounds[0]), float(bounds[1])
    g = np.linspace(lo, hi, int(n))
    if (hi - lo) > 0.25:
        d = np.exp(-0.5 * ((g - 0.5) / 0.05) ** 2)
    else:
        d = (np.exp(-0.5 * ((g - 0.47) / 0.012) ** 2)
             + np.exp(-0.5 * ((g - 0.53) / 0.012) ** 2))
    return d, (g,)


def test_recursion_splits_what_one_pass_cannot_see():
    from pembhb.regions import find_modes
    g = np.linspace(0.0, 1.0, 100)
    coarse = Region((g > 0.3) & (g < 0.7), (g,))

    one_pass, _ = refine_region(coarse, _late_split, ngrid=128,
                                credible=0.9, dilation=1.0)
    tree, _ = find_modes(coarse, _late_split, ngrid=128, max_depth=4,
                         credible=0.9, dilation=1.0)
    assert len(one_pass.components()) == 1      # window still too wide
    assert len(tree.components()) == 2          # recursion tightened, then split


def test_find_modes_is_idempotent_once_converged():
    from pembhb.regions import find_modes
    g = np.linspace(0.0, 1.0, 100)
    coarse = region_from_hpd(_bimodal(g), (g,), 0.999, dilation_factor=1.0)
    once, _ = find_modes(coarse, _evaluator_1d(_bimodal), ngrid=128,
                         credible=0.9999, dilation=1.0)
    twice, _ = find_modes(once, _evaluator_1d(_bimodal), ngrid=128,
                          credible=0.9999, dilation=1.0)
    assert len(twice.parts) == len(once.parts)
    assert twice.volume() == pytest.approx(once.volume(), rel=0.05)


def test_windows_tighten_so_resolution_compounds():
    from pembhb.regions import find_modes
    g = np.linspace(0.0, 1.0, 100)
    region = region_from_hpd(_bimodal(g), (g,), 0.999, dilation_factor=1.0)
    prev_cell = g[1] - g[0]
    for _ in range(3):
        region, _ = find_modes(region, _evaluator_1d(_bimodal), ngrid=128,
                               max_depth=1, credible=0.9999, dilation=1.0)
        cell = max(p.grids[0][1] - p.grids[0][0] for p in region.parts)
        assert cell < prev_cell                       # strictly finer every round
        prev_cell = cell


def test_max_depth_caps_the_work():
    from pembhb.regions import find_modes
    calls = []

    def counting(bounds, n):
        calls.append(bounds)
        return _late_split(bounds, n)

    g = np.linspace(0.0, 1.0, 100)
    coarse = Region((g > 0.3) & (g < 0.7), (g,))
    find_modes(coarse, counting, ngrid=128, max_depth=1, credible=0.9,
               dilation=1.0)
    assert len(calls) == 1                            # exactly one pass


def test_multiregion_can_be_refined_again():
    # the recursion's precondition: refine_region accepts its own output
    from pembhb.regions import MultiRegion
    g = np.linspace(0.0, 1.0, 100)
    coarse = region_from_hpd(_bimodal(g), (g,), 0.999, dilation_factor=1.0)
    once, _ = refine_region(coarse, _evaluator_1d(_bimodal), ngrid=128,
                            credible=0.9999, dilation=1.0)
    assert isinstance(once, MultiRegion)
    twice, _ = refine_region(once, _evaluator_1d(_bimodal), ngrid=128,
                             credible=0.9999, dilation=1.0)
    assert isinstance(twice, MultiRegion)
    assert len(twice.parts) == len(once.parts)
