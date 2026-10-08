"""npz `format_version: 2` — per-mode subgrids round-trip; format 1 still reads.

Format 1 stored one `labels`/`gridx`/`gridy` triple per 2D pair, so every mode
shared a single resolution. Format 2 stores one `mask` + `origin` (bottom-left
vertex) + `extent` per mode, which is what lets a three-pixel mode be refined
without dragging the whole box along with it.
"""
import numpy as np
import pytest

from pembhb.mask_truncation import (
    load_truncation, save_truncation, region_of_pair, truth_violations,
    MaskRejectSampler, _multiregion_from_labels,
)
from pembhb.regions import (
    MultiRegion, Region, grid_from_origin_extent, origin_extent_from_grid,
)

PRIOR = {"a": [0.0, 1.0], "b": [-1.0, 1.0]}


def _paths(tmp_path):
    return str(tmp_path / "prior.yaml"), str(tmp_path / "trunc.npz")


def _two_mode_region():
    """Two modes at 4x different pitch — impossible to store in format 1."""
    gx1 = np.linspace(0.05, 0.25, 21)
    gy1 = np.linspace(-0.8, -0.4, 21)
    gx2 = np.linspace(0.60, 0.80, 6)
    gy2 = np.linspace(0.30, 0.70, 6)
    m1 = np.ones((21, 21), dtype=bool)
    m1[0, 0] = False                                  # a non-rectangular mode
    return MultiRegion([Region(m1, (gx1, gy1)),
                        Region(np.ones((6, 6), dtype=bool), (gx2, gy2))])


# ----------------------------------------------------------------------
# the vertex convention
# ----------------------------------------------------------------------
def test_origin_extent_is_the_grid_edges_not_the_centres():
    g = np.linspace(0.0, 1.0, 11)                     # centres, dx = 0.1
    origin, extent = origin_extent_from_grid(g)
    assert origin == pytest.approx(-0.05)             # half a cell outside
    assert extent == pytest.approx(1.1)               # 11 cells of 0.1
    np.testing.assert_allclose(grid_from_origin_extent(origin, extent, 11), g)


def test_origin_extent_round_trips_at_any_pitch():
    for n in (2, 3, 17, 200):
        g = np.linspace(-3.0, 7.5, n)
        o, e = origin_extent_from_grid(g)
        np.testing.assert_allclose(grid_from_origin_extent(o, e, n), g, atol=1e-12)


def test_one_point_grid_is_rejected():
    with pytest.raises(ValueError):
        origin_extent_from_grid(np.array([0.5]))


# ----------------------------------------------------------------------
# format 2 round-trip
# ----------------------------------------------------------------------
def test_multi_resolution_region_round_trips(tmp_path):
    yaml_path, npz_path = _paths(tmp_path)
    region = _two_mode_region()
    save_truncation(yaml_path, npz_path, PRIOR, {}, [{"idx": (0, 1), "region": region}])

    with np.load(npz_path) as z:
        assert int(z["format_version"]) == 2
        assert int(z["n_modes__0_1"]) == 2

    back = load_truncation(yaml_path, npz_path)["masks_2d"][0]["region"]
    assert len(back.parts) == 2
    assert back.volume() == pytest.approx(region.volume())
    for a, b in zip(back.parts, region.parts):
        np.testing.assert_array_equal(a.mask, b.mask)
        np.testing.assert_allclose(a.grids[0], b.grids[0])
        np.testing.assert_allclose(a.grids[1], b.grids[1])


def test_membership_survives_the_round_trip(tmp_path):
    yaml_path, npz_path = _paths(tmp_path)
    region = _two_mode_region()
    save_truncation(yaml_path, npz_path, PRIOR, {}, [{"idx": (0, 1), "region": region}])
    back = load_truncation(yaml_path, npz_path)["masks_2d"][0]["region"]

    rng = np.random.default_rng(0)
    pts = np.stack([rng.uniform(0.0, 1.0, 3000), rng.uniform(-1.0, 1.0, 3000)])
    np.testing.assert_array_equal(back.contains(pts), region.contains(pts))


def test_periodicity_is_stored(tmp_path):
    yaml_path, npz_path = _paths(tmp_path)
    gx = np.linspace(0.0, 2 * np.pi, 20)
    gy = np.linspace(-1.0, 1.0, 20)
    region = MultiRegion([Region(np.ones((20, 20), dtype=bool), (gx, gy), (True, False))])
    save_truncation(yaml_path, npz_path, PRIOR, {}, [{"idx": (7, 8), "region": region}])
    back = load_truncation(yaml_path, npz_path)["masks_2d"][0]["region"]
    assert back.periodic == (True, False)
    assert back.parts[0].periodic == (True, False)


def test_legacy_entry_without_a_region_key_still_saves(tmp_path):
    # tmnre_joint still builds labels/grid_x/grid_y alongside the region
    yaml_path, npz_path = _paths(tmp_path)
    gx, gy = np.linspace(0.0, 1.0, 20), np.linspace(-1.0, 1.0, 20)
    labels = np.zeros((20, 20), dtype=int)
    labels[2:8, 2:8] = 1
    labels[12:18, 12:18] = 2
    save_truncation(yaml_path, npz_path, PRIOR, {},
                    [{"idx": (0, 1), "labels": labels, "grid_x": gx, "grid_y": gy}])
    back = load_truncation(yaml_path, npz_path)["masks_2d"][0]["region"]
    # saved as written: one part on the grid it came from, nothing invented
    assert len(back.parts) == 1
    np.testing.assert_array_equal(back.parts[0].mask, labels > 0)
    assert back.volume() == pytest.approx(
        Region(labels > 0, (gx, gy)).volume())


# ----------------------------------------------------------------------
# format 1 files still load
# ----------------------------------------------------------------------
def _write_format1(npz_path, labels, gx, gy, i=0, j=1):
    np.savez_compressed(npz_path, **{
        f"labels__{i}_{j}": labels.astype(np.int8),
        f"gridx__{i}_{j}": gx, f"gridy__{i}_{j}": gy})


def test_format1_npz_loads_as_a_multiregion(tmp_path):
    import yaml as _yaml
    yaml_path, npz_path = _paths(tmp_path)
    gx, gy = np.linspace(0.0, 1.0, 40), np.linspace(-1.0, 1.0, 40)
    labels = np.zeros((40, 40), dtype=int)
    labels[4:12, 4:12] = 1
    labels[25:35, 20:30] = 2
    _write_format1(npz_path, labels, gx, gy)
    with open(yaml_path, "w") as f:
        _yaml.safe_dump({"prior": PRIOR, "truncation_mode": "mask",
                         "intervals_1d": {}, "pairs_2d": [[0, 1]]}, f)

    region = load_truncation(yaml_path, npz_path)["masks_2d"][0]["region"]
    assert len(region.parts) == 2
    old = Region(labels > 0, (gx, gy))
    assert region.volume() == pytest.approx(old.volume())

    rng = np.random.default_rng(1)
    pts = np.stack([rng.uniform(0.0, 1.0, 3000), rng.uniform(-1.0, 1.0, 3000)])
    np.testing.assert_array_equal(region.contains(pts), old.contains(pts))


def test_legacy_crop_keeps_a_one_pixel_mode_addressable():
    gx, gy = np.linspace(0.0, 1.0, 20), np.linspace(-1.0, 1.0, 20)
    labels = np.zeros((20, 20), dtype=int)
    labels[10, 10] = 1                                # psi's 1-px ghost mode
    region = _multiregion_from_labels(labels, gx, gy)
    assert len(region.parts) == 1
    assert region.contains(np.array([[gx[10]], [gy[10]]]))[0]
    assert region.volume() == pytest.approx(
        Region(labels > 0, (gx, gy)).volume())


# ----------------------------------------------------------------------
# the consumers
# ----------------------------------------------------------------------
def test_region_of_pair_accepts_both_entry_forms():
    gx, gy = np.linspace(0.0, 1.0, 10), np.linspace(-1.0, 1.0, 10)
    labels = np.zeros((10, 10), dtype=int)
    labels[2:5, 2:5] = 1
    a = region_of_pair({"idx": (0, 1), "labels": labels, "grid_x": gx, "grid_y": gy})
    b = region_of_pair({"idx": (0, 1), "region": Region(labels > 0, (gx, gy))})
    c = region_of_pair({"idx": (0, 1), "region": MultiRegion.from_region(
        Region(labels > 0, (gx, gy)))})
    assert a.volume() == pytest.approx(b.volume()) == pytest.approx(c.volume())


def test_truth_violations_uses_the_region(tmp_path):
    keys = ["a", "b"]
    region = _two_mode_region()
    masks = [{"idx": (0, 1), "region": region}]
    inside = np.array([0.15, -0.6])                   # in the first mode
    assert truth_violations(inside, PRIOR, {}, masks, keys, check_idxs=(0, 1)) == []

    # between the two subgrids: the box never covered the truth
    v = truth_violations(np.array([0.45, 0.0]), PRIOR, {}, masks, keys,
                         check_idxs=(0, 1))
    assert len(v) == 1 and v[0]["kind"] == "2d-mask"
    assert "off the posterior grid" in v[0]["detail"]

    # on mode 0's subgrid but at its one rejected pixel: the contour cut it out
    corner = np.array([region.parts[0].grids[0][0], region.parts[0].grids[1][0]])
    v = truth_violations(corner, PRIOR, {}, masks, keys, check_idxs=(0, 1))
    assert len(v) == 1
    assert "in a gap between components" in v[0]["detail"]


def test_sampler_draws_from_every_mode_of_a_multiresolution_region():
    region = _two_mode_region()
    rng = np.random.default_rng(2)
    x, y = region.draw(4000, rng)
    assert region.contains(np.stack([x, y])).all()
    assert (x < 0.4).any() and (x > 0.5).any()        # both modes populated


def test_mask_reject_sampler_accepts_a_region_entry():
    keys = ["logMchirp", "q", "chi1", "chi2", "dist", "phi", "inc",
            "lambda", "beta", "psi", "Deltat"]
    span = {"inc": [-0.9, 0.9], "beta": [-0.9, 0.9], "chi1": [-0.5, 0.5],
            "chi2": [-0.5, 0.5], "dist": [1.0, 2.0]}
    prior = {k: span.get(k, [1.0, 2.0]) for k in keys}
    gx, gy = np.linspace(1.0, 2.0, 20), np.linspace(1.0, 2.0, 20)
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 5:15] = True
    region = MultiRegion([Region(mask, (gx, gy))])
    masks_2d = [{"idx": (9, 10), "region": region}]      # psi-Deltat
    s = MaskRejectSampler(prior, {}, masks_2d, rng=np.random.default_rng(3),
                          spin_param_basis="chi1chi2")
    _bbhx, tmnre = s.sample(64, 1.0)
    assert tmnre.shape[1] == 64
    assert region.contains(np.stack([tmnre[9], tmnre[10]])).all()
