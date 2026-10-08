"""Volume-ratio denominator: posterior / sampling-proposal, not posterior / box.

The volume ratio drives round-to-round early stopping (`ratio <= 0.5`). In mask
mode the sampling proposal is an irregular mask that can fill only a fraction of
its bounding box, so dividing the posterior by the full grid (the old
`Region.volume_fraction`) pins the ratio to that fraction and it never reaches 1
even when the posterior has converged to the proposal — the observed lambda-beta
"stuck at ~0.2 while unchanged" bug.
"""
import types
import numpy as np
import pytest

from pembhb.callbacks import PlotPosteriorCallback
from pembhb.regions import Region

_proposal_pixels = PlotPosteriorCallback._proposal_pixels


def _sky_like_proposal():
    gx = np.linspace(0.0, 2 * np.pi, 200)
    gy = np.linspace(-1.0, 1.0, 200)
    MX, MY = np.meshgrid(gx, gy, indexing="xy")
    prop = (MX >= 2.2) & (MX <= 4.5) & (MY >= -0.45) & (MY <= 0.9)
    return gx, gy, MX, MY, prop


def test_converged_posterior_gives_ratio_one():
    gx, gy, MX, MY, prop = _sky_like_proposal()
    assert prop.mean() < 0.3          # proposal fills < 30% of its box
    prev = {"kind": "2d", "region": Region(prop, (gx, gy))}
    obj = types.SimpleNamespace(prev_accepted={(7, 8): prev})

    posterior = prop.copy()           # posterior == proposal (converged)
    post_px = int(np.count_nonzero(posterior))
    prop_px = _proposal_pixels(obj, (7, 8), (gx, gy), int(posterior.size))

    # old (buggy) measure would be ~0.25; fixed measure is 1.0
    assert abs(post_px / posterior.size - prop.mean()) < 1e-9
    assert abs(post_px / prop_px - 1.0) < 1e-9


def test_half_proposal_gives_half_ratio():
    gx, gy, MX, MY, prop = _sky_like_proposal()
    prev = {"kind": "2d", "region": Region(prop, (gx, gy))}
    obj = types.SimpleNamespace(prev_accepted={(7, 8): prev})

    half = prop & (MX <= 3.35)        # posterior fills ~half the proposal
    prop_px = _proposal_pixels(obj, (7, 8), (gx, gy), int(prop.size))
    ratio = int(np.count_nonzero(half)) / prop_px
    assert 0.4 < ratio < 0.6          # would (correctly) trigger another round


def test_no_prev_falls_back_to_full_grid():
    gx, gy, MX, MY, prop = _sky_like_proposal()
    obj = types.SimpleNamespace(prev_accepted={})     # round 1 / rectangle mode
    assert _proposal_pixels(obj, (7, 8), (gx, gy), prop.size) == prop.size


def test_no_overlap_falls_back_to_full_grid():
    gx, gy, MX, MY, prop = _sky_like_proposal()
    far = (MX >= 6.0) & (MY >= 0.95)                  # disjoint from any posterior
    prev = {"kind": "2d", "region": Region(far, (gx, gy))}
    obj = types.SimpleNamespace(prev_accepted={(7, 8): prev})
    # a grid that the 'far' proposal maps entirely outside -> no overlap
    gx2 = np.linspace(0.0, 1.0, 50)
    gy2 = np.linspace(-1.0, -0.5, 50)
    total = gx2.size * gy2.size
    assert _proposal_pixels(obj, (7, 8), (gx2, gy2), total) == total


def test_2d_numerator_counts_all_modes_not_main_mode():
    # A bimodal posterior (sky antipode): the main mode alone is ~half the
    # accepted set, so a main-mode numerator trips the 0.5 stop early. The 2D
    # volume-ratio numerator must be the full HPD region (all modes), matching
    # the proposal the sampler draws from.
    from pembhb.regions import region_from_hpd, region_from_main_mode

    gx = np.linspace(0.0, 2 * np.pi, 200)
    gy = np.linspace(-1.0, 1.0, 200)
    MX, MY = np.meshgrid(gx, gy, indexing="xy")

    def blob(cx, cy, s):
        return np.exp(-0.5 * (((MX - cx) / s) ** 2 + ((MY - cy) / s) ** 2))

    dens = blob(2.0, -0.3, 0.25) + 0.9 * blob(4.3, 0.4, 0.25)
    dens /= dens.sum()
    cl, dil, periods = 0.999, 1.1, (2 * np.pi, None)

    proposal = region_from_hpd(dens, (gx, gy), cl, dilation_factor=dil, periods=periods)
    prop_px = int(np.count_nonzero(proposal.mask))
    full_px = int(np.count_nonzero(
        region_from_hpd(dens, (gx, gy), cl, dilation_factor=dil, periods=periods).mask))
    main_px = int(np.count_nonzero(
        region_from_main_mode(dens, (gx, gy), cl, dilation_factor=dil, periods=periods).mask))

    assert full_px / prop_px > 0.95        # converged -> ~1, no early stop
    assert main_px / prop_px < 0.7         # main-mode-only would trip near 0.5


def test_1d_gapped_proposal_denominator():
    g = np.linspace(-1.0, 1.0, 201)
    # cos-ι style two-interval proposal (edge-on band excluded)
    prev = {"kind": "1d", "intervals": [[-1.0, -0.1], [0.1, 1.0]]}
    obj = types.SimpleNamespace(prev_accepted={(6,): prev})
    prop_px = _proposal_pixels(obj, (6,), (g,), g.size)
    keep = (g <= -0.1) | (g >= 0.1)
    assert prop_px == int(np.count_nonzero(keep))
    assert prop_px < g.size                            # gap excluded from denom


def _bimodal_1d():
    g = np.linspace(0.0, 1.0, 100)
    dens = np.exp(-((g - 0.25) / 0.03) ** 2) + np.exp(-((g - 0.75) / 0.03) ** 2)
    return g, dens


def _numerator(dens, g, prev, dilation, clip):
    from pembhb.regions import truncation_region
    return truncation_region(dens, (g,), prev, policy="hard", hysteresis_weight=0.1,
                                credible=0.9999, dilation=dilation,
                                periods=(None,), clip=clip)[0]


def test_1d_numerator_excludes_gap_between_modes():
    from pembhb.regions import region_from_equal_tailed
    g, dens = _bimodal_1d()
    prev = {"kind": "1d", "intervals": [[0.1, 0.4], [0.6, 0.9]]}
    gap = (g > 0.4) & (g < 0.6)

    region = _numerator(dens, g, prev, dilation=1.0, clip=True)
    assert len(region.intervals()) == 2
    assert not (region.mask & gap).any()

    # the old equal-tailed numerator bridges the gap into one interval
    old = region_from_equal_tailed(dens / dens.sum(), (g,), eps=1e-4)
    assert len(old.intervals()) == 1
    assert (old.mask & gap).any()


def test_1d_numerator_clip_keeps_region_inside_prev():
    from pembhb.regions import prev_keep_mask
    g, dens = _bimodal_1d()
    # tight prev: the HPD set fills it, so dilation spills past its edges
    prev = {"kind": "1d", "intervals": [[0.2, 0.3], [0.7, 0.8]]}
    outside = ~prev_keep_mask(prev, (g,))

    clipped = _numerator(dens, g, prev, dilation=1.5, clip=True)
    unclipped = _numerator(dens, g, prev, dilation=1.5, clip=False)
    assert not (clipped.mask & outside).any()
    assert (unclipped.mask & outside).any()


def _cb_with_prev(prev_accepted):
    cb = object.__new__(PlotPosteriorCallback)
    cb.prev_accepted = prev_accepted
    return cb


def test_volume_ratio_matches_pixel_ratio_on_uniform_grid():
    gx, gy, MX, MY, prop = _sky_like_proposal()
    cb = _cb_with_prev({(7, 8): {"kind": "2d", "region": Region(prop, (gx, gy))}})
    post = Region(prop & (MX <= 3.35), (gx, gy))
    ratio, post_vol, prop_vol, _ = cb._volume_ratio((7, 8), post)
    px = np.count_nonzero(post.mask) / cb._proposal_pixels((7, 8), (gx, gy), prop.size)
    assert abs(ratio - px) < 1e-12
    cell = (gx[1] - gx[0]) * (gy[1] - gy[0])
    assert abs(post_vol - np.count_nonzero(post.mask) * cell) < 1e-9


def test_mode_ratios_resolve_each_proposal_mode():
    g = np.linspace(0.0, 1.0, 100)
    m0 = (g >= 0.1) & (g <= 0.4)
    m1 = (g >= 0.6) & (g <= 0.9)
    # derive the proposal from a mask, as a real round does: intervals are then
    # on cell edges and measure exactly n*dx, so an uncut mode reads exactly 1
    prev = {"kind": "1d", "intervals": Region(m0 | m1, (g,)).intervals(0)}
    cb = _cb_with_prev({(6,): prev})
    # posterior keeps all of mode 0, half of mode 1
    post = Region(m0 | ((g >= 0.6) & (g <= 0.75)), (g,))
    ratio, _, _, modes = cb._volume_ratio((6,), post)
    assert len(modes) == 2
    assert abs(modes[0] - 1.0) < 1e-12
    assert 0.45 < modes[1] < 0.6
    assert modes[1] < ratio < modes[0]


def test_no_proposal_gives_box_fraction_and_no_modes():
    g = np.linspace(0.0, 1.0, 100)
    cb = _cb_with_prev({})
    post = Region(g <= 0.25, (g,))
    ratio, _, _, modes = cb._volume_ratio((6,), post)
    assert modes == []
    assert abs(ratio - post.volume_fraction()) < 1e-12


# ----------------------------------------------------------------------
# the denominator is exact: it comes from the proposal's own representation,
# never from resampling it onto whatever grid the posterior was evaluated on
# ----------------------------------------------------------------------
def _multires_proposal():
    from pembhb.regions import MultiRegion
    fine = Region(np.ones((41, 41), dtype=bool),
                  (np.linspace(0.02, 0.22, 41), np.linspace(0.02, 0.22, 41)))
    coarse = Region(np.ones((7, 7), dtype=bool),
                    (np.linspace(0.60, 0.90, 7), np.linspace(0.60, 0.90, 7)))
    return MultiRegion([fine, coarse])


def test_proposal_volume_is_grid_independent():
    prev = _multires_proposal()
    cb = _cb_with_prev({(7, 8): {"kind": "2d", "region": prev}})
    seen, rasterised = set(), set()
    for n in (50, 137, 400):
        gx = gy = np.linspace(0.0, 1.0, n)
        post = Region(prev.contains_grid((gx, gy)), (gx, gy))
        _ratio, _post_vol, prop_vol, _modes = cb._volume_ratio((7, 8), post)
        assert prop_vol == pytest.approx(prev.volume())
        seen.add(round(prop_vol, 12))
        # what a resampled denominator would have given at this resolution
        rasterised.add(round(Region(prev.contains_grid((gx, gy)), (gx, gy)).volume(), 6))
    assert len(seen) == 1                     # one exact value at every pitch
    assert len(rasterised) > 1                # the rasterised one drifts


def test_uncut_1d_mode_reads_exactly_one_at_any_resolution():
    for n in (100, 251, 1000):
        g = np.linspace(0.0, 1.0, n)
        keep = (g >= 0.2) & (g <= 0.5)
        prev = {"kind": "1d", "intervals": Region(keep, (g,)).intervals(0)}
        cb = _cb_with_prev({(6,): prev})
        ratio, _, _, modes = cb._volume_ratio((6,), Region(keep, (g,)))
        assert ratio == pytest.approx(1.0)
        assert modes == pytest.approx([1.0])


def test_mode_ratios_sum_back_to_the_total_under_the_monotone_clip():
    g = np.linspace(0.0, 1.0, 200)
    m0 = (g >= 0.10) & (g <= 0.30)
    m1 = (g >= 0.60) & (g <= 0.95)
    prev = {"kind": "1d", "intervals": Region(m0 | m1, (g,)).intervals(0)}
    cb = _cb_with_prev({(6,): prev})
    post = Region((m0 & (g <= 0.25)) | (m1 & (g >= 0.70)), (g,))
    ratio, post_vol, prop_vol, modes = cb._volume_ratio((6,), post)
    vols = [v for v, _ in cb._proposal_modes((6,))]
    assert sum(r * v for r, v in zip(modes, vols)) == pytest.approx(post_vol)
    assert ratio == pytest.approx(post_vol / prop_vol)
