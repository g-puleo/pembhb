"""Domain-seeded truncation (option B): the proposal keeps >= credible of the
posterior normalised over the domain (round r-1's accepted set)."""
import numpy as np
import pytest

from pembhb.mask_truncation import load_truncation, save_truncation
from pembhb.regions import MultiRegion, Region, find_modes
from pembhb.truncator import Proposal, Truncator, _centres, _seed_from_prev, volume_ratio


def _evaluator(density_fn):
    """Same contract as the Truncator's evaluators: cell centres tiling the window."""
    def evaluate(bounds, n):
        c = _centres(*bounds, n)
        return density_fn(c), (c,)
    return evaluate


def _gauss(mu, sigma):
    return lambda x: np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def test_edge_mode_keeps_credible_and_reaches_the_domain_edge():
    # the cos-iota case: a domain mode 0.017 wide whose posterior peaks near its
    # lower edge; the old coarse pass lost [0.4918, 0.4938] (2.4% of the mass)
    domain = {"kind": "1d", "intervals": [[-0.5033, -0.4891], [0.4918, 0.5092]]}
    density = lambda x: _gauss(0.4975, 0.0018)(x) + 0.04 * _gauss(-0.4975, 0.0018)(x)
    stats = {}
    region, _ = find_modes(_seed_from_prev(domain, 128), _evaluator(density), ngrid=128,
                           max_depth=1, prev=domain, policy="hard", credible=0.9999,
                           dilation=1.0, periods=(None,), clip=True, stats=stats)
    assert stats["kept"] >= 0.9999
    lo = min(lo for lo, hi in region.intervals() if lo > 0)
    assert lo == pytest.approx(0.4918, abs=2e-4)


def test_proposal_never_leaves_the_domain():
    domain = {"kind": "1d", "intervals": [[0.2, 0.3], [0.6, 0.9]]}
    region, _ = find_modes(_seed_from_prev(domain, 128), _evaluator(lambda x: np.ones_like(x)),
                           ngrid=128, max_depth=1, prev=domain, policy="hard",
                           credible=0.999, dilation=1.0, periods=(None,), clip=True)
    for lo, hi in region.intervals():
        assert any(a - 1e-9 <= lo and hi <= b + 1e-9 for a, b in domain["intervals"])


def test_uncut_domain_has_volume_ratio_exactly_one():
    # cell-centred grids tile each interval exactly: no n/(n-1) overshoot
    domain = {"kind": "1d", "intervals": [[0.1, 0.4], [0.6, 0.9]]}
    region = _seed_from_prev(domain, 128)
    ratio, post, prop, modes = volume_ratio(region, domain, box_volume=1.0)
    assert ratio == pytest.approx(1.0, abs=1e-12)
    assert modes == pytest.approx([1.0, 1.0])


def test_propose_is_pure_and_commit_restore_round_trip(tmp_path):
    tr = Truncator({"mode": "mask"}, {}, ["a", "b", "c"])
    g = _centres(0.0, 1.0, 50)
    gx, gy = _centres(0.0, 2.0, 40), _centres(-1.0, 1.0, 30)
    mask2d = np.zeros((30, 40), bool)
    mask2d[5:20, 10:30] = True
    proposal = Proposal(prior={"a": [0.0, 1.0], "b": [0.0, 2.0], "c": [-1.0, 1.0]},
                        regions={(0,): MultiRegion([Region(g < 0.5, (g,))]),
                                 (1, 2): MultiRegion([Region(mask2d, (gx, gy))])})
    assert tr.prev_accepted == {}
    tr.commit(proposal)
    assert np.ravel(tr.prev_accepted[(0,)]["intervals"]) == pytest.approx([0.0, 0.5])

    yml, npz = str(tmp_path / "p.yaml"), str(tmp_path / "t.npz")
    save_truncation(yml, npz, proposal.prior, proposal.intervals_1d, proposal.masks_2d, mode="mask")
    restored = Truncator({"mode": "mask"}, {}, ["a", "b", "c"])
    restored.restore(load_truncation(yml, npz))
    assert np.ravel(restored.prev_accepted[(0,)]["intervals"]) == pytest.approx([0.0, 0.5])
    a, b = tr.prev_accepted[(1, 2)]["region"], restored.prev_accepted[(1, 2)]["region"]
    assert b.volume() == pytest.approx(a.volume())


# ----------------------------------------------------------------------
# proposal_coverage: the next-round proposal built per simulation
# ----------------------------------------------------------------------
def _fake_ratios(centres_by_sim, sigma, calls):
    """Stand-in for truncator._ratios_1d: a Gaussian per simulation, centred at
    that simulation's posterior centre; records the batch size of each call."""
    import torch

    def ratios(model, loader, in_idx, out_idx, bounds, n):
        c = _centres(*bounds, n)
        mu = torch.cat([b["post_centre"] for b in loader]).numpy()
        calls.append(len(mu))
        return np.exp(-0.5 * ((c[None, :] - mu[:, None]) / sigma) ** 2), (c,)
    return ratios


def test_proposal_coverage_counts_truths_inside_each_simulation_proposal(monkeypatch):
    import torch
    from pembhb import truncator as tmod

    truths = np.array([0.40, 0.50, 0.55, 0.45, 0.50])
    centres = truths.copy()
    centres[1] = 0.80          # one overconfident, displaced posterior
    sims = {"source_parameters": torch.tensor(truths[:, None]),
            "post_centre": torch.tensor(centres)}
    calls = []
    monkeypatch.setattr(tmod, "_ratios_1d", _fake_ratios(centres, 0.01, calls))

    tr = Truncator({"mode": "mask", "credible_level_1d": 0.9999, "dilation_1d": 1.0,
                    "ngrid_1d_refined": 128}, {}, ["a"])
    tr.prev_accepted[(0,)] = {"kind": "1d", "intervals": [[0.3, 0.9]]}
    cov = tmod.proposal_coverage(tr, None, sims, [((0,), 0)], {"a": [0.0, 1.0]})

    assert cov[(0,)] == (0.8, 4, 5)
    # with a domain every window is shared: one batched call for all simulations
    assert calls == [5]


def test_proposal_coverage_round_one_refines_per_simulation(monkeypatch):
    import torch
    from pembhb import truncator as tmod

    truths = np.array([0.2, 0.7, 0.45])
    sims = {"source_parameters": torch.tensor(truths[:, None]),
            "post_centre": torch.tensor(truths)}
    calls = []
    monkeypatch.setattr(tmod, "_ratios_1d", _fake_ratios(truths, 0.02, calls))

    tr = Truncator({"mode": "mask", "credible_level_1d": 0.9999, "dilation_1d": 1.0,
                    "ngrid_1d": 100, "ngrid_1d_refined": 128}, {}, ["a"])
    cov = tmod.proposal_coverage(tr, None, sims, [((0,), 0)], {"a": [0.0, 1.0]})

    assert cov[(0,)] == (1.0, 3, 3)
    # the coarse box pass is shared; each simulation's own modes are refined alone
    assert calls[0] == 3 and all(n == 1 for n in calls[1:]) and len(calls) > 1


def test_proposal_coverage_matches_the_real_proposal(monkeypatch):
    # The per-simulation region must be exactly what Truncator.propose would build
    # if that simulation were the observation.
    import torch
    from pembhb import truncator as tmod

    truths = np.array([0.42, 0.61])
    sims = {"source_parameters": torch.tensor(truths[:, None]),
            "post_centre": torch.tensor(truths)}
    monkeypatch.setattr(tmod, "_ratios_1d", _fake_ratios(truths, 0.015, []))
    conf = {"mode": "mask", "credible_level_1d": 0.999, "dilation_1d": 1.0,
            "ngrid_1d_refined": 128}
    tr = Truncator(conf, {}, ["a"])
    tr.prev_accepted[(0,)] = {"kind": "1d", "intervals": [[0.3, 0.8]]}

    regions = []
    real = tr._mask_region

    def spy(*args, **kwargs):
        region, kept = real(*args, **kwargs)
        regions.append(region)
        return region, kept
    monkeypatch.setattr(tr, "_mask_region", spy)
    tmod.proposal_coverage(tr, None, sims, [((0,), 0)], {"a": [0.0, 1.0]})

    for s, region in enumerate(regions):
        expected, _ = real(None, None, (0,), 0, {"a": [0.0, 1.0]}, False,
                           evaluate=lambda b, n, s=s: (tmod._ratios_1d(
                               None, [{"post_centre": sims["post_centre"][s:s + 1]}],
                               0, 0, b, n)[0][0], (_centres(*b, n),)))
        assert np.ravel(region.intervals()) == pytest.approx(np.ravel(expected.intervals()))
