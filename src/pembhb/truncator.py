"""Prior truncation: one procedure for the round-end proposal and the training metric.

``Truncator.propose`` turns the model's posterior at the observation into the
next round's proposal; it is pure (the previous accepted set, the *domain*, is
only read). ``commit`` makes a proposal the domain of the next round. The same
``propose`` serves the round end (``tmnre_joint``) and ``TruncationMonitor``,
whose ``volume_ratio`` therefore measures exactly the set the round proposes.

Two strategies, ``truncation.mode``:
  * ``mask``      — round r's posterior lives on round r-1's accepted set (the
                    domain: it was trained there, and the density is zeroed
                    outside it). Each mode of the domain is evaluated on its own
                    grid; one HPD level, normalised over the whole domain, gives
                    the accepted set (one pass of :func:`pembhb.regions.find_modes`).
                    Round 1 has no domain: a coarse pass over the prior box
                    finds the modes;
  * ``rectangle`` — legacy boxes: 1-eps widest interval per 1D marginal, the
                    main-mode box for the sky, contour boxes for other pairs.
"""

import copy
from dataclasses import dataclass, field

import numpy as np
import torch

from pembhb import utils
from pembhb.mask_truncation import _MASK_PERIOD_BY_NAME
from pembhb.regions import (MultiRegion, Region, _as_multiregion, find_modes,
                            prev_keep_mask, truncation_region)


def _envelope(intervals):
    return [min(lo for lo, _ in intervals), max(hi for _, hi in intervals)]


@dataclass
class Proposal:
    """One round's truncation: accepted regions per marginal and their envelope."""
    prior: dict                                   # {param: [lo, hi]} bounding box
    regions: dict = field(default_factory=dict)   # {marginal_key: MultiRegion} (mask mode)
    kept: dict = field(default_factory=dict)      # {marginal_key: posterior mass kept}

    @property
    def intervals_1d(self):
        return {k[0]: r.intervals() for k, r in self.regions.items() if len(k) == 1}

    @property
    def masks_2d(self):
        return [{"idx": k, "region": r} for k, r in self.regions.items() if len(k) == 2]


class Truncator:
    """Owns the truncation config and the domain (previous accepted set)."""

    def __init__(self, trunc_conf, rect_conf, prior_keys):
        self.conf = trunc_conf
        self.rect_conf = rect_conf          # truncation_veto block: rectangle boxes
        self.prior_keys = prior_keys
        self.mode = trunc_conf.get("mode", "rectangle")
        if self.mode not in ("rectangle", "mask"):
            raise ValueError(f"truncation.mode must be 'rectangle' or 'mask', got {self.mode!r}")
        # {marginal_key: {"kind": "1d", "intervals": [...]} | {"kind": "2d", "region": MultiRegion}}
        self.prev_accepted = {}
        self._monotonic_warned = False

    # ------------------------------------------------------------------ config
    @property
    def policy(self):
        return str(self.conf.get("zero_outside_prev_mask", "hard"))

    def monotonic(self):
        """``A_N ⊆ A_{N-1}``: only meaningful with the ``hard`` zero policy."""
        if not bool(self.conf.get("enforce_monotonic_regions", False)):
            return False
        if self.policy != "hard":
            if not self._monotonic_warned:
                print(f"[Trunc/mask/§2] enforce_monotonic_regions ignored: it "
                      f"requires zero_outside_prev_mask='hard' but policy is "
                      f"{self.policy!r}.", flush=True)
                self._monotonic_warned = True
            return False
        return True

    # ------------------------------------------------------------------ state
    def commit(self, proposal):
        """Make ``proposal`` the domain of the next round."""
        for key, region in proposal.regions.items():
            if len(key) == 1:
                self.prev_accepted[key] = {"kind": "1d", "intervals": region.intervals()}
            else:
                self.prev_accepted[key] = {"kind": "2d", "region": region}

    def restore(self, saved):
        """Domain from :func:`pembhb.mask_truncation.load_truncation` (``--resume``)."""
        for idx, ivs in saved["intervals_1d"].items():
            self.prev_accepted[(int(idx),)] = {"kind": "1d", "intervals": ivs}
        for m in saved["masks_2d"]:
            self.prev_accepted[tuple(m["idx"])] = {"kind": "2d", "region": m["region"]}

    # ------------------------------------------------------------------ propose
    def propose(self, model, obs_loader, marginals, prior, verbose=False):
        """Proposal from ``model`` at the observation; does not change the domain.

        ``marginals`` is an iterable of ``(marginal_key, out_idx)``; ``prior`` the
        current bounding box, updated per marginal into ``Proposal.prior``.
        """
        proposal = Proposal(prior=copy.deepcopy(prior))
        for key, out_idx in marginals:
            if verbose:
                print(f"[Trunc/{self.mode}] {'-'.join(self.prior_keys[i] for i in key)} "
                      f"(head {out_idx}) ...", flush=True)
            if self.mode == "mask":
                region, kept = self._mask_region(model, obs_loader, key, out_idx, prior, verbose)
                proposal.regions[key] = region
                proposal.kept[key] = kept
                if len(key) == 1:
                    proposal.prior[self.prior_keys[key[0]]] = _envelope(region.intervals())
                else:
                    for axis, idx in enumerate(key):
                        proposal.prior[self.prior_keys[idx]] = _envelope(region.intervals(axis))
            else:
                self._rectangle(model, obs_loader, key, out_idx, proposal.prior)
        return proposal

    def _mask_region(self, model, obs_loader, key, out_idx, prior, verbose, evaluate=None):
        """``(MultiRegion, kept)`` of one marginal (mask mode).

        ``evaluate(bounds, n)`` defaults to the network at ``obs_loader``'s single
        observation; :func:`proposal_coverage` injects cached rows instead.
        """
        prev = self.prev_accepted.get(key)
        names = [self.prior_keys[i] for i in key]
        periods = tuple(_MASK_PERIOD_BY_NAME.get(n) for n in names)
        ndim = len(key)
        credible = float(self.conf.get("credible_level_1d" if ndim == 1 else "credible_level_2d", 0.997))
        dilation = float(self.conf.get("dilation_1d" if ndim == 1 else "dilation_2d", 1.2))
        n_ref = int(self.conf.get("ngrid_1d_refined" if ndim == 1 else "ngrid_2d_refined",
                                  128 if ndim == 1 else 64))

        region = _seed_from_prev(prev, n_ref)
        if evaluate is None:
            evaluate = (_evaluator_1d(model, obs_loader, key[0], out_idx) if ndim == 1
                        else _evaluator_2d(model, obs_loader, key, out_idx))
        if region is None:
            # Round 1 (no domain): coarse pass over the prior box finds the modes.
            box = tuple(tuple(prior[n]) for n in names)
            density, grids = evaluate(box[0] if ndim == 1 else box,
                                      int(self.conf.get("ngrid_1d" if ndim == 1 else "ngrid_2d", 100)))
            region, _ = truncation_region(
                density / np.sum(density), grids, None, policy=self.policy,
                hysteresis_weight=None, credible=credible, dilation=dilation,
                periods=periods, clip=False)

        n_before = len(region.components())
        stats = {}
        # One pass: deeper passes re-test domain membership on misaligned grids
        # and drop cells along the domain edge (measured: sky kept 0.9998 ->
        # 0.9945 at depth 4). Resolution still compounds across rounds, since
        # the next round's windows are this round's modes.
        region, _modes = find_modes(
            region, evaluate, ngrid=n_ref, max_depth=1, prev=prev,
            policy=self.policy, credible=credible, dilation=dilation,
            periods=periods, clip=self.monotonic(), stats=stats)
        if verbose:
            src = "domain" if prev else "coarse"
            kept = "n/a" if stats.get("kept") is None else f"{stats['kept']:.5f}"
            print(f"[Trunc/mask] {'-'.join(names)}: {n_before} {src} mode(s) -> "
                  f"{len(region.parts)}, keeps {kept} of the posterior", flush=True)
        return _as_multiregion(region), stats.get("kept")

    def _rectangle(self, model, obs_loader, key, out_idx, prior):
        """Legacy box truncation of one marginal, applied to ``prior`` in place."""
        names = [self.prior_keys[i] for i in key]
        if len(key) == 1:
            interval, _, _, _ = utils.get_widest_interval_1d(
                model, obs_loader, in_param_idx=key[0], out_param_idx=out_idx,
                eps=float(self.rect_conf.get("eps_1d", 1e-4)),
                dilation=float(self.rect_conf.get("dilation_1d", 1.0)))
            prior[names[0]] = [float(interval[0]), float(interval[1])]
        elif key == (7, 8):
            from pembhb.sky_truncation import truncate_sky_prior
            conf = {"prior": prior}
            truncate_sky_prior(
                model, obs_loader, out_param_idx=out_idx, datagen_conf=conf,
                mode="rectangle",
                credible_level=float(self.rect_conf.get("sky_credible_level", 0.999)),
                dilation_factor=float(self.rect_conf.get("sky_dilation", 1.1)))
            prior.update(conf["prior"])
        else:
            norm_2d, _, gx, gy, _, _ = utils.eval_posterior_2d(model, obs_loader, key, out_idx)
            levels, _ = utils.contour_levels(norm_2d)
            boxes, cs = utils.contour_boxes(gx, gy, norm_2d, levels)
            import matplotlib.pyplot as plt
            plt.close(cs.axes.figure)   # contour_boxes leaves its own figure open
            box = boxes[0]
            prior[names[0]] = [float(box[0]), float(box[1])]
            prior[names[1]] = [float(box[2]), float(box[3])]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _seed_from_prev(prev, ngrid):
    """Previous round's accepted set as a ``MultiRegion``, or ``None``.

    1D rebuilds from the stored intervals (lossless: a leaf mode is one
    contiguous interval); 2D already carries a ``MultiRegion``.
    """
    if not prev:
        return None
    if prev.get("kind") == "2d":
        region = prev.get("region")
        return None if region is None else _as_multiregion(region)
    parts = [Region(np.ones(int(ngrid), dtype=bool), (_centres(lo, hi, ngrid),))
             for lo, hi in prev.get("intervals") or [] if float(hi) > float(lo)]
    return MultiRegion(parts) if parts else None


def _centres(lo, hi, n):
    """``n`` cell centres tiling ``[lo, hi]`` exactly (cells, not edges, on the grid)."""
    lo, hi, n = float(lo), float(hi), int(n)
    return lo + (np.arange(n) + 0.5) * (hi - lo) / n


def _ratios_1d(model, loader, in_idx, out_idx, bounds, n):
    """Raw 1D ratios on ``n`` cell centres in ``bounds``, one row per observation."""
    c = _centres(*bounds, n)
    logratios, _inj, _grid = utils.get_logratios_grid(
        loader, model, ngrid_points=int(n), in_param_idx=in_idx,
        out_param_idx=out_idx, low=float(c[0]), high=float(c[-1]), grid_chunk_size=int(n))
    return np.exp(logratios), (c,)


def _ratios_2d(model, loader, in_idx, out_idx, bounds, n):
    """Raw 2D ratios on ``n``×``n`` cell centres in ``bounds``, one map per observation."""
    cx, cy = _centres(*bounds[0], n), _centres(*bounds[1], n)
    logratios, _inj, _gx, _gy = utils.get_logratios_grid_2d(
        loader, model, ngrid_points=int(n), out_param_idx=out_idx,
        in_param_idx=tuple(in_idx), bounds_0=(float(cx[0]), float(cx[-1])),
        bounds_1=(float(cy[0]), float(cy[-1])), grid_chunk_size=int(n) ** 2)
    return np.exp(logratios), (cx, cy)


def _evaluator_1d(model, obs_loader, in_idx, out_idx):
    """``evaluate(bounds, n)`` for :func:`refine_region` — raw 1D ratios.

    Raw, not normalised: the refined modes are pooled into one volume-weighted
    threshold, so each must stay on the common scale.
    """
    def _evaluate(bounds, n):
        density, grids = _ratios_1d(model, obs_loader, in_idx, out_idx, bounds, n)
        return density[0], grids
    return _evaluate


def _evaluator_2d(model, obs_loader, in_idx, out_idx):
    """``evaluate(bounds, n)`` for :func:`refine_region` — raw 2D ratios."""
    def _evaluate(bounds, n):
        density, grids = _ratios_2d(model, obs_loader, in_idx, out_idx, bounds, n)
        return density[0], grids
    return _evaluate


def proposal_coverage(truncator, model, sims, marginals, prior):
    """Coverage of the next-round proposal over labelled simulations.

    For every simulation, build the proposal from its own data with exactly the
    code that builds the real one (:meth:`Truncator._mask_region` on the current
    domain) and test whether its true parameters fall inside. Returns
    ``{marginal_key: (coverage, n_inside, n_total)}``.

    ``sims`` is one materialised batch (fixed noise: every window of a
    simulation must see the same data). Network evaluations are batched: a
    window requested by every simulation (the domain's modes; in round 1 the
    coarse pass over the box) is evaluated once for all of them and each
    simulation reads its own row. Round 1's refined windows depend on each
    simulation's own modes and are evaluated for that simulation alone.
    """
    if truncator.mode != "mask":
        raise ValueError("proposal_coverage supports truncation.mode='mask' only")
    n_total = int(sims["source_parameters"].shape[0])
    truths = sims["source_parameters"].detach().cpu().numpy()
    all_sims = [sims]
    coverage = {}
    for key, out_idx in marginals:
        ndim = len(key)
        ratios = _ratios_1d if ndim == 1 else _ratios_2d
        in_idx = key[0] if ndim == 1 else key
        has_domain = truncator.prev_accepted.get(key) is not None
        cache = {}

        def _key(bounds, n):
            flat = np.asarray(bounds, dtype=float).reshape(-1)
            return (tuple(np.round(flat, 15)), int(n))

        n_inside = 0
        for s in range(n_total):
            calls = [0]

            def evaluate(bounds, n, s=s, calls=calls):
                k = _key(bounds, n)
                shared = has_domain or calls[0] == 0
                calls[0] += 1
                if k in cache:
                    density, grids = cache[k]
                    return density[s], grids
                if shared:
                    cache[k] = ratios(model, all_sims, in_idx, out_idx, bounds, n)
                    return cache[k][0][s], cache[k][1]
                one = [{name: v[s:s + 1] if torch.is_tensor(v) else v
                        for name, v in sims.items()}]
                density, grids = ratios(model, one, in_idx, out_idx, bounds, n)
                return density[0], grids

            region, _kept = truncator._mask_region(
                model, None, key, out_idx, prior, verbose=False, evaluate=evaluate)
            point = truths[s, list(key)].reshape(ndim, 1)
            n_inside += int(region.contains(point)[0])
        coverage[key] = (n_inside / n_total, n_inside, n_total)
    return coverage


def proposal_modes(prev):
    """Per-mode ``(exact volume, membership(grids))`` of the domain ``prev``.

    Volumes come from the domain's own representation (stored intervals in 1D,
    each subgrid's pitch in 2D), never from a resampling. ``[]`` without a domain.
    """
    if prev is None:
        return []
    if prev["kind"] == "1d":
        return [
            (float(hi) - float(lo),
             lambda grids, lo=float(lo), hi=float(hi):
                 (np.asarray(grids[0], dtype=float) >= lo)
                 & (np.asarray(grids[0], dtype=float) <= hi))
            for lo, hi in prev["intervals"]
        ]
    return [(c.volume(), c.contains_grid) for c in prev["region"].components()]


def volume_ratio(region, prev, box_volume):
    """``|A_N| / |A_{N-1}|`` as Σ mask·dV, plus one ratio per domain mode.

    ``region`` is a ``Region`` or ``MultiRegion`` (the proposal of one marginal),
    ``prev`` the domain entry (or None), ``box_volume`` the prior-box volume used
    when there is no domain or it does not overlap the proposal's grids. Mode
    ratios are ``|A_N ∩ mode_k| / |mode_k|`` in the domain's mode order.
    """
    parts = getattr(region, "parts", (region,))
    post_vol = float(sum(p.volume() for p in parts))
    modes = proposal_modes(prev)
    masks = [[membership(p.grids) for p in parts] for _, membership in modes]
    if not any(m.any() for per_mode in masks for m in per_mode):
        return post_vol / box_volume, post_vol, box_volume, []
    prop_vol = float(sum(vol for vol, _ in modes))
    mode_ratios = [sum(p.volume(p.mask & m) for p, m in zip(parts, per_mode)) / vol
                   for (vol, _), per_mode in zip(modes, masks)]
    return post_vol / prop_vol, post_vol, prop_vol, mode_ratios


def box_volume(prior, key, prior_keys):
    return float(np.prod([prior[prior_keys[i]][1] - prior[prior_keys[i]][0] for i in key]))


__all__ = ["Proposal", "Truncator", "volume_ratio", "proposal_modes", "box_volume",
           "prev_keep_mask", "proposal_coverage"]
