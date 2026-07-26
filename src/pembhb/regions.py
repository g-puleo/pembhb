"""One accepted set of a marginal, as a boolean mask on a grid.

Background: ``MASK_TRUNCATION_OPEN_ISSUES.md`` §1 and the design in
``.claude/plans/region-refactor.md``.  The concept *the accepted region of a
marginal* was implemented eleven times across the codebase; this module gives it
a single home.

The central simplification is that **a box is a mask** — a rectangular one.
Once both are boolean arrays on a grid, ``contains``, ``volume_fraction``,
``bounds`` and ``draw`` each have exactly one implementation.  Only *construction*
differs, so there is one :class:`Region` class and several builder functions
(``region_from_hpd``, ``region_from_equal_tailed``, ``region_from_main_mode``,
``region_from_bounds``) — not a class hierarchy.

Axis convention (matches :mod:`pembhb.mask_truncation`):

* 1D: ``grids = (grid_x,)`` with ``grid_x`` the pixel centres; ``mask`` is
  ``(n_x,)``; ``periodic = (periodic_x,)``.
* 2D: ``grids = (grid_x, grid_y)`` (pixel centres per axis); ``mask`` is
  ``(n_y, n_x) = (rows, cols)``; ``periodic = (periodic_x, periodic_y)`` in
  grid-axis order.  The connected-component helpers want periodicity in
  ``(row, col) = (y, x)`` order, so the swap happens internally — see
  ``mask_truncation.py`` line 184 for the same convention.

``contains(values)`` is the primitive: a nearest-pixel lookup into the mask.
Everything else is one line on top of it or on the stored boolean array.
"""

import numpy as np

# Reuse the tested internals verbatim — these already carry the hard-won fixes
# (axis transposition, periodic union-find, torus corners, high-fill-fraction
# rejection sizing) documented in MASK_TRUNCATION_OPEN_ISSUES.md.
from pembhb.mask_truncation import (
    _hpd_threshold,
    _periodic_labelling,
    _periodic_labelling_2d,
    _dilate_mode,
    _dilate_mode_2d,
    _intervals_from_indices,
    components_from_labels,
    _sample_from_intervals,
    _sample_2d_from_components,
)


def _axis_cell_size(grid1d):
    """Pixel pitch of a 1D grid of centres (assumed uniform)."""
    g = np.asarray(grid1d, dtype=float)
    return float(g[1] - g[0])


def _nearest_index(values, grid1d):
    """Round each value to the nearest pixel index on ``grid1d`` (unclipped).

    Grids hold pixel *centres*, so the nearest centre is ``round``, not
    ``floor``.  Returned indices may fall outside ``[0, n-1]``; the caller is
    responsible for the in-range test (an out-of-grid value is *not contained*,
    it must never be clipped onto an edge pixel — see ``truth_violations``).
    """
    g = np.asarray(grid1d, dtype=float)
    dx = g[1] - g[0]
    return np.round((np.asarray(values, dtype=float) - g[0]) / dx).astype(int)


class Region:
    """The accepted set of one marginal (1D or 2D), as a boolean grid mask.

    Parameters
    ----------
    mask : bool array
        ``(n_x,)`` for 1D or ``(n_y, n_x)`` for 2D.
    grids : tuple of 1D arrays
        Pixel centres, axis order ``(grid_x,)`` or ``(grid_x, grid_y)``.
    periodic : tuple of bool, optional
        Per axis, in grid order ``(periodic_x[, periodic_y])``.  This is the
        *effective* periodicity — already resolved by the builder against
        whether the grid still spans the full period.  Defaults to all-False.
    """

    def __init__(self, mask, grids, periodic=None):
        self.grids = tuple(np.asarray(g, dtype=float).reshape(-1) for g in grids)
        self.ndim = len(self.grids)
        if self.ndim not in (1, 2):
            raise ValueError(f"Region supports 1D or 2D, got ndim={self.ndim}")
        self.mask = np.asarray(mask, dtype=bool)

        if self.ndim == 1:
            expected = (self.grids[0].shape[0],)
        else:
            # mask is (rows=y, cols=x)
            expected = (self.grids[1].shape[0], self.grids[0].shape[0])
        if self.mask.shape != expected:
            raise ValueError(
                f"mask shape {self.mask.shape} does not match grids "
                f"(expected {expected}; remember 2D masks are (n_y, n_x))")

        if periodic is None:
            periodic = (False,) * self.ndim
        self.periodic = tuple(bool(p) for p in periodic)
        if len(self.periodic) != self.ndim:
            raise ValueError("periodic must have one entry per axis")

        self._labels_cache = None

    # ------------------------------------------------------------------
    # labelling (lazy) — re-derived from the boolean mask so components,
    # intervals, sampling and persistence can never disagree with `contains`.
    # ------------------------------------------------------------------
    def _labels(self):
        if self._labels_cache is not None:
            return self._labels_cache
        if self.ndim == 1:
            labels, _ = _periodic_labelling(self.mask, self.periodic[0])
        else:
            # helpers index periodicity as (row, col) = (y, x)
            periodic_rowcol = (self.periodic[1], self.periodic[0])
            labels, _ = _periodic_labelling_2d(self.mask, periodic_rowcol)
        self._labels_cache = labels
        return labels

    # ------------------------------------------------------------------
    # the primitive
    # ------------------------------------------------------------------
    def contains(self, values):
        """Membership test by nearest-pixel lookup.

        ``values`` is ``(ndim, N)`` (or ``(N,)``/scalar for 1D).  Returns a
        ``(N,)`` bool array.  A point whose nearest pixel falls outside the grid
        is **not** contained — it is never clipped onto an edge pixel.
        """
        values = np.asarray(values, dtype=float)
        if self.ndim == 1:
            v = values.reshape(-1) if values.ndim <= 1 else values[0]
            col = _nearest_index(v, self.grids[0])
            n = self.grids[0].shape[0]
            in_range = (col >= 0) & (col < n)
            out = np.zeros(col.shape, dtype=bool)
            safe = np.clip(col, 0, n - 1)
            out[in_range] = self.mask[safe][in_range]
            return out

        if values.ndim == 1:
            values = values.reshape(2, 1)
        vx, vy = values[0], values[1]
        col = _nearest_index(vx, self.grids[0])
        row = _nearest_index(vy, self.grids[1])
        n_col = self.grids[0].shape[0]
        n_row = self.grids[1].shape[0]
        in_range = (col >= 0) & (col < n_col) & (row >= 0) & (row < n_row)
        out = np.zeros(col.shape, dtype=bool)
        safe_col = np.clip(col, 0, n_col - 1)
        safe_row = np.clip(row, 0, n_row - 1)
        out[in_range] = self.mask[safe_row, safe_col][in_range]
        return out

    def contains_grid(self, grids):
        """Evaluate membership at every point of another grid.

        ``grids`` is a tuple of new-grid pixel centres in the same axis order.
        Returns a bool array shaped like the *new* grid's mask: ``(m_x,)`` for
        1D or ``(m_y, m_x)`` for 2D.  This is the resampling used by §2 to zero a
        new round's density outside the region the network was trained on.
        """
        new = tuple(np.asarray(g, dtype=float).reshape(-1) for g in grids)
        if len(new) != self.ndim:
            raise ValueError("grid dimensionality mismatch")
        if self.ndim == 1:
            return self.contains(new[0])
        gx, gy = new
        mesh_x, mesh_y = np.meshgrid(gx, gy, indexing="xy")  # (m_y, m_x)
        pts = np.stack([mesh_x.reshape(-1), mesh_y.reshape(-1)], axis=0)
        return self.contains(pts).reshape(mesh_x.shape)

    # ------------------------------------------------------------------
    # measures / summaries — one line each on top of the boolean array
    # ------------------------------------------------------------------
    def volume_fraction(self):
        """Fraction of the grid (== the trained prior box) that is accepted."""
        return float(np.count_nonzero(self.mask)) / float(self.mask.size)

    def bounds(self):
        """Per-axis outer envelope ``[lo, hi]`` of the accepted pixels.

        1D returns ``[lo, hi]``; 2D returns ``[[x_lo, x_hi], [y_lo, y_hi]]``.
        For a wrapped (periodic) axis whose accepted set touches both grid ends
        this is the full axis span — the same value ``_envelope`` of the
        sub-intervals produces.
        """
        if self.ndim == 1:
            idx = np.where(self.mask)[0]
            if idx.size == 0:
                return [float(self.grids[0][0]), float(self.grids[0][0])]
            return [float(self.grids[0][idx[0]]), float(self.grids[0][idx[-1]])]
        col_any = np.where(np.any(self.mask, axis=0))[0]
        row_any = np.where(np.any(self.mask, axis=1))[0]
        if col_any.size == 0 or row_any.size == 0:
            gx0, gy0 = float(self.grids[0][0]), float(self.grids[1][0])
            return [[gx0, gx0], [gy0, gy0]]
        return [
            [float(self.grids[0][col_any[0]]), float(self.grids[0][col_any[-1]])],
            [float(self.grids[1][row_any[0]]), float(self.grids[1][row_any[-1]])],
        ]

    def intervals(self, axis=0):
        """Accepted sub-intervals along ``axis``, wrap-aware.

        Iterates over connected components so a wrapped component yields its two
        sub-intervals (``_intervals_from_indices`` expects contiguous or
        genuinely-wrapping index sets — never a union of separate modes).
        """
        labels = self._labels()
        grid = self.grids[axis]
        out = []
        for k in np.unique(labels):
            if k == 0:
                continue
            if self.ndim == 1:
                idx = np.where(labels == k)[0]
            else:  # x: project over rows
                idx = np.where(np.any(labels == k, axis=axis))[0]

            out.extend(_intervals_from_indices(idx, grid))
        return out

    def components(self):
        """List of single-component :class:`Region` objects, largest not first.

        Each carries the same grids/periodicity; its mask is one connected
        component of ``self``.
        """
        labels = self._labels()
        comps = []
        for k in np.unique(labels):
            if k == 0:
                continue
            comps.append(Region(labels == k, self.grids, self.periodic))
        return comps

    def main_component(self, density=None):
        """The dominant connected component.

        When a ``density`` array on this region's grid is supplied the component
        is chosen by its integrated posterior **mass** (``∫ p`` over the
        component) — the physically meaningful "main mode": a broad, shallow
        blob only wins over a compact, tall one if it truly holds more
        probability.  Without a density it falls back to the largest accepted
        *area* (``volume_fraction``).
        """
        comps = self.components()
        if not comps:
            return self
        if density is None:
            return max(comps, key=lambda r: r.volume_fraction())
        dens = np.asarray(density, dtype=float)
        # The cell volume is constant across components, so summing the density
        # over each component's pixels ranks them by integrated mass (the
        # constant Δ-volume factor cancels in the argmax).
        return max(comps, key=lambda r: float(np.sum(dens[r.mask])))

    # ------------------------------------------------------------------
    # sampling
    # ------------------------------------------------------------------
    def draw(self, n, rng, cube=False):
        """Draw ``n`` points uniformly (in the mask) from the accepted set.

        Returns ``(x,)`` for 1D or ``(x, y)`` for 2D — each a length-``n``
        array.  ``cube=True`` (1D only) draws uniform in ``value**3`` (the
        distance / uniform-in-volume path).
        """
        if self.ndim == 1:
            return _sample_from_intervals(self.intervals(0), n, rng, cube=cube)
        labels = self._labels()
        comps = components_from_labels(labels, self.grids[0], self.grids[1])
        x, y, _acc = _sample_2d_from_components(
            comps, labels, self.grids[0], self.grids[1], n, rng)
        return x, y

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    def to_arrays(self):
        """Serialise to plain arrays: ``{mask (int8), grids..., periodic}``.

        Stores the *labels* (int8) for 2D so downstream per-component sampling
        matches the mask-truncation ``.npz`` layout; 1D stores the boolean mask.
        """
        payload = {
            "ndim": self.ndim,
            "periodic": np.asarray(self.periodic, dtype=bool),
            "grid_x": self.grids[0],
        }
        if self.ndim == 1:
            payload["mask"] = self.mask.astype(np.int8)
        else:
            payload["grid_y"] = self.grids[1]
            labels = self._labels()
            if labels.max() > 127:
                raise ValueError(
                    f"region has {labels.max()} components; int8 storage "
                    f"supports at most 127.")
            payload["labels"] = labels.astype(np.int8)
        return payload

    @classmethod
    def from_arrays(cls, payload):
        ndim = int(payload["ndim"])
        periodic = tuple(bool(p) for p in np.asarray(payload["periodic"]))
        if ndim == 1:
            return cls(np.asarray(payload["mask"]).astype(bool),
                       (payload["grid_x"],), periodic)
        labels = np.asarray(payload["labels"]).astype(int)
        return cls(labels > 0, (payload["grid_x"], payload["grid_y"]), periodic)


# ======================================================================
# Builders — each replaces one scattered implementation.
# ======================================================================

def _effective_periodic(grid1d, period):
    """A grid axis is *effectively* periodic only when it still spans the full
    period.  An already-truncated interior window is not periodic even if the
    parameter is — the seam is no longer physical."""
    if period is None:
        return False
    g = np.asarray(grid1d, dtype=float)
    return bool(np.isclose(g[-1] - g[0], period))


def region_from_hpd(density, grids, credible_level, dilation_factor=1.0,
                    periods=None):
    """HPD level-set region (replaces ``analyse_posterior_1d/2d``).

    ``density`` is the normalised posterior on the grid; ``grids`` the pixel
    centres ``(grid_x[, grid_y])``; ``periods`` the *physical* period per axis in
    grid order (``None`` for non-periodic), from which effective periodicity is
    derived.  The returned :class:`Region` mask is the dilated level set.
    """
    grids = tuple(np.asarray(g, dtype=float).reshape(-1) for g in grids)
    ndim = len(grids)
    if periods is None:
        periods = (None,) * ndim

    if ndim == 1:
        grid_x = grids[0]
        periodic = _effective_periodic(grid_x, periods[0])
        thr = _hpd_threshold(density, credible_level=credible_level)
        mask = np.asarray(density) >= thr
        labelled, _ = _periodic_labelling(mask, periodic)
        dilated = np.zeros_like(mask)
        for k in np.unique(labelled):
            if k == 0:
                continue
            comp = labelled == k
            half_width_pixels = comp.sum() / 2
            n_inflate = max(1, int(np.ceil(half_width_pixels * (dilation_factor - 1))))
            dilated |= _dilate_mode(comp, iterations=n_inflate, periodic=periodic)
        return Region(dilated, (grid_x,), periodic=(periodic,))

    grid_x, grid_y = grids
    periodic_x = _effective_periodic(grid_x, periods[0])
    periodic_y = _effective_periodic(grid_y, periods[1])
    periodic_rowcol = (periodic_y, periodic_x)   # helpers index (row=y, col=x)
    thr = _hpd_threshold(density, credible_level)
    mask = np.asarray(density) >= thr
    labelled, n_comp = _periodic_labelling_2d(mask, periodic_rowcol)
    dilated = np.zeros_like(mask)
    for i in range(1, n_comp + 1):
        comp = labelled == i
        eff_radius = np.sqrt(np.sum(comp) / np.pi)
        n_iter = max(1, int(np.ceil(eff_radius * (dilation_factor - 1))))
        dilated |= _dilate_mode_2d(comp, n_iter, periodic_rowcol)
    return Region(dilated, (grid_x, grid_y), periodic=(periodic_x, periodic_y))


def region_from_equal_tailed(density, grids, eps, dilation=1.0):
    """Equal-tailed 1D interval (replaces ``get_widest_interval_1d`` and its
    inline copy in ``compute_truncation_coverage``).

    ``density`` need not be normalised.  The ``[eps/2, 1-eps/2]`` interval is
    rasterised onto the grid rounding **outward** (floor the low edge, ceil the
    high edge) so the region is always a superset of the old box — never a
    subset that could exclude a truth the old code kept.
    """
    grids = tuple(np.asarray(g, dtype=float).reshape(-1) for g in grids)
    if len(grids) != 1:
        raise ValueError("region_from_equal_tailed is 1D only")
    grid_x = grids[0]
    dp = grid_x[1] - grid_x[0]
    dens = np.asarray(density, dtype=float)
    norm = dens / np.sum(dens * dp)
    cumsum = np.cumsum(norm * dp)
    idx_low = int(np.searchsorted(cumsum, eps / 2))
    idx_high = min(int(np.searchsorted(cumsum, 1 - eps / 2)), grid_x.shape[0] - 1)
    lo, hi = float(grid_x[idx_low]), float(grid_x[idx_high])
    if dilation != 1.0:
        c = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo) * dilation
        lo = max(c - half, float(grid_x[0]))
        hi = min(c + half, float(grid_x[-1]))
        # rasterise the dilated edges outward so the region is a superset
        idx_low = max(0, int(np.floor((lo - grid_x[0]) / dp)))
        idx_high = min(grid_x.shape[0] - 1, int(np.ceil((hi - grid_x[0]) / dp)))
    mask = np.zeros(grid_x.shape[0], dtype=bool)
    mask[idx_low:idx_high + 1] = True
    return Region(mask, (grid_x,))


def region_from_main_mode(density, grids, credible_level, dilation_factor=1.0,
                          periods=None):
    """Highest-mass connected component of the HPD level set (replaces
    ``get_main_mode_box`` for pipeline use).

    A one-line composition: ``region_from_hpd(...).main_component(density)``.
    The main mode is the component holding the most posterior *mass* (∫ p over
    the component), not the largest area.  Unlike the old sky box (which dilated
    the bounding *box*), this dilates the *mask* before selecting the main mode,
    so the result is the level-set component itself, not its axis-aligned
    envelope.
    """
    return region_from_hpd(density, grids, credible_level,
                           dilation_factor=dilation_factor,
                           periods=periods).main_component(density=density)


def region_from_bounds(bounds, grids, periodic=None):
    """Rasterise an axis-aligned box onto the grid (box → mask).

    ``bounds`` is ``[lo, hi]`` for 1D or ``[[x_lo, x_hi], [y_lo, y_hi]]`` for 2D.
    Edges round **outward** so the mask is a superset of the box.  Used by tests
    and the resume path where a plain box must become a :class:`Region`.
    """
    grids = tuple(np.asarray(g, dtype=float).reshape(-1) for g in grids)
    ndim = len(grids)

    def _span_mask(grid1d, lo, hi):
        dp = grid1d[1] - grid1d[0]
        i_lo = max(0, int(np.floor((lo - grid1d[0]) / dp)))
        i_hi = min(grid1d.shape[0] - 1, int(np.ceil((hi - grid1d[0]) / dp)))
        m = np.zeros(grid1d.shape[0], dtype=bool)
        m[i_lo:i_hi + 1] = True
        return m

    if ndim == 1:
        lo, hi = float(bounds[0]), float(bounds[1])
        return Region(_span_mask(grids[0], lo, hi), (grids[0],), periodic)
    (x_lo, x_hi), (y_lo, y_hi) = bounds
    mx = _span_mask(grids[0], float(x_lo), float(x_hi))
    my = _span_mask(grids[1], float(y_lo), float(y_hi))
    mask = np.outer(my, mx)   # (n_y, n_x)
    return Region(mask, (grids[0], grids[1]), periodic)


# ======================================================================
# §2 — evaluate the network only on the region it was trained on.
# ======================================================================

def prev_keep_mask(prev, grids):
    """Boolean "inside the previous round's accepted set" on the CURRENT grid.

    ``prev`` is either ``{"kind": "1d", "intervals": [[lo, hi], ...]}`` (exact
    membership, no interpolation) or ``{"kind": "2d", "region": Region}``
    (nearest-pixel resample of the stored mask).  ``grids`` is ``(grid_x,)`` or
    ``(grid_x, grid_y)`` of the current round.
    """
    if prev["kind"] == "1d":
        g = np.asarray(grids[0], dtype=float).reshape(-1)
        keep = np.zeros(g.shape[0], dtype=bool)
        for lo, hi in prev["intervals"]:
            keep |= (g >= lo) & (g <= hi)
        return keep
    return prev["region"].contains_grid((grids[0], grids[1]))


def apply_prev_mask(density, prev, grids, policy="hard", hysteresis_weight=0.1):
    """Re-weight ``density`` outside a previous round's accepted set (§2).

    NRE estimates ``p(θ|x)/p(θ)`` with ``p(θ)`` the proposal; outside the
    proposal there is no ``p(θ)`` to divide by, so the network's output there is
    unconstrained extrapolation.  This restricts the current round's posterior to
    the region the network was actually trained on, *before* the HPD analysis.

    ``policy``:
      * ``'hard'``       — zero the outside (``accepted_{N+1} ⊆ accepted_N``;
        exclusions become permanent).
      * ``'hysteresis'`` — multiply the outside by ``hysteresis_weight`` (a
        higher bar to re-open than to keep open), recoverable.
      * ``'off'``        — no change.

    Returns ``(new_density, status)`` where ``status`` is one of ``'off'``,
    ``'noprev'`` (round 1 / never truncated), ``'applied'``, or ``'degenerate'``
    (the previous mask does not intersect this grid — the original density is
    returned so there is still something to threshold).  The HPD threshold is
    scale-invariant, so the renormalisation here is cosmetic.
    """
    if policy == "off":
        return density, "off"
    if prev is None:
        return density, "noprev"
    if policy == "hard":
        alpha = 0.0
    elif policy == "hysteresis":
        alpha = float(hysteresis_weight)
    else:
        raise ValueError(
            f"zero_outside_prev_mask must be 'hard', 'hysteresis' or 'off', "
            f"got {policy!r}")

    keep = prev_keep_mask(prev, grids)
    weight = np.where(keep, 1.0, alpha)
    new = np.asarray(density, dtype=float) * weight
    total = float(np.sum(new))
    if total <= 0.0:
        return density, "degenerate"
    return new / total, "applied"


def clip_intervals_to_prev(mask1d, grid1d, prev, period=None):
    """Intersect a fresh 1D accepted mask with the previous round's accepted set
    and return the resulting sub-intervals (§2 monotonicity).

    Zeroing the density *before* the HPD analysis keeps the level set inside the
    previous region, but ``analyse_posterior_1d`` then *dilates* each mode, which
    can grow it back past the previous boundary.  Intersecting the dilated mask
    with the previous set here enforces ``A_N ⊆ A_{N-1}``.  Interval extraction
    reuses the same periodic labelling as the analysis so the two cannot drift.

    ``mask1d`` is the dilated boolean from ``analyse_posterior_1d``; ``prev`` is
    the previous ``_prev_accepted`` entry (``None`` → returned unclipped).  If the
    previous set does not overlap this grid at all, the fresh mask is kept so the
    round is not left with an empty region.
    """
    grid1d = np.asarray(grid1d, dtype=float)
    clipped = np.asarray(mask1d, dtype=bool)
    if prev is not None:
        intersected = clipped & prev_keep_mask(prev, (grid1d,))
        if intersected.any():
            clipped = intersected
    labels, _ = _periodic_labelling(clipped, _effective_periodic(grid1d, period))
    intervals = []
    for k in np.unique(labels):
        if k == 0:
            continue
        idx = np.where(labels == k)[0]
        intervals.extend(_intervals_from_indices(idx, grid1d))
    return intervals


def clip_labels_to_prev(labels2d, grid_x, grid_y, prev, period=(None, None)):
    """Intersect fresh 2D labels with the previous accepted set (§2 monotonicity).

    Returns ``(labels, components)`` with the labels relabelled after the
    intersection and ``components`` rebuilt via :func:`components_from_labels`,
    so the sampler and the recorded region see the monotone set.  ``period`` is
    ``(period_x, period_y)`` in grid-axis order, as passed to
    ``analyse_posterior_2d``.  See :func:`clip_intervals_to_prev` for the why.
    """
    grid_x = np.asarray(grid_x, dtype=float)
    grid_y = np.asarray(grid_y, dtype=float)
    clipped = np.asarray(labels2d) > 0
    if prev is not None:
        intersected = clipped & prev_keep_mask(prev, (grid_x, grid_y))
        if intersected.any():
            clipped = intersected
    # _periodic_labelling_2d indexes periodicity as (row, col) = (y, x).
    periodic_rowcol = (_effective_periodic(grid_y, period[1]),
                       _effective_periodic(grid_x, period[0]))
    labels, _ = _periodic_labelling_2d(clipped, periodic_rowcol)
    comps = components_from_labels(labels, grid_x=grid_x, grid_y=grid_y)
    return labels, comps
