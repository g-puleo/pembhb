"""
Robust sky-parameter truncation on S^2.

The sky posterior lives on the 2-sphere parametrised by:
    lambda  in [0, 2pi]   -- ecliptic longitude (PERIODIC)
    sin(beta) in [-1, 1]  -- sin(ecliptic latitude) (NOT periodic, bounded)

This module replaces the naive axis-aligned bounding-box truncation with
algorithms that correctly handle:
    1. Periodic wrapping of lambda across the 0/2pi boundary
    2. Multimodal posteriors (e.g. LISA antipodal sky degeneracy)
    3. Irregular / non-convex credible regions
    4. Stable thresholding (95% HPD + dilation, instead of noisy 99.99%)

Two operating modes:
    - "main_mode_rectangle": bounding box of the dominant mode (fast, drop-in)
    - "mask": boolean mask on the (lambda, sin beta) grid for rejection sampling
"""

import numpy as np
from scipy import ndimage

# ---------------------------------------------------------------------------
#  Internal helpers
# ---------------------------------------------------------------------------

def _hpd_threshold(density_2d, credible_level):
    """Density threshold enclosing *credible_level* fraction of total mass.

    This is the Highest Posterior Density (HPD) level set: the smallest
    density value t such that {x : p(x) >= t} contains *credible_level*
    of the integrated probability.
    """
    flat = density_2d.ravel()
    idx = np.argsort(flat)[::-1]
    sorted_density = flat[idx]
    cum = np.cumsum(sorted_density)
    cum /= cum[-1]
    i = np.searchsorted(cum, credible_level)
    return sorted_density[min(i, len(sorted_density) - 1)]


def _label_with_periodic_lambda(binary_mask):
    """Connected-component labelling with periodic BCs in the lambda direction.

    The mask has shape (n_beta, n_lambda).  Axis 1 (columns) is lambda and
    is periodic.  Axis 0 (rows) is sin(beta) and is NOT periodic.

    Strategy
    --------
    1. Label the mask normally (no periodicity).
    2. For each row, check if the pixels at column 0 and column n_lam-1
       are both occupied.  If so, their labels are the same physical
       component and must be merged.  Use 8-connectivity: also check
       diagonal neighbours (row +/- 1) across the boundary.
    3. Union-find to merge and relabel.

    Returns
    -------
    labeled : np.ndarray, shape (n_beta, n_lambda)
        Integer label array (0 = background).
    n_components : int
        Number of distinct connected components.
    """
    n_beta, n_lam = binary_mask.shape

    # Step 1: standard labelling (8-connectivity)
    struct = ndimage.generate_binary_structure(2, 2)
    labeled, n_features = ndimage.label(binary_mask, structure=struct)

    if n_features <= 1:
        return labeled, n_features

    # Step 2: union-find to merge labels across the periodic boundary
    parent = list(range(n_features + 1))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    # Check all 8-connected pairs across the lambda seam:
    # pixel (row, n_lam-1) neighbours pixel (row+dr, 0) for dr in {-1, 0, +1}
    for row in range(n_beta):
        l_right = labeled[row, n_lam - 1]
        if l_right == 0:
            continue
        for dr in (-1, 0, 1):
            r2 = row + dr
            if 0 <= r2 < n_beta:
                l_left = labeled[r2, 0]
                if l_left > 0:
                    union(l_right, l_left)

    # Step 3: relabel with merged components
    root_map = {l: find(l) for l in range(1, n_features + 1)}
    unique_roots = sorted(set(root_map.values()))
    root_to_new = {r: i + 1 for i, r in enumerate(unique_roots)}

    remap = np.zeros(n_features + 1, dtype=int)
    for old_label, root in root_map.items():
        remap[old_label] = root_to_new[root]

    labeled = remap[labeled]
    return labeled, len(unique_roots)


def _dilate_mask_isotropic(mask, dilation_factor):
    """Morphologically dilate *mask* so linear extent grows by *dilation_factor*.

    The dilation radius (in grid cells) is estimated from the effective
    radius of the region:  r_eff = sqrt(area / pi).  We dilate by
    (dilation_factor - 1) * r_eff cells.

    Uses 8-connectivity so diagonal expansion is included.
    """
    if dilation_factor <= 1.0:
        return mask.copy()

    area_cells = np.sum(mask)
    if area_cells == 0:
        return mask.copy()

    r_eff = np.sqrt(area_cells / np.pi)
    iterations = max(1, int(np.ceil((dilation_factor - 1.0) * r_eff)))

    struct = ndimage.generate_binary_structure(2, 2)  # 8-connectivity
    return ndimage.binary_dilation(mask, structure=struct, iterations=iterations)


def _dilate_mask_periodic(mask, dilation_factor):
    """Dilate with periodic BCs in lambda (axis 1).

    Pad -> dilate -> trim, so the dilation wraps correctly.
    """
    if dilation_factor <= 1.0:
        return mask.copy()

    n_beta, n_lam = mask.shape
    area_cells = np.sum(mask)
    if area_cells == 0:
        return mask.copy()

    r_eff = np.sqrt(area_cells / np.pi)
    iterations = max(1, int(np.ceil((dilation_factor - 1.0) * r_eff)))
    pad = iterations + 2  # enough so dilation doesn't hit the pad boundary

    padded = np.concatenate([
        mask[:, -pad:],
        mask,
        mask[:, :pad],
    ], axis=1)

    struct = ndimage.generate_binary_structure(2, 2)
    dilated_padded = ndimage.binary_dilation(padded, structure=struct, iterations=iterations)

    # Trim + OR the wrapped parts back
    core = dilated_padded[:, pad:pad + n_lam]
    left_wrap = dilated_padded[:, :pad]       # spilled past right -> left
    right_wrap = dilated_padded[:, pad + n_lam:]  # spilled past left -> right

    result = core.copy()
    result[:, :pad] |= right_wrap
    result[:, -pad:] |= left_wrap
    return result


def _component_bbox_lambda(comp_mask):
    """Bounding interval in the lambda (column) direction, detecting wrapping.

    Parameters
    ----------
    comp_mask : (n_beta, n_lam) bool array

    Returns
    -------
    col_lo, col_hi : int
        Column indices of the bounding interval.
    is_wrapped : bool
        True if the component straddles the lambda=0/2pi boundary and the
        *complement* of the occupied columns contains a single contiguous
        gap (i.e. the component "wraps around").
    """
    n_lam = comp_mask.shape[1]
    cols_occupied = np.any(comp_mask, axis=0)  # shape (n_lam,)
    col_indices = np.where(cols_occupied)[0]

    if len(col_indices) == 0:
        return 0, 0, False
    if len(col_indices) == 1:
        return col_indices[0], col_indices[0], False

    # A wrapped component touches both edges AND has a large interior gap.
    touches_left = cols_occupied[0]
    touches_right = cols_occupied[-1]
    gaps = np.diff(col_indices)
    max_gap = gaps.max()

    # Heuristic: wrapped if it touches both edges and the largest gap
    # is wider than 1/4 of the grid (otherwise it's just a big blob).
    is_wrapped = touches_left and touches_right and max_gap > n_lam // 4

    if is_wrapped:
        # The largest gap separates the "end" of the component from the
        # "start" (going in the positive-lambda direction).
        gap_idx = np.argmax(gaps)
        col_hi = col_indices[gap_idx]       # last column before the gap
        col_lo = col_indices[gap_idx + 1]   # first column after the gap
    else:
        col_lo = col_indices[0]
        col_hi = col_indices[-1]

    return int(col_lo), int(col_hi), bool(is_wrapped)


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------

def analyse_sky_posterior(grid_lam, grid_beta, posterior_2d,
                          credible_level=0.9545, dilation_factor=1.5):
    """Analyse a 2-D sky posterior: find credible regions, label modes.

    Parameters
    ----------
    grid_lam : (n_beta, n_lam) array
        Meshgrid of lambda values (``indexing='xy'``, varies along axis 1).
    grid_beta : (n_beta, n_lam) array
        Meshgrid of sin(beta) values (varies along axis 0).
    posterior_2d : (n_beta, n_lam) array
        Normalised posterior density on the grid.
    credible_level : float
        HPD credible level for the initial threshold (default 0.9545 = 95 %).
    dilation_factor : float
        Factor by which to inflate the credible region linearly (default 1.5).

    Returns
    -------
    result : dict
        ``'mask'``          -- (n_beta, n_lam) bool, dilated credible region.
        ``'mask_undilated'`` -- (n_beta, n_lam) bool, raw HPD region.
        ``'components'``    -- list of component dicts (sorted by mass, descending).
        ``'threshold'``     -- float, HPD density threshold.
        ``'main_mode_idx'`` -- int, index into ``components`` of the dominant mode.

    Each component dict contains:
        ``'label'``       -- int
        ``'mask'``        -- (n_beta, n_lam) bool
        ``'mass'``        -- float, integrated posterior mass in this component
        ``'bbox_lam'``    -- (float, float), bounding interval in lambda
        ``'bbox_beta'``   -- (float, float), bounding interval in sin(beta)
        ``'is_wrapped'``  -- bool, True if component wraps in lambda
    """
    n_beta, n_lam = posterior_2d.shape
    dlam = grid_lam[0, 1] - grid_lam[0, 0] if n_lam > 1 else 1.0
    dbeta = grid_beta[1, 0] - grid_beta[0, 0] if n_beta > 1 else 1.0

    # --- 1. HPD threshold --------------------------------------------------
    threshold = _hpd_threshold(posterior_2d, credible_level)
    binary_mask = posterior_2d >= threshold

    # --- 2. Connected components with periodic lambda ----------------------
    labeled, n_components = _label_with_periodic_lambda(binary_mask)

    # --- 3. Per-component analysis -----------------------------------------
    components = []
    for c in range(1, n_components + 1):
        cmask = labeled == c
        mass = float(np.sum(posterior_2d[cmask]) * dlam * dbeta)

        # Beta bounding box (axis 0, not periodic)
        rows = np.any(cmask, axis=1)
        row_idx = np.where(rows)[0]
        beta_lo = float(grid_beta[row_idx[0], 0])
        beta_hi = float(grid_beta[row_idx[-1], 0])

        # Lambda bounding box (axis 1, periodic)
        col_lo, col_hi, is_wrapped = _component_bbox_lambda(cmask)
        lam_lo = float(grid_lam[0, col_lo])
        lam_hi = float(grid_lam[0, col_hi])

        components.append({
            'label': c,
            'mask': cmask,
            'mass': mass,
            'bbox_lam': (lam_lo, lam_hi),
            'bbox_beta': (beta_lo, beta_hi),
            'is_wrapped': is_wrapped,
        })

    # Sort by posterior mass (largest first)
    components.sort(key=lambda c: c['mass'], reverse=True)
    main_mode_idx = 0 if components else -1

    # --- 4. Dilate the full mask -------------------------------------------
    dilated_mask = _dilate_mask_periodic(binary_mask, dilation_factor)

    return {
        'mask': dilated_mask,
        'mask_undilated': binary_mask,
        'components': components,
        'threshold': threshold,
        'main_mode_idx': main_mode_idx,
    }


def _dilate_box(lam_lo, lam_hi, beta_lo, beta_hi,
                dilation_factor, is_wrapped,
                lam_domain=(0.0, 2 * np.pi),
                beta_domain=(-1.0, 1.0)):
    """Dilate an axis-aligned box by *dilation_factor*, respecting domain bounds.

    For lambda (periodic): if dilated width exceeds the full domain, fall
    back to the full domain.  Otherwise expand symmetrically around the
    centre.

    For sin(beta) (bounded): clamp to [-1, 1].

    Returns (lam_lo, lam_hi, beta_lo, beta_hi, is_wrapped).
    """
    lam_full = lam_domain[1] - lam_domain[0]

    # -- Lambda -------------------------------------------------------------
    if is_wrapped:
        # Wrapped: width = (lam_hi - lam_domain[0]) + (lam_domain[1] - lam_lo)
        width = (lam_hi - lam_domain[0]) + (lam_domain[1] - lam_lo)
    else:
        width = lam_hi - lam_lo

    new_width = width * dilation_factor
    if new_width >= lam_full:
        # Covers the whole circle -> no truncation in lambda
        lam_lo_new = lam_domain[0]
        lam_hi_new = lam_domain[1]
        is_wrapped_new = False
    else:
        expand = (new_width - width) / 2.0
        if is_wrapped:
            lam_lo_new = lam_lo - expand
            lam_hi_new = lam_hi + expand
            # Check if we un-wrapped (gap closed)
            if lam_lo_new <= lam_hi_new:
                lam_lo_new = lam_domain[0]
                lam_hi_new = lam_domain[1]
                is_wrapped_new = False
            else:
                is_wrapped_new = True
        else:
            centre = (lam_lo + lam_hi) / 2.0
            lam_lo_new = centre - new_width / 2.0
            lam_hi_new = centre + new_width / 2.0
            # Check if expansion caused wrapping
            if lam_lo_new < lam_domain[0]:
                # Wrap: shift the deficit to the other side
                deficit = lam_domain[0] - lam_lo_new
                lam_lo_new = lam_domain[1] - deficit
                is_wrapped_new = True
            elif lam_hi_new > lam_domain[1]:
                surplus = lam_hi_new - lam_domain[1]
                lam_hi_new = lam_domain[0] + surplus
                is_wrapped_new = True
                # Swap so convention is: lam_lo > lam_hi for wrapped
                lam_lo_new, lam_hi_new = lam_lo_new, lam_hi_new
            else:
                is_wrapped_new = False

    # -- Beta (clamped to [-1, 1]) ------------------------------------------
    beta_centre = (beta_lo + beta_hi) / 2.0
    beta_half = (beta_hi - beta_lo) / 2.0
    beta_lo_new = max(beta_centre - dilation_factor * beta_half, beta_domain[0])
    beta_hi_new = min(beta_centre + dilation_factor * beta_half, beta_domain[1])

    return lam_lo_new, lam_hi_new, beta_lo_new, beta_hi_new, is_wrapped_new


def get_main_mode_box(grid_lam, grid_beta, posterior_2d,
                      credible_level=0.9545, dilation_factor=1.5):
    """Bounding box of the dominant sky mode, dilated.

    This is the drop-in replacement for the current ``get_widest_box_2d``
    when applied to the (lambda, sin beta) marginal.  It correctly handles
    periodic lambda and multimodal posteriors by selecting only the
    highest-mass connected component.

    Parameters
    ----------
    grid_lam, grid_beta : (n_beta, n_lam) arrays
        Meshgrids (``indexing='xy'``).
    posterior_2d : (n_beta, n_lam) array
        Normalised posterior density.
    credible_level : float
        HPD level for the initial threshold (default 0.9545).
    dilation_factor : float
        Inflate the box linearly by this factor (default 1.5).

    Returns
    -------
    box : dict
        ``'lam'``        -- (float, float)
        ``'beta'``       -- (float, float)
        ``'is_wrapped'`` -- bool (True => lam interval wraps around 2pi)
        ``'mass'``       -- float, posterior mass in the main mode
        ``'n_modes'``    -- int, total number of detected modes
    """
    result = analyse_sky_posterior(
        grid_lam, grid_beta, posterior_2d,
        credible_level=credible_level, dilation_factor=1.0,  # dilate the box, not the mask
    )

    if not result['components']:
        return {
            'lam': (float(grid_lam[0, 0]), float(grid_lam[0, -1])),
            'beta': (float(grid_beta[0, 0]), float(grid_beta[-1, 0])),
            'is_wrapped': False,
            'mass': 0.0,
            'n_modes': 0,
        }

    main = result['components'][result['main_mode_idx']]
    lam_lo, lam_hi = main['bbox_lam']
    beta_lo, beta_hi = main['bbox_beta']

    lam_lo, lam_hi, beta_lo, beta_hi, is_wrapped = _dilate_box(
        lam_lo, lam_hi, beta_lo, beta_hi,
        dilation_factor, main['is_wrapped'],
    )

    return {
        'lam': (lam_lo, lam_hi),
        'beta': (beta_lo, beta_hi),
        'is_wrapped': is_wrapped,
        'mass': main['mass'],
        'n_modes': len(result['components']),
    }


def get_sky_mask(grid_lam, grid_beta, posterior_2d,
                 credible_level=0.9545, dilation_factor=1.5):
    """Boolean mask of the dilated credible region (all modes).

    Use this with rejection sampling: draw (lambda, sin beta) uniformly
    from the rectangular prior, accept only if the mask is True at that
    grid cell.

    Parameters
    ----------
    grid_lam, grid_beta, posterior_2d : arrays, shape (n_beta, n_lam)
    credible_level : float
    dilation_factor : float

    Returns
    -------
    mask : (n_beta, n_lam) bool array
    analysis : dict (full output of ``analyse_sky_posterior``)
    """
    result = analyse_sky_posterior(
        grid_lam, grid_beta, posterior_2d,
        credible_level=credible_level, dilation_factor=dilation_factor,
    )
    return result['mask'], result


# ---------------------------------------------------------------------------
#  Sampler helpers: convert analysis results into sampling instructions
# ---------------------------------------------------------------------------

def sky_box_to_prior_bounds(box, current_prior):
    """Update a prior dict with the main-mode box for lambda and beta.

    Handles the wrapped case by picking the wider interpretation
    (union interval) -- this wastes some prior volume but keeps the
    sampler simple (rectangular bounds).  If sampling efficiency is
    critical, use ``MaskRejectSampler`` instead.

    Parameters
    ----------
    box : dict from ``get_main_mode_box``
    current_prior : dict
        The full prior dict (will be modified in-place and returned).

    Returns
    -------
    current_prior : dict (modified)
    """
    lam_lo, lam_hi = box['lam']
    beta_lo, beta_hi = box['beta']

    if box['is_wrapped']:
        # Wrapped interval: [lam_lo, 2pi] U [0, lam_hi].
        # For a rectangular sampler the simplest safe choice is the full domain.
        # But we can be smarter: express as [lam_lo, lam_hi + 2pi] modulo 2pi
        # and sample in that shifted domain.  For now, keep full range --
        # the mask sampler handles this better.
        import warnings
        warnings.warn(
            f"Sky mode wraps in lambda: [{lam_lo:.3f}, 2pi] U [0, {lam_hi:.3f}]. "
            f"Using full lambda range for rectangular prior. "
            f"Consider using MaskRejectSampler for better efficiency.",
            stacklevel=2,
        )
        current_prior["lambda"] = [0.0, 2 * np.pi]
    else:
        current_prior["lambda"] = [lam_lo, lam_hi]

    current_prior["beta"] = [beta_lo, beta_hi]
    return current_prior


def sample_in_sky_mask(mask, grid_lam, grid_beta, n_samples, rng=None):
    """Draw (lambda, sin_beta) samples uniformly within *mask* via rejection.

    Parameters
    ----------
    mask : (n_beta, n_lam) bool
        Acceptance mask on the grid.
    grid_lam : (n_beta, n_lam) array
        Lambda meshgrid.
    grid_beta : (n_beta, n_lam) array
        sin(beta) meshgrid.
    n_samples : int
    rng : np.random.Generator or None

    Returns
    -------
    lam_samples : (n_samples,) array
    beta_samples : (n_samples,) array
    acceptance_rate : float
    """
    if rng is None:
        rng = np.random.default_rng()

    lam_min, lam_max = grid_lam[0, 0], grid_lam[0, -1]
    beta_min, beta_max = grid_beta[0, 0], grid_beta[-1, 0]
    dlam = grid_lam[0, 1] - grid_lam[0, 0]
    dbeta = grid_beta[1, 0] - grid_beta[0, 0]

    lam_out = np.empty(n_samples)
    beta_out = np.empty(n_samples)
    n_accepted = 0
    n_drawn = 0
    batch = max(n_samples * 4, 10000)  # over-sample to reduce iterations

    while n_accepted < n_samples:
        lam_cand = rng.uniform(lam_min, lam_max + dlam, size=batch)
        beta_cand = rng.uniform(beta_min, beta_max + dbeta, size=batch)

        # Map to grid indices
        col = np.clip(((lam_cand - lam_min) / dlam).astype(int), 0, mask.shape[1] - 1)
        row = np.clip(((beta_cand - beta_min) / dbeta).astype(int), 0, mask.shape[0] - 1)

        accept = mask[row, col]
        idx = np.where(accept)[0]
        n_take = min(len(idx), n_samples - n_accepted)
        lam_out[n_accepted:n_accepted + n_take] = lam_cand[idx[:n_take]]
        beta_out[n_accepted:n_accepted + n_take] = beta_cand[idx[:n_take]]

        n_accepted += n_take
        n_drawn += batch

    acceptance_rate = n_samples / n_drawn
    return lam_out[:n_samples], beta_out[:n_samples], acceptance_rate


# ---------------------------------------------------------------------------
#  High-level truncation step: call from the TMNRE loop
# ---------------------------------------------------------------------------

def truncate_sky_prior(model, obs_loader, out_param_idx,
                       datagen_conf,
                       mode="rectangle",
                       credible_level=0.9545,
                       dilation_factor=1.5):
    """Perform one sky-truncation step and update ``datagen_conf["prior"]``.

    This is the function you call from the TMNRE loop in place of the
    generic bounding-box update for the (lambda, beta) = (7, 8) marginal.

    Parameters
    ----------
    model : InferenceNetwork
        Trained model for this round.
    obs_loader : DataLoader
        Single-event observation loader.
    out_param_idx : int
        Output index of the sky marginal head.
    datagen_conf : dict
        Data generation config; ``datagen_conf["prior"]`` is updated in-place.
    mode : ``"rectangle"`` or ``"mask"``
        - ``"rectangle"``: update lambda/beta bounds with the main-mode box
          (drop-in, no sampler changes needed).
        - ``"mask"``: return a ``MaskRejectSampler`` for the next round.
    credible_level : float
        HPD level (default 0.9545 = 95%).
    dilation_factor : float
        Linear inflation factor (default 1.5).

    Returns
    -------
    If mode == "rectangle":
        datagen_conf : dict (modified in-place)
        info : dict with diagnostic info (n_modes, mass, box, etc.)
    If mode == "mask":
        sampler : MaskRejectSampler
        info : dict
    """
    from pembhb.utils import get_logratios_grid_2d

    in_param_idx = (7, 8)  # (lambda, sin_beta)

    logratios, inj_params, gx, gy = get_logratios_grid_2d(
        obs_loader, model,
        ngrid_points=100,
        in_param_idx=in_param_idx,
        out_param_idx=out_param_idx,
    )

    ratios = np.exp(logratios)
    dp1 = gx[0, 1] - gx[0, 0]
    dp2 = gy[1, 0] - gy[0, 0]
    norm2d = ratios / np.sum(ratios * dp1 * dp2, axis=(1, 2), keepdims=True)
    posterior = norm2d[0]

    analysis = analyse_sky_posterior(
        gx, gy, posterior,
        credible_level=credible_level,
        dilation_factor=dilation_factor,
    )

    box = get_main_mode_box(
        gx, gy, posterior,
        credible_level=credible_level,
        dilation_factor=dilation_factor,
    )

    info = {
        'box': box,
        'analysis': analysis,
        'inj_params': inj_params,
        'grid_lam': gx,
        'grid_beta': gy,
        'posterior': posterior,
    }

    print(f"[sky_truncation] {box['n_modes']} mode(s), "
          f"main mass = {box['mass']:.4f}, wrapped = {box['is_wrapped']}")

    if mode == "rectangle":
        sky_box_to_prior_bounds(box, datagen_conf["prior"])
        lam_lo, lam_hi = datagen_conf["prior"]["lambda"]
        beta_lo, beta_hi = datagen_conf["prior"]["beta"]
        print(f"[sky_truncation] rectangle mode: "
              f"lambda=[{lam_lo:.4f}, {lam_hi:.4f}], "
              f"beta=[{beta_lo:.4f}, {beta_hi:.4f}]")
        return datagen_conf, info

    elif mode == "mask":
        from pembhb.sampler import MaskRejectSampler
        sampler = MaskRejectSampler(
            prior_bounds=datagen_conf["prior"],
            sky_mask=analysis['mask'],
            grid_lam=gx,
            grid_beta=gy,
        )
        return sampler, info

    else:
        raise ValueError(f"Unknown mode: {mode!r}. Use 'rectangle' or 'mask'.")
