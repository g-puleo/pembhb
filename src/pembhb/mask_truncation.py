from pembhb.utils import get_logratios_grid
from scipy import ndimage
import numpy as np
from scipy.stats import nbinom
"""
This small module is designed to provide a truncation scheme which handles multimodality. 
When a MNRE posterior is produced, the threshold corresponding to a certain HPD credibility region is found. 
Pixels in the grid above that threshold are used as a mask. 

In 1d, sampling from the mask is achieved by assuming that the mask is made of disjoint intervals I_k.
Uniform sampling within the mask is achieved by choosing I_k with probability p\propto the width of I_k, then sampling uniformly within I_k. This has 100% efficiency. 

In 2d, sampling from a mask is achievied by first labelling the connected components of the mask, then for each of them taking the max and min coordinates along each axis. 
This results in a set of pairs of intervals P_k = {X_k, Y_k}, one for each connected component. Within the pair, one interval X_k is along the x axis and the other, Y_k, is along the y axis. 
In order to sample uniformly from the union of connected components, we first pick P_k with probability proportional to the number of pixels within that connected component. Then, we generate uniform samples within X_k and Y_k and 
use rejection sampling in order to retain only the samples that effectively fall into the HPD region. 

"""
_MASK_PERIOD_BY_NAME = {"phi": 2 * np.pi, "lambda": 2 * np.pi, "psi": np.pi}
def mask_volume_fraction(mask):
    """Fraction of the prior box occupied by the accepted set.

    The posterior grid spans exactly the trained prior box, so the prior
    "volume" is the whole grid and the cell area cancels.  Pass the DILATED
    mask -- that is what actually gets sampled.
    """
    mask = np.asarray(mask)
    return float(np.count_nonzero(mask)) / float(mask.size)

def truth_violations(true_params, prior_box, intervals_1d, masks_2d, param_keys,
                     check_idxs=None):
    """Is the observation's true parameter vector inside the next proposal?

    Returns ``[]`` when it is, else one dict per miss::

        {'name', 'kind', 'marginal', 'value', 'detail'}

    ``kind`` is ``'1d-mask'``, ``'2d-mask'`` or ``'box'``, naming *what*
    excluded the truth.  A 2D pair yields at most one entry: when (x, y) lands
    outside the mask neither coordinate is individually at fault.

    ``check_idxs`` restricts the plain-box checks to those parameter indices
    (the caller passes the active marginals' indices, so untouched parameters
    that keep their full prior are not reported).  Indices covered by a mask
    are always checked, and are checked against the mask *instead of* the box:
    the mask is strictly tighter, so the box test would be redundant.
    """
    violations = []
    covered = set()

    def _val(idx):
        return float(np.asarray(true_params[idx]).reshape(()))

    for idx, intervals in intervals_1d.items():
        idx = int(idx)
        covered.add(idx)
        v = _val(idx)
        if not any(lo <= v <= hi for lo, hi in intervals):
            violations.append({
                "name": param_keys[idx],
                "kind": "1d-mask",
                "marginal": [idx],
                "value": [v],
                "detail": ("outside every accepted sub-interval "
                           + str([[float(a), float(b)] for a, b in intervals])),
            })

    for m in masks_2d:
        i, j = int(m["idx"][0]), int(m["idx"][1])
        assert not (covered & {i, j}), (
            f"parameter(s) {covered & {i, j}} appear in more than one marginal; "
            "utils.validate_marginals should have caught this")
        covered.update((i, j))

        labels, gx, gy = m["labels"], m["grid_x"], m["grid_y"]
        vx, vy = _val(i), _val(j)
        # grids hold pixel centres -> nearest centre, not floor
        col = int(round((vx - gx[0]) / (gx[1] - gx[0])))
        row = int(round((vy - gy[0]) / (gy[1] - gy[0])))

        entry = {
            "name": f"{param_keys[i]}-{param_keys[j]}",
            "kind": "2d-mask",
            "marginal": [i, j],
            "value": [vx, vy],
        }
        if not (0 <= row < labels.shape[0] and 0 <= col < labels.shape[1]):
            entry["detail"] = (
                f"off the posterior grid (row={row}, col={col}; grid is "
                f"{labels.shape[0]}x{labels.shape[1]} spanning "
                f"x=[{gx[0]:.6g}, {gx[-1]:.6g}], y=[{gy[0]:.6g}, {gy[-1]:.6g}])")
            violations.append(entry)
        elif labels[row, col] <= 0:
            entry["detail"] = (
                f"inside the bounding box but in a gap between components "
                f"(pixel row={row}, col={col} is background)")
            violations.append(entry)

    for idx in sorted(set(check_idxs or ()) - covered):
        idx = int(idx)
        name = param_keys[idx]
        if name not in prior_box:
            continue
        lo, hi = float(prior_box[name][0]), float(prior_box[name][1])
        v = _val(idx)
        if not (lo <= v <= hi):
            violations.append({
                "name": name,
                "kind": "box",
                "marginal": [idx],
                "value": [v],
                "detail": f"outside [{lo:.6g}, {hi:.6g}]",
            })

    return violations


def format_violations(violations):
    """One line per violation, for the round-end error message."""
    return "\n".join(
        f"    [{v['kind']}] {v['name']}: "
        f"true=({', '.join(f'{x:.6g}' for x in v['value'])}) {v['detail']}"
        for v in violations
    )



def components_from_labels(labelled, grid_x, grid_y):
    """Per-component sub-intervals (bounding box, marginalised per axis).

    Returns {k: {'x_intervals': [[lo,hi],...],
                 'y_intervals': [[lo,hi],...]}} for every label k > 0.

    Pure function of (labelled, grid_x, grid_y) -- called both by
    analyse_posterior_2d and by load_truncation, so a reloaded run and a fresh
    one cannot disagree.  Deliberately carries no per-component weight: the
    sampler allocates by mask pixel count, and a stored bounding-box weight
    would be the wrong quantity for that.
    """
    component_intervals_dict = {}
    labels  = np.unique(labelled)
    for i in labels: 
        if i==0: continue
        marginal_x_axis_mask = np.any(labelled==i, axis=0)
        marginal_x_idx = np.where(marginal_x_axis_mask)[0]
        x_intervals = _intervals_from_indices(marginal_x_idx, grid_x)

        marginal_y_axis_mask = np.any(labelled==i, axis=1)
        marginal_y_idx = np.where(marginal_y_axis_mask)[0]
        y_intervals = _intervals_from_indices(marginal_y_idx, grid_y)
        component_intervals_dict[i] = {"x_intervals": x_intervals, "y_intervals": y_intervals}

    return component_intervals_dict

def eval_posterior_1d(model, dataloader, in_param_idx, out_param_idx, ngrid_points=100):
    """Evaluate and normalise a 1D marginal posterior on a grid.

    Returns
    -------
    grid   : (ngrid_points, 1) array  -- parameter values (tmnre coordinate space)
    norm1d : (ngrid_points,) array    -- normalised posterior density
    inj    : injected parameter value(s) for the observation(s)
    """
    logratios, inj_params, grid = get_logratios_grid(
        dataloader,
        model,
        ngrid_points=ngrid_points,
        in_param_idx=in_param_idx,
        out_param_idx=out_param_idx,
    )
    
    ratios = np.exp(logratios[0])  # Take first (only) observation
    dp = grid[1, 0] - grid[0, 0]
    norm1d = ratios / np.sum(ratios * dp)


    return grid, norm1d, inj_params[0]

def _hpd_threshold(density, credible_level):
    """
    Return the threshold value above which a fraction credible_level of the probability density is contained. 
    

    Args:
        density (np.array): the density array to find the HPD threshold
        credible_level (float): the desired fraction to enclose (between 0 and 1)

    Returns:
        float: the value of the threshold.
    """
    flat = density.ravel()
    idx = np.argsort(flat)[::-1]
    sorted_density = flat[idx]
    cum = np.cumsum(sorted_density)
    cum /= cum[-1]
    i = np.searchsorted(cum, credible_level)
    return sorted_density[min(i, len(sorted_density) - 1)]

def _intervals_from_indices(indices, grid1d):

    is_wrapped = np.any(np.diff(indices)>1) # finds a wrapped mode
    N_pixels = grid1d.shape[0]
    intervals = []
    if is_wrapped: 
        jj      = np.where(np.diff(indices)>1)[0][0]
        idx_lo1 = indices[0]
        idx_hi1 = indices[jj]
        idx_lo2 = indices[jj+1]
        idx_hi2 = indices[-1]
        assert idx_lo1 == 0
        assert idx_hi2 == N_pixels - 1
        intervals.append([float(grid1d[idx_lo1]), float(grid1d[idx_hi1])])
        intervals.append([float(grid1d[idx_lo2]), float(grid1d[idx_hi2])])
    
    else: 
        intervals.append([float(grid1d[indices[0]]), float(grid1d[indices[-1]])])

    return intervals

def analyse_posterior_1d(grid, norm1d, credible_level=0.99999,
                         dilation_factor=1.2, period=None):
    """
    Returns
    -------
    dict with:
      'mask'      : (N,) bool   -- cells with density >= HPD threshold (after dilation)
      'intervals' : [[lo, hi], ...]  -- accepted sub-intervals in grid coordinates
      'threshold' : float
    """

    grid1d = grid[:,0]
    thr = _hpd_threshold(norm1d, credible_level=credible_level)
    mask = norm1d>=thr
    # label the modes ensuring periodic bc if needed
    periodic = (period is not None) and bool( np.isclose(grid1d[-1] - grid1d[0], period))
    labelled , n_modes = _periodic_labelling(mask, periodic)
    dilated = np.zeros_like(mask)
    
    # dilate each mode by dilation factor
    for k in np.unique(labelled) :
        if k == 0: 
            continue 
        comp = labelled == k 
        half_width_pixels = comp.sum()/2
        n_pixels_to_inflate = max(1, int(np.ceil(half_width_pixels*(dilation_factor-1))))
        dilated |= _dilate_mode(comp, iterations=n_pixels_to_inflate, periodic=periodic)
    
    # relabel modes (after dilation different modes may merge)
    final_labels, _ = _periodic_labelling(dilated, periodic)
    intervals = []
    for k in np.unique(final_labels):
        if k == 0:
            continue

        indices = np.where(final_labels == k)[0]
        intervals.extend(_intervals_from_indices(indices, grid1d))

    return {"mask": dilated, "intervals": intervals, "threshold": float(thr)}

def _dilate_mode(comp, iterations, periodic): 

    structure = ndimage.generate_binary_structure(1,1)
    if periodic: 
        comp_extended = np.tile(comp, 3)
        size = comp.shape[0]
        dilated_extended = ndimage.binary_dilation(comp_extended, structure, iterations=iterations)
        dilated = np.logical_or(dilated_extended[:size], dilated_extended[size:2*size])
        return dilated
    
    dilated = ndimage.binary_dilation(comp , structure, iterations=iterations)
    return dilated 
    


def _periodic_labelling( mask , periodic): 

    structure = ndimage.generate_binary_structure(1,1) # this is [1,1,1], defines nearest neighbours
    labelled, num_features = ndimage.label(mask, structure=structure)

    if periodic: 
        touches_left = (labelled[0]!=0)
        touches_right = (labelled[-1]!=0)
        if touches_left and touches_right : 
            left_label = labelled[0]
            right_label = labelled[-1]
            labelled[labelled==right_label] = left_label
            num_features-=1
            
    
    return labelled, num_features


def analyse_posterior_2d(grid2d_x, grid2d_y,  norm2d, period: tuple, credible_level=0.999, dilation_factor=1.1):
    grid_x = grid2d_x[0,:]
    grid_y = grid2d_y[:,0]
    period_x = (period[0] is not None) and bool( np.isclose(grid_x[-1] - grid_x[0], period[0]))
    period_y = (period[1] is not None) and bool( np.isclose(grid_y[-1] - grid_y[0], period[1]))
    periodic = (period_y, period_x)
    threshold = _hpd_threshold(norm2d, credible_level)
    mask = norm2d >= threshold
    labelled2d, n_components = _periodic_labelling_2d(mask, periodic)
    
    dilated_mask = np.zeros_like(mask)
    for i in range(1, n_components+1):
        component_mask = labelled2d==i 
        effective_radius_pixels = np.sqrt( np.sum(component_mask)/np.pi)
        n_iterations = max(1, int(np.ceil(effective_radius_pixels*(dilation_factor - 1))))
        dilated_mask |= _dilate_mode_2d( component_mask, n_iterations, periodic)
    
    # relabel after the dilation
    labelled_after_dilation, _ = _periodic_labelling_2d(dilated_mask, periodic)
    component_intervals_dict = components_from_labels(labelled_after_dilation, grid_x=grid_x, grid_y=grid_y)

    return labelled_after_dilation, component_intervals_dict, grid_x, grid_y



def _periodic_labelling_2d(binary_mask, periodic: tuple ):
    """Connected-component labelling, optionally periodic in both directions.


    Strategy
    --------
    1. Label the mask normally (no periodicity).
    2. For each row(column), check if the pixels at column(row) 0 and column(row) n_col-1(n_row-1)
       are both occupied.  If so, their labels are the same physical
       component and must be merged.  Use 8-connectivity: also check
       diagonal neighbours (row +/- 1) across the boundary.
    3. Union-find to merge and relabel.

    Returns
    -------
    labeled : np.ndarray, shape (n_rows, n_cols)
        Integer label array (0 = background).
    n_components : int
        Number of distinct connected components.
    """
    n_ax0, n_ax1 = binary_mask.shape

    # Step 1: standard labelling (8-connectivity)
    struct = ndimage.generate_binary_structure(2, 2)
    labeled, n_features = ndimage.label(binary_mask, structure=struct)

    if n_features <= 1 or not any(periodic):
        return labeled, n_features

    # Step 2: union-find to merge labels across the periodic boundary
    # union-find is an algorithm 
    parent = list(range(n_features + 1))
    # suppose n_features is 5 . parent is init as [0,1,2,3,4,5]
    # every element is the parent of itself. parent will be updated by subsequent union() calls. 
    def find(x):
        # at first find(x) will immediately return x because parent[x] == x for all x
        # we enter the while loop only after running union() at least once. 

        # let's say we have now [0,0,2,1,4,3] . parent[5]==3, and parent[3]==1 and parent[1]==0
        # and we do find(5). this while loop will start because parent[5]!=5 and then
        # 1) parent[5] = 1 ; x=1 ; parent[1]!=1 --> enter again; parent[1]==0 -->parent[parent[1]]=0 --> x=0 ; exit loop because parent[0]==0
        # result: [0,0,2,1,4,1]
        # in practice this CLIMBS THE TREE TO THE ROOT and finds the root of element x's family
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        # say we have [0,0,2,1,4,3] and we do union(4,3): 
        # find(a)==4 , find(b)==0

        a, b = find(a), find(b)
        if a != b:
            # say that a>b 
            # set parent[a] = the root of b's family
            parent[max(a, b)] = min(a, b)

    # Check all 8-connected pairs across the lambda seam:
    # pixel (row, n_lam-1) neighbours pixel (row+dr, 0) for dr in {-1, 0, +1}

    if periodic[1]: # if the parameter along columns (x axis) is periodic

        # for all rows check if the left and right extrema are non-zero, in that case , unites their modes
        for row in range(n_ax0):
            l_right = labeled[row,  - 1]
            if l_right == 0:
                continue
            for dr in (-1, 0, 1):
                r2 = row + dr
                if periodic[0]: # handles case where the other param is periodic to consider also across boundary rows as neighbour
                    r2 %= n_ax0 
                elif not (0<=r2< n_ax0): # if the other param is non periodic, check r2 is in range, if not continue.
                    continue 
                l_left = labeled[r2, 0]
                if l_left > 0:
                    union(l_right, l_left)
    if periodic[0]:   # rows periodic: merge across top/bottom seam
        for col in range(n_ax1):
            l_bottom = labeled[- 1, col]
            if l_bottom == 0:
                continue
            for dc in (-1, 0, 1):
                c2 = col + dc
                if periodic[1]:
                    c2 %= n_ax1
                elif not (0 <= c2 < n_ax1):
                    continue
                l_top = labeled[0, c2]
                if l_top > 0:
                    union(l_bottom, l_top)

    # Step 3: relabel with merged components
    # in practice , if there are roots that are 5, 7, 8 they are relabelled to 0,1,2 and
    root_map = {l: find(l) for l in range(1, n_features + 1)}
    unique_roots = sorted(set(root_map.values()))
    root_to_new = {r: i + 1 for i, r in enumerate(unique_roots)}

    remap = np.zeros(n_features + 1, dtype=int)
    for old_label, root in root_map.items():
        remap[old_label] = root_to_new[root]

    labeled = remap[labeled]
    return labeled, len(unique_roots)

def _dilate_mode_2d(comp, iterations, periodic):
    """
    Dilates mode by padding with wrap mode if a parameter is periodic. 

    """
    struct = ndimage.generate_binary_structure(2, 2)   # 8-connectivity
    pad = iterations
    m = np.pad(comp, ((pad, pad), (0, 0)),
               mode='wrap' if periodic[0] else 'constant')   # rows / axis 0
    m = np.pad(m,    ((0, 0), (pad, pad)),
               mode='wrap' if periodic[1] else 'constant')   # cols / axis 1
    d = ndimage.binary_dilation(m, struct, iterations=iterations)
    n0, n1 = comp.shape
    return d[pad:pad + n0, pad:pad + n1]                # crop back to original



def _sample_from_intervals(intervals, n, rng, cube=False):

    power = 3 if cube else 1.0
    widths = np.array([ (interval[1]**power - interval[0]**power) for interval in intervals ])

    widths/=np.sum(widths)

    chosen_intervals = rng.choice( len(widths), p=widths, size=n)
    count = np.bincount(chosen_intervals, minlength=len(widths)) #count[k] counts how many times k was chosen

    sampled = []

    for k, interval in enumerate(intervals): 
        sampled.append(rng.uniform(low=interval[0]**power, high=interval[1]**power, size=count[k])**(1/power))

    out = np.concatenate(sampled)
    rng.shuffle(out)

    assert out.shape[0] == n 
    return out 

def _sample_2d_from_components(components, labels, grid_x, grid_y, n, rng: np.random.Generator):
    """components:  {k: {'x_intervals':..., 'y_intervals':...}}
       labels: (n_row, n_col) int (== k inside component k).  
       
       
       Returns (x, y, acceptance_rate)."""
    
    ks = list(components) #returns the keys of components
    w = np.array([(labels == k).sum() for k in ks], float)
    w /= w.sum()    
    x0, dx = grid_x[0], grid_x[1]-grid_x[0]
    y0, dy = grid_y[0], grid_y[1]-grid_y[0]
    
    choices = rng.choice(len(w), size=n, p=w)
    counts = np.bincount(choices, minlength=len(w))
    component_samples_x = []
    component_samples_y = []
    accepted=0
    sampled=0
    for pos, i in enumerate(ks):
        # index by position, not by label: labels need not be contiguous 1..n
        N = counts[pos]
        if N==0: continue
        component = components[i]
        mask = labels == i
        # Fill fraction of the projected bounding box, in PIXELS. Counting both
        # sides in pixels is what keeps this <= 1: the intervals hold pixel
        # centres, so a span of n pixels has coordinate width (n-1)*dx, and
        # mixing the two units makes the ratio exceed 1 for compact components
        # (a 2x2 mask would give 4) -> nbinom.ppf(p>1) = NaN.
        # Only used to size the rejection batch; the while-loop below is what
        # guarantees exactly N accepted samples.
        n_cols = int(np.any(mask, axis=0).sum())
        n_rows = int(np.any(mask, axis=1).sum())
        expected_acc = mask.sum() / float(n_cols * n_rows)
        assert 0 < expected_acc <= 1.0
        if expected_acc < 0.05: print(f"WARNING. expected acceptance in mask rejection sampling is {expected_acc}.\n")
        # scipy's nbinom counts failures before the N-th success
        F_max = nbinom.ppf(1 - 1e-3, N, expected_acc)
        # convert failures to total trials
        T_max = int(F_max) + int(N)


        xk_list, yk_list = [], []

        while sum(len(a) for a in xk_list)<N: 
            xk = _sample_from_intervals(component['x_intervals'], T_max, rng)   # cube never needed in 2D
            yk = _sample_from_intervals(component['y_intervals'], T_max, rng)
            col = np.clip(((xk - x0)/dx).astype(int), 0, labels.shape[1]-1)
            row = np.clip(((yk - y0)/dy).astype(int), 0, labels.shape[0]-1)
            keep = labels[row, col] == i
            accepted += keep.sum()
            sampled+=T_max
            kept_idx = np.where( keep )[0] 
            xk_list.append(xk[kept_idx])
            yk_list.append(yk[kept_idx])

        x_ = np.concatenate(xk_list)[:N]
        y_ = np.concatenate(yk_list)[:N]

        component_samples_x.append(x_)
        component_samples_y.append(y_)

    acceptance_rate = accepted/sampled
    x =np.concatenate( component_samples_x )
    y =np.concatenate( component_samples_y )
    return x, y, acceptance_rate

def save_truncation(yaml_path, npz_path, prior_box, intervals_1d, masks_2d,
                    mode="mask", extra=None):
    """Persist a round's truncation state.

    The mask is the source of truth; ``prior_box`` is its bounding box, kept
    for plots and as the base sampler's proposal envelope.

    Writes two files:
      yaml_path -- prior box, mode, 1D intervals, list of 2D pairs
      npz_path  -- per 2D pair: labels (int8) + the two 1D grid axes

    Per-component intervals are NOT stored: they are rebuilt on load via
    components_from_labels, so the array on disk cannot drift from them.
    """
    import os
    import yaml as _yaml

    payload = {
        "prior": {k: [float(v[0]), float(v[1])] for k, v in prior_box.items()},
        "truncation_mode": str(mode),
        "intervals_1d": {
            int(idx): [[float(lo), float(hi)] for lo, hi in ivs]
            for idx, ivs in (intervals_1d or {}).items()
        },
        "pairs_2d": [[int(m["idx"][0]), int(m["idx"][1])] for m in (masks_2d or [])],
    }
    if extra:
        payload.update(extra)

    os.makedirs(os.path.dirname(yaml_path) or ".", exist_ok=True)
    with open(yaml_path, "w") as f:
        _yaml.safe_dump(payload, f)

    if masks_2d:
        arrays = {}
        for m in masks_2d:
            i, j = int(m["idx"][0]), int(m["idx"][1])
            labels = np.asarray(m["labels"])
            if labels.max() > 127:
                raise ValueError(
                    f"2D pair ({i},{j}) has {labels.max()} components; int8 "
                    f"storage supports at most 127.")
            arrays[f"labels__{i}_{j}"] = labels.astype(np.int8)
            arrays[f"gridx__{i}_{j}"] = np.asarray(m["grid_x"], dtype=float)
            arrays[f"gridy__{i}_{j}"] = np.asarray(m["grid_y"], dtype=float)
        os.makedirs(os.path.dirname(npz_path) or ".", exist_ok=True)
        np.savez_compressed(npz_path, **arrays)


def load_truncation(yaml_path, npz_path):
    """Inverse of :func:`save_truncation`.

    Returns ``{"prior", "mode", "intervals_1d", "masks_2d"}`` with ``masks_2d``
    entries shaped exactly as :class:`MaskRejectSampler` consumes them.

    Raises if the run was saved in ``mask`` mode but the ``.npz`` (or a key in
    it) is missing.  Falling back to the bounding box would silently widen the
    proposal relative to the round being resumed, which is the failure this
    whole mechanism exists to prevent.
    """
    import os
    import yaml as _yaml

    with open(yaml_path) as f:
        payload = _yaml.safe_load(f) or {}

    # Runs predating mask truncation have neither key; they are rectangles.
    mode = str(payload.get("truncation_mode", "rectangle"))
    prior = {k: [float(v[0]), float(v[1])] for k, v in payload["prior"].items()}
    intervals_1d = {
        int(k): [[float(lo), float(hi)] for lo, hi in ivs]
        for k, ivs in (payload.get("intervals_1d") or {}).items()
    }
    pairs = [(int(a), int(b)) for a, b in (payload.get("pairs_2d") or [])]

    masks_2d = []
    if pairs:
        if not os.path.exists(npz_path):
            raise RuntimeError(
                f"[Truncation] {yaml_path} declares truncation_mode='{mode}' with "
                f"2D masks for pairs {pairs}, but {npz_path} is missing. Refusing "
                f"to fall back to the bounding box: that would resume from a "
                f"strictly wider prior than the round it continues.")
        with np.load(npz_path) as z:
            for (i, j) in pairs:
                keys = (f"labels__{i}_{j}", f"gridx__{i}_{j}", f"gridy__{i}_{j}")
                missing = [k for k in keys if k not in z]
                if missing:
                    raise RuntimeError(
                        f"[Truncation] {npz_path} is missing {missing} for 2D pair "
                        f"({i},{j}) declared in {yaml_path}.")
                labels = z[keys[0]].astype(int)
                grid_x = z[keys[1]]
                grid_y = z[keys[2]]
                masks_2d.append({
                    "idx": (i, j),
                    "labels": labels,
                    "grid_x": grid_x,
                    "grid_y": grid_y,
                    "components": components_from_labels(labels, grid_x, grid_y),
                })

    return {"prior": prior, "mode": mode,
            "intervals_1d": intervals_1d, "masks_2d": masks_2d}


from pembhb.sampler import UniformSampler   # (use a deferred import inside __init__ if this cycles)

_DIST_IDX = 4   # _ORDERED_PRIOR_KEYS.index("dist")

# Cap on post-mask spin-rejection redraws. Each batch draws n samples, so an
# acceptance above ~1/50 always converges well inside this; hitting it means the
# spin mask is essentially outside the physical region and the loop would hang.
_MAX_SPIN_REJECT_BATCHES = 50

class MaskRejectSampler:
    def __init__(self, prior_bounds, intervals_1d, masks_2d, rng=None,
                 dist_uniform_in_volume=True, spin_param_basis="chieff_chidiff",
                 round_idx=None):
        # intervals_1d : {param_idx: [[lo,hi], ...]}
        # masks_2d     : [ {"idx":(i,j), "labels":..., "components":..., "grid_x":..., "grid_y":...}, ... ]
        # round_idx    : round this sampler proposes for; tags the acceptance log lines
        self.base = UniformSampler(prior_bounds, rng=rng,
                                   dist_uniform_in_volume=dist_uniform_in_volume,
                                   spin_param_basis=spin_param_basis)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.intervals_1d = intervals_1d
        self.masks_2d = masks_2d
        # cube-sample dist only when the prior is uniform-in-volume; if the run
        # samples uniformly in distance, a linear interval draw is the match.
        self.dist_uniform_in_volume = dist_uniform_in_volume
        self.round_idx = round_idx
        self._tag = ("[MaskRejectSampler]" if round_idx is None
                     else f"[MaskRejectSampler r{round_idx}]")

    def _draw_once(self, n, t_obs_end):
        # 1) draw everything from the base prior; we overwrite the truncated dims
        _, tmnre = self.base.sample(n, t_obs_end)          # tmnre: (11, n), dist row already in Gpc

        # 2) 1D-truncated params -> direct interval draw (cube only for dist)
        for idx, intervals in self.intervals_1d.items():
            cube = (idx == _DIST_IDX) and self.dist_uniform_in_volume
            tmnre[idx] = _sample_from_intervals(intervals, n, self.rng, cube=cube)

        # 3) 2D pairs -> per-component draw + mask reject
        for m in self.masks_2d:
            i, j = m["idx"]
            assert i!=4 and j!=4 #distance not supported
            x, y, acc = _sample_2d_from_components(
                m["components"], m["labels"], m["grid_x"], m["grid_y"], n, self.rng)
            tmnre[i], tmnre[j] = x, y
            print(f"{self._tag} 2D pair {m['idx']} acceptance={acc:.3f}")

        return tmnre 
    
    def sample(self, n, t_obs_end):
        """Draw n samples: base prior, with the truncated dims overwritten.

        In the chieff_chidiff basis the spin-validity rejection has to happen
        *after* the overwrite, not inside base.sample(): validity depends on the
        triple (q, chi_eff, chi_diff), so replacing any of slots 1,2,3 from a
        mask invalidates the check the base sampler already did.
        """
        if self.base.spin_param_basis != "chieff_chidiff":
            tmnre = self._draw_once(n, t_obs_end)          # current body
            self.last_acceptance_ratio = 1.0
        else:
            collected, drawn, acc = [], 0, 0
            n_batches = 0
            while sum(c.shape[1] for c in collected) < n:
                if n_batches >= _MAX_SPIN_REJECT_BATCHES:
                    got = sum(c.shape[1] for c in collected)
                    raise RuntimeError(
                        f"{self._tag} spin rejection did not converge: "
                        f"{got}/{n} samples after {n_batches} batches "
                        f"(acceptance {acc / max(drawn, 1):.4f}). The (chi_eff, "
                        f"chi_diff) mask lies mostly outside the physical region "
                        f"|chi1|,|chi2| <= 1 for this q range, so almost every "
                        f"draw is rejected. Check the spin marginal's truncation.")
                batch = self._draw_once(n, t_obs_end)
                keep = self.base._accept_spin(batch)
                drawn += batch.shape[1]; acc += int(keep.sum())
                n_batches += 1
                if keep.any():
                    collected.append(batch[:, keep])
            tmnre = np.concatenate(collected, axis=1)[:, :n]
            self.last_acceptance_ratio = acc / max(drawn, 1)
            # Low here means the mask overlaps the unphysical |chi|>1 region:
            # the loop still returns exactly n, but pays for it in redraws.
            print(f"{self._tag} post-mask spin rejection "
                  f"(chieff_chidiff): acceptance ratio "
                  f"{self.last_acceptance_ratio:.3f} "
                  f"({acc}/{drawn} over {len(collected)} batch(es))")
        bbhx = self.base.samples_to_bbhx_input(tmnre.copy(), t_obs_end)
        return bbhx, tmnre

    def samples_to_bbhx_input(self, samples, t_obs_end):
        return self.base.samples_to_bbhx_input(samples, t_obs_end)
