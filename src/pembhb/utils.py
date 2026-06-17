import yaml
import copy
import torch
import os 
from pembhb import ROOT_DIR, get_torch_dtype, FMIN_FLOOR
import numpy as np
# from pembhb.data import MBHBDataset, mbhb_collate_fn
from glob import glob
from torch.utils.data import DataLoader, TensorDataset
from lightning.pytorch.callbacks import  Callback

from datetime import datetime, timedelta

import matplotlib.pyplot as plt
from tqdm import tqdm
from bbhx import waveformbuild as wfb
DAY_SI = 86400
_ORDERED_PRIOR_KEYS = [
        "logMchirp",
        "q",
        "chi1",
        "chi2",
        "dist",
        "phi",
        "inc",
        "lambda",
        "beta",
        "psi",
        "Deltat"
    ]

def print_params(params: np.array):
    for idx, param in enumerate(params):
        print(f"{_ORDERED_PRIOR_KEYS[idx]}: {params[param]}")

def read_config(fname: str):
    with open(fname, "r", encoding="utf-8") as file:
        conf = yaml.safe_load(file)
    return conf


def apply_pipeline_section(cfg: dict, section_name: str) -> dict:
    """Promote keys from ``cfg[section_name]`` to the top level of ``cfg``.

    The training config is split into a shared part plus pipeline-specific
    sub-sections (``sequential_training``, ``joint_training``).  Model classes
    and scripts still read the historical flat keys (e.g. ``cfg["learning_rate"]``),
    so we flatten the requested section at script startup.

    - Keys from the section override top-level keys (section wins).
    - If the section is missing or empty, ``cfg`` is returned unchanged (so
      older YAMLs that still have the flat keys at the top level keep working).
    - Returned value is the same dict (mutated in place) for chaining.
    """
    section = cfg.get(section_name)
    if not section:
        return cfg
    for k, v in section.items():
        cfg[k] = v
    return cfg

def choose_device_for_pp(required_gib: float = 2.0) -> torch.device:
    """Check if enough CUDA memory is free for pp_plot; fall back to CPU."""
    if torch.cuda.is_available():
        free, _ = torch.cuda.mem_get_info()
        free_gib = free / (1024 ** 3)
        if free_gib >= required_gib:
            print(f"[pp_plot] Using CUDA ({free_gib:.1f} GiB free)")
            return torch.device("cuda")
        else:
            print(f"[pp_plot] Only {free_gib:.1f} GiB free on CUDA (need {required_gib}), falling back to CPU")
    else:
        print("[pp_plot] CUDA not available, using CPU")
    return torch.device("cpu")


def get_logratios_grid(dataloader: torch.utils.data.DataLoader, model: 'InferenceNetwork', ngrid_points: int, in_param_idx : int, out_param_idx: int, low: float=None , high: float=None, device: torch.device = None, grid_chunk_size: int = 100):
    """Generate a grid of logratios for a given observation and model.
    This is useful for plotting the posterior and to make pp plots.

    Processes one observation at a time to keep GPU memory bounded
    regardless of dataloader batch size.

    :param data: observation data
    :type data: torch.Tensor
    :param model: trained model
    :type model: InferenceNetwork
    :param low: lower bound of the grid
    :type low: float
    :param high: upper bound of the grid
    :type high: float
    :param ngrid_points: number of points in the grid, defaults to 100
    :type ngrid_points: int, optional
    :param in_param_idx: index of the parameter to evaluate the logratios for, with respect to the input of the model (i.e. prior order)
    :type in_param_idx: int
    :param out_param_idx: index that identifies the logratios to return , with respect to the output of the model, (i.e. order defined in config_td.yaml)
    :type out_param_idx: int
    :param grid_chunk_size: number of grid points to evaluate per forward pass
    :type grid_chunk_size: int
    :return: logratios for the grid, with shape [batchsize, ngrid_points], injection parameters with shape [batchsize, 11], grid with shape [ngrid_points, 1]
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = []
    injection_params = []

    model.eval()
    model = model.to(device)
    # Use the actual sampling prior (sampler_init_kwargs) as the
    # authoritative source for the grid range.  Fall back to conf["prior"]
    # for backward compatibility with old YAML sidecars.
    _sik = model.hparams["dataset_info"].get("sampler_init_kwargs", {})
    if "prior_bounds" in _sik:
        prior_trained_dict = _sik["prior_bounds"]
    else:
        prior_trained_dict = model.hparams["dataset_info"]["conf"]["prior"]
    prior_bounds = prior_trained_dict[_ORDERED_PRIOR_KEYS[in_param_idx]]
    low = low if low is not None else prior_bounds[0]
    high = high if high is not None else prior_bounds[1]
    grid = torch.linspace(low, high, ngrid_points).to(device).reshape(-1, 1)
    zero_pad1d = torch.zeros(ngrid_points, 10, device=device)
    grid_padded = torch.cat((zero_pad1d[:, :in_param_idx], grid, zero_pad1d[:, in_param_idx:]), dim=1)  # Shape: [ngrid_points, 11]
    with torch.no_grad():
        for batch in dataloader:
            batch = materialize_gpu_noise(batch)
            data_fd = (batch["wave_fd"]+batch["noise_fd"]).to(device)  # Shape: [batchsize, n_channels, n_datapoints]
            source_parameters = batch["source_parameters"]  # Shape: [batchsize, 11]
            has_td = "wave_td" in batch and "noise_td" in batch
            if has_td:
                data_td = (batch["wave_td"]+batch["noise_td"]).to(device)

            batch_size = data_fd.shape[0]

            # Process one observation at a time to avoid materialising
            # [batch_size * ngrid_points, C, F] tensors on GPU.
            for obs_idx in range(batch_size):
                single_fd = data_fd[obs_idx:obs_idx + 1]  # [1, C, F]
                single_td = data_td[obs_idx:obs_idx + 1] if has_td else None

                logratios_chunks = []
                for start in range(0, ngrid_points, grid_chunk_size):
                    end = min(start + grid_chunk_size, ngrid_points)
                    chunk_grid = grid_padded[start:end]  # [chunk, 11]
                    chunk_size = end - start
                    chunk_fd = single_fd.expand(chunk_size, -1, -1)
                    chunk_td = single_td.expand(chunk_size, -1, -1) if has_td else None
                    logits = model(chunk_fd, chunk_td, chunk_grid)[:, out_param_idx]
                    logratios_chunks.append(logits.detach().cpu())

                obs_logratios = torch.cat(logratios_chunks, dim=0)  # [ngrid_points]
                results.append(obs_logratios.unsqueeze(0))  # [1, ngrid_points]
                injection_params.append(source_parameters[obs_idx, in_param_idx].detach().cpu().unsqueeze(0))

        results = torch.cat(results, dim=0).numpy()
        injection_params = torch.cat(injection_params, dim=0).numpy()
        grid = grid.detach().cpu().numpy()
    return results, injection_params, grid

def get_logratios_grid_2d(dataloader: torch.utils.data.DataLoader, model: 'InferenceNetwork', ngrid_points: int, out_param_idx : int, in_param_idx : tuple,
                          bounds_0: tuple = None, bounds_1: tuple = None, device: torch.device = None, grid_chunk_size: int = 500):
    """
    Compute logratios on a 2D grid for two input parameters.

    Processes one observation at a time and chunks the grid to keep GPU
    memory bounded regardless of dataloader batch size.

    :param dataloader: the data loader providing the observations
    :type dataloader: torch.utils.data.DataLoader
    :param model: the inference model
    :type model: InferenceNetwork
    :param ngrid_points: the number of grid points in each dimension
    :type ngrid_points: int
    :param out_param_idx: will fetch the logratios corresponding to output[out_param_idx]
    :type out_param_idx: int
    :param in_param_idx: the indices of the input parameters
    :type in_param_idx: tuple
    :param bounds_0: the bounds of the interval on which the first input parameter is defined, defaults to the trained prior
    :type bounds_0: tuple, optional
    :param bounds_1: the bounds of the interval on which the second input parameter is defined, defaults to trained prior
    :type bounds_1: tuple, optional
    :param device: device to run on, defaults to CUDA if available
    :type device: torch.device, optional
    :param grid_chunk_size: number of grid points to evaluate per forward pass (controls peak GPU memory)
    :type grid_chunk_size: int
    :return: results, injection_params, grid_x, grid_y where results has shape (batch size, ngrid_points, ngrid_points), injection_params has shape (batch size, 2), grid_x and grid_y have shape (ngrid_points, ngrid_points)
    :rtype: _type_
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = []
    injection_params = []
    n_total = ngrid_points ** 2

    with torch.no_grad():
        model.eval()
        model = model.to(device)
        # Use the actual sampling prior (sampler_init_kwargs) as the
        # authoritative source for the grid range.  Fall back to conf["prior"]
        # for backward compatibility with old YAML sidecars.
        _sik = model.hparams["dataset_info"].get("sampler_init_kwargs", {})
        if "prior_bounds" in _sik:
            prior_trained_dict = _sik["prior_bounds"]
        else:
            prior_trained_dict = model.hparams["dataset_info"]["conf"]["prior"]
        if bounds_0 is None:
            bounds_0 = prior_trained_dict[_ORDERED_PRIOR_KEYS[in_param_idx[0]]]
        if bounds_1 is None:
            bounds_1 = prior_trained_dict[_ORDERED_PRIOR_KEYS[in_param_idx[1]]]

        lows = [bounds_0[0], bounds_1[0]]
        highs = [bounds_0[1], bounds_1[1]]
        grid_0 = torch.linspace(lows[0], highs[0], ngrid_points).reshape(-1)
        grid_1 = torch.linspace(lows[1], highs[1], ngrid_points).reshape(-1)

        grid_x, grid_y = torch.meshgrid(grid_0, grid_1, indexing="xy")  # (ngrid, ngrid)
        flattened_x = grid_x.flatten()
        flattened_y = grid_y.flatten()
        grid = torch.stack((flattened_x, flattened_y), dim=1).to(device)  # [n_total, 2]

        # Padded grid input: [n_total, 11]
        grid_padded_input = torch.zeros(n_total, 11, device=device)
        grid_padded_input[:, in_param_idx[0]] = grid[:, 0]
        grid_padded_input[:, in_param_idx[1]] = grid[:, 1]

        for batch in dataloader:
            batch = materialize_gpu_noise(batch)
            data_fd = (batch["wave_fd"] + batch["noise_fd"]).to(device)
            source_parameters = batch["source_parameters"]
            has_td = "wave_td" in batch and "noise_td" in batch
            if has_td:
                data_td = (batch["wave_td"] + batch["noise_td"]).to(device)
            batch_size = data_fd.shape[0]

            # Process one observation at a time to avoid materialising
            # [batch_size * n_total, C, F] tensors on GPU.
            for obs_idx in range(batch_size):
                single_fd = data_fd[obs_idx:obs_idx + 1]   # [1, C, F]
                single_td = data_td[obs_idx:obs_idx + 1] if has_td else None

                logratios_chunks = []
                for start in range(0, n_total, grid_chunk_size):
                    end = min(start + grid_chunk_size, n_total)
                    chunk_grid = grid_padded_input[start:end]  # [chunk, 11]
                    chunk_size = end - start
                    # expand is a view — no memory allocation until forward pass
                    chunk_fd = single_fd.expand(chunk_size, -1, -1)
                    chunk_td = single_td.expand(chunk_size, -1, -1) if has_td else None
                    logits = model(chunk_fd, chunk_td, chunk_grid)[:, out_param_idx]
                    logratios_chunks.append(logits.detach().cpu())

                obs_logratios = torch.cat(logratios_chunks, dim=0)  # [n_total]
                results.append(obs_logratios.unsqueeze(0))  # [1, n_total]
                injection_params.append(source_parameters[obs_idx, in_param_idx].detach().cpu().unsqueeze(0))

        results = torch.cat(results, dim=0).numpy().reshape(-1, ngrid_points, ngrid_points)
        injection_params = torch.cat(injection_params, dim=0).numpy()

    return results, injection_params, grid_x.cpu().numpy(), grid_y.cpu().numpy()

def get_pvalues_1d(logratios: np.array, grid: np.array, inj_param: np.array):
    """Calculate p-values for a 1D logratios array. 
    Recall that exp(logratios) = posterior/prior, and here we assume a uniform prior. 
    
    :param logratios: logratios for the grid , has shape (batch size, ngrid_points)
    :type logratios: np.array
    :param inj_param: injected parameter value
    :type inj_param: float
    :param ngrid_points: number of points in the grid
    :type ngrid_points: int
    :return: p-value for the injected parameter
    """
    
    ratios = np.exp(logratios)
    sorted_ratios = np.sort(ratios, axis=1) 
    sorted_indices =  np.argsort(ratios, axis=1)
    sorted_grid =  grid[sorted_indices]
    inj_param = inj_param.reshape(-1,1,1)
    # find closest value in the grid to the injected parameter
    idx = np.argmin(np.abs(sorted_grid - inj_param), axis=1)[:,0,...]
    idx_rank = np.arange(idx.shape[0])

    cumsum =  np.cumsum(sorted_ratios, axis=1)
    cumsum /= cumsum[:,-1:]  # normalize to get a cumulative distribution
    #print(idx_rank.shape, idx.shape)
    p_values = cumsum[idx_rank, idx]

    return p_values

def get_pvalues_2d(logratios: np.array, grid_0: np.array , grid_1: np.array, inj_param: np.array):
    """Calculate p-values for a 2D logratios array.
    Assume the grid was flattened before. 
    Recall that exp(logratios) = posterior/prior, and here we assume a uniform prior.

    :param logratios: logratios for the grid , has shape (batch size, n_grid_0, n_grid_1)
    :type logratios: np.array
    :param grid_0: grid of values of the parameter 0 where the network was evaluated, has shape (n_grid_0,n_grid_1)
    :type grid_0: np.array
    :param grid_1: grid of values of the parameter 1 where the network was evaluated, has shape (n_grid_0,n_grid_1)
    :type grid_1: np.array
    :param inj_param: injected parameter value
    :type inj_param: np.array with shape (batch size, 2)
    """


    # convert logratios to unnormalised probabilities
    probs = np.exp(logratios)  # shape (batch_size, n_grid_0, n_grid_1)

    # flatten spatial dimensions for sorting
    probs_flat = probs.reshape(probs.shape[0], -1)  # shape (batch_size, n_grid_0*n_grid_1)
    # sort probabilities in descending order
    sort_idx = np.argsort(-probs_flat, axis=1)  # shape (batch_size, n_grid_0*n_grid_1)
    # apply sorting
    probs_sorted = np.take_along_axis(probs_flat, sort_idx, axis=1)  # shape (batch_size, n_grid_0*n_grid_1)
    # normalise so total probability sums to 1
    probs_sorted /= probs_sorted.sum(axis=1, keepdims=True)  # shape (batch_size, n_grid_0*n_grid_1)
    # cumulative distribution
    cumsum_probs = np.cumsum(probs_sorted, axis=1)  # shape (batch_size, n_grid_0*n_grid_1)
    # flatten the grid into list of coordinates
    grid_points = np.stack([grid_0.flatten(), grid_1.flatten()], axis=-1)  # shape (n_grid_0*n_grid_1, 2)
    # for each injection, find nearest grid point index
    diffs = inj_param[:, None, :] - grid_points[None, :, :]  # shape (batch_size, n_grid_0*n_grid_1, 2)
    dists = np.linalg.norm(diffs, axis=-1)  # shape (batch_size, n_grid_0*n_grid_1)
    inj_idx = np.argmin(dists, axis=1)  # shape (batch_size,)
    # map from original flat index to sorted index
    # first, invert sorting to get rank positions
    inv_sort_idx = np.argsort(sort_idx, axis=1)  # shape (batch_size, n_grid_0*n_grid_1)
    # find the rank of the injected point inside the sorted array
    inj_rank = np.take_along_axis(inv_sort_idx, inj_idx[:, None], axis=1).squeeze(-1)  # shape (batch_size,)
    # extract p-values from cumulative sums
    pvalues  = np.take_along_axis(cumsum_probs, inj_rank[:, None], axis=1).squeeze(-1)  # shape (batch_size,)

    return pvalues  # shape (batch_size,)


def grid_posterior_moments_1d(logratios: np.array, grid: np.array):
    """First two moments of a 1D NRE marginal posterior on its grid.

    The grid posterior is ``p(θ) ∝ exp(logratios)`` under the uniform-prior
    convention used throughout (see :func:`get_pvalues_1d`). Because the grid
    is uniformly spaced, normalising the discrete weights to sum to 1 gives the
    correct discrete moment estimates (the spacing Δθ cancels in the ratio).

    :param logratios: log-ratios on the grid, shape ``(batch, ngrid)``
    :type logratios: np.array
    :param grid: grid values, shape ``(ngrid,)`` or ``(ngrid, 1)``
    :type grid: np.array
    :return: ``(mean, std)`` each of shape ``(batch,)``
    :rtype: tuple[np.array, np.array]
    """
    g = np.asarray(grid, dtype=np.float64).reshape(-1)            # (ngrid,)
    lr = np.asarray(logratios, dtype=np.float64)                  # (batch, ngrid)
    # Posterior weights on the grid. Subtracting the per-row max is a pure
    # numerical-stability trick: it cancels in the normalisation, so the
    # moments are unchanged.
    w = np.exp(lr - lr.max(axis=1, keepdims=True))
    w /= w.sum(axis=1, keepdims=True)                             # (batch, ngrid)

    mean = (w * g[None, :]).sum(axis=1)                           # (batch,)
    var = (w * g[None, :] ** 2).sum(axis=1) - mean ** 2
    # var is E[θ²]-E[θ]² ≥ 0 in exact arithmetic; for a very narrow posterior
    # the subtraction of two large near-equal numbers can land slightly
    # negative in float, which would make sqrt return nan. Floor it at 0.
    var = np.maximum(var, 0.0)
    return mean, np.sqrt(var)


def grid_posterior_moments_2d(logratios: np.array, grid_0: np.array, grid_1: np.array):
    """Per-axis (marginal) moments of a 2D NRE posterior on its grid.

    Returns the mean and standard deviation of *each* parameter — i.e. the
    ``sqrt`` of the diagonal entries of the 2D covariance — which is what the
    per-parameter λ / τ statistics need. Axis order matches
    :func:`get_logratios_grid_2d`: ``grid_0`` carries the ``in_param_idx[0]``
    value at every cell, ``grid_1`` the ``in_param_idx[1]`` value.

    The marginal mean of θ₀ is its joint-posterior-weighted average over the
    whole grid, ``Σ_cells w·θ₀``. Summing over both grid axes already performs
    the marginalisation over θ₁ (the weights of every cell sharing a θ₀ value
    add up to that value's marginal weight), so no explicit axis-collapse is
    needed. Using ``grid_0``/``grid_1`` — which hold the value per cell — keeps
    this independent of the meshgrid orientation.

    :param logratios: log-ratios on the grid, shape ``(batch, ngrid, ngrid)``
    :type logratios: np.array
    :param grid_0: parameter-0 value at each cell, shape ``(ngrid, ngrid)``
    :type grid_0: np.array
    :param grid_1: parameter-1 value at each cell, shape ``(ngrid, ngrid)``
    :type grid_1: np.array
    :return: ``(mean_0, std_0, mean_1, std_1)`` each of shape ``(batch,)``
    :rtype: tuple[np.array, np.array, np.array, np.array]
    """
    g0 = np.asarray(grid_0, dtype=np.float64)[None, ...]          # (1, ngrid, ngrid)
    g1 = np.asarray(grid_1, dtype=np.float64)[None, ...]
    lr = np.asarray(logratios, dtype=np.float64)                 # (batch, ngrid, ngrid)

    w = np.exp(lr - lr.max(axis=(1, 2), keepdims=True))
    w /= w.sum(axis=(1, 2), keepdims=True)                       # normalise per batch

    # Weighted average of each parameter's per-cell value over the joint
    # posterior == its marginal mean/variance (see docstring).
    mean_0 = (w * g0).sum(axis=(1, 2))
    mean_1 = (w * g1).sum(axis=(1, 2))
    var_0 = np.maximum((w * g0 ** 2).sum(axis=(1, 2)) - mean_0 ** 2, 0.0)
    var_1 = np.maximum((w * g1 ** 2).sum(axis=(1, 2)) - mean_1 ** 2, 0.0)
    return mean_0, np.sqrt(var_0), mean_1, np.sqrt(var_1)




def update_bounds(model: 'InferenceNetwork', observation_loader: DataLoader, priordict: dict, in_param_idx: int, n_gridpoints: int, out_param_idx: int, eps: float = 1e-5):
    """Update the prior bounds based on the posterior obtained from a model on a single observation. 
    Used to do truncation in MNRE. 

    :param model: trained inference model
    :type model: InferenceNetwork
    :param observation_dataset: dataset containing the (single) obs
    :type observation_dataset: MBHBDataset
    :param priordict: dictionary containing the prior bounds for each parameter. 
    :type priordict: dict
    :param in_param_idx: index of the parameter to update
    :type in_param_idx: int
    :param n_gridpoints: number of points in the grid, defaults to 100
    :type n_gridpoints: int, optional
    :return: updated prior bounds
    :rtype: dict
    """
    print(f"Updating prior bounds for {_ORDERED_PRIOR_KEYS[in_param_idx]}...")
    # evaluate the model over a decently fine grid, which requires knowledge of previous prior region 
    prior_bounds_found = False
    while prior_bounds_found==False:
        try:
            
            logratios, injection_params, grid = get_logratios_grid(observation_loader, model, n_gridpoints, in_param_idx=in_param_idx, out_param_idx=out_param_idx)
            # find the 95% two tail interval of the posterior 
            print(f"injection_params are: {injection_params}")
            cumsum = np.cumsum(np.exp(logratios))
            cumsum /= cumsum[-1]  
            idx_low = np.argwhere(cumsum < eps/2)[-1]
            idx_high = np.argwhere(cumsum > 1-eps/2)[0]
            print(f"prior bounds found with ngridpoints {n_gridpoints}")
            prior_bounds_found = True
        except IndexError:
            print(f"cannot update prior bounds with current number of grid points: {n_gridpoints} and eps {eps}")
            n_gridpoints *= 2
        
    new_low = grid[idx_low]
    new_high = grid[idx_high]

    updated_prior = priordict.copy()
    updated_prior[_ORDERED_PRIOR_KEYS[in_param_idx]] = [new_low.item(), new_high.item()]
    print(f"Updated prior for {_ORDERED_PRIOR_KEYS[in_param_idx]}: {updated_prior[_ORDERED_PRIOR_KEYS[in_param_idx]]}")
    
    return updated_prior

def update_bounds_2d(model: 'InferenceNetwork', observation_loader: DataLoader, priordict: dict, in_param_idx: tuple, n_gridpoints: int, out_param_idx: int, eps: float = 1e-5):

    # predict the model on the loader by getting the logratios 
    logratios = get_logratios_grid_2d(observation_loader, model, n_gridpoints, out_param_idx=out_param_idx, in_param_idx=in_param_idx)

    # get the grid of 

def _pp_curve(ax, ranks: np.ndarray, label: str | None = None, **plot_kw):
    """Draw one P-P curve from a 1-D rank array onto an existing Axes.

    Sorted ranks on x (HPD level), normalised empirical CDF on y (always
    ``(0, 1]``).  Used by both :func:`pp_plot` (single-curve, per-marginal
    file) and :func:`pp_plot_overlay` (multi-curve overlay).
    """
    ranks = np.asarray(ranks)
    if ranks.size == 0:
        return
    sorted_p = np.sort(ranks)
    empirical = np.arange(1, sorted_p.size + 1) / sorted_p.size
    ax.plot(sorted_p, empirical, label=label, **plot_kw)


def pp_plot( dataloader, model , in_param_idx: int, name: str, out_param_idx: int, output_dir: str = None, device: torch.device = None):
    """Generate a pp plot using the examples in dataset, and the posteriors obtained by the model .
    :param dataset: dataset used to make the pp plot
    :type dataset: MBHBDataset
    :param model:  trained model used to make the pp plot
    :type model: InferenceNetwork
    :param low: lower bound of the prior used to generate the dataset
    :type low: float
    :param high: upper bound of the prior used to generate the dataset
    :type high: float
    :param inj_param_idx: index of the parameter that you want to make the pp plot for, with respect to the output of the model.
    :type inj_param_idx: int
    :param name: name of the plot, defaults to None
    :type name: str, optional
    :param output_dir: directory to save the plot to. Defaults to ROOT_DIR/plots.
    :type output_dir: str, optional
    :param device: device to run inference on. Defaults to CUDA if available.
    :type device: torch.device, optional
    """
    print(f"Making pp plot for {name}...")
    logratios, injection_params, grid = get_logratios_grid(dataloader, model, ngrid_points=100, in_param_idx=in_param_idx, out_param_idx=out_param_idx, device=device)
    p_values = get_pvalues_1d(logratios, grid, injection_params)
    fig, ax  = plt.subplots(figsize=(10, 6))
    _pp_curve(ax, p_values, marker='o', linestyle='-', markersize=3)
    ax.plot([0, 1], [0, 1], color="black", lw=0.6, ls="--", alpha=0.6)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('HPD level')
    ax.set_ylabel('empirical coverage')
    ax.set_title(f'P-P plot, {name}')
    ax.grid(visible=True)
    if name is not None:
        if output_dir is None:
            output_dir = os.path.join(ROOT_DIR, "plots")
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, f"{name}_pp_plot.png"))
    plt.close()


def pp_plot_overlay(ranks_per_marginal: dict, output_path: str, title: str = ""):
    """Overlay 1-D P-P curves for several marginals on a single figure.

    Each entry of ``ranks_per_marginal`` is ``{label: ranks_1d_array}`` —
    typically collected by the PP-KS callback after one ``get_pvalues_1d``
    call per marginal.  The figure shows one color-cycled curve per marginal
    plus a ``y = x`` reference (perfect calibration).  The parent directory
    of ``output_path`` is created if missing.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 7))
    for label, ranks in sorted(ranks_per_marginal.items()):
        _pp_curve(ax, ranks, label=label, lw=1.0)
    ax.plot([0, 1], [0, 1], color="black", lw=0.6, ls="--", alpha=0.6,
            label="ideal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("HPD level")
    ax.set_ylabel("empirical coverage")
    ax.set_title(title or "P-P overlay")
    ax.grid(visible=True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=110)
    plt.close(fig)

def pp_plot_2d(dataloader, model,  in_param_idx: tuple, out_idx: int, name: str, output_dir: str = None, device: torch.device = None):
    print(f"Making pp plot for {name}...")
    logratios, injection_params, grid_x, grid_y = get_logratios_grid_2d(dataloader, model, ngrid_points=50, out_param_idx=out_idx, in_param_idx=in_param_idx, device=device)
    p_values = get_pvalues_2d(logratios, grid_x, grid_y, injection_params)
    sorted_pvalues = np.sort(p_values)
    sorted_normalised_rank = np.arange(sorted_pvalues.shape[0])/sorted_pvalues.shape[0]
    fig, ax  = plt.subplots(figsize=(10, 6))
    ax.plot( sorted_pvalues, sorted_normalised_rank, marker='o', linestyle='-', markersize=3)
    ax.plot([0,1],[0,1], linestyle='--', color='red')
    ax.set_xlabel('HPD level')
    ax.set_ylabel('empirical coverage')
    ax.set_title(f'P-P plot, {name}')
    ax.grid(visible=True)
    if name is not None:
        if output_dir is None:
            output_dir = os.path.join(ROOT_DIR, "plots")
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, f"{name}_pp_plot_2d.png"))
    plt.close()


def chirp_mass_from_m1m2(m1, m2):
    """Calculate the chirp mass from the component masses.
    :param m1: mass of the primary black hole
    :type m1: float or np.array
    :param m2: mass of the secondary black hole
    :type m2: float or np.array
    :return: chirp mass
    :rtype: float or np.array
    """
    return (m1*m2)**(3/5) / (m1+m2)**(1/5)                            

class BBHWaveformTD(wfb.BBHxParallelModule):
    """Generate waveforms put through response functions

    This class generates waveforms put through the LISA response function. In the
    future, ground-based analysis may be added. Therefore, it currently
    returns the TDI variables according the response keyword arguments given.

    If you use this class, please cite `arXiv:2005.01827 <https://arxiv.org/abs/2005.01827>`_
    and `arXiv:2111.01064 <https://arxiv.org/abs/2111.01064>`_, as well as the papers
    listed for the waveform and response given just below.

    Right now, it is hard coded to produce the waveform with
    :class:`PhenomHMAmpPhase <bbhx.waveforms.phenomhm.PhenomHMAmpPhase>`. This can also be used
    to produce PhenomD. See the docs for that waveform. The papers describing PhenomHM/PhenomD
    waveforms are here: `arXiv:1708.00404 <https://arxiv.org/abs/1708.00404>`_,
    `arXiv:1508.07250 <https://arxiv.org/abs/1508.07250>`_, and
    `arXiv:1508.07253 <https://arxiv.org/abs/1508.07253>`_.

    The response function is the fast frequency domain response function
    from `arXiv:1806.10734 <https://arxiv.org/abs/1806.10734>`_ and
    `arXiv:2003.00357 <https://arxiv.org/abs/2003.00357>`_. It is implemented in
    :class:`LISATDIResponse <bbhx.response.fastfdresponse.LISATDIResponse`.

    This class is GPU accelerated.

    This is a small modification to the BBHx waveform generation which outputs in FD
    here we output in TD through IFFT

    Args:
        amp_phase_kwargs (dict, optional): Keyword arguments for the
            initialization of the ampltidue-phase waveform class: :class:`PhenomHMAmpPhase <bbhx.waveforms.phenomhm.PhenomHMAmpPhase>`.
        response_kwargs (dict, optional): Keyword arguments for the initialization
            of the response class: :class:`LISATDIResponse <bbhx.response.fastfdresponse.LISATDIResponse`.
        interp_kwargs (dict, optional): Keyword arguments for the initialization
            of the interpolation class: :class:`TemplateInterpFD`.
        use_gpu (bool, optional): If ``True``, use a GPU. (Default: ``False``)

    Attributes:
        amp_phase_gen (obj): Waveform generation class.
        data_length (int): Length of the final output data.
        interp_response (obj): Interpolation class.
        length (int): Length of initial evaluations of waveform and response.
        num_bin_all (int): Total number of binaries analyzed.
        num_interp_params (int): Number of parameters to interpolate (9).
        num_modes (int): Number of harmonic modes.
        out_buffer_final (xp.ndarray): Array with buffer information with shape:
            ``(self.num_interp_params, self.num_bin_all, self.num_modes, self.length)``.
            The order of the parameters is amplitude, phase, t-f, transferL1re, transferL1im,
            transferL2re, transferL2im, transferL3re, transferL3im.

    """
    def __init__(
        self,
        amp_phase_kwargs={},
        response_kwargs={},
        interp_kwargs={},
        force_backend=None,
    ):
        super().__init__(force_backend=force_backend)
        self.force_backend = force_backend
        # initialize waveform and response funtions
        self.amp_phase_gen = wfb.PhenomHMAmpPhase(**amp_phase_kwargs, force_backend=force_backend)
        self.response_gen = wfb.LISATDIResponse(**response_kwargs, force_backend=force_backend)

        self.num_interp_params = 9

        # setup the final interpolant
        self.interp_response = wfb.TemplateInterpFD(**interp_kwargs, force_backend=force_backend)

    @property
    def xp(self) -> object:
        """Numpy or Cupy"""
        return self.backend.xp

    @classmethod
    def supported_backends(cls) -> list:
        return ["bbhx_" + _tmp for _tmp in cls.GPU_RECOMMENDED()]

    @property
    def waveform_gen(self) -> callable:
        """C/CUDA wrapped function for computing waveforms"""
        return self.backend.direct_sum_wrap

    def __call__(
        self,
        m1,
        m2,
        chi1z,
        chi2z,
        distance,
        phi_ref,
        f_ref,
        inc,
        lam,
        beta,
        psi,
        t_ref,
        t_obs_start=0.0, # in years
        t_obs_end=1.0,   # in years
        dt = 5.0,        # in second
        out_channel = None,
        length=None,
        modes=None,
        shift_t_limits=True,
        compress=True, # TODO
        squeeze=False, # TODO
    ):
        r"""Generate the binary black hole frequency-domain TDI waveforms


        Args:
            m1 (double scalar or np.ndarray): Mass 1 in Solar Masses :math:`(m1 > m2)`.
            m2 (double or np.ndarray): Mass 2 in Solar Masses :math:`(m1 > m2)`.
            chi1z (double or np.ndarray): Dimensionless spin 1 (for Mass 1) in Solar Masses.
            chi2z (double or np.ndarray): Dimensionless spin 2 (for Mass 1) in Solar Masses.
            distance (double or np.ndarray): Luminosity distance in m.
            phi_ref (double or np.ndarray): Phase at ``f_ref``.
            f_ref (double or np.ndarray): Reference frequency at which ``phi_ref`` and ``t_ref`` are set.
                If ``f_ref == 0``, it will be set internally by the PhenomHM code
                to :math:`f_\\text{max} = \\text{max}(f^2A_{22}(f))`.
            inc (double or np.ndarray): Inclination of the binary in radians :math:`(\iota\in[0.0, \pi])`.
            lam (double or np.ndarray): Ecliptic longitude :math:`(\lambda\in[0.0, 2\pi])`.
            beta (double or np.ndarray): Ecliptic latitude :math:`(\\beta\in[-\pi/2, \pi/2])`.
            psi (double or np.ndarray): Polarization angle in radians :math:`(\psi\in[0.0, \pi])`.
            t_ref (double or np.ndarray): Reference time in seconds. It is set at ``f_ref``.
            t_obs_start (double, optional): Start time of observation in years
                in the LISA constellation reference frame. This is with reference to :math:`t=0`.
                (Default: 0.0)
            t_obs_end (double, optional): End time of observation in years in the
                LISA constellation reference frame. This is with reference to :math:`t=0`.
                (Default: 1.0)
            dt (double, optional): Sampling rate at which to evaluate the final waveform.
                set in seconds (Default: 10.0)
            out_channel (list, optional): Specify how many channels to output. If None 
            length (int, optional): Number of frequencies to use in sparse array for
                interpolation.
            modes (list, optional): Harmonic modes to use. If not given, they will
                default to those available in the waveform model. For PhenomHM:
                [(2,2), (3,3), (4,4), (2,1), (3,2), (4,3)]. For PhenomD: [(2,2)].
                (Default: ``None``)
            shift_t_limits (bool, optional): If ``False``, ``t_obs_start`` and ``t_obs_end``
                are relative to ``t_ref`` counting backwards in time. If ``True``,
                those quantities are relative to :math:`t=0`. (Default: ``False``)
            compress (bool, optional): If ``True``, combine harmonics into single channel
                waveforms. (Default: ``True``)



        Returns:
            xp.ndarray: Shape ``(3, self.length, self.num_bin_all)``.
                Final waveform for each binary. If  ``compress==True``.
                # TODO: switch dimensions?
            xp.ndarray:  Shape ``(3, self.num_modes, self.length, self.num_bin_all)``.
                Final waveform for each binary. If ``compress==False``.
        Raises:
            ValueError: ``length`` and ``freqs`` not given. Modes are given but not in a list.

        """
        # make sure everything is at least a 1D array
        m1 = np.atleast_1d(m1)
        m2 = np.atleast_1d(m2)
        chi1z = np.atleast_1d(chi1z)
        chi2z = np.atleast_1d(chi2z)
        distance = np.atleast_1d(distance)
        phi_ref = np.atleast_1d(phi_ref)
        inc = np.atleast_1d(inc)
        lam = np.atleast_1d(lam)
        beta = np.atleast_1d(beta)
        psi = np.atleast_1d(psi)
        t_ref = np.atleast_1d(t_ref)

        self.num_bin_all = len(m1)

        # TODO: add sanity checks for t_start, t_end
        # how to set up time limits
        if shift_t_limits is False:
            wfb.warnings.warn(
                "Deprecated: shift_t_limits. Previously shift_t_limits defaulted to False. This option is now removed and permanently set to shift_t_limits=True."
            )
            # t_ref_L = tLfromSSBframe(t_ref, lam, beta)

            # # start and end times are defined in the LISA reference frame
            # t_obs_start_L = t_ref_L - t_obs_start * YRSID_SI
            # t_obs_end_L = t_ref_L - t_obs_end * YRSID_SI

            # # convert to SSB frame
            # t_obs_start_SSB = tSSBfromLframe(t_obs_start_L, lam, beta, 0.0)
            # t_obs_end_SSB = tSSBfromLframe(t_obs_end_L, lam, beta, 0.0)

            # # fix zeros and less than zero
            # t_start = (
            #     t_obs_start_SSB if t_obs_start > 0.0 else np.zeros(self.num_bin_all)
            # )
            # t_end = t_obs_end_SSB if t_obs_end > 0.0 else np.zeros_like(t_start)

        # else:
        # start and end times are defined in the LISA reference frame
        t_obs_start_L   = t_obs_start * wfb.YRSID_SI
        t_obs_end_L     = t_obs_end * wfb.YRSID_SI
        # index at which to cut the signal
        n_cut           = int((t_obs_end_L - t_obs_start_L) / dt)
        # convert to SSB frame
        t_obs_start_SSB = wfb.tSSBfromLframe(t_obs_start_L, lam, beta, 0.0)
        # set up the end of response observation one day after merger
        t_obs_final = np.where(
                t_ref > t_obs_end_L,
                t_ref + DAY_SI,
                t_obs_end_L)
        
        t_obs_end_SSB = wfb.tSSBfromLframe(t_obs_final, lam, beta, 0.0)
        
        #t_obs_end_SSB = tSSBfromLframe(t_obs_end_L, lam, beta, 0.0)
        # To avoid issues in cut signal we take t_ref + 1*DAY_SI
        t_start = np.atleast_1d(t_obs_start_SSB)
        t_end = np.atleast_1d(t_obs_end_SSB)

        Tresponse = t_end - t_start
        
        self.length = length


        # setup harmonic modes
        if modes is None:
            # default mode setup
            self.num_modes = len(self.amp_phase_gen.allowable_modes)
        else:
            if not isinstance(modes, list):
                raise ValueError("modes must be a list.")
            self.num_modes = len(modes)

        self.num_bin_all = len(m1)

        out_buffer = self.xp.zeros(
            (self.num_interp_params * self.length * self.num_modes * self.num_bin_all)
        )


        phi_ref_amp_phase = np.zeros_like(m1)


        self.amp_phase_gen(
            m1,
            m2,
            chi1z,
            chi2z,
            distance,
            phi_ref_amp_phase,
            f_ref,
            t_ref,
            length,
            freqs=None,
            out_buffer=out_buffer,
            modes=modes,
            Tobs=Tresponse,
            direct=False,
        )
        
        
        # setup buffer to carry around all the quantities of interest
        # params are amp, phase, tf, transferL1re, transferL1im, transferL2re, transferL2im, transferL3re, transferL3im
        out_buffer = out_buffer.reshape(
            self.num_interp_params, self.num_bin_all, self.num_modes, self.length
        )
        out_buffer = out_buffer.flatten().copy()

        # compute response function
        self.response_gen(
            self.amp_phase_gen.freqs,
            inc,
            lam,
            beta,
            psi,
            phi_ref,
            length,
            out_buffer=out_buffer,  # fill into this buffer
            modes=self.amp_phase_gen.modes,
            direct=False,
        )

        # for checking
        self.out_buffer_final = out_buffer.reshape(
            9, self.num_bin_all, self.num_modes, self.length
        ).copy()

        
        #create time structure
        f22_start = self.amp_phase_gen.freqs_shaped[:,0,0]
        Mtots     = (m1 + m2 ) * wfb.MTSUN_SI
        nu        = m1 * m2 / (m1 + m2)**2
        T_step    = 1.5 * 5 / 256 / nu * (np.pi * Mtots * 0.8 * f22_start)**(-8./3.) * Mtots
        # pad freq to powers of 2
        # Unsure whether this is that much useful 
        n         = 2**int(np.max(np.ceil(np.log2(T_step / dt))))
        df        = 1. / n / dt
        freqs     = np.arange(0, n//2 + 1) * df
        # setup interpolant

        self.freqs = freqs
        spline = wfb.CubicSplineInterpolant(
            self.amp_phase_gen.freqs,
            out_buffer,
            length=self.length,
            num_interp_params=self.num_interp_params,
            num_modes=self.num_modes,
            num_bin_all=self.num_bin_all,
            force_backend=self.force_backend
        )
        # TODO: try single block reduction for likelihood (will probably be worse for smaller batch, but maybe better for larger batch)?
        template_channels = self.interp_response(
            freqs,
            spline.container,
            t_start,
            t_end,
            self.length,
            self.num_modes,
            3,
        )
        # fill the data stream

        if compress:
            # put in separate data streams
            data_FD = self.xp.zeros(
                (self.num_bin_all, 3, n//2 + 1), dtype=self.xp.complex128
            )
            for bin_i, (temp, start_i, length_i) in enumerate(
                zip(
                    template_channels,
                    self.interp_response.start_inds,
                    self.interp_response.lengths,
                )
            ):
                data_FD[bin_i, :, start_i : start_i + length_i] = temp    
                #data_FD[bin_i, :, start_i : start_i + length_i] = temp    
        else:
            raise NotImplementedError("Not implemented")
        if out_channel is None:
            # Turn the object to TD
            ifftseries= self.xp.fft.ifft(
                np.dstack(
            (data_FD[:,:,:-1], np.flip(data_FD[:,:,1:].conj(),axis=2))),
            axis = 2).real / dt
            # Rebuild time series from positive and negative times
            #timeseries = np.dstack((ifftseries[:,:,n//2:], ifftseries[:,:,:n//2])).real
            if n_cut <= n:
                return ifftseries[:,:,:n_cut]
            else:
                return self.xp.pad(ifftseries, [(0,0),(0,0),(0,n_cut - n)])
        else:
            # Turn the object to TD
            if isinstance(out_channel,(list,np.ndarray,tuple)):
                ifftseries= self.xp.fft.ifft(
                    np.dstack(
                (data_FD[:,out_channel,:-1], np.flip(data_FD[:,out_channel,1:].conj(),axis = 2))),
                axis = 2).real / dt
                # Rebuild time series from positive and negative times
                #timeseries = np.dstack((ifftseries[:,:,n//2:], ifftseries[:,:,:n//2])).real
                if n_cut <= n:
                    return ifftseries[:,:,:n_cut]
                else:
                    return self.xp.pad(ifftseries, [(0,0),(0,0),(0,n_cut - n)])

            else:
                ifftseries= self.xp.fft.ifft(
                    np.hstack(
                (data_FD[:,out_channel,:-1], np.flip(data_FD[:,out_channel,1:].conj(),axis = 1))),
                axis = 1).real / dt
                # Rebuild time series from positive and negative times
                #timeseries = np.dstack((ifftseries[:,:,n//2:], ifftseries[:,:,:n//2])).real             
                if n_cut <= n:
                    return ifftseries[:,:n_cut]
                else:
                    return self.xp.pad(ifftseries, [(0,0),(0,n_cut - n)])



def plot_posterior_1d(grid: np.array,  normalised_ratios: np.array, true_value: float,  ax_buffer: plt.Axes, parameter_name: str, title: str=None, **plot_kwargs):
    ax_buffer.plot(grid, normalised_ratios, **plot_kwargs)
    ax_buffer.axvline(x=true_value, color='r', linestyle='--')
    if title is not None:
        ax_buffer.set_title(title)
    ax_buffer.set_xlabel(parameter_name)
    ax_buffer.set_ylabel("Posterior Density")
    ax_buffer.grid()

# def plot_posterior_2d(grid_x: np.array, grid_y: np.array, ratios: np.array, true_values: list, ax_buffer: plt.Axes, parameter_names: list, title: str=None):
#     # dx = grid_x[1]-grid_x[0] # assuming uniform spacing
#     # dy = grid_y[1]-grid_y[0]
#     # #normalised_ratios = ratios / np.sum(ratios*dx*dy)
#     c = ax_buffer.pcolormesh(grid_x, grid_y, ratios, shading='auto', cmap="inferno")
    
#     # add contour lines at a few percentile levels of the density
#     flat = ratios.flatten()

#     # sort by density, high -> low
#     idx = np.argsort(flat)[::-1]
#     sorted_density = flat[idx]

#     # cumulative mass (area factor omitted; cancels on uniform grid)
#     cum = np.cumsum(sorted_density)
#     cum /= cum[-1]

#     # credible levels
#     targets = [0.6827, 0.9545, 0.9973]#1-1e-4]

#     # density thresholds
#     thresh = []
#     for t in targets:
#         i = np.searchsorted(cum, t)
#         thresh.append(sorted_density[i])

#     # contour wants increasing levels: widest -> narrowest
#     levels = np.sort(thresh)
#     sorted_index_levels = np.argsort(thresh)

#     targets_sorted = np.array(targets)[sorted_index_levels]
#     cont = ax_buffer.contour(
#         grid_x, grid_y, ratios,
#         levels=levels,
#         colors='white',
#         linewidths=0.8
#     )
#     fmt = {lev: f"{p:.3f}" for lev, p in zip(cont.levels, targets_sorted)}
#     boxes = []


#     for lvl_segs in cont.allsegs:
#         xs = []
#         ys = []
#         for seg in lvl_segs:
#             xs.append(seg[:, 0])
#             ys.append(seg[:, 1])
#         xs = np.concatenate(xs)
#         ys = np.concatenate(ys)
#         boxes.append((xs.min(), xs.max(), ys.min(), ys.max()))

#     # for lvl, box in zip(levels, boxes):
#     #     print(lvl, box)

#     # for lvl, (xmin, xmax, ymin, ymax) in zip(levels, boxes):
#     #     print(f"level {lvl}: xmin={xmin}, xmax={xmax}, ymin={ymin}, ymax={ymax}")
#     ax_buffer.clabel(cont, fmt=fmt, fontsize=8)
#     ax_buffer.axvline(x=true_values[0], color='r', linestyle='--', label='True Value')
#     ax_buffer.axhline(y=true_values[1], color='r', linestyle='--')
#     cbar = plt.colorbar(c, ax=ax_buffer)
#     cbar.set_label('Posterior Density')
#     if title is not None:
#         ax_buffer.set_title(title)
#     ax_buffer.set_xlabel(parameter_names[0])
#     ax_buffer.set_ylabel(parameter_names[1])
#     #plt.colorbar(c, ax=ax_buffer, label='Posterior Density')

#     ax_buffer.grid()
#     return boxes[-1]

def contour_levels(ratios, targets=(0.6827, 0.9545, 0.9973, 0.9999)):
    flat = ratios.ravel()
    idx = np.argsort(flat)[::-1]
    sorted_density = flat[idx]

    cum = np.cumsum(sorted_density)
    cum /= cum[-1]

    thresh = []
    for t in targets:
        i = np.searchsorted(cum, t)
        thresh.append(sorted_density[i])

    sorted_levels = np.sort(thresh)
    sorted_targets = np.array(targets)[np.argsort(thresh)]
    return sorted_levels, sorted_targets

def contour_boxes(grid_x, grid_y, ratios, levels, ax=None, colors=None, linestyles=None, linewidths=None, alpha=None):
    """
    Compute contour boxes and optionally plot contours with custom styling.
    
    Parameters:
    -----------
    colors : str or list, optional
        Color(s) for contour lines
    linestyles : str or list, optional
        Linestyle(s) for contour lines
    linewidths : float or list, optional
        Linewidth(s) for contour lines
    alpha : float or list, optional
        Alpha value(s) for contour lines
    """
    contour_kwargs = {}
    if colors is not None:
        contour_kwargs['colors'] = colors
    if linestyles is not None:
        contour_kwargs['linestyles'] = linestyles
    if linewidths is not None:
        contour_kwargs['linewidths'] = linewidths
    if alpha is not None:
        contour_kwargs['alpha'] = alpha
    
    if ax:
        #print(f"using provided ax for contour boxes")
        cs = ax.contour(grid_x, grid_y, ratios, levels=levels, **contour_kwargs) 
    else:
        #print(f"creating new fig for contour boxes")
        fig, ax = plt.subplots()
        cs = ax.contour(grid_x, grid_y, ratios, levels=levels, **contour_kwargs)

    boxes = []
    for lvl_segs in cs.allsegs:
        xs = np.concatenate([seg[:,0] for seg in lvl_segs])
        ys = np.concatenate([seg[:,1] for seg in lvl_segs])
        boxes.append((xs.min(), xs.max(), ys.min(), ys.max()))
    if not ax: 
        plt.close(fig)
    return boxes, cs

def posterior_heatmap_2d(grid_x: np.array, grid_y: np.array, ratios: np.array, true_values: list, ax_buffer: plt.Axes, parameter_names: list, title: str=None, show_colormap=True, **plot_kwargs):
    """Draw the baseline 2D posterior: pcolormesh + injection cross-hairs + labels.

    Split out from :func:`posterior_contours_2d` so callers that need the
    heatmap to appear even when contour levels are ill-defined can draw it
    independently of the contour overlay.
    """
    if show_colormap:
        c = ax_buffer.pcolormesh(grid_x, grid_y, ratios, shading='auto', cmap="inferno", **plot_kwargs)
        fig = ax_buffer.get_figure()
        fig.colorbar(c, ax=ax_buffer)

    ax_buffer.axvline(x=true_values[0], color='r', linestyle='--', label='True Value')
    ax_buffer.axhline(y=true_values[1], color='r', linestyle='--')

    if title is not None:
        ax_buffer.set_title(title)
    ax_buffer.set_xlabel(parameter_names[0])
    ax_buffer.set_ylabel(parameter_names[1])
    ax_buffer.grid()


def posterior_contours_2d(grid_x: np.array, grid_y: np.array, ratios: np.array, true_values: list, ax_buffer: plt.Axes, parameter_names: list, levels: np.array, levels_labels: list[str], title: str=None, do_plot=False, show_colormap=True, contour_colors=None, contour_linestyles=None, contour_linewidths=None, contour_alpha=None, **plot_kwargs):
    """
    Find the bounding box of the contour levels specified in levels. 

    :param grid_x: the grid of x coordinates over which the ratios are evaluated
    :type grid_x: np.array
    :param grid_y: the grid of y coordinates over which the ratios are evaluated
    :type grid_y: np.array
    :param ratios: the values of the function to be contoured
    :type ratios: np.array
    :param true_values: the true values of the parameters being estimated
    :type true_values: list
    :param ax_buffer: the axes on which to plot the contours
    :type ax_buffer: plt.Axes
    :param parameter_names: the names of the parameters being estimated
    :type parameter_names: list
    :param levels: the contour levels to be plotted
    :type levels: np.array
    :param levels_labels: the labels for the contour levels
    :type levels_labels: list[str]
    :param title: the title of the plot, defaults to None
    :type title: str, optional
    :param do_plot: if True, make also a plot of the contour on the axis defined by ax_buffer, defaults to False
    :type do_plot: bool, optional
    :param show_colormap: if True (default), show pcolormesh and colorbar. Set to False for contour-only plots
    :type show_colormap: bool, optional
    :param contour_colors: color(s) for contour lines
    :param contour_linestyles: linestyle(s) for contour lines
    :param contour_linewidths: linewidth(s) for contour lines
    :param contour_alpha: alpha value(s) for contour lines
    :return: the bounding box of the contour levels
    :rtype: tuple
    """
    if do_plot:
        posterior_heatmap_2d(grid_x, grid_y, ratios, true_values, ax_buffer,
                             parameter_names, title=title,
                             show_colormap=show_colormap, **plot_kwargs)

        boxes, cs = contour_boxes(grid_x, grid_y, ratios, levels, ax=ax_buffer,
                                  colors=contour_colors, linestyles=contour_linestyles,
                                  linewidths=contour_linewidths, alpha=contour_alpha)
        fmt = {lev: f"{p:.3f}" for lev, p in zip(levels, levels_labels)}
        ax_buffer.clabel(cs, fmt=fmt, fontsize=8)
    else:
        boxes , cs = contour_boxes(grid_x, grid_y, ratios, levels, ax=None)
        plt.close()

    return boxes

def posterior_contours_2d_imshow(grid_x: np.array, grid_y: np.array, ratios: np.array, true_values: list, ax_buffer: plt.Axes, parameter_names: list, levels: np.array, levels_labels: list[str], title: str=None, do_plot=False, **plot_kwargs):
    """
    Find the bounding box of the contour levels specified in levels. 

    :param grid_x: the grid of x coordinates over which the ratios are evaluated
    :type grid_x: np.array
    :param grid_y: the grid of y coordinates over which the ratios are evaluated
    :type grid_y: np.array
    :param ratios: the values of the function to be contoured
    :type ratios: np.array
    :param true_values: the true values of the parameters being estimated
    :type true_values: list
    :param ax_buffer: the axes on which to plot the contours
    :type ax_buffer: plt.Axes
    :param parameter_names: the names of the parameters being estimated
    :type parameter_names: list
    :param levels: the contour levels to be plotted
    :type levels: np.array
    :param levels_labels: the labels for the contour levels
    :type levels_labels: list[str]
    :param title: the title of the plot, defaults to None
    :type title: str, optional
    :param do_plot: if True, make also a plot of the contour on the axis defined by ax_buffer, defaults to False
    :type do_plot: bool, optional
    :return: the bounding box of the contour levels
    :rtype: tuple
    """
    if do_plot: 
        # make a colormesh on the ax_buffer
        
        # add contour lines
        boxes, cs = contour_boxes(grid_x, grid_y, ratios, levels, ax=ax_buffer)
        fmt = {lev: f"{p:.3f}" for lev, p in zip(levels, levels_labels)}
        ax_buffer.clabel(cs, fmt=fmt, fontsize=8)
        ax_buffer.axvline(x=true_values[0], color='r', linestyle='--', label='True Value')
        ax_buffer.axhline(y=true_values[1], color='r', linestyle='--')
        fig = ax_buffer.get_figure()
        if title is not None:
            ax_buffer.set_title(title)
        ax_buffer.set_xlabel(parameter_names[0])
        ax_buffer.set_ylabel(parameter_names[1])
        ax_buffer.grid()
    else: 
        boxes , cs = contour_boxes(grid_x, grid_y, ratios, levels, ax=None)
        plt.close()
    
    return boxes





def materialize_gpu_noise(batch):
    """Complete a deferred (``gpu_noise``) batch: tile + draw coloured noise.

    When the data module runs with ``gpu_noise=True`` the collate fn ships only
    the ``B`` distinct waveforms + the ``(C, F)`` ``noise_scale`` plus the marker
    keys ``_gpu_noise`` / ``_n_noise`` / ``_noise_factor`` (see
    :func:`mbhb_collate_fn`).  This expands the batch to the ``n*B`` effective
    size (matching the collate's ``[batch | batch | ...]`` tiling, with a per-slot
    independent realisation) and draws the noise on the tensors' **current
    device** -- so it works both in Lightning's ``on_after_batch_transfer`` (GPU)
    and in manual CPU loops (e.g. encoder normalisation fits).

    Idempotent: returns ``batch`` unchanged for stored-noise or already
    materialised batches.  No TD noise is generated (FD-only pipelines); a
    present ``wave_td`` is tiled for consistency.
    """
    if not (isinstance(batch, dict) and batch.get("_gpu_noise", False)
            and "noise_fd" not in batch):
        return batch
    n = int(batch["_n_noise"])
    nf = float(batch["_noise_factor"])
    ns = batch["noise_scale"]                   # (C, F)
    wave = batch["wave_fd"]                      # (B, C, F) distinct
    params = batch["source_parameters"]          # (B, P) distinct
    if n > 1:
        wave = wave.repeat(n, 1, 1)
        params = params.repeat(n, 1)
        if "wave_td" in batch:
            batch["wave_td"] = batch["wave_td"].repeat(n, 1, 1)
    Bn, C, Fr = wave.shape
    re = torch.randn(Bn, C, Fr, device=wave.device, dtype=ns.dtype)
    im = torch.randn(Bn, C, Fr, device=wave.device, dtype=ns.dtype)
    batch["noise_fd"] = nf * torch.complex(re, im) * ns.unsqueeze(0)
    batch["wave_fd"] = wave
    batch["source_parameters"] = params
    return batch


class GPUNoiseMixin:
    """Mixin for LightningModules trained on MBHB batches.

    Materialises the coloured noise + tiling on-device after the host->device
    transfer (see :func:`materialize_gpu_noise`).  Mix in *before* ``LightningModule``
    so this hook overrides the framework default::

        class MyNet(GPUNoiseMixin, LightningModule): ...

    No-op when the data module runs with ``gpu_noise=False`` or for stored-noise
    batches, so it is always safe to add.
    """

    def on_after_batch_transfer(self, batch, dataloader_idx):
        return materialize_gpu_noise(batch)


def mbhb_collate_fn(batch, noise_scale, noise_factor, noise_shuffling=True, td_params=None,
                    n_noise_realisations=1, gpu_noise=False):
    """Collate a batch, generating FD (and optionally TD) noise on the fly.

    :param batch: list of sample dicts from MBHBDataset.__getitem__
    :param noise_scale: real tensor of shape (n_channels, n_freqs) equal to
        ``filtered_asd / sqrt(4 * df)``; multiplied bin-wise by complex draws
        ``re + j im`` with ``re, im ~ N(0, 1)`` i.i.d. (Re and Im of the
        whitened noise then each have unit variance — the convention used
        throughout the codebase).
        Unused when the batch already carries a stored ``noise_fd`` per sample.
    :param noise_factor: scalar multiplier applied to the noise amplitude
        (works for both freshly-generated and stored noise)
    :param noise_shuffling: kept for API compatibility, has no effect — noise is
        always freshly generated per call when not stored on disk
    :param td_params: tuple ``(dt, n_time)`` needed to derive TD noise via IFFT,
        or ``None`` when only FD data is required
    :param gpu_noise: when True (and the batch carries no stored noise), defer
        both the noise generation and the ``n_noise_realisations`` batch tiling
        to the GPU.  Only the ``B`` distinct waveforms, the params and the
        ``(C, F)`` ``noise_scale`` are returned, together with the marker keys
        ``_gpu_noise`` / ``_n_noise`` / ``_noise_factor``; the actual coloured
        noise + tiling are produced in
        ``FMPEInferenceNetwork.on_after_batch_transfer``.  This slashes the
        per-batch CPU ``randn`` cost and the host->device transfer volume.
        Ignored when the batch carries a stored ``noise_fd``.
    :param n_noise_realisations: number of distinct noise realisations to draw
        and show the model for *each* waveform. When > 1 the batch is expanded
        from ``B`` to ``B * n_noise_realisations`` examples by tiling
        ``n_noise_realisations`` copies of the original batch and pairing each
        of the ``B * n_noise_realisations`` slots with its own *independently
        drawn* noise realisation: every waveform is therefore shown with
        ``n_noise_realisations`` independent realisations unique to it (no
        realisation is shared across waveforms). Copies are tiled — i.e. the
        expanded batch is ``[batch | batch | ...]`` so adjacent examples remain
        distinct waveforms, which keeps the roll-by-1 contrastive scrambling in
        the model valid. Defaults to 1 (no expansion). Ignored when the batch
        carries stored noise, so observation/test data with persisted noise is
        never expanded.
    """
    B = len(batch)
    wave_fd = torch.stack([b["wave_fd"] for b in batch])
    params  = torch.stack([b["params"] for b in batch])
    has_td = "wave_td" in batch[0]
    has_stored_noise = "noise_fd" in batch[0]

    if has_stored_noise:
        # Use the noise realisation persisted on disk (e.g. observation files).
        # This is what makes the obs deterministic across calls and consistent
        # with post-hoc visualisation scripts. No expansion is applied here.
        noise_fd = noise_factor * torch.stack([b["noise_fd"] for b in batch])
        wave_td = torch.stack([b["wave_td"] for b in batch]) if has_td else None
    else:
        n = max(1, int(n_noise_realisations))
        if gpu_noise:
            # Defer noise draw + tiling to the GPU. Ship only the B distinct
            # waveforms + the (C, F) noise_scale (~96 KB) instead of the tiled
            # waveforms + noise (~hundreds of MB). The expansion to (n*B) and
            # the coloured-noise draw happen in
            # FMPEInferenceNetwork.on_after_batch_transfer.
            out = {
                "source_parameters": params,            # (B, P) distinct
                "wave_fd": wave_fd,                      # (B, C, F) distinct
                "noise_scale": noise_scale,              # (C, F)
                "_gpu_noise": True,
                "_n_noise": n,
                "_noise_factor": float(noise_factor),
            }
            if has_td:
                # FD-only pipelines (FMPE) don't use this, but keep it untiled
                # for completeness; TD noise is not generated in this path.
                out["wave_td"] = torch.stack([b["wave_td"] for b in batch])
            return out
        # Draw an *independent* coloured complex Gaussian noise realisation for
        # every (copy, waveform) slot in the expanded batch:
        # z = (re + j im) * noise_scale, with re, im ~ N(0, 1) i.i.d.
        C, F = noise_scale.shape
        re = torch.randn(n * B, C, F, dtype=noise_scale.dtype)
        im = torch.randn(n * B, C, F, dtype=noise_scale.dtype)
        noise_fd = noise_factor * torch.complex(re, im) * noise_scale.unsqueeze(0)  # (n*B, C, F)

        wave_td = torch.stack([b["wave_td"] for b in batch]) if has_td else None
        # Tile n copies of the batch ([batch | batch | ...]) so that example
        # k = j*B + i corresponds to (waveform_i, noise_k). Each waveform is
        # therefore paired with n *independent* realisations (its own, not a
        # set shared across the batch). Tiling keeps adjacent examples
        # distinct waveforms, so the roll-by-1 contrastive scrambling in the
        # model stays valid.
        wave_fd = wave_fd.repeat(n, 1, 1)                       # (n*B, C, F)
        params  = params.repeat(n, 1)                          # (n*B, P)
        if has_td:
            wave_td = wave_td.repeat(n, 1, 1)

    out = {
        "source_parameters": params,
        "wave_fd": wave_fd,
        "noise_fd": noise_fd,
    }

    if has_td:
        out["wave_td"] = wave_td
        if td_params is not None:
            dt, n_time = td_params
            # Reconstruct two-sided FD spectrum (DC + positive + conjugate-flipped negative).
            # Derive shapes from noise_fd so this is correct after any expansion.
            Bn, C = noise_fd.shape[0], noise_fd.shape[1]
            dc = torch.zeros(Bn, C, 1, dtype=noise_fd.dtype)
            pos2 = torch.cat([dc, noise_fd], dim=2)
            neg = torch.flip(pos2[..., 1:].conj(), dims=[-1])
            two_sided = torch.cat([pos2, neg], dim=2)
            noise_td = torch.fft.ifft(two_sided, dim=-1).real / dt
            out["noise_td"] = noise_td[..., :n_time]

    return out

def whiten_fd(data, asd):
    """Whiten FD data by dividing by the ASD (amplitude spectral density).

    Zero-ASD bins are guarded: ``0 → inf`` so that whitened values → 0
    rather than inf/nan.

    Parameters
    ----------
    data : torch.Tensor
        Complex tensor of shape ``(B, C, F)`` or ``(C, F)``.
    asd : torch.Tensor
        Real tensor of shape ``(C, F)``.

    Returns
    -------
    torch.Tensor
        Whitened tensor, same shape and dtype as *data*.
    """
    safe_asd = asd.clone().to(data.device)
    safe_asd[safe_asd == 0] = float('inf')
    if data.ndim == 3:
        return data / safe_asd.unsqueeze(0)
    return data / safe_asd


def fd_inner(a, b, df):
    r"""Noise-weighted FD inner product (Gram matrix) on pre-whitened data.

    Computes the GW inner product

    .. math::

        G_{ij} = \langle a_i \mid b_j \rangle
        = 4 \, \operatorname{Re} \sum_c \sum_k
          \bar{a}_{i,c,k} \, b_{j,c,k} \, \Delta f_k

    where *a* and *b* have already been divided by the ASD.  The sum
    runs over channels *c* and frequency bins *k*.

    Always returns an ``(M, N)`` Gram matrix — use :func:`fd_norm` for
    efficient diagonal-only computation.

    Parameters
    ----------
    a : torch.Tensor
        Pre-whitened complex tensor of shape ``(M, C, F)``.
    b : torch.Tensor
        Pre-whitened complex tensor of shape ``(N, C, F)``.
    df : float or torch.Tensor
        Frequency bin width(s).  Scalar for uniform spacing, or a 1-D
        tensor of shape ``(F,)`` for non-uniform spacing.

    Returns
    -------
    torch.Tensor
        Real tensor of shape ``(M, N)`` with entry ``G_{ij} = ⟨a_i | b_j⟩``.
    """
    M, N = a.shape[0], b.shape[0]
    C, F = a.shape[-2], a.shape[-1]

    # Build per-bin weight
    if isinstance(df, (int, float)):
        w = 4.0 * df
    else:
        w = 4.0 * df.to(a.device)  # (F,)

    # Flatten (C, F) → D = C*F, tile weight across channels
    a_flat = a.reshape(M, C * F)          # (M, D)
    b_flat = b.reshape(N, C * F)          # (N, D)
    if isinstance(w, float):
        aw = a_flat.conj() * w
    else:
        w_tiled = w.repeat(C)             # (D,)
        aw = a_flat.conj() * w_tiled
    return (aw @ b_flat.mT).real          # (M, N)


def fd_norm(a, df):
    r"""Norm induced by :func:`fd_inner`: :math:`\|a\| = \sqrt{\langle a \mid a \rangle}`.

    Computes the diagonal elements only — never allocates an ``(M, M)``
    matrix.

    Parameters
    ----------
    a : torch.Tensor
        Pre-whitened complex tensor of shape ``(B, C, F)`` or ``(C, F)``.
    df : float or torch.Tensor
        Frequency bin width(s).  Same convention as :func:`fd_inner`.

    Returns
    -------
    torch.Tensor
        Shape ``(B,)`` or scalar.
    """
    squeeze = a.ndim == 2
    if squeeze:
        a = a.unsqueeze(0)

    if isinstance(df, (int, float)):
        w = 4.0 * df
    else:
        w = 4.0 * df.to(a.device)  # (F,)

    power = (a.conj() * a).real  # (B, C, F)
    if isinstance(w, float):
        result = (w * power).sum(dim=(-2, -1)).sqrt()
    else:
        result = (power * w).sum(dim=(-2, -1)).sqrt()

    return result.squeeze(0) if squeeze else result


# ---------------------------------------------------------------------------
# Fisher Information Matrix utilities
# ---------------------------------------------------------------------------

# Per-parameter finite-difference step (TMNRE coords) used by the Fisher matrix.
# These are the steps RECOMMENDED by the convergence diagnostic
# (test/test_fisher_derivative.py): for each parameter it sweeps dx, finds the
# roundoff elbow, and recommends the point one ladder step into the clean dx²
# truncation region (≈ 2× the elbow). The Fisher always uses these values — see
# compute_fisher_matrix_waveform_deriv. Re-run the diagnostic and update here if
# the simulator grid / waveform settings change. (Values for a 1-week MBHB on the
# default datagen grid, seed 42 expansion point.)
FISHER_ABSOLUTE_STEP_DEFAULTS = {
    "logMchirp": 8.0e-7,   # log10(Mchirp[Msun])
    "q":         8.0e-5,
    "chi1":      2.0e-5,
    "chi2":      8.0e-5,
    "dist":      5.0e-5,   # Gpc
    "phi":       6.25e-6,  # rad
    "inc":       1.25e-5,  # cos(inc)
    "lambda":    2.5e-5,   # rad
    "beta":      1.25e-5,  # sin(beta)
    "psi":       6.25e-6,  # rad (mod π)
    "Deltat":    5.0e-8,   # days
}


def generate_waveforms_at(simulator, tmnre_batch):
    """Generate FD waveforms for a batch of TMNRE parameter vectors.

    Single source of truth for *how* a waveform is produced from TMNRE
    coordinates — used both by the Fisher matrix and by the finite-difference
    convergence test. The nonlinear TMNRE→bbhx transform
    (``samples_to_bbhx_input``: logMc→m1m2, cos→inc, sin→beta, Deltat→t_ref) is
    applied here, so derivatives taken via this function are end-to-end in TMNRE
    space.

    Parameters
    ----------
    simulator : MBHBSimulatorFD
        Provides ``sampler``, ``wfd``, ``waveform_kwargs``, ``freqs``,
        ``channels_idx``, ``t_obs_end_SI`` and ``xp``.
    tmnre_batch : np.ndarray, shape (11, n_cols)
        Columns are TMNRE parameter vectors in ``_ORDERED_PRIOR_KEYS`` order.

    Returns
    -------
    np.ndarray, shape (n_cols, n_ch, n_freq), complex
        Noise-free waveforms on the simulator's own frequency grid.
    """
    bbhx_batch = simulator.sampler.samples_to_bbhx_input(
        tmnre_batch, t_obs_end=simulator.t_obs_end_SI
    )
    # Pin "freqs" explicitly rather than reuse simulator.generate so the result
    # is immune to any in-place slicing of waveform_kwargs (e.g. high_freq_only
    # in the emcee path) that would otherwise desync the grid from asd/df.
    wf_kw = dict(simulator.waveform_kwargs)
    wf_kw["freqs"] = simulator.xp.asarray(np.asarray(simulator.freqs))
    waves = simulator.wfd(*bbhx_batch, **wf_kw)
    if hasattr(waves, "get"):
        waves = waves.get()
    return np.asarray(waves)[:, simulator.channels_idx, :]


def waveform_central_difference(simulator, true_tmnre_params, param_name, dx):
    """Central-difference derivative ``∂h/∂θ`` of the waveform w.r.t. one param.

    Computes ``f'_approx(x, dx) = (h(θ+dx) − h(θ−dx)) / (2·dx)`` where only the
    TMNRE coordinate ``param_name`` is perturbed. This is exactly the quantity
    that builds the Fisher Jacobian, factored out so its convergence in ``dx``
    can be tested independently (see ``test/test_fisher_derivative.py``).

    Parameters
    ----------
    simulator : MBHBSimulatorFD
        Simulator built consistently with the analysis grid.
    true_tmnre_params : array-like, shape (11,)
        Expansion point in ``_ORDERED_PRIOR_KEYS`` order.
    param_name : str
        Name of the parameter to differentiate w.r.t.
    dx : float
        Central-difference step in TMNRE coordinates.

    Returns
    -------
    np.ndarray, shape (n_ch, n_freq), complex
        The central-difference derivative ``∂h/∂θ_{param_name}``.
    """
    theta0 = np.asarray(true_tmnre_params, dtype=np.float64).reshape(-1)
    p = _ORDERED_PRIOR_KEYS.index(param_name)
    batch = np.repeat(theta0[:, None], 2, axis=1)  # (11, 2): [plus, minus]
    batch[p, 0] = theta0[p] + dx
    batch[p, 1] = theta0[p] - dx
    waves = generate_waveforms_at(simulator, batch)  # (2, n_ch, n_freq)
    return (waves[0] - waves[1]) / (2.0 * dx)


def waveform_richardson_derivative(simulator, true_tmnre_params, param_name, dx):
    """Richardson-extrapolated waveform derivative, accurate to ``O(dx⁴)``.

    Combines two central differences to cancel the leading ``dx²`` truncation
    term (see the derivation in ``test/test_fisher_derivative.py``):

        f'_gt(dx) = (4·f'_approx(dx) − f'_approx(2·dx)) / 3 = ∂h/∂θ + O(dx⁴) .

    This is the "ground truth" derivative the convergence diagnostic recommends
    evaluating at the per-parameter elbow step. Returns array (n_ch, n_freq).
    """
    f1 = waveform_central_difference(simulator, true_tmnre_params, param_name, dx)
    f2 = waveform_central_difference(simulator, true_tmnre_params, param_name, 2.0 * dx)
    return (4.0 * f1 - f2) / 3.0


def compute_fisher_matrix_waveform_deriv(
    simulator,
    true_tmnre_params,
    varying_params: list,
    freq_mask=None,
    use_richardson: bool = True,
):
    """Waveform-derivative Fisher matrix ``F_ij = <∂_i h | ∂_j h>``.

    The *standard* GW Fisher matrix, built from first derivatives of the
    **waveform** (not second derivatives of the log-likelihood). The inner
    product is the noise-weighted overlap

        ⟨a|b⟩ = 4·Re Σ_ch Σ_{f≥FMIN_FLOOR} a* b · df / S_n ,    S_n = asd² ,

    identical to :func:`pembhb.simulator.compute_snr_fd`. Because
    ``F = Jᵀ N⁻¹ J`` it is **positive semi-definite by construction** — no
    negative eigenvalues, no dependence on the noise realisation, and far
    better conditioned than the log-likelihood Hessian (so no small-step NaN
    blow-up). Derivatives are taken in TMNRE parameter space, so the nonlinear
    transform applied by ``simulator.sampler.samples_to_bbhx_input``
    (logMc→m1m2, cos→inc, sin→beta, Deltat→t_ref) is differentiated
    end-to-end — no explicit Jacobian needed.

    Parameters
    ----------
    simulator : MBHBSimulatorFD
        Simulator built *consistently with the observed data* (same frequency
        grid, modes, ``waveform_kwargs``). Provides ``wfd``/``generate``,
        ``sampler``, ``asd``, ``df``, ``freqs``, ``channels_idx`` and
        ``t_obs_end_SI``.
    true_tmnre_params : array-like, shape (11,)
        Expansion point in ``_ORDERED_PRIOR_KEYS`` order.
    varying_params : list of str
        Parameters to include in the Fisher matrix. The finite-difference step
        for each is taken from ``FISHER_ABSOLUTE_STEP_DEFAULTS`` (the convergence-
        diagnostic recommendation); there is no prior-width-relative option.
    freq_mask : array-like of bool, optional
        Boolean mask over ``simulator.freqs`` selecting which bins enter the
        inner product. Use this to mirror an analysis that restricts the
        likelihood to a sub-band (e.g. ``high_freq_only``). Combined (AND) with
        the always-applied ``freqs ≥ FMIN_FLOOR`` / ``asd > 0`` mask.
    use_richardson : bool
        If True (default), build the Jacobian from the Richardson-extrapolated
        derivative (``O(dx⁴)``, see :func:`waveform_richardson_derivative`)
        rather than the plain ``O(dx²)`` central difference. Costs two extra
        waveform evaluations per parameter; cancels the leading truncation term.

    Returns
    -------
    fisher : np.ndarray, shape (n, n)
        Fisher matrix (ordered as *varying_params*).
    param_uncertainties : np.ndarray, shape (n,)
        ``sqrt(diag(F⁻¹))`` (NaN if singular; the eigenvalues are printed).
    """
    theta0 = np.asarray(true_tmnre_params, dtype=np.float64).reshape(-1)
    assert theta0.shape[0] == len(_ORDERED_PRIOR_KEYS), (
        f"[Fisher] expected {len(_ORDERED_PRIOR_KEYS)} params, got {theta0.shape[0]}"
    )
    idx = {name: _ORDERED_PRIOR_KEYS.index(name) for name in varying_params}
    n = len(varying_params)

    # Only inc/beta enter the waveform through arccos/arcsin, so their TMNRE
    # coordinate (cos inc, sin beta) has a hard [-1, 1] domain; a step past the
    # edge yields NaN. Everything else tolerates a small excursion fine.
    DOMAIN = {"inc": (-1.0, 1.0), "beta": (-1.0, 1.0)}
    SAFETY = 1e-6

    # Per-parameter finite-difference step (TMNRE coords). We ALWAYS use the
    # diagnostic-recommended absolute steps in FISHER_ABSOLUTE_STEP_DEFAULTS —
    # never `step_frac * prior_width`. Those recommendations come from the
    # convergence diagnostic (test/test_fisher_derivative.py), which places each
    # step in the clean dx² truncation region (one ladder point above the
    # roundoff elbow); a prior-width-relative step has no such guarantee and
    # previously produced grotesquely large steps (e.g. Deltat → hundreds of GW
    # cycles in the band) that silently corrupted the Fisher diagonals.
    eps = np.zeros(n)
    for k, name in enumerate(varying_params):
        v = theta0[idx[name]]
        if name not in FISHER_ABSOLUTE_STEP_DEFAULTS:
            raise KeyError(
                f"[Fisher] no FISHER_ABSOLUTE_STEP_DEFAULTS entry for '{name}'; "
                f"run the convergence diagnostic and add a recommended step."
            )
        e = FISHER_ABSOLUTE_STEP_DEFAULTS[name]
        dom = DOMAIN.get(name)
        if dom is not None:  # keep both v±e strictly inside the domain
            lo_d, hi_d = dom
            e = min(e, (v - lo_d) * (1.0 - SAFETY), (hi_d - v) * (1.0 - SAFETY))
        if e <= 0:
            raise ValueError(
                f"[Fisher] non-positive step for '{name}' (value {v} on a domain edge)."
            )
        eps[k] = e
    # Show the actual finite-difference steps used — this is the variable most
    # responsible for any Fisher mis-estimation, so make it visible.
    print("[Fisher] Finite-difference steps (TMNRE coords):")
    for name, e in zip(varying_params, eps):
        print(f"  h({name}) = {e:.3e}")

    # Jacobian: J[k] = ∂h/∂θ_k, each (n_ch, n_freq). By default uses the
    # Richardson-extrapolated derivative (O(dx⁴), cancels the dx² truncation
    # term) evaluated at the per-parameter step `eps`, which the convergence
    # diagnostic (test/test_fisher_derivative.py) recommends placing at the
    # error elbow. Set use_richardson=False for the plain O(dx²) central
    # difference. Both share the exact derivative code path the diagnostic tests.
    deriv_fn = waveform_richardson_derivative if use_richardson else waveform_central_difference
    J = np.stack(
        [deriv_fn(simulator, theta0, name, eps[k])
         for k, name in enumerate(varying_params)],
        axis=0,
    )

    # Noise weighting 4·df/asd² on high-passed, finite bins (matches compute_snr_fd).
    freqs = np.asarray(simulator.freqs)
    asd = np.asarray(simulator.asd)
    df = simulator.df
    df_arr = df if np.ndim(df) > 0 else np.full(freqs.shape, df)
    mask = (freqs >= FMIN_FLOOR) & np.all(asd > 0, axis=0)
    if freq_mask is not None:
        mask &= np.asarray(freq_mask, dtype=bool)
    weight = np.zeros_like(asd)
    weight[:, mask] = 4.0 * df_arr[mask] / asd[:, mask] ** 2

    fisher = np.zeros((n, n))
    for a in range(n):
        for b in range(a, n):
            val = float(np.sum((np.conj(J[a]) * J[b]) * weight).real)
            fisher[a, b] = val
            fisher[b, a] = val

    print("[Fisher] Waveform-derivative Fisher matrix:")
    print(fisher)
    try:
        fisher_inv = np.linalg.inv(fisher)
        diag = np.diag(fisher_inv)
        param_uncertainties = np.sqrt(diag)
        if np.any(diag < 0):
            bad = [varying_params[i] for i in np.where(diag < 0)[0]]
            print(f"[Fisher] WARNING: negative F⁻¹ diagonal for {bad} — "
                  f"Fisher is ill-conditioned (eigenvalues {np.linalg.eigvalsh(fisher)}).")
        print("[Fisher] Parameter uncertainties (Cramér-Rao lower bound):")
        for name, sigma in zip(varying_params, param_uncertainties):
            print(f"  σ({name}) = {sigma:.6e}")
    except np.linalg.LinAlgError:
        evals = np.linalg.eigvalsh(fisher)
        print(f"[Fisher] WARNING: Fisher matrix is singular – eigenvalues = {evals}")
        param_uncertainties = np.full(n, np.nan)

    return fisher, param_uncertainties


def compute_fisher_prior_bounds(
    datagen_config: dict,
    observation_file: str,
    event_idx: int,
    varying_params: list,
    fixed_params: list,
    n_sigma: float = 5.0,
    param_n_sigma: dict = None,
) -> dict:
    """Build prior bounds for data generation using the Fisher Information Matrix.

    For each parameter in *varying_params* the bounds are set to
    ``[true_val ± n_sigma * σ_FIM]`` where *σ_FIM* is the Cramér-Rao
    1-sigma uncertainty.  Parameters in *fixed_params* are pinned to their
    **true value read from the observation file** (zero-width prior).  All
    other parameters keep the ``datagen_config["prior"]`` bounds unchanged.

    Parameters
    ----------
    datagen_config : dict
        Datagen configuration (as returned by :func:`read_config`).
    observation_file : str
        Path to the HDF5 observation file.
    event_idx : int
        Index of the event to use as the FIM expansion point.
    varying_params : list of str
        Parameter names for which FIM-based bounds are computed.
    fixed_params : list of str
        Names of parameters to hold fixed.  Their values are read from the
        observation file (``source_parameters[event_idx]``), **not** from the
        YAML config.
    n_sigma : float
        Half-width of the generated prior in units of the FIM σ.

    Returns
    -------
    dict
        Complete prior-bounds dict compatible with
        ``sampler_init_kwargs={"prior_bounds": ...}``.
    """
    import h5py  # h5py is already a project dependency
    # Lazy import to avoid circular dependency with pembhb.simulator
    from pembhb.simulator import MBHBSimulatorFD

    # Load observation first so we can use true values for fixed params.
    print(f"[Fisher] Loading event {event_idx} from {observation_file} ...")
    import h5py as _h5
    with _h5.File(observation_file, "r") as f:
        freqs_obs       = f["frequencies"][:]
        true_params_arr = f["source_parameters"][event_idx]  # shape (11,)
        wave_fd         = f["wave_fd"][event_idx]            # shape (n_ch, n_freqs), noise-free

    # Build fixed_values dict from the observation file.
    fixed_values = {
        key: float(true_params_arr[_ORDERED_PRIOR_KEYS.index(key)])
        for key in fixed_params
    }
    print("[Fisher] Fixed parameter values read from observation file:")
    for k, v in fixed_values.items():
        print(f"  {k} = {v:.6e}")

    # Build simulator-friendly dummy prior (correct shape, no crash).
    # We only need the simulator for its waveform generator and frequency array.
    dummy_prior = copy.deepcopy(datagen_config["prior"])
    for key, val in fixed_values.items():
        dummy_prior[key] = [val, val]

    print("[Fisher] Initializing simulator for FIM evaluation ...")
    fisher_config = copy.deepcopy(datagen_config)
    fisher_config["backend"] = "cpu"  # CPU keeps wfd/freqs deterministic for the FIM
    wp = fisher_config["waveform_params"]
    simulator = MBHBSimulatorFD(
        fisher_config,
        sampler_init_kwargs={"prior_bounds": dummy_prior},
        seed=42,
        n_freq_bins=wp.get("n_freq_bins", 4096),
        freq_spacing=wp.get("freq_spacing", "linear"),
    )
    frequencies = simulator.freqs

    assert np.allclose(freqs_obs, frequencies), (
        "[Fisher] Frequency mismatch between observation file and simulator!"
    )

    # --- Hard consistency check: the waveform we differentiate must be the one
    #     that generated the observation. wave_fd in the HDF5 is noise-free
    #     (noise lives separately in noise_fd), so regenerating h(θ_true) with
    #     the simulator's waveform_kwargs must reproduce it. A grid / modes /
    #     length / t_obs mismatch produces an order-1 discrepancy here.
    true_full = np.asarray(true_params_arr, dtype=np.float64).reshape(-1, 1)
    bbhx_true = simulator.sampler.samples_to_bbhx_input(
        true_full, t_obs_end=simulator.t_obs_end_SI
    )
    h_true = simulator.generate(bbhx_true)[0]  # (n_ch, n_freq)
    rel_diff = np.linalg.norm(h_true - wave_fd) / (np.linalg.norm(wave_fd) + 1e-30)
    if rel_diff > 1e-2:
        raise AssertionError(
            f"[Fisher] Regenerated waveform does not match stored wave_fd "
            f"(relative diff {rel_diff:.2e} > 1e-2). The simulator's "
            f"waveform_kwargs / grid / modes are inconsistent with how the "
            f"observation was generated."
        )
    print(f"[Fisher] Consistency check passed: ‖h(θ_true) − wave_fd‖/‖wave_fd‖ "
          f"= {rel_diff:.2e}.")

    # FIM via waveform derivatives (noise-independent, PSD-by-construction).
    _, param_uncertainties = compute_fisher_matrix_waveform_deriv(
        simulator,
        true_params_arr,
        varying_params,
    )

    # Assemble final prior bounds.
    prior_bounds = copy.deepcopy(datagen_config["prior"])

    # Pin fixed parameters to the true values from the observation file.
    for key, val in fixed_values.items():
        prior_bounds[key] = [val, val]

    # FIM-based bounds for varying parameters.
    param_n_sigma = param_n_sigma or {}
    for key, sigma in zip(varying_params, param_uncertainties):
        true_val = float(true_params_arr[_ORDERED_PRIOR_KEYS.index(key)])
        n_sig_eff = param_n_sigma.get(key, n_sigma)
        if np.isfinite(sigma):
            lo = float(true_val - n_sig_eff * sigma)
            hi = float(true_val + n_sig_eff * sigma)
        else:
            print(f"[Fisher] WARNING: σ({key}) is NaN – keeping datagen_config bounds.")
            lo, hi = datagen_config["prior"][key]
        # Clip to the physical prior range — those are hard constraints
        # (e.g. cos(inc) ∈ [-1, 1], chi ∈ [-1, 1]). Fisher tails beyond them
        # would produce NaN waveforms via downstream arcsin/arccos.
        phys_lo, phys_hi = datagen_config["prior"][key]
        lo_c = max(lo, float(phys_lo))
        hi_c = min(hi, float(phys_hi))
        if (lo_c, hi_c) != (lo, hi):
            print(f"[Fisher] {key}: clipped to physical prior [{phys_lo}, {phys_hi}] "
                  f"(was [{lo:.6e}, {hi:.6e}])")
        prior_bounds[key] = [lo_c, hi_c]
        print(f"[Fisher] {key}: true={true_val:.6e}, σ={sigma:.3e}, n_sigma={n_sig_eff} → [{lo_c:.6e}, {hi_c:.6e}]")

    print(f"[Fisher] Final prior bounds: {prior_bounds}")
    return prior_bounds


def compute_fisher_sigmas_for_testset(
    datagen_config: dict,
    true_params: np.ndarray,
    varying_params: list,
    wave_fd_check: np.ndarray = None,
    chunk_size: int = 25,
    backend: str = "cpu",
):
    """Cramér-Rao 1-σ per parameter for a *set* of expansion points (batched).

    Builds one CPU simulator (consistent with ``datagen_config``) and evaluates
    the waveform-derivative Fisher matrix at every row of *true_params*. The
    numerics are identical to looping :func:`compute_fisher_matrix_waveform_deriv`
    — same per-parameter steps (``FISHER_ABSOLUTE_STEP_DEFAULTS`` with the same
    inc/beta domain clamp), same Richardson ``O(dx⁴)`` derivative, same
    nominal-PSD inner product — but all the waveform evaluations are assembled
    into one ``(11, n_test·n·4)`` batch and handed to
    :func:`generate_waveforms_at` (which bbhx vectorises over binaries) in
    memory-bounded chunks, instead of issuing ``2n`` tiny calls per point. The
    Fisher matrices are then formed with a single ``einsum`` and inverted in a
    batch. For each point it returns ``sqrt(diag(F⁻¹))`` — the marginal 1-σ for
    every parameter (marginalised over the others through the matrix inverse).

    This is the epoch-independent denominator of the τ statistic: it depends
    only on the true parameter point and the noise PSD, never on the NRE, so the
    caller computes it once per round and caches it.

    :param datagen_config: datagen configuration (as :func:`read_config` returns)
    :type datagen_config: dict
    :param true_params: expansion points, shape ``(n_test, 11)`` in
        ``_ORDERED_PRIOR_KEYS`` order
    :type true_params: np.ndarray
    :param varying_params: parameters to include in the Fisher matrix (the full
        inferred set for the round); σ is read off the diagonal of ``F⁻¹``
    :type varying_params: list of str
    :param wave_fd_check: optional noise-free ``wave_fd`` (shape ``(n_ch,
        n_freq)``) for the *first* expansion point; if given, the simulator's
        regenerated ``h(θ_true[0])`` is asserted to match it (grid/modes guard,
        as in :func:`compute_fisher_prior_bounds`).
    :type wave_fd_check: np.ndarray, optional
    :param chunk_size: number of expansion points whose waveforms are generated
        per bbhx call (each call produces ``chunk_size·n·4`` waveforms); bounds
        peak memory. Does not affect the result.
    :type chunk_size: int
    :param backend: bbhx backend for the throwaway Fisher simulator. ``"cpu"``
        (default) matches :func:`compute_fisher_prior_bounds` and needs no GPU;
        ``"cuda12x"`` is ~10× faster on the batched generation and agrees with
        the CPU result to ~1e-8 relative, so the τ callback can opt into it
        during training (the GPU is already resident).
    :type backend: str
    :return: ``(sigmas, varying_params)`` where ``sigmas`` has shape
        ``(n_test, len(varying_params))`` (NaN where the Fisher is singular)
    :rtype: tuple[np.ndarray, list]
    """
    from pembhb.simulator import MBHBSimulatorFD  # lazy: avoid circular import

    true_params = np.asarray(true_params, dtype=np.float64)
    assert true_params.ndim == 2 and true_params.shape[1] == len(_ORDERED_PRIOR_KEYS), (
        f"[Fisher-τ] expected (n_test, {len(_ORDERED_PRIOR_KEYS)}), got {true_params.shape}"
    )
    n_test = true_params.shape[0]
    n = len(varying_params)

    fisher_config = copy.deepcopy(datagen_config)
    fisher_config["backend"] = backend
    wp = fisher_config["waveform_params"]
    simulator = MBHBSimulatorFD(
        fisher_config,
        sampler_init_kwargs={"prior_bounds": copy.deepcopy(datagen_config["prior"])},
        seed=42,
        n_freq_bins=wp.get("n_freq_bins", 4096),
        freq_spacing=wp.get("freq_spacing", "linear"),
    )

    # One-off consistency guard: the waveform we differentiate must be the one
    # that generated the test data. A grid/modes/length mismatch shows up here.
    if wave_fd_check is not None:
        theta0 = true_params[0].reshape(-1, 1)
        bbhx_true = simulator.sampler.samples_to_bbhx_input(
            theta0, t_obs_end=simulator.t_obs_end_SI
        )
        h_true = simulator.generate(bbhx_true)[0]
        rel = np.linalg.norm(h_true - wave_fd_check) / (np.linalg.norm(wave_fd_check) + 1e-30)
        if rel > 1e-2:
            raise AssertionError(
                f"[Fisher-τ] regenerated waveform mismatch (rel diff {rel:.2e} > 1e-2): "
                f"simulator grid/modes inconsistent with the test data."
            )
        print(f"[Fisher-τ] consistency check passed (rel diff {rel:.2e}).")

    # Per-(point, param) steps, mirroring compute_fisher_matrix_waveform_deriv:
    # the diagnostic step, clamped for inc/beta whose cos/sin coord lives on
    # [-1, 1] (clamp is per-point, so eps is (n_test, n)).
    DOMAIN = {"inc": (-1.0, 1.0), "beta": (-1.0, 1.0)}
    SAFETY = 1e-6
    col_of = {name: _ORDERED_PRIOR_KEYS.index(name) for name in varying_params}
    eps = np.zeros((n_test, n), dtype=np.float64)
    for k, name in enumerate(varying_params):
        if name not in FISHER_ABSOLUTE_STEP_DEFAULTS:
            raise KeyError(
                f"[Fisher-τ] no FISHER_ABSOLUTE_STEP_DEFAULTS entry for '{name}'; "
                f"run the convergence diagnostic and add a recommended step."
            )
        e = np.full(n_test, FISHER_ABSOLUTE_STEP_DEFAULTS[name], dtype=np.float64)
        if name in DOMAIN:
            lo_d, hi_d = DOMAIN[name]
            v = true_params[:, col_of[name]]
            e = np.minimum.reduce([
                e, (v - lo_d) * (1.0 - SAFETY), (hi_d - v) * (1.0 - SAFETY),
            ])
        if np.any(e <= 0):
            bad = np.where(e <= 0)[0]
            raise ValueError(
                f"[Fisher-τ] non-positive step for '{name}' at points {bad.tolist()} "
                f"(value on a domain edge)."
            )
        eps[:, k] = e

    # Noise weighting 4·df/asd² on high-passed bins (as in compute_snr_fd);
    # nominal PSD, so one weight serves all points.
    freqs = np.asarray(simulator.freqs)
    asd = np.asarray(simulator.asd)
    df = simulator.df
    df_arr = df if np.ndim(df) > 0 else np.full(freqs.shape, df)
    mask = (freqs >= FMIN_FLOOR) & np.all(asd > 0, axis=0)
    weight = np.zeros_like(asd)
    weight[:, mask] = 4.0 * df_arr[mask] / asd[:, mask] ** 2     # (n_ch, n_freq)

    # Batched Richardson Jacobian + Fisher, chunked over points. Stencil
    # columns per (point, param): [+dx, -dx, +2dx, -2dx].
    sigmas = np.full((n_test, n), np.nan, dtype=np.float64)
    for start in range(0, n_test, chunk_size):
        end = min(start + chunk_size, n_test)
        c = end - start
        base = true_params[start:end]                            # (c, 11)
        batch = np.broadcast_to(base[:, None, None, :], (c, n, 4, 11)).copy()
        e_chunk = eps[start:end]                                 # (c, n)
        for k, name in enumerate(varying_params):
            col = col_of[name]
            ek = e_chunk[:, k]                                   # (c,)
            batch[:, k, 0, col] = base[:, col] + ek
            batch[:, k, 1, col] = base[:, col] - ek
            batch[:, k, 2, col] = base[:, col] + 2.0 * ek
            batch[:, k, 3, col] = base[:, col] - 2.0 * ek

        tmnre_batch = batch.reshape(-1, 11).T                    # (11, c·n·4)
        waves = generate_waveforms_at(simulator, tmnre_batch)    # (c·n·4, n_ch, n_freq)
        n_ch, n_freq = waves.shape[1], waves.shape[2]
        waves = waves.reshape(c, n, 4, n_ch, n_freq)

        e_b = e_chunk[:, :, None, None]                          # (c, n, 1, 1)
        # Central differences at dx and 2dx, then Richardson O(dx⁴):
        f1 = (waves[:, :, 0] - waves[:, :, 1]) / (2.0 * e_b)
        f2 = (waves[:, :, 2] - waves[:, :, 3]) / (4.0 * e_b)
        J = (4.0 * f1 - f2) / 3.0                                # (c, n, n_ch, n_freq)

        # F[p, a, b] = Re Σ_ch Σ_f conj(J[p,a]) · J[p,b] · weight
        F = np.real(np.einsum("pahf,pbhf,hf->pab", np.conj(J), J, weight))

        # Batched inverse; flag singular blocks individually.
        for j in range(c):
            try:
                diag = np.diag(np.linalg.inv(F[j]))
                sigmas[start + j] = np.sqrt(np.where(diag >= 0, diag, np.nan))
            except np.linalg.LinAlgError:
                pass  # leave NaN row
        print(f"[Fisher-τ] {end}/{n_test} expansion points done.")

    return sigmas, list(varying_params)


def transfer_classifier_weights(old_model, new_model):
    """Transfer classifier weights from *old_model* to *new_model* for matching marginals.

    Marginals are matched by their parameter-index tuple (e.g. ``(0,)`` or ``(7, 8)``).
    For every marginal that exists in both models the corresponding classifier
    weights are copied; new marginals keep their random initialisation.

    Works with both ``InferenceNetwork`` and ``JointAEInferenceNetwork``.
    """
    # Build a lookup: marginal_tuple -> (domain_key, position_in_domain) for old model
    old_lookup = {}
    for domain, marginal_list in old_model.marginals_dict.items():
        for pos, marginal in enumerate(marginal_list):
            old_lookup[tuple(marginal)] = (domain, pos)

    transferred, fresh = [], []
    for domain, marginal_list in new_model.marginals_dict.items():
        for pos, marginal in enumerate(marginal_list):
            key = tuple(marginal)
            if key in old_lookup:
                old_domain, old_pos = old_lookup[key]
                src = old_model.logratios_model_dict[old_domain].classifiers[old_pos]
                dst = new_model.logratios_model_dict[domain].classifiers[pos]
                dst.load_state_dict(src.state_dict())
                transferred.append(key)
            else:
                fresh.append(key)

    # Also transfer GradNorm loss weights if both models use them
    if (hasattr(old_model, "weights_loss_logits") and
            hasattr(new_model, "weights_loss_logits")):
        # Map old weight values by marginal index
        old_marg_list = old_model.marginals_list
        new_marg_list = new_model.marginals_list
        for new_idx, marg in enumerate(new_marg_list):
            key = tuple(marg)
            if key in old_lookup:
                old_idx = old_marg_list.index(list(key))
                new_model.weights_loss_logits.data[new_idx] = (
                    old_model.weights_loss_logits.data[old_idx]
                )

    print(f"[transfer] Carried over classifiers for {transferred}")
    print(f"[transfer] Freshly initialised classifiers for {fresh}")


def resolve_marginals_for_round(train_conf: dict, round_idx: int) -> dict:
    """Return the marginals dict to use for a given round.

    If ``train_conf`` contains a ``marginal_schedule`` list, the last entry
    whose ``from_round`` is <= *round_idx* wins.  Otherwise falls back to
    ``train_conf["marginals"]``.
    """
    schedule = train_conf.get("marginal_schedule")
    if not schedule:
        return train_conf["marginals"]

    # Sort by from_round so the last match is the tightest
    sorted_schedule = sorted(schedule, key=lambda e: e["from_round"])
    active = train_conf["marginals"]  # default fallback
    for entry in sorted_schedule:
        if round_idx >= entry["from_round"]:
            active = entry["marginals"]
    return active


def validate_marginals(marginals_config: dict):
    """Validate that no parameter index appears in multiple marginals.
    
    :param marginals_config: dictionary containing marginal lists from train_config
    :raises ValueError: if a parameter index is repeated across marginals
    """
    all_indices = []
    for key, marginal_list in marginals_config.items():
        for marginal in marginal_list:
            for idx in marginal:
                if idx in all_indices:
                    raise ValueError(
                        f"Parameter index {idx} ({_ORDERED_PRIOR_KEYS[idx]}) appears in multiple marginals. "
                        f"Each parameter index can only appear in one marginal for prior truncation."
                    )
                all_indices.append(idx)

def get_widest_interval_1d(model, dataloader, in_param_idx, out_param_idx, eps=0.0001):
    """Get the widest credible interval for a 1D marginal posterior.
    
    :param model: trained inference model
    :param dataloader: dataloader containing the observation
    :param in_param_idx: index of the input parameter
    :param out_param_idx: index of the output (logratio)
    :param eps: credible level (default 0.0001 for 99.99% interval)
    :return: (widest_interval, norm1d, grid, inj_params) where widest_interval is [low, high]
    """
    logratios, inj_params, grid = get_logratios_grid(
        dataloader,
        model,
        ngrid_points=100,
        in_param_idx=in_param_idx,
        out_param_idx=out_param_idx,
    )
    
    ratios = np.exp(logratios[0])  # Take first (only) observation
    dp = grid[1, 0] - grid[0, 0]
    norm1d = ratios / np.sum(ratios * dp)
    
    # Find credible interval using cumulative sum
    cumsum = np.cumsum(norm1d * dp)
    idx_low = np.searchsorted(cumsum, eps / 2)
    idx_high = np.searchsorted(cumsum, 1 - eps / 2)
    
    widest_interval = [float(grid[idx_low, 0]), float(grid[idx_high, 0])]
    return widest_interval, norm1d, grid, inj_params

def get_widest_box_2d(model, dataloader, in_param_idx, out_param_idx, ax_buffer=None, do_plot=False,
                      return_norm2d=False):
    """Get the widest credible box for a 2D marginal posterior.

    :param model: trained inference model
    :param dataloader: dataloader containing the observation
    :param in_param_idx: tuple of indices for the two input parameters
    :param out_param_idx: index of the output (logratio)
    :param ax_buffer: matplotlib axis to plot on (optional)
    :param do_plot: whether to create the contour plot
    :param return_norm2d: if True, also return (norm2d, dp1, dp2, gx, gy)
    :return: (widest_box, inj_params) or (widest_box, inj_params, norm2d, dp1, dp2, gx, gy)
    """
    norm2d, inj_params, gx, gy, dp1, dp2 = eval_posterior_2d(model, dataloader, in_param_idx, out_param_idx)
    levels, labels = contour_levels(norm2d)
    boxes = posterior_contours_2d(
        gx, gy, norm2d,
        inj_params[0],
        ax_buffer=ax_buffer,
        parameter_names=[_ORDERED_PRIOR_KEYS[in_param_idx[0]], _ORDERED_PRIOR_KEYS[in_param_idx[1]]],
        levels=levels,
        levels_labels=labels,
        do_plot=do_plot
    )
    widest_box = boxes[0]
    if return_norm2d:
        return widest_box, inj_params, norm2d, float(dp1), float(dp2), gx, gy
    return widest_box, inj_params


def get_widest_box_sky(model, dataloader, in_param_idx, out_param_idx,
                       credible_level=0.9545, dilation_factor=1.5,
                       ax_buffer=None, do_plot=False):
    """Robust sky truncation: bounding box of the main mode on S^2.

    Drop-in replacement for ``get_widest_box_2d`` when the marginal is the
    sky (lambda, sin beta).  Uses the 95 % HPD contour dilated by 1.5x
    (more stable than the raw 99.99 % contour) and handles periodic lambda.

    Parameters
    ----------
    model, dataloader, in_param_idx, out_param_idx :
        Same as ``get_widest_box_2d``.
    credible_level : float
        HPD level for thresholding (default 0.9545 = 95 %).
    dilation_factor : float
        Linear inflation factor applied to the 95 % region (default 1.5).
    ax_buffer : matplotlib Axes or None
    do_plot : bool

    Returns
    -------
    widest_box : tuple (x_low, x_high, y_low, y_high)
        Compatible with the existing truncation pipeline.
    inj_params : np.ndarray
    sky_analysis : dict
        Full output of ``analyse_sky_posterior`` (includes mask, components, etc.)
    """
    from pembhb.sky_truncation import get_main_mode_box, get_sky_mask, analyse_sky_posterior

    logratios, inj_params, gx, gy = get_logratios_grid_2d(
        dataloader, model,
        ngrid_points=100,
        in_param_idx=in_param_idx,
        out_param_idx=out_param_idx,
    )

    ratios = np.exp(logratios)
    dp1 = gx[0, 1] - gx[0, 0]
    dp2 = gy[1, 0] - gy[0, 0]
    norm2d = ratios / np.sum(ratios * dp1 * dp2, axis=(1, 2), keepdims=True)

    # Use first (only) observation
    posterior = norm2d[0]

    # --- Robust sky analysis -----------------------------------------------
    box = get_main_mode_box(gx, gy, posterior,
                            credible_level=credible_level,
                            dilation_factor=dilation_factor)
    sky_analysis = analyse_sky_posterior(gx, gy, posterior,
                                         credible_level=credible_level,
                                         dilation_factor=dilation_factor)

    lam_lo, lam_hi = box['lam']
    beta_lo, beta_hi = box['beta']

    print(f"[sky_truncation] {box['n_modes']} mode(s) detected, "
          f"main mode mass = {box['mass']:.4f}, "
          f"wrapped = {box['is_wrapped']}")
    print(f"[sky_truncation] lambda = [{lam_lo:.4f}, {lam_hi:.4f}], "
          f"sin(beta) = [{beta_lo:.4f}, {beta_hi:.4f}]")

    # --- Optional plot -----------------------------------------------------
    if do_plot and ax_buffer is not None:
        ax_buffer.pcolormesh(gx, gy, posterior, shading='auto', cmap='inferno')
        # Overlay the dilated mask as a semi-transparent region
        mask = sky_analysis['mask']
        masked_arr = np.ma.masked_where(~mask, np.ones_like(mask, dtype=float))
        ax_buffer.pcolormesh(gx, gy, masked_arr, shading='auto',
                             cmap='Greens', alpha=0.25, vmin=0, vmax=1)
        # Mark the main-mode bounding box
        from matplotlib.patches import Rectangle
        if not box['is_wrapped']:
            rect = Rectangle((lam_lo, beta_lo), lam_hi - lam_lo, beta_hi - beta_lo,
                              linewidth=1.5, edgecolor='lime', facecolor='none',
                              linestyle='--', label='main mode box')
            ax_buffer.add_patch(rect)
        else:
            # Two rectangles for wrapped interval
            TWO_PI = 2 * np.pi
            rect1 = Rectangle((lam_lo, beta_lo), TWO_PI - lam_lo, beta_hi - beta_lo,
                               linewidth=1.5, edgecolor='lime', facecolor='none', linestyle='--')
            rect2 = Rectangle((0, beta_lo), lam_hi, beta_hi - beta_lo,
                               linewidth=1.5, edgecolor='lime', facecolor='none', linestyle='--')
            ax_buffer.add_patch(rect1)
            ax_buffer.add_patch(rect2)
        # Injection
        ax_buffer.axvline(inj_params[0, 0], color='r', ls='--', lw=0.8)
        ax_buffer.axhline(inj_params[0, 1], color='r', ls='--', lw=0.8)
        ax_buffer.set_xlabel(_ORDERED_PRIOR_KEYS[in_param_idx[0]])
        ax_buffer.set_ylabel(_ORDERED_PRIOR_KEYS[in_param_idx[1]])
        ax_buffer.legend(fontsize=7)

    # Return in the same (x_low, x_high, y_low, y_high) format as get_widest_box_2d
    widest_box = (lam_lo, lam_hi, beta_lo, beta_hi)
    return widest_box, inj_params, sky_analysis


def eval_posterior_2d(model, dataloader, in_param_idx, out_param_idx, ngrid_points=100,
                      bounds_0=None, bounds_1=None, keep_batch_dim=False):
    """Evaluate and normalize a 2D marginal posterior on a grid.

    When *bounds_0* / *bounds_1* are provided, the grid spans those ranges
    instead of using the model's stored bounds (useful for plotting on a
    round-specific prior box).

    *keep_batch_dim* controls the shape of the returned ``norm2d``: ``False``
    drops the batch axis (returns ``(ngrid, ngrid)``), ``True`` keeps it
    (returns ``(batch, ngrid, ngrid)``).  ``dp0`` and ``dp1`` are NOT returned
    when the batch axis is kept, since callers in that mode (visualisation)
    can compute them from ``gx``/``gy``.
    """
    logratios, inj_params, gx, gy = get_logratios_grid_2d(
        dataloader, model,
        ngrid_points=ngrid_points,
        in_param_idx=in_param_idx,
        out_param_idx=out_param_idx,
        bounds_0=bounds_0, bounds_1=bounds_1,
    )
    ratios = np.exp(logratios)
    dp0 = float(gx[0, 1] - gx[0, 0])
    dp1 = float(gy[1, 0] - gy[0, 0])
    norm2d = ratios / np.sum(ratios * dp0 * dp1, axis=(1, 2), keepdims=True)
    if keep_batch_dim:
        return norm2d, inj_params, gx, gy
    return norm2d[0], inj_params, gx, gy, dp0, dp1
