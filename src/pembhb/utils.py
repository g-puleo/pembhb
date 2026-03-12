import yaml
import copy
import torch
import os 
from pembhb import ROOT_DIR
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
    with open(fname, "r") as file:
        conf = yaml.safe_load(file)
    return conf

def get_logratios_grid(dataloader: torch.utils.data.DataLoader, model: 'InferenceNetwork', ngrid_points: int, in_param_idx : int, out_param_idx: int, low: float=None , high: float=None):
    """Generate a grid of logratios for a given observation and model.
    This is useful for plotting the posterior and to make pp plots
    
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
    :return: logratios for the grid, with shape [batchsize, ngrid_points], injection parameters with shape [batchsize, 11], grid with shape [ngrid_points, 1]
    """

    results = []
    injection_params = []

    model.eval()
    model = model.to("cuda")
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
    grid = torch.linspace(low, high, ngrid_points).to("cuda").reshape(-1, 1)
    zero_pad1d = torch.zeros(ngrid_points, 10).to("cuda")
    grid_padded = torch.cat((zero_pad1d[:, :in_param_idx], grid, zero_pad1d[:, in_param_idx:]), dim=1)  # Shape: [ngrid_points, 11]
    with torch.no_grad():
        for batch in tqdm( dataloader ) :
            
            data_fd = (batch["wave_fd"]+batch["noise_fd"]).to("cuda")  # Shape: [batchsize, n_channels, n_datapoints]
            data_td = (batch["wave_td"]+batch["noise_td"]).to("cuda")  # Shape: [batchsize, n_channels, n_datapoints]
            source_parameters = batch["source_parameters"]  # Shape: [batchsize, 11]


            batch_size = data_fd.shape[0]

            data_td_expanded = data_td.unsqueeze(1).expand(batch_size, ngrid_points, -1, -1)  # Shape: [batchsize, ngrid_points, n_channels, n_datapoints]
            data_fd_expanded = data_fd.unsqueeze(1).expand(batch_size, ngrid_points, -1, -1)  # Shape: [batchsize, ngrid_points, n_channels, n_datapoints]
            grid_expanded = grid_padded.unsqueeze(0).expand(batch_size, -1, -1) # shape is [batchsize, ngrid_points, 11]

            batched_data_td = data_td_expanded.reshape(-1, data_td_expanded.shape[-2], data_td_expanded.shape[-1])  # Flatten batch and ngrid_points
            batched_data_fd = data_fd_expanded.reshape(-1, data_fd_expanded.shape[-2], data_fd_expanded.shape[-1])  # Flatten batch and ngrid_points
            batched_grid = grid_expanded.reshape(-1, grid_expanded.shape[-1])  # Flatten batch and ngrid_points

            batched2dataset = TensorDataset(batched_data_fd, batched_data_td, batched_grid)
            batched2dataloader = DataLoader(batched2dataset, batch_size=50, shuffle=False)
            logratios_list = []
            for batch2 in batched2dataloader:
                batched2_data_fd, batched2_data_td, batched2_grid = batch2
                batched2_data_fd = batched2_data_fd.to("cuda")
                batched2_data_td = batched2_data_td.to("cuda")
                batched2_grid = batched2_grid.to("cuda")
                logratios= model(batched2_data_fd, batched2_data_td, batched2_grid)[:, out_param_idx]  
                logratios_list.append(logratios)
            logratios = torch.cat(logratios_list, dim=0)            # view them as [batchsize, ngrid_points]
            logratios = logratios.reshape(batch_size, ngrid_points)

            results.append(logratios.detach().cpu())
            injection_params.append(source_parameters[:, in_param_idx].detach().cpu())

        results = torch.cat(results, dim=0).numpy()
        injection_params = torch.cat(injection_params, dim=0).numpy()
        grid = grid.detach().cpu().numpy()
    return results, injection_params, grid

def get_logratios_grid_2d(dataloader: torch.utils.data.DataLoader, model: 'InferenceNetwork', ngrid_points: int, out_param_idx : int, in_param_idx : tuple, 
                          bounds_0: tuple = None, bounds_1: tuple = None):
    """
    Compute logratios on a 2D grid for two input parameters.

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
    :param prior_bounds_0: the bounds of the interval on which the first input parameter is defined, defaults to the trained prior
    :type prior_bounds_0: tuple, optional
    :param prior_bounds_1: the bounds of the interval on which the second input parameter is defined, defaults to trained prior
    :type prior_bounds_1: tuple, optional
    :return: results, injection_params, grid_x, grid_y where results has shape (batch size, ngrid_points, ngrid_points), injection_params has shape (batch size, 2), grid_x and grid_y have shape (ngrid_points, ngrid_points)
    :rtype: _type_
    """
    results = []
    injection_params = []
    with torch.no_grad():
        model.eval()
        model = model.to("cuda")
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

        grid_x, grid_y = torch.meshgrid(grid_0, grid_1, indexing="xy")# grid_x and grid_y have shape (ngrid_points, ngrid_points)
        flattened_x = grid_x.flatten() # values of param_0 to test
        flattened_y = grid_y.flatten() # values of param_1 to test
        grid = torch.stack((flattened_x, flattened_y), dim=1).to("cuda")  # Shape: [ngrid_points^2, 2]
        
        # Create the padded input with parameters at correct positions
        grid_padded_input = torch.zeros(ngrid_points**2, 11).to("cuda")
        grid_padded_input[:, in_param_idx[0]] = grid[:, 0]
        grid_padded_input[:, in_param_idx[1]] = grid[:, 1]

        for batch in tqdm(dataloader):
            data_fd = (batch["wave_fd"] + batch["noise_fd"]).to("cuda")  # Shape: [batchsize, n_channels, n_datapoints]
            source_parameters = batch["source_parameters"]
            data_td  = (batch["wave_td"] + batch["noise_td"]).to("cuda")  
            batch_size = data_fd.shape[0]
            data_fd_expanded = data_fd.unsqueeze(1).expand(-1, ngrid_points**2, -1, -1)  # Shape: [batchsize, ngrid_points^2, n_channels, n_datapoints]
            grid_expanded = grid_padded_input.unsqueeze(0).expand(batch_size, -1, -1) # shape is [batchsize, ngrid_points^2, 11]
            data_td_expanded = data_td.unsqueeze(1).expand(-1, ngrid_points**2, -1,-1)  # Shape: [batchsize, ngrid_points^2, n_datapoints]
            batched_data_fd = data_fd_expanded.reshape(-1, data_fd_expanded.shape[-2], data_fd_expanded.shape[-1])
            batched_grid = grid_expanded.reshape(-1, grid_expanded.shape[-1])  # Flatten batch and ngrid_points
            batched_data_td = data_td_expanded.reshape(-1, data_td_expanded.shape[-2], data_td_expanded.shape[-1])
            batched2dataset = TensorDataset(batched_data_fd, batched_data_td, batched_grid)
            batched2dataloader = DataLoader(batched2dataset, batch_size=20, shuffle=False)
            logratios_list = []
            for batch2 in batched2dataloader:
                batched2_data_fd, batched2_data_td, batched2_grid = batch2
                batched2_data_fd = batched2_data_fd.to("cuda")
                batched2_data_td = batched2_data_td.to("cuda")
                batched2_grid = batched2_grid.to("cuda")
                logratios= model(batched2_data_fd, batched2_data_td, batched2_grid)[:, out_param_idx]  
                logratios_list.append(logratios)
            logratios = torch.cat(logratios_list, dim=0)
            # logratios = model(batched_data_fd, batched_data_td,  batched_grid)[:, out_idx] # shape is [batchsize*ngrid_points^2, ]
            logratios = logratios.reshape(batch_size, ngrid_points**2)  # Reshape to [batchsize, ngrid_points^2]

            results.append(logratios.detach().cpu())
            injection_params.append(source_parameters[:, in_param_idx].detach().cpu())

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

def pp_plot( dataloader, model , in_param_idx: int, name: str, out_param_idx: int):  
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
    """
    print(f"Making pp plot for {name}...")
    logratios, injection_params, grid = get_logratios_grid(dataloader, model, ngrid_points=100, in_param_idx=in_param_idx, out_param_idx=out_param_idx)
    p_values = get_pvalues_1d(logratios, grid, injection_params)
    sorted_pvalues = np.sort(p_values)
    sorted_rank = np.arange(sorted_pvalues.shape[0])
    fig, ax  = plt.subplots(figsize=(10, 6))
    ax.plot( sorted_pvalues, sorted_rank, marker='o', linestyle='-', markersize=3)
    ax.set_xlabel('HPD level')
    ax.set_ylabel('empirical coverage')
    ax.set_title(f'P-P plot, {name}')
    ax.grid(visible=True)
    if name is not None:
        fig.savefig(os.path.join(ROOT_DIR, "plots", f"{name}_pp_plot.png"))
    plt.close()

def pp_plot_2d(dataloader, model,  in_param_idx: tuple, out_idx: int, name: str):
    print(f"Making pp plot for {name}...")
    logratios, injection_params, grid_x, grid_y = get_logratios_grid_2d(dataloader, model, ngrid_points=50, out_param_idx=out_idx, in_param_idx=in_param_idx)
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
        fig.savefig(os.path.join(ROOT_DIR, "plots", f"{name}_pp_plot_2d.png"))
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
        # make a colormesh on the ax_buffer (optional, for debugging)
        if show_colormap:
            c = ax_buffer.pcolormesh(grid_x, grid_y, ratios, shading='auto', cmap="inferno", **plot_kwargs)
        
        # add contour lines
        boxes, cs = contour_boxes(grid_x, grid_y, ratios, levels, ax=ax_buffer,
                                  colors=contour_colors, linestyles=contour_linestyles,
                                  linewidths=contour_linewidths, alpha=contour_alpha)
        fmt = {lev: f"{p:.3f}" for lev, p in zip(levels, levels_labels)}
        ax_buffer.clabel(cs, fmt=fmt, fontsize=8)
        
        ax_buffer.axvline(x=true_values[0], color='r', linestyle='--', label='True Value')
        ax_buffer.axhline(y=true_values[1], color='r', linestyle='--')
        
        if show_colormap:
            fig = ax_buffer.get_figure()
            cbar = fig.colorbar(c, ax=ax_buffer)
        
        if title is not None:
            ax_buffer.set_title(title)
        ax_buffer.set_xlabel(parameter_names[0])
        ax_buffer.set_ylabel(parameter_names[1])
        ax_buffer.grid()
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





def mbhb_collate_fn(batch, subset: torch.utils.data.Subset, noise_factor, noise_shuffling=True):
    B = len(batch)
    wave_fd = torch.stack([b["wave_fd"] for b in batch])
    wave_td = torch.stack([b["wave_td"] for b in batch])
    params  = torch.stack([b["params"] for b in batch])

    # pick noise indices randomly
    if noise_shuffling:
        subset_idxs = torch.tensor(subset.indices)
        pick = subset_idxs[torch.randint(0, len(subset_idxs), (B,))]
    else:
        pick = torch.tensor([b["idx"] for b in batch])


    noise_fd = noise_factor*torch.stack([subset.dataset._load("noise_fd", i) for i in pick])
    noise_td = noise_factor*torch.stack([subset.dataset._load("noise_td", i) for i in pick])

    # delivers waveform and a random instance of the noise. 
    return {
        "source_parameters": params,
        "wave_fd": wave_fd,
        "wave_td": wave_td,
        "noise_fd": noise_fd,
        "noise_td": noise_td,
        "noise_index": pick,
    }

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

def compute_fisher_information_matrix(
    likelihood_fn,
    true_params_dict: dict,
    param_names: list,
    delta_frac: float = 1e-4,
):
    """Compute the Fisher Information Matrix via numerical second derivatives.

    Uses finite differences of the log-likelihood at *true_params_dict* to fill
    the FIM, then returns the Cramér-Rao lower bounds on parameter
    uncertainties as ``sqrt(diag(FIM^{-1}))``.

    Parameters
    ----------
    likelihood_fn : callable
        Function ``(dict) -> float`` returning the log-likelihood for a dict of
        ``{param_name: value}`` covering at least *param_names*.
    true_params_dict : dict
        Expansion-point parameter values (typically the injected / true values).
    param_names : list of str
        Names of the parameters to include in the FIM.
    delta_frac : float
        Fractional step size for finite differences
        (``delta_i = |val_i| * delta_frac`` if ``val_i != 0``, else
        ``delta_frac`` directly).

    Returns
    -------
    fisher : np.ndarray, shape (n, n)
        Fisher Information Matrix.
    param_uncertainties : np.ndarray, shape (n,)
        1-sigma uncertainties from the Cramér-Rao bound
        (``NaN`` if the FIM is singular).
    """
    n = len(param_names)
    fisher = np.zeros((n, n))

    logl_0 = likelihood_fn(true_params_dict)
    print(f"[FIM] Log-likelihood at expansion point: {logl_0:.4f}")
    print("[FIM] Computing Fisher Information Matrix ...")

    for i, pi in enumerate(param_names):
        for j, pj in enumerate(param_names):
            if j < i:
                fisher[i, j] = fisher[j, i]
                continue

            vi = true_params_dict[pi]
            vj = true_params_dict[pj]
            di = abs(vi) * delta_frac if vi != 0 else delta_frac
            dj = abs(vj) * delta_frac if vj != 0 else delta_frac

            if i == j:
                p_plus  = {**true_params_dict, pi: vi + di}
                p_minus = {**true_params_dict, pi: vi - di}
                d2 = (likelihood_fn(p_plus) - 2.0 * logl_0 + likelihood_fn(p_minus)) / di ** 2
            else:
                p_pp = {**true_params_dict, pi: vi + di, pj: vj + dj}
                p_pm = {**true_params_dict, pi: vi + di, pj: vj - dj}
                p_mp = {**true_params_dict, pi: vi - di, pj: vj + dj}
                p_mm = {**true_params_dict, pi: vi - di, pj: vj - dj}
                d2 = (
                    likelihood_fn(p_pp) - likelihood_fn(p_pm)
                    - likelihood_fn(p_mp) + likelihood_fn(p_mm)
                ) / (4.0 * di * dj)

            fisher[i, j] = -d2
            print(f"  F[{pi}, {pj}] = {fisher[i, j]:.3e}")

    print("[FIM] Fisher Information Matrix:")
    print(fisher)

    try:
        fisher_inv = np.linalg.inv(fisher)
        param_uncertainties = np.sqrt(np.diag(fisher_inv))
        print("[FIM] Parameter uncertainties (Cramér-Rao lower bound):")
        for name, sigma in zip(param_names, param_uncertainties):
            print(f"  σ({name}) = {sigma:.6e}")
    except np.linalg.LinAlgError:
        print("[FIM] WARNING: Fisher matrix is singular – cannot invert.")
        param_uncertainties = np.full(n, np.nan)

    return fisher, param_uncertainties


def compute_fisher_prior_bounds(
    datagen_config: dict,
    observation_file: str,
    event_idx: int,
    varying_params: list,
    fixed_params: list,
    n_sigma: float = 5.0,
    delta_frac: float = 1e-4,
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
    delta_frac : float
        Fractional step size passed to :func:`compute_fisher_information_matrix`.

    Returns
    -------
    dict
        Complete prior-bounds dict compatible with
        ``sampler_init_kwargs={"prior_bounds": ...}``.
    """
    import h5py  # h5py is already a project dependency
    # Lazy imports to avoid circular dependency with pembhb.simulator
    from pembhb.simulator import MBHBSimulatorFD_TD
    from bbhx.likelihood import Likelihood as BBHXLikelihoodFn

    # Load observation first so we can use true values for fixed params.
    print(f"[Fisher] Loading event {event_idx} from {observation_file} ...")
    import h5py as _h5
    with _h5.File(observation_file, "r") as f:
        freqs_obs       = f["frequencies"][:]
        true_params_arr = f["source_parameters"][event_idx]  # shape (11,)
        wave_fd         = f["wave_fd"][event_idx]            # shape (n_ch, n_freqs)
        noise_fd        = f["noise_fd"][event_idx]           # shape (n_ch, n_freqs)

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
    simulator = MBHBSimulatorFD_TD(
        datagen_config,
        sampler_init_kwargs={"prior_bounds": dummy_prior},
        seed=42,
    )
    frequencies = simulator.freqs_pos

    assert np.allclose(freqs_obs, frequencies), (
        "[Fisher] Frequency mismatch between observation file and simulator!"
    )

    # Build AET data and PSD.
    data_fd_complex = wave_fd + noise_fd
    psd_AE  = simulator.asd ** 2
    psd_T   = np.ones((1, psd_AE.shape[1]))
    psd_AET = np.concatenate([psd_AE, psd_T], axis=0)
    data_T  = np.zeros((1, data_fd_complex.shape[1]), dtype=np.complex128)
    data_fd = np.concatenate([data_fd_complex, data_T], axis=0)

    print("[Fisher] Creating BBHX likelihood ...")
    bbhx_ll = BBHXLikelihoodFn(
        simulator.wfd,
        frequencies,
        data_fd,
        psd_AET,
        force_backend="cpu",
    )

    # All parameters not in varying_params are fixed to the true observed value.
    all_fixed = {
        key: float(true_params_arr[_ORDERED_PRIOR_KEYS.index(key)])
        for key in _ORDERED_PRIOR_KEYS
        if key not in varying_params
    }

    def _log_likelihood(params_dict: dict) -> float:
        """Evaluate BBHX log-likelihood for the varying parameters only."""
        tmnre = np.array(
            [params_dict[k] if k in params_dict else all_fixed[k]
             for k in _ORDERED_PRIOR_KEYS],
            dtype=np.float64,
        ).reshape(-1, 1)
        bbhx_p = simulator.sampler.samples_to_bbhx_input(
            tmnre, t_obs_end=simulator.t_obs_end_SI
        )
        wf_kw = simulator.waveform_kwargs.copy()
        return float(bbhx_ll.get_ll(bbhx_p, **wf_kw)[0])

    # FIM computation.
    true_varying = {
        k: float(true_params_arr[_ORDERED_PRIOR_KEYS.index(k)])
        for k in varying_params
    }
    _, param_uncertainties = compute_fisher_information_matrix(
        _log_likelihood, true_varying, varying_params, delta_frac=delta_frac
    )

    # Assemble final prior bounds.
    prior_bounds = copy.deepcopy(datagen_config["prior"])

    # Pin fixed parameters to the true values from the observation file.
    for key, val in fixed_values.items():
        prior_bounds[key] = [val, val]

    # FIM-based bounds for varying parameters.
    for key, sigma in zip(varying_params, param_uncertainties):
        true_val = float(true_params_arr[_ORDERED_PRIOR_KEYS.index(key)])
        if np.isfinite(sigma):
            lo = float(true_val - n_sigma * sigma)
            hi = float(true_val + n_sigma * sigma)
        else:
            print(f"[Fisher] WARNING: σ({key}) is NaN – keeping datagen_config bounds.")
            lo, hi = datagen_config["prior"][key]
        prior_bounds[key] = [lo, hi]
        print(f"[Fisher] {key}: true={true_val:.6e}, σ={sigma:.3e} → [{lo:.6e}, {hi:.6e}]")

    print(f"[Fisher] Final prior bounds: {prior_bounds}")
    return prior_bounds


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

def get_widest_box_2d(model, dataloader, in_param_idx, out_param_idx, ax_buffer=None, do_plot=False):
    """Get the widest credible box for a 2D marginal posterior.
    
    :param model: trained inference model
    :param dataloader: dataloader containing the observation
    :param in_param_idx: tuple of indices for the two input parameters
    :param out_param_idx: index of the output (logratio)
    :param ax_buffer: matplotlib axis to plot on (optional)
    :param do_plot: whether to create the contour plot
    :return: (widest_box, inj_params) where widest_box is [x_low, x_high, y_low, y_high]
    """
    logratios, inj_params, gx, gy = get_logratios_grid_2d(
        dataloader,
        model,
        ngrid_points=100,
        in_param_idx=in_param_idx,
        out_param_idx=out_param_idx,
    )

    ratios = np.exp(logratios)
    dp1 = gx[0, 1] - gx[0, 0]  # param_0 spacing (x varies along columns with xy indexing)
    dp2 = gy[1, 0] - gy[0, 0]  # param_1 spacing (y varies along rows with xy indexing)
    norm2d = ratios / np.sum(ratios * dp1 * dp2, axis=(1, 2), keepdims=True)
    levels, labels = contour_levels(norm2d)
    boxes = posterior_contours_2d(
        gx, gy, norm2d[0],
        inj_params[0], 
        ax_buffer=ax_buffer, 
        parameter_names=[_ORDERED_PRIOR_KEYS[in_param_idx[0]], _ORDERED_PRIOR_KEYS[in_param_idx[1]]],
        levels=levels, 
        levels_labels=labels,
        do_plot=do_plot
    )
    widest_box = boxes[0]
    return widest_box, inj_params

class PlotPosteriorCallback(Callback):
    def __init__(self, timestamp: str, obs_loader: DataLoader, input_idx_list: list, output_idx_list: list, round_idx: int , call_every_n_epochs=1): 
        self.epochs_elapsed = 0
        self.call_every_n_epochs = call_every_n_epochs
        self.timestamp = timestamp
        self.obs_loader = obs_loader
        self.input_idx_list = input_idx_list
        self.output_idx_list = output_idx_list
        self.n_marginals = len(input_idx_list)
        self.init_time = datetime.now()
        self.round_idx = round_idx
        # Storage for volume ratio diagnostics
        self.volume_ratios = {}
    
    def _compute_posterior_volume_2d(self, widest_box):
        """
        Compute the area/volume of the posterior from the widest contour box.
        
        Parameters:
        -----------
        widest_box : tuple
            The bounding box of the 99.99% contour.
            Currently: (x_min, x_max, y_min, y_max) for axis-aligned boxes.
            
        Returns:
        --------
        float
            Area enclosed by the posterior contour.
            
        Notes:
        ------
        FUTURE EXTENSION FOR TILTED BOXES:
        - If posterior contours become non-axis-aligned, widest_box format may change
          to a list of vertices [(x1,y1), (x2,y2), ...]
        - In that case, use Shoelace formula or similar for polygon area:
          area = 0.5 * abs(sum(x[i]*y[i+1] - x[i+1]*y[i] for i in range(n)))
        - Consider using shapely.geometry.Polygon for robust area calculation
        """
        # Current implementation: axis-aligned box
        # widest_box = (x_min, x_max, y_min, y_max)
        posterior_area = (widest_box[1] - widest_box[0]) * (widest_box[3] - widest_box[2])
        return posterior_area
    
    def _compute_prior_volume_2d(self, pl_module, in_param_idx):
        """
        Compute the area/volume of the prior for a 2D marginal.
        
        Parameters:
        -----------
        pl_module : LightningModule
            The model containing prior information in hparams.
        in_param_idx : tuple
            Indices of the two parameters defining the 2D marginal.
            
        Returns:
        --------
        float
            Area of the prior region.
            
        Notes:
        ------
        **MODIFY THIS METHOD WHEN SWITCHING TO TILTED BOUNDING BOXES**
        
        Current implementation assumes axis-aligned rectangular priors.
        Prior bounds are stored as:
            prior_dict[param_name] = [min_value, max_value]
        
        For tilted/rotated bounding boxes:
        1. Prior specification will change (e.g., vertices, rotation matrix, etc.)
        2. Access prior from: pl_module.hparams["dataset_info"]["conf"]["prior"]
        3. Compute area based on new representation:
           - If vertices: use Shoelace formula or shapely.geometry.Polygon
           - If rotation + bounds: compute area of rotated rectangle
           - Example with vertices:
             ```python
             vertices = prior_dict[marginal_key]  # [(x1,y1), (x2,y2), ...]
             from shapely.geometry import Polygon
             prior_area = Polygon(vertices).area
             ```
        4. Ensure consistency with sampler_init_kwargs format in sampler.py
        
        Potential issues to address:
        - Normalization: If grid evaluation doesn't align with tilted prior,
          posterior normalization may be affected
        - Grid coverage: Axis-aligned grids may inefficiently cover tilted regions
        - Coordinate transforms: May need to transform between rotated and
          canonical coordinate systems
        """
        # Current implementation: axis-aligned rectangular prior
        # Use the actual sampling prior (sampler_init_kwargs) as the
        # authoritative source.  Fall back to conf["prior"] for backward compat.
        _sik = pl_module.hparams["dataset_info"].get("sampler_init_kwargs", {})
        if "prior_bounds" in _sik:
            prior_dict = _sik["prior_bounds"]
        else:
            prior_dict = pl_module.hparams["dataset_info"]["conf"]["prior"]
        
        # Get bounds for each parameter
        param_name_0 = _ORDERED_PRIOR_KEYS[in_param_idx[0]]
        param_name_1 = _ORDERED_PRIOR_KEYS[in_param_idx[1]]
        
        prior_bounds_0 = prior_dict[param_name_0]
        prior_bounds_1 = prior_dict[param_name_1]
        
        # Compute area as product of widths
        prior_area = (prior_bounds_0[1] - prior_bounds_0[0]) * (prior_bounds_1[1] - prior_bounds_1[0])
        
        return prior_area

    def _compute_posterior_volume_1d(self, widest_interval):
        """Compute the width of the posterior credible interval for a 1D marginal.
        
        Parameters:
        -----------
        widest_interval : list
            [low, high] bounds of the credible interval.
            
        Returns:
        --------
        float
            Width of the posterior interval.
        """
        return widest_interval[1] - widest_interval[0]
    
    def _compute_prior_volume_1d(self, pl_module, in_param_idx):
        """Compute the width of the prior for a 1D marginal.
        
        Parameters:
        -----------
        pl_module : LightningModule
            The model containing prior information in hparams.
        in_param_idx : int
            Index of the parameter.
            
        Returns:
        --------
        float
            Width of the prior range.
        """
        # Use the actual sampling prior (sampler_init_kwargs) as the
        # authoritative source.  Fall back to conf["prior"] for backward compat.
        _sik = pl_module.hparams["dataset_info"].get("sampler_init_kwargs", {})
        if "prior_bounds" in _sik:
            prior_dict = _sik["prior_bounds"]
        else:
            prior_dict = pl_module.hparams["dataset_info"]["conf"]["prior"]
        param_name = _ORDERED_PRIOR_KEYS[in_param_idx]
        prior_bounds = prior_dict[param_name]
        return prior_bounds[1] - prior_bounds[0]

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.epochs_elapsed == 0: 
            os.makedirs(os.path.join(ROOT_DIR, "plots", self.timestamp), exist_ok=True)

        self.epochs_elapsed += 1
        if (self.epochs_elapsed-2) % self.call_every_n_epochs == 0:
            #print("plotting posteriors on observed data")
            train_time = datetime.now() - self.init_time
            td_trunc = train_time - timedelta(microseconds=train_time.microseconds)
            title_plot = f"training time={td_trunc}s"
            # plot the posterior on the observed data , using the current model
            for i in range(self.n_marginals):
                in_param_idx = self.input_idx_list[i]
                out_param_idx = self.output_idx_list[i]

                # Initialize widest_boxes dict if not present
                if not hasattr(pl_module, 'widest_boxes'):
                    pl_module.widest_boxes = {}
                marginal_key = tuple(in_param_idx)

                if len(in_param_idx) == 1:
                    # Handle 1D marginals
                    param_idx = in_param_idx[0]
                    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
                    fig.suptitle(
                        f"Round {self.round_idx} - Epoch {trainer.current_epoch} - {title_plot}",
                        fontsize=10,
                    )
                    
                    try:
                        epsilon_value = 1e-4
                        widest_interval, norm1d, grid, inj_params = get_widest_interval_1d(
                            pl_module,
                            self.obs_loader,
                            in_param_idx=param_idx,
                            out_param_idx=out_param_idx,
                            eps=epsilon_value
                        )
                        
                        # Plot
                        ax.plot(grid.flatten(), norm1d, 'b-', linewidth=1.5)
                        ax.axvline(inj_params[0], color='r', linestyle='--', label='Injection')
                        ax.axvline(widest_interval[0], color='g', linestyle=':', label=f'{100*(1-epsilon_value):.2f}% CI')
                        ax.axvline(widest_interval[1], color='g', linestyle=':')
                        ax.fill_between(grid.flatten(), 0, norm1d, 
                                       where=(grid.flatten() >= widest_interval[0]) & (grid.flatten() <= widest_interval[1]),
                                       alpha=0.3, color='green')
                        ax.set_xlabel(_ORDERED_PRIOR_KEYS[param_idx])
                        ax.set_ylabel('Posterior density')
                        ax.legend()
                        
                        # Store the widest interval
                        pl_module.widest_boxes[marginal_key] = widest_interval
                        
                        # Compute posterior-to-prior volume (width) ratio for 1D marginal
                        posterior_width = self._compute_posterior_volume_1d(widest_interval)
                        prior_width = self._compute_prior_volume_1d(pl_module, param_idx)
                        volume_ratio = posterior_width / prior_width
                        
                        # Store and log the volume ratio
                        if marginal_key not in self.volume_ratios:
                            self.volume_ratios[marginal_key] = []
                        self.volume_ratios[marginal_key].append({
                            'epoch': trainer.current_epoch,
                            'ratio': volume_ratio,
                            'posterior_width': posterior_width,
                            'prior_width': prior_width
                        })
                        
                        # Log metric to tensorboard if logger exists
                        if trainer.logger is not None:
                            metric_name = f"volume_ratio/{_ORDERED_PRIOR_KEYS[param_idx]}"
                            trainer.logger.log_metrics({metric_name: volume_ratio}, step=trainer.current_epoch)
                        
                        # Print diagnostic
                        param_name = _ORDERED_PRIOR_KEYS[param_idx]
                        print(f"Round {self.round_idx}, Epoch {trainer.current_epoch}, {param_name}: "
                              f"Volume ratio (posterior/prior) = {volume_ratio:.6f} "
                              f"(posterior width: {posterior_width:.6e}, prior width: {prior_width:.6e})")
                        
                        out = os.path.join(ROOT_DIR, "plots", self.timestamp, 
                                          f"posterior_round_{self.round_idx}_epoch_{trainer.current_epoch}_{_ORDERED_PRIOR_KEYS[param_idx]}.pdf")
                        fig.savefig(out, bbox_inches="tight")
                    except Exception as e:
                        print(f"Error plotting 1D marginal for {_ORDERED_PRIOR_KEYS[param_idx]}: {e}")
                    finally:
                        plt.close(fig)

                elif len(in_param_idx) == 2:
                    # Handle 2D marginals
                    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
                    fig.tight_layout()
                    fig.suptitle(
                        f"Round {self.round_idx} - Epoch {trainer.current_epoch} - {title_plot}",
                        fontsize=10,
                    )

                    try:
                        widest_box, inj_params = get_widest_box_2d(
                            pl_module,
                            self.obs_loader,
                            in_param_idx=in_param_idx,
                            out_param_idx=out_param_idx,
                            ax_buffer=ax,
                            do_plot=True
                        )

                        # Store widest_box keyed by the marginal (tuple of input parameter indices)
                        pl_module.widest_boxes[marginal_key] = widest_box
                        
                        # Compute posterior-to-prior volume ratio for 2D marginal
                        posterior_area = self._compute_posterior_volume_2d(widest_box)
                        prior_area = self._compute_prior_volume_2d(pl_module, in_param_idx)
                        volume_ratio = posterior_area / prior_area
                        
                        # Store and log the volume ratio
                        if marginal_key not in self.volume_ratios:
                            self.volume_ratios[marginal_key] = []
                        self.volume_ratios[marginal_key].append({
                            'epoch': trainer.current_epoch,
                            'ratio': volume_ratio,
                            'posterior_area': posterior_area,
                            'prior_area': prior_area
                        })
                        
                        # Log metric to tensorboard if logger exists
                        if trainer.logger is not None:
                            metric_name = f"volume_ratio/{_ORDERED_PRIOR_KEYS[in_param_idx[0]]}_{_ORDERED_PRIOR_KEYS[in_param_idx[1]]}"
                            trainer.logger.log_metrics({metric_name: volume_ratio}, step=trainer.current_epoch)
                        
                        # Print diagnostic
                        param_names = f"{_ORDERED_PRIOR_KEYS[in_param_idx[0]]}-{_ORDERED_PRIOR_KEYS[in_param_idx[1]]}"
                        print(f"Round {self.round_idx}, Epoch {trainer.current_epoch}, {param_names}: "
                              f"Volume ratio (posterior/prior) = {volume_ratio:.6f} "
                              f"(posterior area: {posterior_area:.6e}, prior area: {prior_area:.6e})")
                        
                        out = os.path.join(ROOT_DIR, "plots", self.timestamp,
                                          f"posterior_round_{self.round_idx}_epoch_{trainer.current_epoch}_{_ORDERED_PRIOR_KEYS[in_param_idx[0]]}_{_ORDERED_PRIOR_KEYS[in_param_idx[1]]}.pdf")
                        fig.savefig(out, bbox_inches="tight")
                    except ValueError as ve:
                        print(f"caught ValueError: {ve} during contour plotting, skipping this plot")
                    finally:
                        plt.close(fig)

    def on_train_end(self, trainer, pl_module):
        self.on_validation_epoch_end(trainer, pl_module)
        print(f"Total training time: {datetime.now() - self.init_time}")