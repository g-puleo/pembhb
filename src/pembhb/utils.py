import yaml
import torch
import os 
from pembhb import ROOT_DIR
import numpy as np
from pembhb.data import MBHBDataset
from torch.utils.data import DataLoader, TensorDataset
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

def get_logratios_grid(dataset: MBHBDataset, model: 'InferenceNetwork', ngrid_points: int, in_param_idx : int, out_param_idx: int):
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
    dataloader = DataLoader(dataset, batch_size=min(10, len(dataset)), shuffle=False)

    results = []
    injection_params = []

    model.eval()
    model = model.to("cuda")
    prior_trained_dict = model.hparams["dataset_info"]["conf"]["prior"]
    prior_bounds = prior_trained_dict[_ORDERED_PRIOR_KEYS[in_param_idx]]
    low = prior_bounds[0]
    high = prior_bounds[1]
    grid = torch.linspace(low, high, ngrid_points).to("cuda").reshape(-1, 1)
    zero_pad1d = torch.zeros(ngrid_points, 10).to("cuda")
    grid_padded = torch.cat((zero_pad1d[:, :in_param_idx], grid, zero_pad1d[:, in_param_idx:]), dim=1)  # Shape: [ngrid_points, 11]
    with torch.no_grad():
        for batch in tqdm( dataloader ) :
            data_fd = batch["data_fd"].to("cuda")  # Shape: [batchsize, n_channels, n_datapoints]
            data_td = batch["data_td"].to("cuda")  # Shape: [batchsize, n_channels, n_datapoints]
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

def get_logratios_grid_2d(dataset: MBHBDataset, model: 'InferenceNetwork', ngrid_points: int, out_param_idx : int, in_param_idx : tuple):
    dataloader = DataLoader(dataset, batch_size=min(10, len(dataset)), shuffle=False)
    results = []
    injection_params = []

    model.eval()
    model = model.to("cuda")
    prior_trained_dict = model.hparams["dataset_info"]["conf"]["prior"]
    prior_bounds_0 = prior_trained_dict[_ORDERED_PRIOR_KEYS[in_param_idx[0]]]
    prior_bounds_1 = prior_trained_dict[_ORDERED_PRIOR_KEYS[in_param_idx[1]]]
    lows = [prior_bounds_0[0], prior_bounds_1[0]]
    highs = [prior_bounds_0[1], prior_bounds_1[1]]
    grid_0 = torch.linspace(lows[0], highs[0], ngrid_points).reshape(-1)
    grid_1 = torch.linspace(lows[1], highs[1], ngrid_points).reshape(-1)

    grid_x, grid_y = torch.meshgrid(grid_0, grid_1)# grid_x and grid_y have shape (ngrid_points, ngrid_points)
    flattened_x = grid_x.flatten() # values of param_0 to test
    flattened_y = grid_y.flatten() # values of param_1 to test
    grid = torch.stack((flattened_x, flattened_y), dim=1).to("cuda")  # Shape: [ngrid_points^2, 2]
    padder = torch.zeros(ngrid_points**2, 9).to("cuda") # pass zero as other parameters
    grid_padded_input = torch.cat((grid, padder), dim=1)  # Shape: [ngrid_points^2, 11]

    with torch.no_grad():
        for batch in tqdm(dataloader):
            data_fd = batch["data_fd"].to("cuda")  # Shape: [batchsize, n_channels, n_datapoints]
            source_parameters = batch["source_parameters"]
            data_td  = batch["data_td"].to("cuda")  
            batch_size = data_fd.shape[0]
            data_fd_expanded = data_fd.unsqueeze(1).expand(-1, ngrid_points**2, -1, -1)  # Shape: [batchsize, ngrid_points^2, n_channels, n_datapoints]
            grid_expanded = grid_padded_input.unsqueeze(0).expand(batch_size, -1, -1) # shape is [batchsize, ngrid_points^2, 11]
            data_td_expanded = data_td.unsqueeze(1).expand(-1, ngrid_points**2, -1,-1)  # Shape: [batchsize, ngrid_points^2, n_datapoints]
            batched_data_fd = data_fd_expanded.reshape(-1, data_fd_expanded.shape[-2], data_fd_expanded.shape[-1])
            batched_grid = grid_expanded.reshape(-1, grid_expanded.shape[-1])  # Flatten batch and ngrid_points
            batched_data_td = data_td_expanded.reshape(-1, data_td_expanded.shape[-2], data_td_expanded.shape[-1])
            batched2dataset = TensorDataset(batched_data_fd, batched_data_td, batched_grid)
            batched2dataloader = DataLoader(batched2dataset, batch_size=75, shuffle=False)
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

    return results, injection_params, grid_x , grid_y

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
    print(idx_rank.shape, idx.shape)
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




    




def update_bounds(model: 'InferenceNetwork', observation_dataset: MBHBDataset, priordict: dict, in_param_idx: int, n_gridpoints: int, out_param_idx: int, eps: float = 1e-5):
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
            logratios, injection_params, grid = get_logratios_grid(observation_dataset, model, n_gridpoints, in_param_idx=in_param_idx, out_param_idx=out_param_idx)
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

def pp_plot( dataset, model , in_param_idx: int, name: str, out_param_idx: int):  
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
    logratios, injection_params, grid = get_logratios_grid(dataset, model, ngrid_points=100, in_param_idx=in_param_idx, out_param_idx=out_param_idx)
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

def pp_plot_2d(dataset, model,  in_param_idx: tuple, out_idx: int, name: str):
    print(f"Making pp plot for {name}...")
    logratios, injection_params, grid_x, grid_y = get_logratios_grid_2d(dataset, model, ngrid_points=50, out_param_idx=out_idx, in_param_idx=in_param_idx)
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
