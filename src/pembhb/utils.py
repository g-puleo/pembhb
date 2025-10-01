import yaml
import torch
import os 
from pembhb import ROOT_DIR
import numpy as np
from pembhb.data import MBHBDataset
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm
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
    prior_trained_dict = model.hparams["prior"]
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

def get_logratios_grid_2d(dataset: MBHBDataset, model: 'InferenceNetwork', ngrid_points: int, out_param_idx : int, in_param_idx : tuple):
    dataloader = DataLoader(dataset, batch_size=min(10, len(dataset)), shuffle=False)
    results = []
    injection_params = []

    model.eval()
    model = model.to("cuda")
    prior_trained_dict = model.hparams["prior"]
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




    




def update_bounds(model: 'InferenceNetwork', observation_dataset: MBHBDataset, priordict: dict, in_param_idx: int, n_gridpoints: int = 100000, out_param_idx: int = None, eps: float = 1e-5):
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
    logratios, injection_params, grid = get_logratios_grid(observation_dataset, model, n_gridpoints, in_param_idx=in_param_idx, out_param_idx=out_param_idx)
    # find the 95% two tail interval of the posterior 
    print(f"injection_params are: {injection_params}")
    cumsum = np.cumsum(np.exp(logratios))
    cumsum /= cumsum[-1]  
    idx_low = np.argwhere(cumsum < eps/2)[-1]
    idx_high = np.argwhere(cumsum > 1-eps/2)[0]

    new_low = grid[idx_low]
    new_high = grid[idx_high]

    updated_prior = priordict.copy()
    updated_prior[_ORDERED_PRIOR_KEYS[in_param_idx]] = [new_low.item(), new_high.item()]
    print(f"Updated prior for {_ORDERED_PRIOR_KEYS[in_param_idx]}: {updated_prior[_ORDERED_PRIOR_KEYS[in_param_idx]]}")
    
    return updated_prior

def pp_plot( dataset, model , in_param_idx: int, name: str = None, out_param_idx: int = None):  
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
    logratios, injection_params, grid_x, grid_y = get_logratios_grid_2d(dataset, model, ngrid_points=50, in_param_idx=in_param_idx, out_idx=out_idx)
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
                                      