import yaml
import torch
import os 
from pembhb import ROOT_DIR
import numpy as np
from pembhb.model import InferenceNetwork
from pembhb.data import MBHBDataset
from torch.utils.data import DataLoader
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

def get_logratios_grid(dataset: MBHBDataset, model: InferenceNetwork, low: float, high: float, ngrid_points: int, inj_param_idx : int):
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
    :return: logratios for the grid, with shape   [batchsize, ngrid_points], injection parameters with shape [batchsize, 11], grid with shape [ngrid_points, 1]
    """
    dataloader = DataLoader(dataset, batch_size=min(10, len(dataset)), shuffle=False)

    results = []
    injection_params = []

    model.eval()
    model = model.to("cuda")
    grid = torch.linspace(low, high, ngrid_points).to("cuda").reshape(-1, 1)
    zero_pad1d = torch.zeros(ngrid_points, 10).to("cuda")
    grid_padded = torch.cat((zero_pad1d[:, :inj_param_idx], grid, zero_pad1d[:, inj_param_idx:]), dim=1)  # Shape: [ngrid_points, 11]
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

            logratios= model(batched_data_fd, batched_data_td, batched_grid)[:, inj_param_idx]  # Get logratios for mchirp
            # view them as [batchsize, ngrid_points]
            logratios = logratios.reshape(batch_size, ngrid_points)

            results.append(logratios.detach().cpu())
            injection_params.append(source_parameters[:, inj_param_idx].detach().cpu())

        results = torch.cat(results, dim=0).numpy()
        injection_params = torch.cat(injection_params, dim=0).numpy()
    return results, injection_params, grid.detach().cpu()

def get_logratios_grid_2d(dataset: MBHBDataset, model: InferenceNetwork, lows: float, highs: float, ngrid_points: int, out_idx : int, inj_param_idx : tuple):
    dataloader = DataLoader(dataset, batch_size=min(10, len(dataset)), shuffle=False)
    results = []
    injection_params = []

    model.eval()
    model = model.to("cuda")
    grid_0 = torch.linspace(lows[0], highs[0], ngrid_points).to("cuda").reshape(-1, 1)
    grid_1 = torch.linspace(lows[1], highs[1], ngrid_points).to("cuda").reshape(-1, 1)

    grid_x, grid_y = np.meshgrid(grid_0, grid_1) # grid_x and grid_y have shape (ngrid_points, ngrid_points)
    flattened_x = grid_x.flatten() # values of param_0 to test
    flattened_y = grid_y.flatten() # values of param_1 to test
    grid = torch.stack((flattened_x, flattened_y), dim=1).to("cuda")  # Shape: [ngrid_points^2, 2]
    padder = torch.zeros(ngrid_points**2, 9).to("cuda") # pass zero as other parameters
    grid_padded_input = torch.cat((grid, padder), dim=1)  # Shape: [ngrid_points^2, 11]

    with torch.no_grad():
        for batch in tqdm(dataloader):
            data_fd = batch["data_fd"].to("cuda")  # Shape: [batchsize, n_channels, n_datapoints]
            source_parameters = batch["source_parameters"]
            batch_size = data_fd.shape[0]
            data_fd_expanded = data_fd.unsqueeze(1).expand(-1, ngrid_points**2, -1, -1)  # Shape: [batchsize, ngrid_points^2, n_channels, n_datapoints]
            grid_expanded = grid_padded_input.unsqueeze(0).expand(batch_size, -1, -1) # shape is [batchsize, ngrid_points^2, 11]

            batched_data = data_fd_expanded.reshape(-1, data_fd_expanded.shape[-2], data_fd_expanded.shape[-1])
            batched_grid = grid_expanded.reshape(-1, grid_expanded.shape[-1])  # Flatten batch and ngrid_points

            logratios = model(batched_data, batched_grid)[:, out_idx] # shape is [batchsize*ngrid_points^2, ]
            logratios = logratios.reshape(batch_size, ngrid_points**2)  # Reshape to [batchsize, ngrid_points^2]

            results.append(logratios.detach().cpu())
            injection_params.append(source_parameters[:, inj_param_idx].detach().cpu())

        results = torch.cat(results, dim=0).numpy()
        injection_params = torch.cat(injection_params, dim=0).numpy()

    return results, injection_params, grid.detach().cpu()

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
    idx = np.argmin(np.abs(sorted_grid - inj_param), axis=1).numpy().squeeze(1)
    idx_rank = np.arange(idx.shape[0])

    cumsum =  np.cumsum(sorted_ratios, axis=1)
    cumsum /= cumsum[:,-1:]  # normalize to get a cumulative distribution
    print(idx_rank.shape, idx.shape)
    p_values = cumsum[idx_rank, idx]

    return p_values

def get_pvalues_2d(logratios: np.array, grid: np.array , inj_param: np.array):
    """Calculate p-values for a 2D logratios array.
    Assume the grid was flattened before. 
    Recall that exp(logratios) = posterior/prior, and here we assume a uniform prior.

    :param logratios: logratios for the grid , has shape (batch size, ngrid_points)
    :type logratios: np.array
    :param grid: grid of parameters, has shape (ngrid_points, 2)
    :type grid: np.array
    :param inj_param: injected parameter value
    :type inj_param: np.array with shape (batch size, 2)
    """

    ratios = np.exp(logratios)
    sorted_ratios = np.sort(ratios, axis=1)
    sorted_indices =  np.argsort(ratios, axis=1)
    sorted_grid =  grid[sorted_indices] # shape is 
    inj_param = inj_param.reshape(-1,1,2)


    




def update_bounds(model: InferenceNetwork, observation_dataset: MBHBDataset, priordict: dict, parameter_idx: int, n_gridpoints: int = 100):
    """Update the prior bounds based on the posterior obtained from a model on a single observation. 
    Used to do truncation in MNRE. 

    :param model: trained inference model
    :type model: InferenceNetwork
    :param observation_dataset: dataset containing the (single) obs
    :type observation_dataset: MBHBDataset
    :param priordict: dictionary containing the prior bounds for each parameter. 
    :type priordict: dict
    :param parameter_idx: index of the parameter to update
    :type parameter_idx: int
    :param n_gridpoints: number of points in the grid, defaults to 100
    :type n_gridpoints: int, optional
    :return: updated prior bounds
    :rtype: dict
    """
    # evaluate the model over a decently fine grid, which requires knowledge of previous prior region 
    prior_low, prior_high = priordict[_ORDERED_PRIOR_KEYS[parameter_idx]]
    logratios, injection_params, grid = get_logratios_grid(observation_dataset, model, prior_low, prior_high, n_gridpoints, inj_param_idx=parameter_idx)
    print(f"prior_low: {prior_low}, prior_high: {prior_high}")
    # find the 95% two tail interval of the posterior 
    print(f"injection_params are: {injection_params}")
    cumsum = np.cumsum(np.exp(logratios))
    cumsum /= cumsum[-1]  
    idx_low = np.argwhere(cumsum < 0.01)[-1]
    idx_high = np.argwhere(cumsum > 0.99)[0]

    new_low = grid[idx_low]
    new_high = grid[idx_high]

    updated_prior = priordict.copy()
    updated_prior[_ORDERED_PRIOR_KEYS[parameter_idx]] = [new_low.item(), new_high.item()]
    print(f"Updated prior for {_ORDERED_PRIOR_KEYS[parameter_idx]}: {updated_prior[_ORDERED_PRIOR_KEYS[parameter_idx]]}")
    
    return updated_prior

def pp_plot( dataset, model , low: float, high: float, inj_param_idx: int, name: str = None):  
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
    logratios, injection_params, grid = get_logratios_grid(dataset, model, low=low, high=high, ngrid_points=100, inj_param_idx=inj_param_idx)
    p_values = get_pvalues_1d(logratios, grid, injection_params)
    sorted_pvalues = np.sort(p_values)
    sorted_rank = np.arange(sorted_pvalues.shape[0])
    fig, ax  = plt.subplots(figsize=(10, 6))
    ax.plot(sorted_rank, sorted_pvalues, marker='o', linestyle='-', markersize=3)
    ax.set_xlabel('Rank')
    ax.set_ylabel('P-value')
    ax.set_title(f'Sorted P-values, {name}')
    ax.grid(visible=True)
    if name is not None:
        fig.savefig(os.path.join(ROOT_DIR, "plots", f"{name}_pp_plot.png"))
    plt.close()

def pp_plot_2d(dataset, model, lows: tuple, highs: tuple, inj_param_idx: tuple, out_idx: int, name: str):

    dataset = MBHBDataset(dataset, channels="AE")


if __name__ == "__main__":
    fname = "/u/g/gpuleo/pembhb/logs_0729/peregrine_norm/version_1/checkpoints/epoch=93-step=16920.ckpt"
    conf = read_config(os.path.join(ROOT_DIR,"config.yaml"))
    model = InferenceNetwork.load_from_checkpoint(fname, conf=conf)
    model.eval()

    dataset = MBHBDataset(os.path.join(ROOT_DIR, "data/observation.h5"))

    updated_prior = update_bounds(model, dataset, conf, parameter_idx=1, n_gridpoints=100)
    print(updated_prior)


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
                                      