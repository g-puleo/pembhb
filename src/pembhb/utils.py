import yaml
import torch
import os 
from pembhb import ROOT_DIR
import numpy as np
from pembhb.model import InferenceNetwork
from pembhb.data import MBHBDataset
from torch.utils.data import DataLoader

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
    dataloader = DataLoader(dataset, batch_size=min(50, len(dataset)), shuffle=False)

    results = []
    injection_params = []

    model.eval()
    model = model.to("cuda")
    grid = torch.linspace(low, high, ngrid_points).to("cuda").reshape(-1, 1)

    with torch.no_grad():
        for batch in dataloader:
            data_fd = batch["data_fd"].to("cuda")  # Shape: [batchsize, n_channels, n_datapoints]
            source_parameters = batch["source_parameters"]  # Shape: [batchsize, 11]

            # q_grid = torch.linspace(low, high, ngrid_points).to("cuda").reshape(-1, 1)
            zero_pad1d = torch.zeros(ngrid_points, 10).to("cuda")

            lmc_grid_padded = torch.cat((grid, zero_pad1d), dim=1)  # Shape: [ngrid_points, 11]
            # q_grid_padded = torch.cat((zero_pad1d[:, 0:1], q_grid, zero_pad1d[:, 1:]), dim=1)  # Shape: [ngrid_points, 11]

            batch_size = data_fd.shape[0]
            data_fd_expanded = data_fd.unsqueeze(1).expand(batch_size, ngrid_points, -1, -1)  # Shape: [batchsize, ngrid_points, n_channels, n_datapoints]
            mc_grid_expanded = lmc_grid_padded.unsqueeze(0).expand(batch_size, -1, -1) # shape is [batchsize, ngrid_points, 11]

            batched_data = data_fd_expanded.reshape(-1, data_fd_expanded.shape[-2], data_fd_expanded.shape[-1])  # Flatten batch and ngrid_points
            batched_mc_grid = mc_grid_expanded.reshape(-1, mc_grid_expanded.shape[-1])  # Flatten batch and ngrid_points

            #print(f"data_fd_expanded shape: {data_fd_expanded.shape}, mc_grid shape: {mc_grid.shape}")
            logratios_mchirp = model(batched_data, batched_mc_grid)[:, 0]  # Get logratios for mchirp
            # view them as [batchsize, ngrid_points]
            logratios_mchirp = logratios_mchirp.reshape(batch_size, ngrid_points)
            #logratios_q = model(data_fd_expanded, q_grid_padded.unsqueeze(0).expand(batch_size, -1, -1))[:, :, 1]

            results.append(logratios_mchirp.detach().cpu())
            injection_params.append(source_parameters[:, inj_param_idx].detach().cpu())

        results = torch.cat(results, dim=0).numpy()
        injection_params = torch.cat(injection_params, dim=0).numpy()
    return results, injection_params, grid.detach().cpu()

def update_bounds(model: InferenceNetwork, observation_dataset: MBHBDataset, conf: dict, parameter_idx: int, n_gridpoints: int = 100):

    # evaluate the model over a decently fine grid, which requires knowledge of previous prior region 
    prior_low, prior_high = conf["prior"][_ORDERED_PRIOR_KEYS[parameter_idx]]
    logratios, injection_params, grid = get_logratios_grid(observation_dataset, model, prior_low, prior_high, n_gridpoints, inj_param_idx=parameter_idx)
    # find the 95% two tail interval of the posterior 
    print(f"injection_params are: {injection_params}")
    cumsum = np.cumsum(np.exp(logratios))
    cumsum /= cumsum[-1]  
    idx_low = np.argwhere(cumsum < 0.05)[-1]
    idx_high = np.argwhere(cumsum > 0.95)[0]

    new_low = grid[idx_low]
    new_high = grid[idx_high]

    updated_prior = conf["prior"].copy()
    updated_prior[_ORDERED_PRIOR_KEYS[parameter_idx]] = [new_low.item(), new_high.item()]
    print(f"Updated prior for {_ORDERED_PRIOR_KEYS[parameter_idx]}: {updated_prior[_ORDERED_PRIOR_KEYS[parameter_idx]]}")
    
    return updated_prior



if __name__ == "__main__":
    fname = "/u/g/gpuleo/pembhb/logs_0729/peregrine_norm/version_1/checkpoints/epoch=93-step=16920.ckpt"
    conf = read_config(os.path.join(ROOT_DIR,"config.yaml"))
    model = InferenceNetwork.load_from_checkpoint(fname, conf=conf)
    model.eval()

    dataset = MBHBDataset(os.path.join(ROOT_DIR, "data/observation.h5"))

    updated_prior = update_bounds(model, dataset, conf, parameter_idx=1, n_gridpoints=100)
    print(updated_prior)
