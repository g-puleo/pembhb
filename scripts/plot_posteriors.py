import argparse
from pembhb import utils
from pembhb.data import MBHBDataset
from pembhb.model import InferenceNetwork
from glob import glob
from torch.utils.data import Subset
import numpy as np
import  matplotlib.pyplot as plt
from pembhb.utils import _ORDERED_PRIOR_KEYS
def plot_posterior_1d(grid: np.array,  ratios: np.array, true_value: float,  ax_buffer: plt.Axes, parameter_name: str, idx: int=0):
    dtheta = grid[1]-grid[0] # assuming uniform spacing
    normalised_ratios = ratios / np.sum(ratios*dtheta)
    ax_buffer.plot(grid, normalised_ratios, label='NRE')
    ax_buffer.axvline(x=true_value, color='r', linestyle='--', label='True Value')
    #ax_buffer.set_title(f"Event {idx}")
    ax_buffer.set_xlabel(parameter_name)
    ax_buffer.set_ylabel("Posterior Density")
    ax_buffer.grid()
    return ax_buffer

def plot_posterior_2d(grid_x: np.array, grid_y: np.array, ratios: np.array, true_values: list, ax_buffer: plt.Axes, parameter_names: list, idx: int=0):
    dx = grid_x[1]-grid_x[0] # assuming uniform spacing
    dy = grid_y[1]-grid_y[0]
    #normalised_ratios = ratios / np.sum(ratios*dx*dy)
    c = ax_buffer.pcolormesh(grid_x, grid_y, ratios, shading='auto')
    ax_buffer.axvline(x=true_values[0], color='r', linestyle='--', label='True Value')
    ax_buffer.axhline(y=true_values[1], color='r', linestyle='--')
    # ax_buffer.set_title(f"Event {idx}")
    ax_buffer.set_xlabel(parameter_names[0])
    ax_buffer.set_ylabel(parameter_names[1])
    #plt.colorbar(c, ax=ax_buffer, label='Posterior Density')

    ax_buffer.grid()
    return ax_buffer
PAGE_WIDTH_INCHES = 4.791
def main():
    parser = argparse.ArgumentParser(description="Plot posteriors from MBHB dataset.")
    parser.add_argument("filename", type=str, help="Path to dataset file")
    parser.add_argument("-n", "--num_events", type=int, default=10, help="Number of events to process")
    args = parser.parse_args()

    # Load dataset
    dataset = MBHBDataset(args.filename, transform_fd='log', device='cuda')
    N_events = min(args.num_events, len(dataset))
    dataset_to_compute = Subset(dataset, range(N_events))
    filenames_dict = {'q': glob("/u/g/gpuleo/pembhb/logs/20250917_093320_round_0/version_0/checkpoints/*.ckpt")[0], 
                      'Mc': glob("/u/g/gpuleo/pembhb/logs/20250920_162436_round_0/version_0/checkpoints/*.ckpt")[0],
                      'tc': glob("/u/g/gpuleo/pembhb/logs/20250919_141212_round_0/version_0/checkpoints/*.ckpt")[0],
                      'qMc': glob("/u/g/gpuleo/pembhb/logs/20250919_155210_round_0/version_0/checkpoints/*.ckpt")[0]}#mc, q 2d, 0.9 threshold
    xlabel_dict = {'q': 'mass ratio', 'Mc': r'$\log_{10}(\mathcal{M}_{\rm c}/M_\odot)$', 'tc': r'$\Delta t$ (days)', 'lam': r'$\lambda$', 'beta': r'$\beta$'}
    param_idx_dict = {'q': 1, 'Mc': 0, 'tc': 10, 'qMc': [0,1], 'lam': 7, 'beta': 8}
    out_param_idx_dict = {'lam': 0, 'beta': 1}

    ######## UNCOMMENT THE FOLLOWING CODE IF YOU WANT JUST ONE PARAMETER POSTERIORS. 
    # fname = "logs/20250919_133507_round_0/version_0/checkpoints/epoch=6-step=2975.ckpt" # tc model, 0.8 threshold
    #fname = glob("/u/g/gpuleo/pembhb/logs/20250917_093320_round_0/version_0/checkpoints/*.ckpt")[0]#mass ratio only
    # fname = glob("/u/g/gpuleo/pembhb/logs/20250919_141212_round_0/version_0/checkpoints/*.ckpt")[0]#tc only, 0.9 threshold
    # fname = glob("/u/g/gpuleo/pembhb/logs/20250919_155210_round_0/version_0/checkpoints/*.ckpt")[0]#mc, q 2d, 0.9 threshold
    fname = "/u/g/gpuleo/pembhb/logs/20251111_100948_round_0/version_0/checkpoints/epoch=13-step=2380.ckpt"

    # Compute logratios

    # Plot 1D posteriors
    trained_model = InferenceNetwork.load_from_checkpoint(fname)
    for param in ['lam', 'beta']:
        xlabel = xlabel_dict[param]
        inj_param_idx = param_idx_dict[param]   
        out_param_idx = out_param_idx_dict[param]
        print(f"Evaluating logratios: {param}, using model from {fname}")
        logratios, params, grid = utils.get_logratios_grid(dataset_to_compute, trained_model, ngrid_points=1000, in_param_idx=inj_param_idx, out_param_idx=out_param_idx)
        ratios = np.exp(logratios)
        for i in range(N_events):
            fig, ax = plt.subplots(figsize=(8, 5))
            plot_posterior_1d(grid, ratios[i], params[i], ax, xlabel, idx=i)
            plt.savefig(f"posterior_{param}_event_{i+1}.png")
            plt.close()
    
    ######## THIS CODE PLOTS BOTH 1D AND 2D POSTERIORS FOR THE PARAMETERS OF INTEREST. 
    # for param in ['q', 'Mc', 'tc', 'qMc']:
    #     fname = filenames_dict[param]
    #     trained_model = InferenceNetwork.load_from_checkpoint(fname)
    #     inj_param_idx = param_idx_dict[param]   
    #     print(f"Evaluating logratios: {param}, using model from {fname}")
    #     if param == 'qMc':
    #         logratios, params, grid_x, grid_y = utils.get_logratios_grid_2d(dataset_to_compute, trained_model, ngrid_points=100, in_param_idx=inj_param_idx, out_param_idx=0)
    #         ratios = np.exp(logratios)
    #         for i in range(N_events):
    #             fig, ax = plt.subplots(figsize=(PAGE_WIDTH_INCHES/2, PAGE_WIDTH_INCHES/2))
    #             plot_posterior_2d(grid_x.cpu().numpy(), grid_y.cpu().numpy(), ratios[i], params[i], ax, [xlabel_dict['Mc'], xlabel_dict['q']], idx=i)
    #             fig.savefig(f"plots/posterior_{param}_event_{i+1}.png", dpi=300, bbox_inches='tight')
    #             plt.close(fig)
    #     else:
    #         xlabel = xlabel_dict[param]
    #         logratios, params, grid = utils.get_logratios_grid(dataset_to_compute, trained_model, ngrid_points=1000, in_param_idx=inj_param_idx, out_param_idx=0)
    #         ratios = np.exp(logratios)
    #         for i in range(N_events):
    #             fig, ax = plt.subplots(figsize=(PAGE_WIDTH_INCHES/2, PAGE_WIDTH_INCHES/2))
    #             plot_posterior_1d(grid, ratios[i], params[i], ax, xlabel, idx=i)
    #             fig.savefig(f"plots/posterior_{param}_event_{i+1}.png", dpi=300, bbox_inches='tight')
    #             plt.close(fig)

if __name__ == "__main__":
    main()