import argparse, os
from pembhb import utils, ROOT_DIR
from pembhb.data import MBHBDataset, mbhb_collate_fn
from pembhb.model import InferenceNetwork
from glob import glob
from torch.utils.data import Subset, DataLoader
import numpy as np
import  matplotlib.pyplot as plt
from pembhb.utils import plot_posterior_1d, plot_posterior_2d
from pembhb.import_utils import import_model

PAGE_WIDTH_INCHES = 4.791
def main():
    parser = argparse.ArgumentParser(description="Plot posteriors from MBHB dataset.")
    parser.add_argument("filename", type=str, help="Path to dataset file")
    parser.add_argument("-n", "--num_events", type=int, default=10, help="Number of events to process")
    args = parser.parse_args()
    dataset = MBHBDataset(args.filename, transform_fd='log', device='cpu')
    N_events = min(args.num_events, len(dataset))
    dataset_to_compute = Subset(dataset, range(N_events))
    # filenames_dict = {'q': glob("/u/g/gpuleo/pembhb/logs/20250917_093320_round_0/version_0/checkpoints/*.ckpt")[0], 
    #                   'Mc': glob("/u/g/gpuleo/pembhb/logs/20250920_162436_round_0/version_0/checkpoints/*.ckpt")[0],
    #                   'tc': glob("/u/g/gpuleo/pembhb/logs/20250919_141212_round_0/version_0/checkpoints/*.ckpt")[0],
    #                   'qMc': glob("/u/g/gpuleo/pembhb/logs/20250919_155210_round_0/version_0/checkpoints/*.ckpt")[0]}#mc, q 2d, 0.9 threshold
    xlabel_dict = {'q': 'mass ratio', 'Mc': r'$\log_{10}(\mathcal{M}_{\rm c}/M_\odot)$', 'tc': r'$\Delta t$ (days)', 'lam': r'$\lambda$', 'beta': r'$\beta$'}
    param_idx_dict = {'q': 1, 'Mc': 0, 'tc': 10, 'qMc': [0,1], 'lam': 7, 'beta': 8}
    out_param_idx_dict = {'Mc': 0, 'q': 1, 'qMc': 0}

    ######## UNCOMMENT THE FOLLOWING CODE IF YOU WANT JUST ONE PARAMETER POSTERIORS. 
    # fname = "logs/20250919_133507_round_0/version_0/checkpoints/epoch=6-step=2975.ckpt" # tc model, 0.8 threshold
    #fname = glob("/u/g/gpuleo/pembhb/logs/20250917_093320_round_0/version_0/checkpoints/*.ckpt")[0]#mass ratio only
    # fname = glob("/u/g/gpuleo/pembhb/logs/20250919_141212_round_0/version_0/checkpoints/*.ckpt")[0]#tc only, 0.9 threshold
    # fname = glob("/u/g/gpuleo/pembhb/logs/20250919_155210_round_0/version_0/checkpoints/*.ckpt")[0]#mc, q 2d, 0.9 threshold
    #fname = "/u/g/gpuleo/pembhb/logs/20251111_100948_round_0/version_0/checkpoints/epoch=13-step=2380.ckpt"
    # fname = "/u/g/gpuleo/pembhb/logs/20251112_164554_round_0/version_0/checkpoints/epoch=24-step=4250.ckpt"
    # fname = "/u/g/gpuleo/pembhb/logs/20251117_093429_round_0/version_0/checkpoints/epoch=121-step=20740.ckpt"
    #timestamp = "20251128_160310"
    timestamp = "20251205_100028"

    # Compute logratios

    # Plot 1D posteriors
    trained_model = import_model(timestamp)
    # for param in ['Mc', 'q']:
    #     xlabel = xlabel_dict[param]
    #     inj_param_idx = param_idx_dict[param]   
    #     out_param_idx = out_param_idx_dict[param]
    #     print(f"Evaluating logratios: {param}, using model from {fname}")
    #     logratios, params, grid = utils.get_logratios_grid(dataset_to_compute, trained_model, ngrid_points=1000, in_param_idx=inj_param_idx, out_param_idx=out_param_idx)
    #     ratios = np.exp(logratios)
    #     for i in range(N_events):
    #         fig, ax = plt.subplots(figsize=(8, 5))
    #         plot_posterior_1d(grid, ratios[i], params[i], ax, xlabel, idx=i)
    #         plt.savefig(f"posterior_{param}_event_{i+1}.png")
    #         plt.close()
    
    ####### THIS CODE PLOTS BOTH 1D AND 2D POSTERIORS FOR THE PARAMETERS OF INTEREST. 
        # trained_model = InferenceNetwork.load_from_checkpoint(fname)
    dataloader = DataLoader(dataset_to_compute, batch_size=1, shuffle=False, collate_fn=lambda b: mbhb_collate_fn(b, dataset_to_compute))

    # logratios_q , params_q , grid_q = utils.get_logratios_grid(dataloader, trained_model, ngrid_points=1000, in_param_idx=param_idx_dict['q'], out_param_idx=out_param_idx_dict['q'])
    # logratios_Mc, params_Mc, grid_Mc = utils.get_logratios_grid(dataloader, trained_model, ngrid_points=1000, in_param_idx=param_idx_dict['Mc'], out_param_idx=out_param_idx_dict['Mc'])
    N_grid_points = 100
    logratios_qMc, params_qMc, grid_x, grid_y = utils.get_logratios_grid_2d(dataloader, trained_model, ngrid_points=N_grid_points, in_param_idx=param_idx_dict['qMc'], out_param_idx=out_param_idx_dict['qMc'])

    # ratios_q = np.exp(logratios_q)
    # ratios_Mc = np.exp(logratios_Mc)

    # dq_1d = grid_q[1]-grid_q[0] # assuming uniform spacing
    # dMc_1d = grid_Mc[1]-grid_Mc[0]
    # normalised_ratios_q = ratios_q / np.sum(ratios_q * dq_1d, axis=1, keepdims=True)
    # normalised_ratios_Mc = ratios_Mc / np.sum(ratios_Mc * dMc_1d, axis=1, keepdims=True)
    ratios_qMc = np.exp(logratios_qMc) # shape (N_events, ngrid, ngrid)

    dq = (grid_y[0,1]-grid_y[0,0]) # assuming uniform spacing
    dMc = (grid_x[1,0]-grid_x[0,0])
    normalised_ratios_qmc = ratios_qMc / np.sum(ratios_qMc* dq * dMc, axis=(1,2), keepdims=True)
    marginalised_q_from2d = np.sum(normalised_ratios_qmc*dMc, axis=1)
    marginalised_Mc_from2d = np.sum(normalised_ratios_qmc*dq, axis=2)

    grid_mc_from2d = grid_x[:, 0].reshape(-1)
    grid_q_from2d = grid_y[0, :].reshape(-1)
    for i in range(N_events):
        fig, ax = plt.subplots(1,1,figsize=(1.3*PAGE_WIDTH_INCHES/2, PAGE_WIDTH_INCHES/2))
        ax.set_facecolor('black')
        fig.patch.set_facecolor('black')

        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        # keep only left and bottom spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # style remaining spines
        ax.spines['left'].set_color('white')
        ax.spines['bottom'].set_color('white')

        # ticks only on left/bottom
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        for spine in ax.spines.values():
            spine.set_color('white')
        fig.tight_layout()
        # breakpoint()
        # plot_posterior_1d(grid_Mc, normalised_ratios_Mc[i], params_Mc[i], ax[0], xlabel_dict['Mc'], idx=i, color='blue', label='From 1D NRE')
        # plot_posterior_1d(grid_q, normalised_ratios_q[i], params_q[i], ax[1], xlabel_dict['q'], idx=i, color='blue', label='From 1D NRE')
        # plot_posterior_1d(grid_mc_from2d, marginalised_Mc_from2d[i], params_qMc[i][0], ax[0], xlabel_dict['Mc'], idx=i, color='orange', linestyle='--', label='From 2D NRE')
        # plot_posterior_1d(grid_q_from2d, marginalised_q_from2d[i], params_qMc[i][1], ax[1], xlabel_dict['q'], idx=i, color='orange', linestyle='--', label='From 2D NRE')
        plot_posterior_2d(grid_x, grid_y, normalised_ratios_qmc[i], params_qMc[i], ax, [xlabel_dict['Mc'], xlabel_dict['q']], title=f"Neural Posterior")
        cax = fig.axes[-1]   # colorbar axis
        cax.tick_params(colors='white')

        for spine in cax.spines.values():
            spine.set_color('white')
        ax.legend()
        #ax[1].legend()
        # also plot the true likelihood for the event 
        #out = np.load("plots/likelihood_values_fullgrid.npy")
        #Z = out.reshape((N_grid_points, N_grid_points))
        #Z-= np.max(Z)  # for numerical stability
        #plot_posterior_2d(grid_x, grid_y, np.exp(Z-np.max(Z)),  params_qMc[i],  ax[1], [r"$\log_{10}(\mathcal{M}_c/M_{\odot})$", r"$q$"], title="true loglikelihood")
        fig.savefig(f"plots/{timestamp}/posterior1d2d_exponential_fixallparams_event_{i+1}.png", bbox_inches='tight', dpi=300)
if __name__ == "__main__":
    main()