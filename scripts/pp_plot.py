import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import yaml
import torch
from pembhb import ROOT_DIR
from pembhb import utils
from pembhb.model import InferenceNetwork,PeregrineModel
from pembhb.sampler import UniformSampler
from pembhb.simulator import LISAMBHBSimulator
from pembhb.data import MBHBDataset
from pembhb.utils import get_pvalues_1d
from torch.utils.data import DataLoader
from glob import glob

# def load_wronglysaved_model():
#     model_checkpoint = torch.load("/u/g/gpuleo/pembhb/logs_0725/peregrine_norm/version_8/checkpoints/epoch=960-step=43245.ckpt")
#     config_path = os.path.join(ROOT_DIR, "config.yaml")
#     # load conf and init untrained model
#     with open(config_path, "r") as file:
#         conf = yaml.safe_load(file)
#     model = PeregrineModel(conf=conf)

#     # engineer the checkpoint to load the model state dict even if it was ignored
#     x = model_checkpoint["state_dict"]
#     from collections import OrderedDict
#     y = OrderedDict()
#     for key in x.keys():
#         target_key = key.removeprefix("model.")
#         y[target_key] = x[key]
#     model_checkpoint["state_dict"] = y
#     model.load_state_dict(state_dict=model_checkpoint["state_dict"])
#     lr = model_checkpoint["hyper_parameters"]["lr"]
#     trained_model = InferenceNetwork(lr=lr, classifier_model=model)
#     return trained_model 
def plot_posterior(data: np.array, injected_params: np.array, model: InferenceNetwork):

    N_examples = data.shape[0]
    print(data.shape)
    marginals = model.marginals
    N_marginals = len(model.marginals)
    assert N_examples==1,  "N_examples must be less than 10 to limit the number of plots."
    # generate samples from each prior
    sampler = UniformSampler(model.bounds_trained)
    Nsamples = 1000
    _, samples = sampler.sample(n_samples=Nsamples)
    samples= samples.T
    samples_torch =  torch.tensor(samples, device='cuda', dtype=torch.float32)
    data_torch = torch.tensor(data,device='cuda').expand(Nsamples, -1,-1)
    print(samples_torch.dtype, data_torch.dtype)

    logratios = model(data_torch, samples_torch)


    for i in range(N_examples):
        for j in range(N_marginals):
            current_marginal = marginals[j]
            # print(current_marginal, type(current_marginal))
            current_param = injected_params[current_marginal]
            Ndim = len(current_marginal)
            assert Ndim < 3, "unable to plot marginal with >=3 dimensions"
            logr = logratios[:,j]    
            posterior_weight = np.exp(-logr.detach().cpu().numpy())
            if Ndim == 1: 
                print(samples[:,current_marginal].squeeze(1).shape, posterior_weight.shape)
                plt.hist(samples[:,current_marginal].squeeze(1), weights=posterior_weight, histtype='step', label='approximate posterior', bins=100)
                plt.axvline(x=current_param, color='red',  label='true value')
                plt.title(f"marginal {current_marginal}")
                plt.legend()
                plt.xlabel(f"value")
                plt.ylabel(f"pdf")
                plt.savefig(f"marginal_{i}_{j}")
                plt.close()
            if Ndim == 2: 
                print(samples[:,current_marginal].shape, posterior_weight.shape)
                plt.hist2d(x=samples[:,current_marginal][:,0], y=samples[:,current_marginal][:,1], weights=posterior_weight, bins=100)
                plt.scatter(x=current_param[0], y=current_param[1], marker='*', color='red')
                plt.title(f"marginal {current_marginal}")
                plt.xlabel(f"value_A")
                plt.ylabel(f"value_B")
                plt.savefig(f"marginal_{i}_{j}")
                plt.close()


def plot_posterior_grid(data: np.array, injected_params: np.array, model: InferenceNetwork, idx: int ,  ngrid_points=100):

    plt.plot(lmc_grid.cpu().numpy().flatten(), np.exp(logratios_mchirp.numpy()), label="approximate posterior")
    plt.axvline(x=injected_params[0], color='red', label='true value')
    plt.title("Posterior for mchirp")
    plt.xlabel(r"$\log_{10}(\mathrm{chirp\ mass}/M_\odot)$")
    plt.ylabel("unnormalised pdf")
    plt.legend()
    plt.savefig(os.path.join(ROOT_DIR, "plots",f"test_event_{idx}_posterior_mchirp.png"))
    plt.close()

    plt.plot(q_grid.cpu().numpy().flatten(), np.exp(logratios_q.numpy()), label="approximate posterior")
    plt.axvline(x=injected_params[1], color='red', label='true value')
    plt.title("Posterior for q")
    plt.xlabel(r"$\mathrm{q}$")
    plt.ylabel("unnormalised pdf")
    plt.legend()
    plt.savefig(os.path.join(ROOT_DIR, "plots", f"test_event_{idx}_posterior_q.png"))
    plt.close()
    # del data_fd_torch, lmc_grid_padded, q_grid_padded
    # torch.cuda.empty_cache()
if __name__ == "__main__":

    dataset = MBHBDataset(os.path.join(ROOT_DIR, "data", "test_examples_seed314.h5"), transform_fd='log', transform_td='normalise_max', device='cuda')

    # for i in range(20):
    #     datum = dataset.__getitem__(i)

    #     data_fd = datum["data_fd"][np.newaxis, :,:]
    #     source_par = datum["source_parameters"]
    fname = "logs/20250919_133507_round_0/version_0/checkpoints/epoch=6-step=2975.ckpt" # tc model, 0.8 threshold
    fname = glob("/u/g/gpuleo/pembhb/logs/20250917_093320_round_0/version_0/checkpoints/*.ckpt")[0]#mass ratio only
    #fname = glob("/u/g/gpuleo/pembhb/logs/20250919_141212_round_0/version_0/checkpoints/*.ckpt")[0]#tc only, 0.9 threshold

    #conf = utils.read_config(os.path.join(ROOT_DIR,"config_td.yaml"))
    model = InferenceNetwork.load_from_checkpoint(fname)
    model.eval()

    #     plot_posterior_grid(data_fd, source_par , model, i)
    #logratios, injection_params, grid = utils.get_logratios_grid(dataset, model, low=-3, high=-1, ngrid_points=1000, in_param_idx=10, out_param_idx=0)
    logratios, injection_params, grid = utils.get_logratios_grid(dataset, model, low=1, high=5, ngrid_points=1000, in_param_idx=1, out_param_idx=0)
    p_values = get_pvalues_1d(logratios, grid, injection_params)
    sorted_pvalues = np.sort(p_values)
    sorted_rank = np.arange(sorted_pvalues.shape[0])
    sorted_rank_normalised = sorted_rank / sorted_rank.max()
    fig, ax  = plt.subplots(figsize=(10, 6))
    ax.plot(sorted_pvalues, sorted_rank_normalised , marker='o', linestyle='-', markersize=3)
    ax.set_xlabel('credibility level (HPD)')
    ax.set_ylabel('empirical coverage')
    ax.set_title('p-p plot for mass ratio')
    ax.grid(visible=True)
    fig.savefig(os.path.join(ROOT_DIR, "plots", "pvalues_plot_q.png"))
    plt.close()