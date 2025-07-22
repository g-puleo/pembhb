import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import yaml
from pembhb import ROOT_DIR
from pembhb.model import InferenceNetwork
from pembhb.sampler import UniformSampler
from pembhb.simulator import LISAMBHBSimulator
from pembhb.data import MBHBDataset
def plot_posterior(data: np.array, injected_params: np.array, model: InferenceNetwork):

    N_examples = data.shape[0]
    marginals = model.marginals
    N_marginals = len(model.marginals)
    assert N_examples < 10, "N_examples must be less than 10 to limit the number of plots."
    # generate samples from each prior
    sampler = UniformSampler(model.bounds_trained)
    _, samples = sampler.sample(n_samples=10000)
    logratios = model(data, samples)


    for i in range(N_examples):
        logratios_this = logratios[i]
        inj_par_this = injected_params[i]
        for j in range(N_marginals):
            current_marginal = marginals[j]
            current_param = inj_par_this[current_marginal]
            Ndim = len(current_marginal)
            assert Ndim < 3, "unable to plot marginal with >=3 dimensions"
            logr = logratios_this[j]    
            posterior_weight = np.exp(-logr)
            if Ndim == 1: 
                plt.hist(samples[:,current_marginal], weights=posterior_weight)
                plt.axvline(x=current_param[0])
                plt.title(f"marginal {current_marginal}")
                plt.xlabel(f"value")
                plt.ylabel(f"approximate posterior")
                plt.savefig(f"marginal_{i}_{j}")
            if Ndim == 2: 
                plt.hist(samples[:,current_marginal], weights=posterior_weight)
                plt.scatter(x=current_param[0], y=current_param[1])
                plt.title(f"marginal {current_marginal}")
                plt.xlabel(f"value_A")
                plt.ylabel(f"value_B")
                plt.savefig(f"marginal_{i}_{j}")

            

if __name__ == "__main__":

    dataset = MBHBDataset("example.h5")
    datum = dataset.__getitem__(0)
    model = InferenceNetwork.load_from_checkpoint("/u/g/gpuleo/pembhb/logs_0721/peregrine/version_1/checkpoints/epoch=369-step=1110.ckpt")
    
    plot_posterior(datum["data_fd"], datum["source_parameters"], model)