import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import yaml
import torch
from pembhb import ROOT_DIR
from pembhb.model import InferenceNetwork,PeregrineModel
from pembhb.sampler import UniformSampler
from pembhb.simulator import LISAMBHBSimulator
from pembhb.data import MBHBDataset

def load_wronglysaved_model():
    model_checkpoint = torch.load("/u/g/gpuleo/pembhb/logs_0723/peregrine_norm/version_1/checkpoints/epoch=844-step=38025.ckpt")
    config_path = os.path.join(ROOT_DIR, "config.yaml")
    # load conf and init untrained model
    with open(config_path, "r") as file:
        conf = yaml.safe_load(file)
    model = PeregrineModel(conf=conf)

    # engineer the checkpoint to load the model state dict even if it was ignored
    x = model_checkpoint["state_dict"]
    from collections import OrderedDict
    y = OrderedDict()
    for key in x.keys():
        target_key = key.removeprefix("model.")
        y[target_key] = x[key]
    model_checkpoint["state_dict"] = y
    model.load_state_dict(state_dict=model_checkpoint["state_dict"])
    lr = model_checkpoint["hyper_parameters"]["lr"]
    trained_model = InferenceNetwork(lr=lr, classifier_model=model)
    return trained_model 
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
            

if __name__ == "__main__":

    dataset = MBHBDataset("example.h5")
    datum = dataset.__getitem__(0)
    data_fd = datum["data_fd"][np.newaxis, :,:]
    source_par = datum["source_parameters"]
    model = load_wronglysaved_model().to("cuda")
    plot_posterior(data_fd, source_par , model)
