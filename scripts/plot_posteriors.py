import argparse
from pembhb import utils
from pembhb.data import MBHBDataset
from pembhb.model import InferenceNetwork
from glob import glob
import numpy as np

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Plot posteriors from MBHB dataset.")
    parser.add_argument("filename", type=str, help="Path to dataset file")
    parser.add_argument("-n", "--num_events", type=int, default=10, help="Number of events to process")
    args = parser.parse_args()

    # Load dataset
    dataset = MBHBDataset(args.filename, transform_fd='log', transform_td='normalise_max', device='cuda')
    model_fname = "/u/g/gpuleo/pembhb/logs/20250919_133507_round_0/version_0/checkpoints/epoch=6-step=2975.ckpt"
    trained_model = InferenceNetwork.load_from_checkpoint(model_fname, conf=utils.read_config("config_td.yaml"))
    logratios, params, grid = utils.get_logratios_grid(dataset, trained_model, -3,-1, ngrid_points=1000, in_param_idx=10, out_param_idx=0)
    
    ratios = np.exp(logratios)
    
    print(f"ratios shape: {ratios.shape}")
    print(f"params shape: {params.shape}")
    # Compute logratios

    # Plot posteriors
    for i in range(min(args.num_events, len(dataset))):
        plt.figure(figsize=(8, 5))
        plt.plot(grid, ratios[i], label='NRE')
        plt.axvline(x=params[i], color='r', linestyle='--', label='True Value')
        plt.title(f"Posterior for Event {i+1}")
        plt.xlabel("mass ratio")
        plt.ylabel("Posterior Density (unnormalised)")
        plt.grid()
        plt.savefig(f"posterior_tc_event_{i+1}.png")
        plt.close()

if __name__ == "__main__":
    main()