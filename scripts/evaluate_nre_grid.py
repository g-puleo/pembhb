import os 
import numpy as np
from glob import glob
from pembhb import ROOT_DIR, utils
from pembhb.model import InferenceNetwork
from pembhb.import_utils import import_model
from pembhb.data import MBHBDataset, mbhb_collate_fn
from torch.utils.data import Subset, DataLoader
from grid_evaluation_config import logmc_width, q_width
# parse dataset name from command line
import argparse
parser = argparse.ArgumentParser(description="Plot posteriors from MBHB dataset.")
parser.add_argument("filename", type=str, help="Path to dataset file")
parser.add_argument("-i", "--event_idx", type=int, default=0, help="Index of the event to process")
args = parser.parse_args()
# Load dataset
dataset = MBHBDataset(args.filename, cache_in_memory=True)
dataset_to_compute = Subset(dataset, [args.event_idx])
true_params = dataset.source_parameters[args.event_idx].cpu().numpy()
true_logmc = true_params[0]
true_q = true_params[1]
# logmc_max = true_logmc + logmc_width/2
# logmc_min = true_logmc - logmc_width/2
# q_max = true_q + q_width/2
# q_min = true_q - q_width/2

# logmc_min = 5.2663613230806785
# logmc_max = 5.266382694244385
# q_min = 4.6824813093
# q_max = 4.68416443136473

logmc_min = 5.2662 
logmc_max = 5.2665
q_min = 4.682
q_max = 4.6855
out_dir = os.path.join(ROOT_DIR, "plots", "npe_evaluation", f"event_{args.event_idx}")
os.makedirs(out_dir, exist_ok=True)
dataloader = DataLoader(dataset_to_compute, batch_size=1, shuffle=False, collate_fn=lambda b: mbhb_collate_fn(b, dataset_to_compute, noise_factor=1.0))
# import the model
timestamp = "20260204_narrowprior_v2"
trained_model = import_model(timestamp)
N_grid_points = 100
xlabel_dict = {'q': 'mass ratio', 'Mc': r'$\log_{10}(\mathcal{M}_{\rm c}/M_\odot)$', 'tc': r'$\Delta t$ (days)', 'lam': r'$\lambda$', 'beta': r'$\beta$'}
in_param_idx_dict = {'q': 1, 'Mc': 0, 'tc': 10, 'qMc': [0,1], 'lam': 7, 'beta': 8}
out_param_idx_dict = {'Mc': 0, 'q': 1, 'qMc': 0}

# compute logratios 
print("Computing logratios on grid...")
logratios_qMc, params_qMc, grid_x, grid_y = utils.get_logratios_grid_2d(dataloader, trained_model, ngrid_points=N_grid_points, in_param_idx=in_param_idx_dict['qMc'], out_param_idx=out_param_idx_dict['qMc'],
                                                                        bounds_0=[logmc_min, logmc_max], bounds_1=[q_min, q_max])
# save to npy 
np.save(os.path.join(out_dir, "logratios_qMc.npy"), logratios_qMc)
np.save(os.path.join(out_dir, "grid_x.npy"), grid_x)
np.save(os.path.join(out_dir, "grid_y.npy"), grid_y)
np.save(os.path.join(out_dir, "true_params.npy"), params_qMc)