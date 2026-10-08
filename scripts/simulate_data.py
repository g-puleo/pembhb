"""Simulate N MBHB signals with bbhx and store them to HDF5 (+ YAML sidecar).

Typical use — a single noisy observation at the configured injection:

    python scripts/simulate_data.py --n 1 --injection --store-noise \
        --fname /path/to/obs.h5 --seed 0
"""
import argparse
import os

import yaml

from pembhb import ROOT_DIR
from pembhb.simulator import MBHBSimulatorFD

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--n", type=int, default=1, help="number of samples")
parser.add_argument("--fname", required=True, help="output HDF5 path")
parser.add_argument("-s", "--seed", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=None)
parser.add_argument("--datagen_config", default=os.path.join(ROOT_DIR, "configs", "datagen_config.yaml"))
parser.add_argument("--injection", action="store_true",
                    help="pin every parameter to the config's `injection` block instead of sampling the prior")
parser.add_argument("--store-noise", action="store_true",
                    help="store a fixed noise realisation (noise_fd); required for the --obs-path of tmnre_joint.py")
parser.add_argument("--noise-seed", type=int, default=0)
args = parser.parse_args()

with open(os.path.join(ROOT_DIR, args.datagen_config), "r") as file:
    conf = yaml.safe_load(file)

prior = conf["prior"]
if args.injection:
    inj = conf["injection"]
    assert set(inj) == set(prior), f"injection keys {sorted(inj)} != prior keys {sorted(prior)}"
    prior = {k: [float(inj[k]), float(inj[k])] for k in prior}
    conf["prior"] = prior

sampler_init_kwargs = {"prior_bounds": prior,
                       "spin_param_basis": conf.get("spin_param_basis", "chi1chi2")}
wp = conf["waveform_params"]
sim = MBHBSimulatorFD(conf, sampler_init_kwargs=sampler_init_kwargs, seed=args.seed,
                      n_freq_bins=wp.get("n_freq_bins", 4096), freq_spacing=wp.get("freq_spacing", "linear"))
sim.sample_and_store(filename=args.fname, N=args.n, batch_size=args.batch_size,
                     store_noise=args.store_noise, noise_seed=args.noise_seed)
