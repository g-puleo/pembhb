import argparse
import yaml
import os
from pembhb import ROOT_DIR
from pembhb.simulator import MBHBSimulatorFD

parser = argparse.ArgumentParser(
            prog='Simulate N data and store the output into file')

parser.add_argument("--n", default=1)
parser.add_argument("--fname", required=True)
parser.add_argument("-s", "--seed", required=True)
parser.add_argument("--batch_size", default=None)
config_path = os.path.join(ROOT_DIR, "configs/datagen_config.yaml")
with open(config_path, "r") as file:
    conf = yaml.safe_load(file) 
args = parser.parse_args()
print(type(args.n))
sampler_init_kwargs={'prior_bounds': conf["prior"]}
sim = MBHBSimulatorFD(conf, sampler_init_kwargs=sampler_init_kwargs, seed=int(args.seed), n_freq_bins=conf["waveform_params"]["n_freq_bins"], freq_spacing=conf["waveform_params"]["freq_spacing"])
sim.sample_and_store(filename=args.fname, N=int(args.n), batch_size=int(args.batch_size) if args.batch_size is not None else None)
