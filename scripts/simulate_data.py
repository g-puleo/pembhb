import argparse
import yaml
import os
from pembhb import ROOT_DIR
from pembhb.simulator import LISAMBHBSimulator

parser = argparse.ArgumentParser(
            prog='Simulate N data and store the output into file')

parser.add_argument("--n", default=1)
parser.add_argument("--fname", required=True)

config_path = os.path.join(ROOT_DIR, "config.yaml")
with open(config_path, "r") as file:
    conf = yaml.safe_load(file) 
args = parser.parse_args()
print(type(args.n))
sampler_init_kwargs={'prior_bounds': conf["prior"]}
sim = LISAMBHBSimulator(conf, sampler_init_kwargs=sampler_init_kwargs)
sim.sample_and_store(filename=args.fname, N=int(args.n))
