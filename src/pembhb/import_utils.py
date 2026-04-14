import yaml, os 
from pembhb.model import InferenceNetwork
from pembhb import ROOT_DIR
from glob import glob
def read_config(fname: str): 
    with open(fname, "r") as file:
        conf = yaml.safe_load(file)
    return conf

def import_model( timestamp: str ) : 
    fname = glob(f"/data/gpuleo/mbhb/logs/{timestamp}_round_2/version_0/checkpoints/*.ckpt")[0]
    trained_model = InferenceNetwork.load_from_checkpoint(fname)
    return trained_model
