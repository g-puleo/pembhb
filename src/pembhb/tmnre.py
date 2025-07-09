import os
from pembhb.simulator import LISAMBHBSimulator
from pembhb.model import InferenceNetwork
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch import Trainer
import numpy as np
from pembhb import ROOT_DIR
import yaml 
def round(conf:dict):

    sim = LISAMBHBSimulator(conf)
    
    print("Sampling from the simulator...")
    samples = sim.sample(100)
    logger = TensorBoardLogger(os.path.join(ROOT_DIR, "logs"), name="tmnre")
    trainer = Trainer()
    model = InferenceNetwork(num_features = 10000, hlayersizes=(500,20), lr=1e-3)
    trainer.fit(model, dm)
    trainer.save_checkpoint(os.path.join(ROOT_DIR, "checkpoints", "tmnre_model.ckpt"))
    prior_samples = sim.sample(N = 10000, targets = ['z_tot'])
    obs = sim.sample(N = 1 , targets = ['data_fd'])
    np.save(os.path.join(ROOT_DIR, "data", "observation_params.npy"), obs['z_tot'])
    np.save(os.path.join(ROOT_DIR, "data", "observation_data.npy"), obs['data_fd'])
    predictions = trainer.infer( model , obs , prior_samples)
    print("New bounds:", new_bounds)
    return predictions, new_bounds, samples


if __name__ == "__main__":

    # Load configuration from config.yaml
    config_path = os.path.join(ROOT_DIR, "config.yaml")
    with open(config_path, "r") as file:
        conf = yaml.safe_load(file)

    predictions, new_bounds, samples = round(conf)