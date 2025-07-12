import os
from pembhb.simulator import LISAMBHBSimulator
from pembhb.model import InferenceNetwork
from pembhb.data import MBHBDataModule
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch import Trainer
import numpy as np
from pembhb import ROOT_DIR
import yaml 

def round(conf:dict, sampler_init_kwargs:dict = None):

    ######## DATA GENERATION #########
    sim = LISAMBHBSimulator(conf, sampler_init_kwargs=sampler_init_kwargs)
    fname = os.path.join(ROOT_DIR, "data", "simulated_data.h5")
    print("Sampling from the simulator...")
    os.makedirs(os.path.join(ROOT_DIR, "data"), exist_ok=True)
    try: 
        sim.sample_and_store(fname, N=1000, batch_size=100)
        print("Data saved to", fname)
    except ValueError:
        pass


    ######## DATA LOADING AND TRAINING THE MODEL #########
    data_module = MBHBDataModule(
        filename=fname,
        targets=['data_fd', 'source_parameters'],
        batch_size=100
    )

    logger = TensorBoardLogger(os.path.join(ROOT_DIR, "logs"), name="tmnre")
    trainer = Trainer(logger=logger)
    model = InferenceNetwork(num_features = 10000, hlayersizes=(500,20), lr=1e-3)
    trainer.fit(model, data_module)
    trainer.save_checkpoint(os.path.join(ROOT_DIR, "checkpoints", "tmnre_model.ckpt"))
    prior_samples = sim.sampler.sample(N = 10000)
    obs = sim._sample(N = 1)
    np.save(os.path.join(ROOT_DIR, "data", "observation_params.npy"), obs[0])
    np.save(os.path.join(ROOT_DIR, "data", "observation_data.npy"), obs[1])
    predictions = model({'data_fd': obs[1], 'source_parameters': obs[0]})
    return predictions


if __name__ == "__main__":

    # Load configuration from config.yaml
    config_path = os.path.join(ROOT_DIR, "config.yaml")
    with open(config_path, "r") as file:
        conf = yaml.safe_load(file)

    predictions = round(conf, sampler_init_kwargs={'prior_bounds': conf["prior"]})