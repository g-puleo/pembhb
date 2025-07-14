import os
from pembhb.simulator import LISAMBHBSimulator, DummySimulator
from pembhb.model import InferenceNetwork
from pembhb.data import MBHBDataModule
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch import Trainer
import numpy as np
from pembhb import ROOT_DIR
import yaml 

def round(conf:dict, sampler_init_kwargs:dict, lr:float, idx:int=0):

    ######## DATA GENERATION #########
    #sim = LISAMBHBSimulator(conf, sampler_init_kwargs=sampler_init_kwargs)
    sim = DummySimulator(sampler_init_kwargs=sampler_init_kwargs)
    fname = os.path.join(ROOT_DIR, "data", f"simulated_data_straightline.h5")
    print("Sampling from the simulator...")
    os.makedirs(os.path.join(ROOT_DIR, "data"), exist_ok=True)
    try: 
        sim.sample_and_store(fname, N=1000, batch_size=200)
        print("Data saved to", fname)
    except ValueError:
        print("File might already exist, skipping sampling.")
        pass


    ######## DATA LOADING AND TRAINING THE MODEL #########
    data_module = MBHBDataModule(
        filename=fname,
        targets=['data_fd', 'source_parameters'],
        batch_size=conf["training"]["batch_size"]
        )

    logger = TensorBoardLogger(os.path.join(ROOT_DIR, f"logs"), name=f"dummy_{idx}")
    trainer = Trainer(logger=logger, max_epochs=conf["training"]["epochs"], accelerator="gpu", devices=1, enable_progress_bar=True)
    model = InferenceNetwork(num_features=10, num_channels=6, hlayersizes=(50, 15, 5), lr=lr)
    trainer.fit(model, data_module)
    #trainer.save_checkpoint(os.path.join(ROOT_DIR, "checkpoints", f"tmnre_model_{idx}.ckpt"))
    return 


if __name__ == "__main__":  

    # Load configuration from config.yaml
    config_path = os.path.join(ROOT_DIR, "config.yaml")
    with open(config_path, "r") as file:
        conf = yaml.safe_load(file)

    #round(conf, sampler_init_kwargs={'prior_bounds': conf["prior"]}, lr=conf["training"]["learning_rate"], idx=0)
    round(conf, sampler_init_kwargs={'low': 0.5, 'high': 1.0}, lr=conf["training"]["learning_rate"], idx=0)