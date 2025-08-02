import os
from pembhb.simulator import LISAMBHBSimulator, DummySimulator
from pembhb.model import InferenceNetwork, PeregrineModel
from pembhb.data import MBHBDataModule, MBHBDataset
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader , random_split
import numpy as np
from pembhb import ROOT_DIR
from pembhb.utils import read_config, update_bounds, pp_plot
import yaml 

def round(conf:dict, sampler_init_kwargs:dict, lr:float, idx:int=0):

    ######## DATA GENERATION #########
    sim = LISAMBHBSimulator(conf, sampler_init_kwargs=sampler_init_kwargs)
    #sim = DummySimulator(sampler_init_kwargs=sampler_init_kwargs)
    fname = os.path.join(ROOT_DIR, "data", f"simulated_data_round_{idx}.h5")
    print("Sampling from the simulator...")
    os.makedirs(os.path.join(ROOT_DIR, "data"), exist_ok=True)
    try: 
        sim.sample_and_store(fname, N=10500, batch_size=200)
        print("Data saved to", fname)
    except ValueError:
        print("File might already exist, skipping sampling.")
        


    ######## DATA LOADING AND TRAINING THE MODEL #########


    checkpoint_callback = ModelCheckpoint(monitor='val_loss', mode='min')
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=15, mode='min')
    logger = TensorBoardLogger(os.path.join(ROOT_DIR, f"logs_0801"), name=f"peregrine_norm")
    trainer = Trainer(
                    logger=logger,
                    max_epochs=conf["training"]["epochs"], 
                    accelerator="gpu", devices=1,
                    enable_progress_bar=True, 
                    callbacks=[checkpoint_callback, early_stopping_callback]
                    )
    model = InferenceNetwork(conf)
    data_module = MBHBDataModule(fname, conf)
    test_dataset = data_module.test
    #model = InferenceNetwork(num_features=10, num_channels=6, hlayersizes=(100, 20), marginals=conf["tmnre"]["marginals"], marginal_hidden_size=10, lr=lr)
    trainer.fit(model, data_module)
    #trainer.save_checkpoint(os.path.join(ROOT_DIR, "checkpoints", f"tmnre_model_{idx}.ckpt"))
    return model, test_dataset


if __name__ == "__main__":  

    # Load configuration from config.yaml
    config_path = os.path.join(ROOT_DIR, "config.yaml")
    conf = read_config(config_path)

    dataset_observation = MBHBDataset(os.path.join(ROOT_DIR, "data/observation.h5"))
    # load observation
    for i in range(1): 
        trained_model, test_dataset = round(conf, sampler_init_kwargs={'prior_bounds': conf["prior"]}, lr=conf["training"]["learning_rate"], idx=0)
        pp_plot(test_dataset, trained_model, low=conf["prior"]["logMchirp"][0], high=conf["prior"]["logMchirp"][1], inj_param_idx=0, name=f"round_{i}_logMchirp")
        pp_plot(test_dataset, trained_model, low=conf["prior"]["q"][0], high=conf["prior"]["q"][1], inj_param_idx=1, name=f"round_{i}_q")

        updated_prior = update_bounds(trained_model, dataset_observation, conf["prior"], parameter_idx=0, n_gridpoints=100)
        updated_prior = update_bounds(trained_model, dataset_observation, updated_prior, parameter_idx=1, n_gridpoints=100)
        print(f"Updated prior after round {i}:\nlog10Mchirp: {updated_prior["logMchirp"]},\nmass ratio: {updated_prior["q"]}\n")
        conf["prior"] = updated_prior
        
    #round(conf, sampler_init_kwargs={'low': 0.5, 'high': 1.0} , lr=conf["training"]["learning_rate"], idx=0)