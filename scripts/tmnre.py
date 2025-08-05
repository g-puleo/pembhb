import os
import torch
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
from datetime import datetime
import matplotlib.pyplot as plt

torch.set_float32_matmul_precision("medium")
def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")
TIME_OF_EXECUTION = get_timestamp()
def round(conf:dict, sampler_init_kwargs:dict, lr:float, idx:int=0):

    ######## DATA GENERATION #########
    sim = LISAMBHBSimulator(conf, sampler_init_kwargs=sampler_init_kwargs)
    #sim = DummySimulator(sampler_init_kwargs=sampler_init_kwargs)
    fname = os.path.join(ROOT_DIR, "data", f"restricted_simulated_data_round_{idx}.h5")
    print("Sampling from the simulator...")
    os.makedirs(os.path.join(ROOT_DIR, "data"), exist_ok=True)
    try: 
        sim.sample_and_store(fname, N=21000, batch_size=200)
        print("Data saved to", fname)
    except ValueError:
        print("File might already exisht, skipping sampling.")
        


    ######## DATA LOADING AND TRAINING THE MODEL #########


    checkpoint_callback = ModelCheckpoint(monitor='val_loss', mode='min')
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=100, mode='min')
    logger = TensorBoardLogger(os.path.join(ROOT_DIR, f"logs/{TIME_OF_EXECUTION}"), name=f"round_{idx}")
    trainer = Trainer(
                    logger=logger,
                    max_epochs=conf["training"]["epochs"], 
                    accelerator="gpu", devices=1,
                    enable_progress_bar=True, 
                    callbacks=[checkpoint_callback, early_stopping_callback]
                    )
    model = InferenceNetwork(conf)
    data_module = MBHBDataModule(fname, conf)
    #model = InferenceNetwork(num_features=10, num_channels=6, hlayersizes=(100, 20), marginals=conf["tmnre"]["marginals"], marginal_hidden_size=10, lr=lr)
    trainer.fit(model, data_module)
    test_dataset = data_module.test
    #trainer.save_checkpoint(os.path.join(ROOT_DIR, "checkpoints", f"tmnre_model_{idx}.ckpt"))
    return model, test_dataset


if __name__ == "__main__":  

    # Load configuration from config.yaml
    config_path = os.path.join(ROOT_DIR, "config.yaml")
    conf = read_config(config_path)

    dataset_observation = MBHBDataset(os.path.join(ROOT_DIR, "data/observation.h5"), conf["training"]["transform"])
    # load observation

    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].set_title("logMchirp Prior Bounds")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("logMchirp")
    axes[1].set_title("q Prior Bounds")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("q")

    # Initialize lists to store prior bounds for plotting
    logMchirp_lower = [conf["prior"]["logMchirp"][0]]
    logMchirp_upper = [conf["prior"]["logMchirp"][1]]
    q_lower = [conf["prior"]["q"][0]]
    q_upper = [conf["prior"]["q"][1]]

    for i in range(1): 
        print(f"Running round {i}...")
        # if i == 0: 
        #     trained_model = InferenceNetwork.load_from_checkpoint("/u/g/gpuleo/pembhb/logs/logs_0804/peregrine_norm/version_1/checkpoints/epoch=136-step=23290.ckpt")
        #     datamodule = MBHBDataModule(os.path.join(ROOT_DIR, "data", f"simulated_data_round_{i}.h5"), conf=conf)
        #     datamodule.setup(stage='fit')
        #     test_dataset = datamodule.test
        # # elif i == 1: 
        # #     trained_model = InferenceNetwork.load_from_checkpoint("/u/g/gpuleo/pembhb/logs/20250804_125709/round_1/version_0/checkpoints/epoch=36-step=3330.ckpt")
        # #     datamodule = MBHBDataModule(os.path.join(ROOT_DIR, "data", f"simulated_data_round_{i}.h5"), conf=conf)
        # #     datamodule.setup(stage='fit')
        # #     test_dataset = datamodule.test
        # else:
        trained_model, test_dataset = round(conf, sampler_init_kwargs={'prior_bounds': conf["prior"]}, lr=conf["training"]["learning_rate"], idx=i)
        
        pp_plot(test_dataset, trained_model, low=conf["prior"]["logMchirp"][0], high=conf["prior"]["logMchirp"][1], inj_param_idx=0, name=f"round_{i}_logMchirp")
        pp_plot(test_dataset, trained_model, low=conf["prior"]["q"][0], high=conf["prior"]["q"][1], inj_param_idx=1, name=f"round_{i}_q")

        updated_prior = update_bounds(trained_model, dataset_observation, conf["prior"], parameter_idx=0, n_gridpoints=100)
        updated_prior = update_bounds(trained_model, dataset_observation, updated_prior, parameter_idx=1, n_gridpoints=100)
        print(f"Updated prior after round {i}:\nlog10Mchirp: {updated_prior['logMchirp']},\nmass ratio: {updated_prior['q']}\n")
        conf["prior"] = updated_prior
        
        # Update the lists with new prior bounds
        logMchirp_lower.append(updated_prior["logMchirp"][0])
        logMchirp_upper.append(updated_prior["logMchirp"][1])
        q_lower.append(updated_prior["q"][0])
        q_upper.append(updated_prior["q"][1])

        # Clear the axes before updating the plots
        axes[0].cla()
        axes[1].cla()

        # Update the titles and labels
        axes[0].set_title("logMchirp Prior Bounds")
        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel("logMchirp")
        axes[1].set_title("q Prior Bounds")
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("q")

        # Update the plots
        axes[0].plot(range(len(logMchirp_lower)), logMchirp_lower, label="Lower Bound", color="blue")
        axes[0].plot(range(len(logMchirp_upper)), logMchirp_upper, label="Upper Bound", color="orange")
        axes[1].plot(range(len(q_lower)), q_lower, label="Lower Bound", color="blue")
        axes[1].plot(range(len(q_upper)), q_upper, label="Upper Bound", color="orange")

        # Add legends
        axes[0].legend()
        axes[1].legend()

        # Save the updated figure
        plt.tight_layout()
        plt.savefig(os.path.join(ROOT_DIR, f"plots/prior_bounds_iteration_{i}.png"))

        # Free CUDA memory
        del trained_model
        torch.cuda.empty_cache()

    #round(conf, sampler_init_kwargs={'low': 0.5, 'high': 1.0} , lr=conf["training"]["learning_rate"], idx=0)