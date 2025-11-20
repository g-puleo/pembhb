import os
import torch
from pembhb.simulator import MBHBSimulatorFD_TD, DummySimulator
from pembhb.model import InferenceNetwork, PeregrineModel
from pembhb.data import MBHBDataModule, MBHBDataset

from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.tuner import Tuner
from torch.utils.data import DataLoader , random_split
import numpy as np
from pembhb import ROOT_DIR
from pembhb.utils import read_config, update_bounds, pp_plot, pp_plot_2d, _ORDERED_PRIOR_KEYS
import argparse 
from datetime import datetime
import matplotlib.pyplot as plt

torch.set_float32_matmul_precision("medium")
def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")
TIME_OF_EXECUTION = get_timestamp()
class SequentialTrainer:
    def __init__(self, train_conf, datagen_conf, dataset_obs_path):
        self.train_conf = train_conf
        self.datagen_conf = datagen_conf
        self.dataset_obs_path = dataset_obs_path
        self.dataset_observation = MBHBDataset(dataset_obs_path, transform_fd=train_conf["transform_fd"])

        self.logMchirp_lower = [datagen_conf["prior"]["logMchirp"][0]]
        self.logMchirp_upper = [datagen_conf["prior"]["logMchirp"][1]]
        self.q_lower = [datagen_conf["prior"]["q"][0]]
        self.q_upper = [datagen_conf["prior"]["q"][1]]

        self._setup_plot()

    def _setup_plot(self):
        self.fig, self.axes = plt.subplots(1, 2, figsize=(12, 6))
        for ax, title, ylabel in zip(
            self.axes,
            ["logMchirp Prior Bounds", "q Prior Bounds"],
            ["logMchirp", "q"],
        ):
            ax.set_title(title)
            ax.set_xlabel("Iteration")
            ax.set_ylabel(ylabel)

    def round(self, idx, sampler_init_kwargs):
        
        sim = MBHBSimulatorFD_TD(self.datagen_conf, sampler_init_kwargs=sampler_init_kwargs)
        fname_base = "fix_all_notmcq_newdata"
        fname_h5 = os.path.join(ROOT_DIR, "data", f"{fname_base}.h5")
        data_fname_yaml = os.path.join(ROOT_DIR, "data", f"{fname_base}.yaml")
        datagen_info = read_config(data_fname_yaml)

        checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min")
        early_stopping_callback = EarlyStopping(
            monitor="val_accuracy",
            patience=self.train_conf["early_stop_patience"],
            mode="max",
            stopping_threshold=0.9,
        )
        logger = TensorBoardLogger(
            os.path.join(ROOT_DIR, "logs"), name=f"{TIME_OF_EXECUTION}_round_{idx}"
        )

        trainer = Trainer(
            logger=logger,
            max_epochs=self.train_conf["epochs"],
            accelerator=self.train_conf["device"],
            devices=1,
            enable_progress_bar=True,
            callbacks=[checkpoint_callback, early_stopping_callback],
        )

        data_module = MBHBDataModule(fname_h5, self.train_conf)
        print(datagen_info.keys())
        model = InferenceNetwork(train_conf=self.train_conf, dataset_info=datagen_info)

        trainer.fit(model, data_module)
        test_dataset = data_module.test
        return model, test_dataset

    def _update_prior_and_plot(self, updated_prior):
        self.logMchirp_lower.append(updated_prior["logMchirp"][0])
        self.logMchirp_upper.append(updated_prior["logMchirp"][1])
        self.q_lower.append(updated_prior["q"][0])
        self.q_upper.append(updated_prior["q"][1])

        for ax in self.axes:
            ax.cla()

        self.axes[0].set_title("logMchirp Prior Bounds")
        self.axes[0].set_xlabel("Iteration")
        self.axes[0].set_ylabel("logMchirp")
        self.axes[1].set_title("q Prior Bounds")
        self.axes[1].set_xlabel("Iteration")
        self.axes[1].set_ylabel("q")

        self.axes[0].plot(range(len(self.logMchirp_lower)), self.logMchirp_lower, label="Lower Bound", color="blue")
        self.axes[0].plot(range(len(self.logMchirp_upper)), self.logMchirp_upper, label="Upper Bound", color="orange")
        self.axes[1].plot(range(len(self.q_lower)), self.q_lower, label="Lower Bound", color="blue")
        self.axes[1].plot(range(len(self.q_upper)), self.q_upper, label="Upper Bound", color="orange")

        self.axes[0].legend()
        self.axes[1].legend()
        plt.tight_layout()
        plt.savefig(os.path.join(ROOT_DIR, "plots", f"prior_bounds_iteration_{len(self.logMchirp_lower)-1}.png"))

    def run(self, n_rounds=1):
        for i in range(n_rounds):
            print(f"Running round {i}...")
            model, test_dataset = self.round(
                idx=i,
                sampler_init_kwargs={"prior_bounds": self.datagen_conf["prior"]},
            )
            # model = InferenceNetwork.load_from_checkpoint("/u/g/gpuleo/pembhb/logs/20251111_100948_round_0/version_0/checkpoints/epoch=13-step=2380.ckpt")
            updated_prior = self.datagen_conf["prior"].copy()
            out_idx = 0
            for key, marginal_list in self.train_conf["marginals"].items():
                for marginal in marginal_list:
                    if len(marginal) == 1:
                        inj_idx = marginal[0]
                        pp_plot(test_dataset, model, in_param_idx=inj_idx, name=f"round_{i}_{key}", out_param_idx=out_idx)
                        updated_prior = update_bounds(
                            model, self.dataset_observation, updated_prior,
                            in_param_idx=inj_idx, n_gridpoints=10000, out_param_idx=out_idx
                        )
                    elif len(marginal) == 2:
                        inj1, inj2 = marginal
                        pp_plot_2d(test_dataset, model, in_param_idx=marginal, out_idx=out_idx,
                                   name=f"round_{_ORDERED_PRIOR_KEYS[inj1]}_{_ORDERED_PRIOR_KEYS[inj2]}")
                    out_idx += 1

            exit(1)
            print(f"Updated prior after round {i}:\nlog10Mchirp: {updated_prior['logMchirp']},\nq: {updated_prior['q']}")
            self.datagen_conf["prior"] = updated_prior
            self._update_prior_and_plot(updated_prior)

            del model
            if self.train_conf["device"] == "cuda":
                torch.cuda.empty_cache()

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--config", type=str, required=True)
    # args = parser.parse_args()
    train_config_filename = "train_config.yaml"
    datagen_config_filename = "datagen_config.yaml"
    
    train_config = read_config(os.path.join(ROOT_DIR, train_config_filename))
    datagen_config = read_config(os.path.join(ROOT_DIR, datagen_config_filename))
                               
    trainer = SequentialTrainer(train_conf=train_config, datagen_conf=datagen_config, dataset_obs_path=os.path.join(ROOT_DIR, "data/observation_fix_all_notmcq_newdata.h5"))
    trainer.run(n_rounds=1)

    #round(conf, sampler_init_kwargs={'low': 0.5, 'high': 1.0} , lr=conf["training"]["learning_rate"], idx=0)