import os
import torch
from pembhb.simulator import MBHBSimulatorFD_TD, DummySimulator
from pembhb.model import InferenceNetwork, PeregrineModel
from pembhb.data import MBHBDataModule, MBHBDataset
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint, Callback
from lightning.pytorch.tuner import Tuner
from torch.utils.data import DataLoader , random_split, Subset
import numpy as np
from pembhb import ROOT_DIR
from pembhb import utils

import argparse 
from datetime import datetime
import matplotlib.pyplot as plt

torch.set_float32_matmul_precision("medium")
def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")
TIME_OF_EXECUTION = get_timestamp()

class PlotPosteriorCallback(Callback):
    def __init__(self, timestamp: str, obs_loader: DataLoader, input_idx_list: list, output_idx_list: list, call_every_n_epochs=1): 
        self.epochs_elapsed = 0
        self.call_every_n_epochs = call_every_n_epochs
        self.timestamp = timestamp
        self.obs_loader = obs_loader
        self.input_idx_list = input_idx_list
        self.output_idx_list = output_idx_list
        self.n_marginals = len(input_idx_list)
        self.init_time = datetime.now()

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.epochs_elapsed == 0: 
            os.makedirs(os.path.join(ROOT_DIR, "plots", TIME_OF_EXECUTION), exist_ok=True)

        self.epochs_elapsed += 1
        if self.epochs_elapsed % self.call_every_n_epochs == 0:
            print("plotting posteriors on observed data")
            # plot the posterior on the observed data , using the current model
            self.train_time = datetime.now() - self.init_time
            title_plot = f"training time={self.train_time}s"
            for i in range(self.n_marginals):
                fig, ax = plt.subplots(figsize=(5, 5))
                in_param_idx = self.input_idx_list[i]
                out_param_idx = self.output_idx_list[i]
                if len(in_param_idx) == 1:
                    logratios, inj_params, grid = utils.get_logratios_grid(self.obs_loader, pl_module, ngrid_points=1000, in_param_idx=in_param_idx[0], out_param_idx=out_param_idx)
                    ratios = np.exp(logratios)
                    dparam = (grid[1] - grid[0]).reshape(-1)
                    normalised_ratios = (ratios /np.sum(ratios * dparam, axis=1, keepdims=True)).reshape(-1)
                    utils.plot_posterior_1d(grid, normalised_ratios, true_value=inj_params[0], ax_buffer=ax, parameter_name=utils._ORDERED_PRIOR_KEYS[in_param_idx[0]], title=title_plot)
                    fig.savefig(os.path.join(ROOT_DIR, f"plots/{self.timestamp}", f"posterior_epoch_{trainer.current_epoch}_{utils._ORDERED_PRIOR_KEYS[in_param_idx[0]]}.png"))
                    plt.close(fig)
                elif len(in_param_idx) == 2:
                    logratios, inj_params, grid_x, grid_y = utils.get_logratios_grid_2d(self.obs_loader, pl_module, ngrid_points=100, in_param_idx=in_param_idx, out_param_idx=out_param_idx)
                    ratios = np.exp(logratios)
                    dp1 = grid_x[1,0] - grid_x[0,0]
                    dp2 = grid_y[0,1] - grid_y[0,0]
                    normalised_ratios = ratios / np.sum(ratios * dp2 * dp1, axis=(1,2), keepdims=True)
                    widest_box = utils.plot_posterior_2d(grid_x, grid_y, normalised_ratios[0], inj_params[0], ax_buffer=ax, parameter_names=[utils._ORDERED_PRIOR_KEYS[in_param_idx[0]], utils._ORDERED_PRIOR_KEYS[in_param_idx[1]]], title_plot=title_plot)
                    pl_module.widest_box = widest_box
                    fname_current = os.path.join(ROOT_DIR, f"plots/{self.timestamp}", f"posterior_epoch_{trainer.current_epoch}_{utils._ORDERED_PRIOR_KEYS[in_param_idx[0]]}_{utils._ORDERED_PRIOR_KEYS[in_param_idx[1]]}.png")
                    fig.savefig(fname_current)
                    plt.close(fig)
            print("done plotting posteriors")

    def on_train_end(self, trainer, pl_module):
        self.on_validation_epoch_end(trainer, pl_module)
        print(f"Total training time: {datetime.now() - self.init_time}")
class SequentialTrainer:
    def __init__(self, train_conf, datagen_conf, dataset_obs_path):
        self.train_conf = train_conf
        self.datagen_conf = datagen_conf
        self.dataset_obs_path = dataset_obs_path
        # Subset is there because utils.mbhb_collate_fn expects a Subset, it will access its dataset attribute
        self.dataset_observation = Subset(MBHBDataset(dataset_obs_path, transform_fd=train_conf["transform_fd"]), indices=[0])
        self.dataloader_obs = DataLoader(self.dataset_observation, batch_size=train_conf["batch_size"], shuffle=False, collate_fn=lambda b: utils.mbhb_collate_fn(b, self.dataset_observation, noise_shuffling=False))
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
        
        fname_base = f"fix_all_notmcq_newdata_round_{idx}"
        fname_h5 = os.path.join(ROOT_DIR, "data", f"{fname_base}.h5")

        sim = MBHBSimulatorFD_TD(self.datagen_conf, sampler_init_kwargs=sampler_init_kwargs)
        if not os.path.exists(fname_h5):
            sim.sample_and_store(fname_h5, N=10000, batch_size=250)
        else: 
            print(f"Using existing dataset at {fname_h5}")
        data_fname_yaml = os.path.join(ROOT_DIR, "data", f"{fname_base}.yaml")
        datagen_info = utils.read_config(data_fname_yaml)
        data_module = MBHBDataModule(fname_h5, self.train_conf)
        print(datagen_info.keys())
        
        #self.model = InferenceNetwork.load_from_checkpoint("logs/20251121_122224_round_0/version_0/checkpoints/epoch=205-step=35020.ckpt")
        if not hasattr(self, 'model'):
            self.model = InferenceNetwork(train_conf=self.train_conf, dataset_info=datagen_info)
        elif not self.train_conf["keep_ckpt"]:
            self.model = InferenceNetwork(train_conf=self.train_conf, dataset_info=datagen_info)
        else : 
            pass
        checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min")
        early_stopping_callback = EarlyStopping(
            monitor="val_accuracy",
            patience=self.train_conf["early_stop_patience"],
            mode="max",
            stopping_threshold=self.train_conf["early_stop_threshold"],
        )


        logger = TensorBoardLogger(
            os.path.join(ROOT_DIR, "logs"), name=f"{TIME_OF_EXECUTION}_round_{idx}"
        )

        plot_posterior_callback = PlotPosteriorCallback(
            timestamp=TIME_OF_EXECUTION,
            obs_loader=self.dataloader_obs,
            input_idx_list=self.model.marginals_list,
            output_idx_list=list(range(len(self.model.marginals_list))),
            call_every_n_epochs=10)

        trainer = Trainer(
            logger=logger,
            max_epochs=self.train_conf["epochs"],
            accelerator=self.train_conf["device"],
            devices=1,
            enable_progress_bar=True,
            callbacks=[checkpoint_callback, early_stopping_callback, plot_posterior_callback],
        )

        # elif not self.train_conf["keep_ckpt"]:
        #     self.model = InferenceNetwork(train_conf=self.train_conf, dataset_info=datagen_info)
        # else: 
        #     pass

        trainer.fit(self.model, data_module)
        test_dataloader = data_module.test_dataloader()
        return test_dataloader

    def _plot_updated_prior_bounds(self, updated_prior):
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
            test_dataloader = self.round(
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
                        utils.pp_plot(test_dataloader, self.model, in_param_idx=inj_idx, name=f"round_{i}_{key}", out_param_idx=out_idx)
                        # updated_prior = utils.update_bounds(
                        #     self.model, self.dataloader_obs, updated_prior,
                        #     in_param_idx=inj_idx, n_gridpoints=10000, out_param_idx=out_idx
                        # )
                    elif len(marginal) == 2:
                        inj1, inj2 = marginal
                        utils.pp_plot_2d(self.dataloader_obs, self.model, in_param_idx=marginal, out_idx=out_idx,
                                   name=f"round_{utils._ORDERED_PRIOR_KEYS[inj1]}_{utils._ORDERED_PRIOR_KEYS[inj2]}")
                        tmp_prior = self.datagen_conf["prior"].copy()
                        tmp_prior[utils._ORDERED_PRIOR_KEYS[inj1]] = [self.model.widest_box[0], self.model.widest_box[1]]
                        tmp_prior[utils._ORDERED_PRIOR_KEYS[inj2]] = [self.model.widest_box[2], self.model.widest_box[3]]
                        self.datagen_conf["prior"] = tmp_prior
                    out_idx += 1

            print(f"Updated prior after round {i}:\nlog10Mchirp: {self.datagen_conf['prior']['logMchirp']},\nq: {self.datagen_conf['prior']['q']}")
            self._plot_updated_prior_bounds(self.datagen_conf["prior"])

            if self.train_conf["device"] == "cuda":
                torch.cuda.empty_cache()

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--config", type=str, required=True)
    # args = parser.parse_args()
    train_config_filename = "train_config.yaml"
    datagen_config_filename = "datagen_config.yaml"
    
    train_config   = utils.read_config(os.path.join(ROOT_DIR, train_config_filename))
    datagen_config = utils.read_config(os.path.join(ROOT_DIR, datagen_config_filename))
                               
    trainer = SequentialTrainer(train_conf=train_config, datagen_conf=datagen_config, dataset_obs_path=os.path.join(ROOT_DIR, "data/testes_newdata_fixall_notmcq.h5"))
    trainer.run(n_rounds=3)

    #round(conf, sampler_init_kwargs={'low': 0.5, 'high': 1.0} , lr=conf["training"]["learning_rate"], idx=0)