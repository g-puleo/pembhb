import os, shutil
import torch
from pembhb.simulator import MBHBSimulatorFD_TD, DummySimulator
from pembhb.model import InferenceNetwork, PeregrineModel, ReducedOrderModel, ROMWrapper
from pembhb.data import MBHBDataModule, MBHBDataset, mbhb_collate_fn
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint, Callback
from lightning.pytorch.tuner import Tuner
from torch.utils.data import DataLoader , random_split, Subset
import numpy as np
from pembhb import ROOT_DIR, DATA_ROOT_DIR
from pembhb import utils
from glob import glob
import copy


import argparse 
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

torch.set_float32_matmul_precision("medium")
def get_timestamp():
    return datetime.now().strftime("%Y%m%d")
TIME_OF_EXECUTION = get_timestamp()+"_narrowprior_v0"

class PlotPosteriorCallback(Callback):
    def __init__(self, timestamp: str, obs_loader: DataLoader, input_idx_list: list, output_idx_list: list, round_idx: int , call_every_n_epochs=1): 
        self.epochs_elapsed = 0
        self.call_every_n_epochs = call_every_n_epochs
        self.timestamp = timestamp
        self.obs_loader = obs_loader
        self.input_idx_list = input_idx_list
        self.output_idx_list = output_idx_list
        self.n_marginals = len(input_idx_list)
        self.init_time = datetime.now()
        self.round_idx = round_idx
    def on_validation_epoch_end(self, trainer, pl_module):
        if self.epochs_elapsed == 0: 
            os.makedirs(os.path.join(ROOT_DIR, "plots", TIME_OF_EXECUTION), exist_ok=True)

        self.epochs_elapsed += 1
        if (self.epochs_elapsed-2) % self.call_every_n_epochs == 0:
            print("plotting posteriors on observed data")
            train_time = datetime.now() - self.init_time
            td_trunc = train_time - timedelta(microseconds=train_time.microseconds)
            title_plot = f"training time={td_trunc}s"
            # plot the posterior on the observed data , using the current model
            for i in range(self.n_marginals):
                in_param_idx = self.input_idx_list[i]
                out_param_idx = self.output_idx_list[i]

                # if pure 1D, skip (unchanged)
                if len(in_param_idx) == 1:
                    continue

                fig, ax = plt.subplots(1, 1, figsize=(4, 4))
                fig.tight_layout()
                fig.suptitle(
                    f"Round {self.round_idx} - Epoch {trainer.current_epoch} - {title_plot}",
                    fontsize=10,
                )

                logratios, inj_params, gx, gy = utils.get_logratios_grid_2d(
                    self.obs_loader,
                    pl_module,
                    ngrid_points=100,
                    in_param_idx=in_param_idx,
                    out_param_idx=out_param_idx
                )

                ratios = np.exp(logratios)
                dp1 = gx[1, 0] - gx[0, 0]
                dp2 = gy[0, 1] - gy[0, 0]
                norm2d = ratios / np.sum(ratios * dp1 * dp2, axis=(1, 2), keepdims=True)

                # only 2D → ax is a single axis
                ax_2d = ax
                levels, labels = utils.contour_levels(norm2d)
                # find the levels corresponding to 68%, 95%, 99.7, %
                try: 
                    boxes = utils.posterior_contours_2d(
                        gx,
                        gy,
                        norm2d[0],
                        inj_params[0],
                        ax_buffer=ax_2d,
                        parameter_names=[
                            utils._ORDERED_PRIOR_KEYS[in_param_idx[0]],
                            utils._ORDERED_PRIOR_KEYS[in_param_idx[1]],
                        ],
                        title="",
                        levels=levels,
                        levels_labels=labels,
                        do_plot=True
                    )

                except ValueError: 
                    print("caught ValueError during contour plotting, skipping this plot")
                    continue
                pl_module.widest_box = boxes[-1]
                out = os.path.join( ROOT_DIR,"plots",self.timestamp,f"posterior_round_{self.round_idx}_epoch_{trainer.current_epoch}_{utils._ORDERED_PRIOR_KEYS[in_param_idx[0]]}_{utils._ORDERED_PRIOR_KEYS[in_param_idx[1]]}.pdf")
                fig.savefig(out, bbox_inches="tight")
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
        self.dataset_observation = Subset(MBHBDataset(dataset_obs_path, cache_in_memory=True), indices=[2])
        self.dataloader_obs = DataLoader(self.dataset_observation, batch_size=train_conf["batch_size"], shuffle=False, collate_fn=lambda b: mbhb_collate_fn(b, self.dataset_observation, noise_shuffling=False))
        self.logMchirp_lower = [datagen_conf["prior"]["logMchirp"][0]]
        self.logMchirp_upper = [datagen_conf["prior"]["logMchirp"][1]]
        self.q_lower = [datagen_conf["prior"]["q"][0]]
        self.q_upper = [datagen_conf["prior"]["q"][1]]
        self._setup_plot()
        
        # load baseline model to update prior before training new model
        if self.train_conf["baseline_model"]["use"]:
            self.model = InferenceNetwork.load_from_checkpoint(self.train_conf["baseline_model"]["filename"])
            widest_box = self._get_widest_box(self.model, self.dataloader_obs)

            # update prior based on model performance
            self.datagen_conf["prior"]["logMchirp"] = [widest_box[0], widest_box[1]]
            self.datagen_conf["prior"]["q"] = [widest_box[2], widest_box[3]]
            print(f"Updated prior based on baseline model:\nlog10Mchirp: {self.datagen_conf['prior']['logMchirp']},\nq: {self.datagen_conf['prior']['q']}")
    
    def _get_widest_box(self, model, dataloader):
        logratios, inj_params, gx, gy = utils.get_logratios_grid_2d(
            dataloader,
            model,
            ngrid_points=100,
            in_param_idx=(0,1),
            out_param_idx=0,
        )

        ratios = np.exp(logratios)
        dp1 = gx[1, 0] - gx[0, 0]
        dp2 = gy[0, 1] - gy[0, 0]
        norm2d = ratios / np.sum(ratios * dp1 * dp2)
        levels, labels = utils.contour_levels(norm2d)
        boxes = utils.posterior_contours_2d(gx, gy, norm2d[0],
                                            inj_params[0], ax_buffer=None, parameter_names=['log10Mc', 'q'],
                                            levels=levels, levels_labels=labels)
        widest_box = boxes[-1]
        return widest_box
    
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
    
    def _generate_data(self, round_idx, sampler_init_kwargs) : 
        fname_base = f"simulation_round_{round_idx}"
        fname_h5 = os.path.join(DATA_ROOT_DIR, TIME_OF_EXECUTION,  f"{fname_base}.h5")
        os.makedirs(os.path.dirname(fname_h5), exist_ok=True)
        sim = MBHBSimulatorFD_TD(self.datagen_conf, sampler_init_kwargs=sampler_init_kwargs)
        if not os.path.exists(fname_h5):
            sim.sample_and_store(fname_h5, N=50000, batch_size=250)
        else: 
            try:
                resp = input(f"Dataset file {fname_h5} already exists. Resample and overwrite? [y/N]: ").strip().lower()
            except Exception:
                # non-interactive environment, default to not retrain
                resp = "n"
            if resp in ("y", "yes"):
                os.remove(fname_h5)
                sim.sample_and_store(fname_h5, N=50000, batch_size=250)
                print(f"Resampled dataset and saved to {fname_h5}")
            else:
                print(f"Using existing dataset at {fname_h5}")
        self.data_fname_yaml = fname_h5.replace(".h5", ".yaml")
        self.datagen_info = utils.read_config(self.data_fname_yaml)
        self.data_module = MBHBDataModule(fname_h5, train_config["batch_size"], cache_in_memory=True)
        self.data_module.setup(stage="fit")
        self.test_dataloader = self.data_module.test_dataloader()

    def _train_rom ( self, round_idx) : 
        print("Training ROM...")
        rom = ReducedOrderModel(tolerance=5e-6, device="cuda", batch_size=5000)
        filename = os.path.join(DATA_ROOT_DIR, TIME_OF_EXECUTION, f"rom_round_{round_idx}.pt")
        if not os.path.exists(filename):
            rom.train(self.data_module.train)
            rom.to_file(filename)
        else: 
            # existing ROM file found: ask whether to retrain or reuse
            resp = None
            try:
                resp = input(f"ROM file {filename} already exists. Retrain and overwrite? [y/N]: ").strip().lower()
            except Exception:
                # non-interactive environment, default to not retrain
                resp = "n"
            if resp in ("y", "yes"):
                rom.train(self.data_module.train)
                rom.to_file(filename)
                print(f"Retrained ROM and saved to {filename}")
            else:
                print(f"Using existing ROM at {filename}")

        self.data_summary = ROMWrapper(filename=filename, device=self.train_conf["device"])
        self.data_module.full_dataset.to("cpu") # go back to loading from cpu

    def _train_inference_network ( self, round_idx, data_summary=None) :
        #  initialise data summarizer (ROM) 
        mean, std = self.data_module.get_params_mean_std()
        normalisation = {"td_normalisation": np.array(self.data_module.get_max_td()), 
                         "param_mean": np.array(mean),
                         "param_std": np.array(std)}
        self.model = InferenceNetwork(train_conf=self.train_conf, dataset_info=self.datagen_info, normalisation=normalisation, data_summarizer=data_summary)
        self.model.train()
        copy_bounds_file = True

        logger = TensorBoardLogger(
            os.path.join(DATA_ROOT_DIR, "logs"), name=f"{TIME_OF_EXECUTION}_round_{round_idx}"
        )
        
        checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min")
        early_stopping_callback = EarlyStopping(
            monitor="val_accuracy",
            patience=self.train_conf["early_stop_patience"],
            mode="max",
            min_delta=self.train_conf["early_stop_min_delta"],
            stopping_threshold=self.train_conf["early_stop_threshold"],
        )
        
        
        plot_posterior_callback = PlotPosteriorCallback(
            timestamp=TIME_OF_EXECUTION,
            obs_loader=self.dataloader_obs,
            input_idx_list=self.model.marginals_list,
            output_idx_list=list(range(len(self.model.marginals_list))),
            round_idx=round_idx,
            call_every_n_epochs=1)

        trainer = Trainer(
            logger=logger,
            max_epochs=self.train_conf["epochs"],
            accelerator=self.train_conf["device"],
            devices=1,
            enable_progress_bar=True,
            callbacks=[checkpoint_callback, early_stopping_callback, plot_posterior_callback],
        )

        # if round_idx == 0: 
        #     self.model = InferenceNetwork.load_from_checkpoint("/data/gpuleo/mbhb/logs/20260107_v1_round_4/version_0/checkpoints/epoch=78-step=11060.ckpt")
        #     self.model.widest_box = self._get_widest_box(self.model, self.dataloader_obs)
        # else: 
        trainer.fit(self.model, self.data_module)
        print(f"Widest box after training: {self.model.widest_box}")
        if copy_bounds_file: 
            shutil.copy(self.data_fname_yaml, logger.log_dir)


    def round(self, idx, sampler_init_kwargs):
        self._generate_data(round_idx=idx, sampler_init_kwargs=sampler_init_kwargs)
        if self.train_conf["architecture"]["data_summary"]["type"] == "ROM":
            self._train_rom(round_idx=idx)
            self._train_inference_network(round_idx=idx, data_summary=self.data_summary)
        else:
            self._train_inference_network(round_idx=idx, data_summary=None)
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
        self.fig.tight_layout()
        self.fig.savefig(os.path.join(ROOT_DIR, "plots", TIME_OF_EXECUTION, f"prior_bounds_iteration_{len(self.logMchirp_lower)-1}.png"))

    def run(self, n_rounds=1):
        for i in range(n_rounds):
            print(f"Running round {i}...")
            self.round(
                idx=i,
                sampler_init_kwargs={"prior_bounds": self.datagen_conf["prior"]},
            )
            # model = InferenceNetwork.load_from_checkpoint("/u/g/gpuleo/pembhb/logs/20251111_100948_round_0/version_0/checkpoints/epoch=13-step=2380.ckpt")
            out_idx = 0
            for key, marginal_list in self.train_conf["marginals"].items():
                for marginal in marginal_list:
                    if len(marginal) == 1:
                        inj_idx = marginal[0]
                        utils.pp_plot(self.test_dataloader, self.model, in_param_idx=inj_idx, name=f"round_{i}_{key}", out_param_idx=out_idx)
                        # updated_prior = utils.update_bounds(
                        #     self.model, self.dataloader_obs, updated_prior,
                        #     in_param_idx=inj_idx, n_gridpoints=10000, out_param_idx=out_idx
                        # )
                    elif len(marginal) == 2:
                        inj1, inj2 = marginal
                        #utils.pp_plot_2d(self.test_dataloader, self.model, in_param_idx=marginal, out_idx=out_idx,
                        #           name=f"{ROOT_DIR}/plots/{TIME_OF_EXECUTION}/round_{i}_{utils._ORDERED_PRIOR_KEYS[inj1]}_{utils._ORDERED_PRIOR_KEYS[inj2]}")
            
                        tmp_prior = copy.deepcopy(self.datagen_conf["prior"])
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
    trainer.run(n_rounds=2)

    #round(conf, sampler_init_kwargs={'low': 0.5, 'high': 1.0} , lr=conf["training"]["learning_rate"], idx=0)