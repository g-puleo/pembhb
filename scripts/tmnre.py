import os, shutil
import numpy as np
import torch
from torch.utils.data import DataLoader , Subset
import copy
from datetime import datetime
import matplotlib.pyplot as plt
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint


from pembhb.simulator import MBHBSimulatorFD_TD, MBHBSimulatorFD
from pembhb.model import InferenceNetwork, PerMarginalInferenceNetwork
from pembhb.rom import ReducedOrderModel, ROMWrapper
from pembhb.autoencoder import (
    DenoisingAutoencoder, AutoencoderWrapper,
    MarginalEncoderTrainer, MarginalEncoderWrapper,
)
from pembhb.data import MBHBDataModule, MBHBDataset, mbhb_collate_fn
from pembhb import ROOT_DIR, DATA_ROOT_DIR, set_precision
from pembhb import utils
from pembhb.utils import validate_marginals, resolve_marginals_for_round, transfer_classifier_weights, get_widest_interval_1d, get_widest_box_2d, choose_device_for_pp
from pembhb.callbacks import PlotPosteriorCallback, VolumeRatioEarlyStopping, PeriodicProgressCallback

def get_timestamp():
    return datetime.now().strftime("%Y%m%d")

TIME_OF_EXECUTION = get_timestamp() + "_sequential_volratio_earlystop"

class SequentialTrainer:
    def __init__(self, train_conf, datagen_conf, dataset_obs_path):
        self.train_conf = train_conf
        self.datagen_conf = datagen_conf
        self.dataset_obs_path = dataset_obs_path
        self.training_start = datetime.now()

        # Validate that no parameter index appears in multiple marginals
        validate_marginals(train_conf["marginals"])
        
        #self.dataset_observation = Subset(MBHBDataset(dataset_obs_path, cache_in_memory=True), indices=[2])
        self.dataset_observation = Subset(MBHBDataset(dataset_obs_path, cache_in_memory=True),indices =[0])
        obs_noise_scale = self.dataset_observation.dataset.noise_scale
        obs_td_params = self.dataset_observation.dataset.td_params
        self.dataloader_obs = DataLoader(self.dataset_observation, batch_size=1, shuffle=False, collate_fn=lambda b: mbhb_collate_fn(b, obs_noise_scale, noise_factor=self.train_conf["noise_factor"], noise_shuffling=False, td_params=obs_td_params))
        self.logMchirp_lower = [datagen_conf["prior"]["logMchirp"][0]]
        self.logMchirp_upper = [datagen_conf["prior"]["logMchirp"][1]]
        self.q_lower = [datagen_conf["prior"]["q"][0]]
        self.q_upper = [datagen_conf["prior"]["q"][1]]
        self._setup_plot()
        
        # load baseline model to update prior before training new model
        if self.train_conf["baseline_model"]["use"]:
            self.model = InferenceNetwork.load_from_checkpoint(self.train_conf["baseline_model"]["filename"])
            
            # update prior based on model performance for all marginals (1D and 2D)
            out_idx = 0
            for key, marginal_list in self.train_conf["marginals"].items():
                for marginal in marginal_list:
                    if len(marginal) == 1:
                        # 1D marginal
                        widest_interval, _, _, _ = get_widest_interval_1d(
                            self.model, self.dataloader_obs,
                            in_param_idx=marginal[0], out_param_idx=out_idx, eps=1e-4
                        )
                        param_name = utils._ORDERED_PRIOR_KEYS[marginal[0]]
                        self.datagen_conf["prior"][param_name] = widest_interval
                        print(f"Updated prior based on baseline model for 1D marginal ({param_name})")
                    elif len(marginal) == 2:
                        # 2D marginal
                        widest_box, _ = get_widest_box_2d(
                            self.model, self.dataloader_obs, 
                            in_param_idx=tuple(marginal), out_param_idx=out_idx
                        )
                        inj1, inj2 = marginal
                        self.datagen_conf["prior"][utils._ORDERED_PRIOR_KEYS[inj1]] = [widest_box[0], widest_box[1]]
                        self.datagen_conf["prior"][utils._ORDERED_PRIOR_KEYS[inj2]] = [widest_box[2], widest_box[3]]
                        print(f"Updated prior based on baseline model for 2D marginal ({utils._ORDERED_PRIOR_KEYS[inj1]}, {utils._ORDERED_PRIOR_KEYS[inj2]})")
                    out_idx += 1
            print(f"Updated prior after baseline model: {self.datagen_conf['prior']}")

        # Optionally compute Fisher-matrix-based prior bounds for round 1.
        # When train_conf["fisher_prior"]["enabled"] is True the bounds are
        # derived from the FIM at the target event; otherwise this is None and
        # datagen_conf["prior"] is used as usual.
        self.fisher_prior_bounds = None
        fp_conf = self.train_conf.get("fisher_prior", {})
        if fp_conf.get("enabled", False):
            print("[Fisher] Computing Fisher Information Matrix for prior initialisation ...")
            self.fisher_prior_bounds = utils.compute_fisher_prior_bounds(
                datagen_config=self.datagen_conf,
                observation_file=dataset_obs_path,
                event_idx=fp_conf["event_idx"],
                varying_params=fp_conf["varying_params"],
                fixed_params=fp_conf["fixed_params"],
                n_sigma=fp_conf.get("n_sigma", 5.0),
                delta_frac=fp_conf.get("delta_frac", 1e-4),
            )

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
        domain = self.datagen_conf.get("waveform_params", {}).get("domain", "fd_td")
        if domain == "fd":
            wp = self.datagen_conf["waveform_params"]
            sim = MBHBSimulatorFD(
                self.datagen_conf,
                sampler_init_kwargs=sampler_init_kwargs,
                n_freq_bins=wp.get("n_freq_bins", 4096),
                freq_spacing=wp.get("freq_spacing", "linear"),
            )
        else:
            sim = MBHBSimulatorFD_TD(self.datagen_conf, sampler_init_kwargs=sampler_init_kwargs)
        N_simulations = 50000
        batch_size_generation = 250
        if not os.path.exists(fname_h5):
            sim.sample_and_store(fname_h5, N=N_simulations, batch_size=batch_size_generation)
        else: 
            try:
                resp = input(f"Dataset file {fname_h5} already exists. Resample and overwrite? [y/N]: ").strip().lower()
            except Exception:
                # non-interactive environment, default to not retrain
                resp = "n"
            if resp in ("y", "yes"):
                os.remove(fname_h5)
                sim.sample_and_store(fname_h5, N=N_simulations, batch_size=batch_size_generation)
                print(f"Resampled dataset and saved to {fname_h5}")
            else:
                print(f"Using existing dataset at {fname_h5}")
        self.data_fname_yaml = fname_h5.replace(".h5", ".yaml")
        self.datagen_info = utils.read_config(self.data_fname_yaml)
        self.data_module = MBHBDataModule(fname_h5, train_config["batch_size"], num_workers=4, cache_in_memory=True, noise_factor=self.train_conf["noise_factor"])
        assert self.data_module.median_snr > 8, f"Median SNR lower than 8. Please make sure this is what you want. "
        self.data_module.setup(stage="fit")
        self.test_dataloader = self.data_module.test_dataloader()

    def _train_rom ( self, round_idx) : 
        print("Training ROM...")
        rom_training_config = self.train_conf["architecture"]["data_summary"]["ROM"]
        freqs = self.data_module.get_freqs()
        df = (freqs[1] - freqs[0]).item()
        rom = ReducedOrderModel(
            tolerance=rom_training_config.get("tolerance", 1e-3),
            device="cuda",
            batch_size=rom_training_config.get("batch_train", 1000),
            patience=rom_training_config.get("patience", 1),
            freq_cutoff_idx=rom_training_config.get("freq_cutoff_idx", None),
            df=df,
            normalize_by_max=rom_training_config.get("normalize_by_max", True),
            debugging=rom_training_config.get("debugging", False),
        )
        filename = os.path.join(DATA_ROOT_DIR, TIME_OF_EXECUTION, f"rom_round_{round_idx}.pt")
        
        use_pinned_memory = rom_training_config.get("use_pinned_memory", False)
        prefetch_batches = rom_training_config.get("prefetch_batches", 1)
        convergence_on = rom_training_config.get("convergence_on", "sigma_data")
        
        if not os.path.exists(filename):
            train_dl = self.data_module.train_dataloader(num_workers=0, pin_memory=use_pinned_memory)
            rom.train(train_dl, use_pinned_memory=use_pinned_memory, prefetch_batches=prefetch_batches,
                      convergence_on=convergence_on)
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
                train_dl = self.data_module.train_dataloader(num_workers=0, pin_memory=use_pinned_memory)
                rom.train(train_dl, use_pinned_memory=use_pinned_memory, prefetch_batches=prefetch_batches,
                          convergence_on=convergence_on)
                rom.to_file(filename)
                print(f"Retrained ROM and saved to {filename}")
            else:
                print(f"Using existing ROM at {filename}")

        self.data_summary = ROMWrapper(
            filename=filename,
            device=self.train_conf["device"],
            max_basis_elems=rom_training_config.get("max_basis_elems", None),
        )
        
        # Free cached data and switch to disk-based loading to reduce memory
        self.data_module.full_dataset.clear_cache()
    def _load_rom(self, round_idx):
        assert round_idx == 1, "Are you sure this is the ROM for this round?"
        rom_config = self.train_conf["architecture"]["data_summary"]["ROM"]
        filename = rom_config["filename"]
        print(f"Loading ROM from {filename}...")
        self.data_summary = ROMWrapper(
            filename=filename,
            device=self.train_conf["device"],
            max_basis_elems=rom_config.get("max_basis_elems", None),
        )
    

    def _load_autoencoder(self, round_idx):
        """Load a trained autoencoder from checkpoint and wrap it for use as data_summary."""
        print("Loading trained Autoencoder...")
        ae_config = self.train_conf["architecture"]["data_summary"]["Autoencoder"]
        ckpt_path = ae_config["filename"]
        device = ae_config.get("device", self.train_conf["device"])
        
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"Autoencoder checkpoint not found at {ckpt_path}. "
                "Please train an autoencoder first using scripts/train_autoencoder.py"
            )
        
        # Load the trained autoencoder
        autoencoder = DenoisingAutoencoder.load_from_checkpoint(ckpt_path)
        autoencoder.to(device)
        autoencoder.eval()
        
        # Wrap it for use as data_summarizer
        self.data_summary = AutoencoderWrapper(autoencoder, freeze=True)
        self.data_summary.to(device)
        
        print(f"Loaded autoencoder from {ckpt_path}")
        print(f"Bottleneck dimensionality: {self.data_summary.get_n_features()}")
        
        # Free cached data and switch to disk-based loading to reduce memory
        self.data_module.full_dataset.clear_cache()

    def _train_autoencoder(self, round_idx, finetune_previous=False):
        """Train a DenoisingAutoencoder on the current round's data and wrap it as data_summary.

        Mirrors ``_train_rom``: trains from scratch on the data generated by
        ``_generate_data`` for this round, saves a checkpoint, and wraps the
        best model in an :class:`AutoencoderWrapper`.

        The hyperparameters are read from
        ``self.train_conf["architecture"]["data_summary"]["Autoencoder"]``
        so that they match what :mod:`scripts.train_autoencoder` accepts.
        """
        print(f"Training Autoencoder for round {round_idx}...")
        ae_conf = self.train_conf["architecture"]["data_summary"]["Autoencoder"]
        device = ae_conf.get("device", self.train_conf["device"])

        ckpt_dir = os.path.join(DATA_ROOT_DIR, TIME_OF_EXECUTION)
        ckpt_tag = f"ae_round_{round_idx}"
        ckpt_sentinel = os.path.join(ckpt_dir, f"{ckpt_tag}_done.txt")

        # --- check for existing trained AE for this round -----------------
        if os.path.exists(ckpt_sentinel):
            # Read best checkpoint path written by a previous run
            best_ckpt = open(ckpt_sentinel).read().strip()
            if os.path.exists(best_ckpt):
                resp = None
                try:
                    resp = input(
                        f"AE checkpoint for round {round_idx} already exists at "
                        f"{best_ckpt}. Retrain and overwrite? [y/N]: "
                    ).strip().lower()
                except Exception:
                    resp = "n"
                if resp not in ("y", "yes"):
                    print(f"Reusing existing AE checkpoint: {best_ckpt}")
                    autoencoder = DenoisingAutoencoder.load_from_checkpoint(best_ckpt)
                    autoencoder.to(device)
                    autoencoder.eval()
                    self.data_summary = AutoencoderWrapper(autoencoder, freeze=True)
                    self.data_summary.to(device)
                    print(f"Bottleneck dimensionality: {self.data_summary.get_n_features()}")
                    self.data_module.full_dataset.clear_cache()
                    return

        # --- retrieve prior bounds from YAML sidecar ----------------------
        prior_bounds = self.datagen_info.get("conf", {}).get("prior", None)

        # --- build model --------------------------------------------------
        hidden_channels = ae_conf.get("hidden_channels", (32, 64, 128, 256, 256))
        if isinstance(hidden_channels, list):
            hidden_channels = tuple(hidden_channels)

        # --- if finetune_previous , then do not reinitialise the autoencoder and use the one from the previous round

        if not finetune_previous or round_idx == 1:
            autoencoder = DenoisingAutoencoder(
                n_channels=ae_conf.get("n_channels", 2),
                n_freqs=ae_conf.get("n_freqs", 4096),
                architecture=ae_conf.get("architecture", "conv"),
                bottleneck_dim=ae_conf.get("bottleneck_dim", 128),
                hidden_channels=hidden_channels,
                kernel_size=ae_conf.get("kernel_size", 4),
                stride=ae_conf.get("stride", 2),
                dropout=ae_conf.get("dropout", 0.0),
                residual=ae_conf.get("residual", False),
                lr=ae_conf.get("lr", 1e-3),
                weight_decay=ae_conf.get("weight_decay", 1e-5),
                scheduler_patience=ae_conf.get("scheduler_patience", 10),
                scheduler_factor=ae_conf.get("scheduler_factor", 0.3),
                representation=ae_conf.get("representation", "amp_phase"),
                high_freq_only=ae_conf.get("high_freq_only", False),
                freq_split_idx=ae_conf.get("freq_split_idx", 2048),
                prior_bounds=prior_bounds,
            )
            autoencoder = autoencoder.to(device)

            # --- fit normalisation from training split (clean signals) --------
            print("[AE] Fitting normalisation statistics ...")
            norm_loader = self.data_module.train_dataloader(shuffle=False, num_workers=0)
            autoencoder.fit_normalisation(norm_loader)

            # --- set noise ASD for noise-weighted reconstruction loss --------
            asd = self.data_module.get_asd()
            if asd is not None:
                autoencoder.set_noise_asd(asd)

        else:
            autoencoder = self.data_summary.autoencoder
            self.data_summary.unfreeze_parameters()
            print(f"Finetuning autoencoder from previous round with bottleneck dim {autoencoder.hparams.bottleneck_dim} and prior bounds {autoencoder.hparams.prior_bounds}")

            # Re-fit normalisation on the new round's data and re-set noise
            # ASD so the noise-weighted loss is used consistently across rounds.
            print("[AE] Re-fitting normalisation for new round data ...")
            norm_loader = self.data_module.train_dataloader(shuffle=False, num_workers=0)
            autoencoder.fit_normalisation(norm_loader)
            asd = self.data_module.get_asd()
            if asd is not None:
                autoencoder.set_noise_asd(asd)
        # --- callbacks ----------------------------------------------------
        checkpoint_cb = ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=2,
            filename=f"{ckpt_tag}" + "-{epoch:03d}-{val_loss:.4e}",
        )
        early_stop_cb = EarlyStopping(
            monitor="val_loss",
            patience=ae_conf.get("early_stop_patience", 50),
            mode="min",
        )

        from pembhb.diagnostics import AutoencoderDiagnosticsCallback
        ae_diag_cb = AutoencoderDiagnosticsCallback()

        # --- logger -------------------------------------------------------
        log_name = ae_conf.get("log_name", "autoencoder")
        logger = TensorBoardLogger(
            save_dir=os.path.join(DATA_ROOT_DIR, "logs"),
            name=f"{TIME_OF_EXECUTION}_{log_name}_round_{round_idx}",
        )

        # --- trainer ------------------------------------------------------
        ae_trainer = Trainer(
            logger=logger,
            max_epochs=ae_conf.get("epochs", 500),
            accelerator=device,
            devices=1,
            enable_progress_bar=False,
            callbacks=[checkpoint_cb, early_stop_cb, ae_diag_cb,
                       PeriodicProgressCallback(print_every=20, label="AE")],
            gradient_clip_val=ae_conf.get("gradient_clip_val", None),
        )

        # --- train --------------------------------------------------------

        ae_trainer.fit(autoencoder, self.data_module)

        best_ckpt = checkpoint_cb.best_model_path
        print(f"[AE] Best checkpoint : {best_ckpt}")
        print(f"[AE] Best val_loss   : {checkpoint_cb.best_model_score:.6e}")

        # Write sentinel so we can skip retraining on re-runs
        with open(ckpt_sentinel, "w") as f:
            f.write(best_ckpt)

        # --- wrap best model for NRE --------------------------------------
        autoencoder = DenoisingAutoencoder.load_from_checkpoint(best_ckpt)
        autoencoder.to(device)
        autoencoder.eval()
        self.data_summary = AutoencoderWrapper(autoencoder, freeze=True)
        self.data_summary.to(device)

        print(f"[AE] Bottleneck dimensionality: {self.data_summary.get_n_features()}")

        # Free cached data and switch to disk-based loading to reduce memory
        self.data_module.full_dataset.clear_cache()

    def _train_marginal_encoders(self, round_idx, finetune_previous=False):
        """Train one ConvEncoder + RegressionHead per marginal and wrap the
        result as a :class:`MarginalEncoderWrapper` stored in ``self.data_summary``.

        Mirrors ``_train_autoencoder`` in structure so the overall round flow
        is unchanged.
        """
        print(f"Training Marginal Encoders for round {round_idx}...")
        me_conf = self.train_conf["architecture"]["data_summary"]["MarginalEncoder"]
        device  = me_conf.get("device", self.train_conf["device"])

        ckpt_dir     = os.path.join(DATA_ROOT_DIR, TIME_OF_EXECUTION)
        ckpt_tag     = f"marginal_enc_round_{round_idx}"
        ckpt_sentinel = os.path.join(ckpt_dir, f"{ckpt_tag}_done.txt")

        # --- check for existing trained encoder for this round ----------------
        if os.path.exists(ckpt_sentinel):
            best_ckpt = open(ckpt_sentinel).read().strip()
            if os.path.exists(best_ckpt):
                resp = None
                try:
                    resp = input(
                        f"Marginal encoder checkpoint for round {round_idx} already exists at "
                        f"{best_ckpt}. Retrain and overwrite? [y/N]: "
                    ).strip().lower()
                except Exception:
                    resp = "n"
                if resp not in ("y", "yes"):
                    print(f"Reusing existing marginal encoder checkpoint: {best_ckpt}")
                    enc_trainer = MarginalEncoderTrainer.load_from_checkpoint(best_ckpt)
                    enc_trainer.to(device).eval()
                    self.data_summary = MarginalEncoderWrapper(enc_trainer, freeze=True)
                    self.data_summary.to(device)
                    print(f"Bottleneck dim  : {self.data_summary.get_n_features()}")
                    print(f"Num marginals   : {self.data_summary.get_n_marginals()}")
                    self.data_module.full_dataset.clear_cache()
                    return

        # --- flatten marginals across all domains (domain is irrelevant here) --
        marginals_flat = [
            marginal
            for marginal_list in self.train_conf["marginals"].values()
            for marginal in marginal_list
        ]

        # --- retrieve normalisation statistics --------------------------------
        param_mean_np, param_std_np = self.data_module.get_params_mean_std()
        prior_bounds = self.datagen_info.get("conf", {}).get("prior", None)

        hidden_channels = me_conf.get("hidden_channels", (32, 64, 128, 256, 256))
        if isinstance(hidden_channels, list):
            hidden_channels = tuple(hidden_channels)
        regressor_hidden = me_conf.get("regressor_hidden_sizes", (128, 64))
        if isinstance(regressor_hidden, list):
            regressor_hidden = tuple(regressor_hidden)

        # --- build model or reuse previous encoder for fine-tuning -----------
        if not finetune_previous or round_idx == 1:
            enc_trainer = MarginalEncoderTrainer(
                n_channels=me_conf.get("n_channels", 2),
                n_freqs=me_conf.get("n_freqs", 4096),
                marginals=marginals_flat,
                bottleneck_dim=me_conf.get("bottleneck_dim", 200),
                hidden_channels=hidden_channels,
                kernel_size=me_conf.get("kernel_size", 5),
                stride=me_conf.get("stride", 2),
                dropout=me_conf.get("dropout", 0.0),
                residual=me_conf.get("residual", False),
                regressor_hidden_sizes=regressor_hidden,
                param_mean=param_mean_np.tolist(),
                param_std=param_std_np.tolist(),
                lr=me_conf.get("lr", 1e-3),
                weight_decay=me_conf.get("weight_decay", 1e-5),
                scheduler_patience=me_conf.get("scheduler_patience", 75),
                scheduler_factor=me_conf.get("scheduler_factor", 0.3),
                representation=me_conf.get("representation", "real_imag"),
                prior_bounds=prior_bounds,
            )
            enc_trainer = enc_trainer.to(device)

            print("[MarginalEncoder] Fitting normalisation statistics ...")
            norm_loader = self.data_module.train_dataloader(shuffle=False, num_workers=0)
            enc_trainer.fit_normalisation(norm_loader)
        else:
            # Fine-tune from previous round (marginals must be unchanged)
            prev_wrapper = getattr(self, "data_summary", None)
            if prev_wrapper is None or not isinstance(prev_wrapper, MarginalEncoderWrapper):
                raise RuntimeError(
                    "finetune_previous=True but no previous MarginalEncoderWrapper found."
                )
            if [list(m) for m in prev_wrapper.trainer.marginals] != marginals_flat:
                print(
                    "[MarginalEncoder] WARNING: marginals changed between rounds; "
                    "falling back to training from scratch."
                )
                # Recurse with finetune_previous=False
                self._train_marginal_encoders(round_idx, finetune_previous=False)
                return
            enc_trainer = prev_wrapper.trainer
            prev_wrapper.unfreeze_parameters()
            print(
                f"[MarginalEncoder] Fine-tuning from previous round "
                f"(bottleneck_dim={enc_trainer.bottleneck_dim})"
            )

        # --- callbacks --------------------------------------------------------
        checkpoint_cb = ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=2,
            filename=f"{ckpt_tag}" + "-{epoch:03d}-{val_loss:.4e}",
        )
        early_stop_cb = EarlyStopping(
            monitor="val_loss",
            patience=me_conf.get("early_stop_patience", 100),
            mode="min",
        )
        log_name = me_conf.get("log_name", "marginal_encoder")
        logger = TensorBoardLogger(
            save_dir=os.path.join(DATA_ROOT_DIR, "logs"),
            name=f"{TIME_OF_EXECUTION}_{log_name}_round_{round_idx}",
        )

        # --- trainer ----------------------------------------------------------
        me_trainer = Trainer(
            logger=logger,
            max_epochs=me_conf.get("epochs", 500),
            accelerator=device,
            devices=1,
            enable_progress_bar=False,
            callbacks=[checkpoint_cb, early_stop_cb,
                       PeriodicProgressCallback(print_every=20, label="MargEnc")],
            gradient_clip_val=me_conf.get("gradient_clip_val", None),
        )
        me_trainer.fit(enc_trainer, self.data_module)

        best_ckpt = checkpoint_cb.best_model_path
        print(f"[MarginalEncoder] Best checkpoint : {best_ckpt}")
        print(f"[MarginalEncoder] Best val_loss   : {checkpoint_cb.best_model_score:.6e}")

        # Write sentinel for re-run skipping
        os.makedirs(ckpt_dir, exist_ok=True)
        with open(ckpt_sentinel, "w") as f:
            f.write(best_ckpt)

        # --- wrap best model --------------------------------------------------
        enc_trainer = MarginalEncoderTrainer.load_from_checkpoint(best_ckpt)
        enc_trainer.to(device).eval()
        self.data_summary = MarginalEncoderWrapper(enc_trainer, freeze=True)
        self.data_summary.to(device)

        print(f"[MarginalEncoder] Bottleneck dim  : {self.data_summary.get_n_features()}")
        print(f"[MarginalEncoder] Num marginals   : {self.data_summary.get_n_marginals()}")

        self.data_module.full_dataset.clear_cache()

    def _load_marginal_encoders(self, round_idx):
        """Load a trained MarginalEncoderTrainer from the checkpoint path
        specified in ``train_config.yaml`` and wrap it as data_summary.
        """
        me_conf   = self.train_conf["architecture"]["data_summary"]["MarginalEncoder"]
        ckpt_path = me_conf["filename"]
        device    = me_conf.get("device", self.train_conf["device"])

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"MarginalEncoder checkpoint not found at {ckpt_path}."
            )
        print(f"Loading MarginalEncoder from {ckpt_path}...")
        enc_trainer = MarginalEncoderTrainer.load_from_checkpoint(ckpt_path)
        enc_trainer.to(device).eval()
        self.data_summary = MarginalEncoderWrapper(enc_trainer, freeze=True)
        self.data_summary.to(device)
        print(f"Bottleneck dim : {self.data_summary.get_n_features()}")
        print(f"Num marginals  : {self.data_summary.get_n_marginals()}")
        self.data_module.full_dataset.clear_cache()

    def _train_inference_network ( self, round_idx, data_summary=None) :
        #  initialise data summarizer (ROM)
        mean, std = self.data_module.get_params_mean_std()
        normalisation = {"td_normalisation": np.array(self.data_module.get_max_td()),
                         "param_mean": np.array(mean),
                         "param_std": np.array(std)}
        old_model = getattr(self, "model", None)
        ds_type = self.train_conf["architecture"]["data_summary"]["type"]
        NetworkClass = PerMarginalInferenceNetwork if ds_type == "MarginalEncoder" else InferenceNetwork
        self.model = NetworkClass(train_conf=self.train_conf, dataset_info=self.datagen_info, normalisation=normalisation, data_summarizer=data_summary, periodic_bc_params=self.train_conf["periodic_bc_params"])
        # Carry over classifier weights from the previous round (finetune)
        if old_model is not None:
            transfer_classifier_weights(old_model, self.model)
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
            call_every_n_epochs=2,
            training_start_time=self.training_start)

        callbacks_list = [checkpoint_callback, early_stopping_callback, plot_posterior_callback]
        vr_conf = self.train_conf.get("volume_ratio_early_stop", {})
        if vr_conf.get("enabled", False):
            vr_callback = VolumeRatioEarlyStopping(
                warmup_epochs=vr_conf.get("warmup_epochs", 50),
                patience=vr_conf.get("patience", 10),
                rel_tol=vr_conf.get("rel_tol", 0.02),
                ema_alpha=vr_conf.get("ema_alpha", 0.3),
                min_ratio_threshold=vr_conf.get("min_ratio_threshold", 0.5),
            )
            callbacks_list.append(vr_callback)

        callbacks_list.append(PeriodicProgressCallback(print_every=20, label="NRE"))
        trainer = Trainer(
            logger=logger,
            max_epochs=self.train_conf["epochs"],
            accelerator=self.train_conf["device"],
            devices=1,
            enable_progress_bar=False,
            callbacks=callbacks_list,
        )

        # Reserve GPU memory on the first round so other processes cannot
        # steal it during the overnight run.
        if self.train_conf["device"] == "cuda" and not getattr(self, "_gpu_reserved", False):
            from pembhb.gpu_utils import reserve_gpu_memory
            safety = self.train_conf.get("gpu_reserve_safety_factor", 1.25)
            reserve_gpu_memory(self.model, self.data_module.train_dataloader(),
                               safety_factor=safety)
            self._gpu_reserved = True

        trainer.fit(self.model, self.data_module)
        if hasattr(self.model, 'widest_boxes'):
            print(f"Widest boxes after training: {self.model.widest_boxes}")
        if copy_bounds_file: 
            shutil.copy(self.data_fname_yaml, logger.log_dir)


    def round(self, idx, sampler_init_kwargs):
        self._generate_data(round_idx=idx, sampler_init_kwargs=sampler_init_kwargs)
        data_summary_type = self.train_conf["architecture"]["data_summary"]["type"]
        data_summary_load = self.train_conf["architecture"]["data_summary"][data_summary_type]["load"]
        if data_summary_type == "ROM":
            if data_summary_load:
                #load the rom from file specified in train_config.yaml

                self._load_rom(round_idx=idx)
            else:
                self._train_rom(round_idx=idx)
            # if idx ==1: 
            #     self.model = InferenceNetwork.load_from_checkpoint("/data/gpuleo/mbhb/logs/20260217rom_1000_fullsky_narrowmc_tc_v0_round_1/version_1/checkpoints/epoch=26-step=3780.ckpt")
            #     print("warning: using pretrained inference network for round 1, skipping training for this round")
            #else: 
                self._train_inference_network(round_idx=idx, data_summary=self.data_summary)
        elif data_summary_type == "Autoencoder":
            if data_summary_load:
                self._load_autoencoder(round_idx=idx)
            else:
                self._train_autoencoder(round_idx=idx, finetune_previous=self.train_conf["architecture"]["data_summary"]["Autoencoder"]["finetune_previous"])
            self._train_inference_network(round_idx=idx, data_summary=self.data_summary)
        elif data_summary_type == "MarginalEncoder":
            if data_summary_load:
                self._load_marginal_encoders(round_idx=idx)
            else:
                self._train_marginal_encoders(
                    round_idx=idx,
                    finetune_previous=self.train_conf["architecture"]["data_summary"]["MarginalEncoder"].get("finetune_previous", True),
                )
            self._train_inference_network(round_idx=idx, data_summary=self.data_summary)
        else:
            # No data summary (direct inference on raw data)
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
        for i in range(1,n_rounds+1):
            print(f"Running round {i}...")
            # Resolve scheduled marginals for this round
            active_marginals = resolve_marginals_for_round(self.train_conf, i)
            self.train_conf["marginals"] = active_marginals
            validate_marginals(active_marginals)
            print(f"Active marginals for round {i}: {active_marginals}")

            if i == 1 and self.fisher_prior_bounds is not None:
                # Update datagen_conf["prior"] so that conf and
                # sampler_init_kwargs stay consistent in the YAML sidecar.
                self.datagen_conf["prior"].update(copy.deepcopy(self.fisher_prior_bounds))
                sampler_kwargs = {"prior_bounds": self.fisher_prior_bounds}
                print("[Fisher] Using Fisher-based prior for round 1.")
                import yaml as _yaml
                _out = os.path.join(DATA_ROOT_DIR, TIME_OF_EXECUTION, "fisher_prior_round_1.yaml")
                os.makedirs(os.path.dirname(_out), exist_ok=True)
                with open(_out, "w") as _f:
                    _yaml.safe_dump({"fisher_prior_bounds": self.fisher_prior_bounds}, _f)
                print(f"[Fisher] Saved Fisher prior bounds to {_out}")
            else:
                sampler_kwargs = {"prior_bounds": self.datagen_conf["prior"]}
            self.round(
                idx=i,
                sampler_init_kwargs=sampler_kwargs,
            )
            if self.train_conf["device"] == "cuda":
                torch.cuda.empty_cache()
            pp_device = choose_device_for_pp(required_gib=2.0)
            pp_output_dir = os.path.join(ROOT_DIR, "plots", TIME_OF_EXECUTION)
            out_idx = 0
            for key, marginal_list in self.train_conf["marginals"].items():
                for marginal in marginal_list:
                    marginal_key = tuple(marginal)

                    if len(marginal) == 1:
                        # 1D marginal
                        inj_idx = marginal[0]
                        param_name = utils._ORDERED_PRIOR_KEYS[inj_idx]
                        utils.pp_plot(self.test_dataloader, self.model, in_param_idx=inj_idx, name=f"round_{i}_{key}", out_param_idx=out_idx, output_dir=pp_output_dir, device=pp_device)
                        
                        # Update prior bounds from widest_boxes
                        if hasattr(self.model, 'widest_boxes') and marginal_key in self.model.widest_boxes:
                            widest_interval = self.model.widest_boxes[marginal_key]
                            tmp_prior = copy.deepcopy(self.datagen_conf["prior"])
                            tmp_prior[param_name] = [widest_interval[0], widest_interval[1]]
                            self.datagen_conf["prior"] = tmp_prior
                        else:
                            print(f"Warning: No widest_interval found for 1D marginal {marginal_key} ({param_name})")
                            
                    elif len(marginal) == 2:
                        # 2D marginal
                        inj1, inj2 = marginal
                        if marginal_key == (7, 8):  # sky marginal
                            from pembhb.sky_truncation import truncate_sky_prior
                            _, sky_info = truncate_sky_prior(
                                self.model, self.dataloader_obs, out_param_idx=out_idx,
                                datagen_conf=self.datagen_conf,
                                mode="rectangle",
                                credible_level=0.9545, dilation_factor=1.5,
                            )
                        # Get widest_box for this specific marginal from the dictionary
                        elif hasattr(self.model, 'widest_boxes') and marginal_key in self.model.widest_boxes:
                            widest_box = self.model.widest_boxes[marginal_key]
                            tmp_prior = copy.deepcopy(self.datagen_conf["prior"])
                            tmp_prior[utils._ORDERED_PRIOR_KEYS[inj1]] = [widest_box[0], widest_box[1]]
                            tmp_prior[utils._ORDERED_PRIOR_KEYS[inj2]] = [widest_box[2], widest_box[3]]
                            self.datagen_conf["prior"] = tmp_prior
                        else:
                            print(f"Warning: No widest_box found for 2D marginal {marginal_key}")
                    out_idx += 1

            print(f"Updated prior after round {i}: {self.datagen_conf['prior']}")
            self._plot_updated_prior_bounds(self.datagen_conf["prior"])

            if self.train_conf["device"] == "cuda":
                torch.cuda.empty_cache()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="Name for this run, used in logging and checkpointing")
    parser.add_argument("--train-config", default="train_config.yaml",
                        help="Train config filename inside configs/")
    

    args = parser.parse_args()

    train_config_filename = args.train_config
    run_name = args.name
    datagen_config_filename = "datagen_config.yaml"

    train_config   = utils.read_config(os.path.join(ROOT_DIR, "configs", train_config_filename))
    datagen_config = utils.read_config(os.path.join(ROOT_DIR, "configs", datagen_config_filename))

    # Activate the configured precision (default: float32)
    set_precision(train_config.get("precision", "float32"))

    # Dynamic timestamp incorporating the data summary type
    ds_type = train_config["architecture"]["data_summary"]["type"].lower()
    TIME_OF_EXECUTION = get_timestamp() + f"_{ds_type}_{run_name}"

    trainer = SequentialTrainer(train_conf=train_config, datagen_conf=datagen_config, dataset_obs_path="/data/gpuleo/mbhb/obs_logfreq_q3_t.h5")
    trainer.run(n_rounds=9)

    #round(conf, sampler_init_kwargs={'low': 0.5, 'high': 1.0} , lr=conf["training"]["learning_rate"], idx=0)