import os, shutil
import torch
from pembhb.simulator import MBHBSimulatorFD_TD, DummySimulator
from pembhb.model import InferenceNetwork, PeregrineModel, ReducedOrderModel, ROMWrapper
from pembhb.autoencoder import DenoisingAutoencoder, AutoencoderWrapper
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
TIME_OF_EXECUTION = get_timestamp()+"autoenc_fullsky_narrowmc_tc_v0"

def validate_marginals(marginals_config: dict):
    """Validate that no parameter index appears in multiple marginals.
    
    :param marginals_config: dictionary containing marginal lists from train_config
    :raises ValueError: if a parameter index is repeated across marginals
    """
    all_indices = []
    for key, marginal_list in marginals_config.items():
        for marginal in marginal_list:
            for idx in marginal:
                if idx in all_indices:
                    raise ValueError(
                        f"Parameter index {idx} ({utils._ORDERED_PRIOR_KEYS[idx]}) appears in multiple marginals. "
                        f"Each parameter index can only appear in one marginal for prior truncation."
                    )
                all_indices.append(idx)

def get_widest_interval_1d(model, dataloader, in_param_idx, out_param_idx, eps=0.003):
    """Get the widest credible interval for a 1D marginal posterior.
    
    :param model: trained inference model
    :param dataloader: dataloader containing the observation
    :param in_param_idx: index of the input parameter
    :param out_param_idx: index of the output (logratio)
    :param eps: credible level (default 0.003 for 99.7% interval)
    :return: (widest_interval, norm1d, grid, inj_params) where widest_interval is [low, high]
    """
    logratios, inj_params, grid = utils.get_logratios_grid(
        dataloader,
        model,
        ngrid_points=1000,
        in_param_idx=in_param_idx,
        out_param_idx=out_param_idx,
    )
    
    ratios = np.exp(logratios[0])  # Take first (only) observation
    dp = grid[1, 0] - grid[0, 0]
    norm1d = ratios / np.sum(ratios * dp)
    
    # Find credible interval using cumulative sum
    cumsum = np.cumsum(norm1d * dp)
    idx_low = np.searchsorted(cumsum, eps / 2)
    idx_high = np.searchsorted(cumsum, 1 - eps / 2)
    
    widest_interval = [float(grid[idx_low, 0]), float(grid[idx_high, 0])]
    return widest_interval, norm1d, grid, inj_params

def get_widest_box_2d(model, dataloader, in_param_idx, out_param_idx, ax_buffer=None, do_plot=False):
    """Get the widest credible box for a 2D marginal posterior.
    
    :param model: trained inference model
    :param dataloader: dataloader containing the observation
    :param in_param_idx: tuple of indices for the two input parameters
    :param out_param_idx: index of the output (logratio)
    :param ax_buffer: matplotlib axis to plot on (optional)
    :param do_plot: whether to create the contour plot
    :return: (widest_box, inj_params) where widest_box is [x_low, x_high, y_low, y_high]
    """
    logratios, inj_params, gx, gy = utils.get_logratios_grid_2d(
        dataloader,
        model,
        ngrid_points=100,
        in_param_idx=in_param_idx,
        out_param_idx=out_param_idx,
    )

    ratios = np.exp(logratios)
    dp1 = gx[1, 0] - gx[0, 0]
    dp2 = gy[0, 1] - gy[0, 0]
    norm2d = ratios / np.sum(ratios * dp1 * dp2, axis=(1, 2), keepdims=True)
    levels, labels = utils.contour_levels(norm2d)
    boxes = utils.posterior_contours_2d(
        gx, gy, norm2d[0],
        inj_params[0], 
        ax_buffer=ax_buffer, 
        parameter_names=[utils._ORDERED_PRIOR_KEYS[in_param_idx[0]], utils._ORDERED_PRIOR_KEYS[in_param_idx[1]]],
        levels=levels, 
        levels_labels=labels,
        do_plot=do_plot
    )
    widest_box = boxes[0]
    return widest_box, inj_params

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
        # Storage for volume ratio diagnostics
        self.volume_ratios = {}
    
    def _compute_posterior_volume_2d(self, widest_box):
        """
        Compute the area/volume of the posterior from the widest contour box.
        
        Parameters:
        -----------
        widest_box : tuple
            The bounding box of the 99.99% contour.
            Currently: (x_min, x_max, y_min, y_max) for axis-aligned boxes.
            
        Returns:
        --------
        float
            Area enclosed by the posterior contour.
            
        Notes:
        ------
        FUTURE EXTENSION FOR TILTED BOXES:
        - If posterior contours become non-axis-aligned, widest_box format may change
          to a list of vertices [(x1,y1), (x2,y2), ...]
        - In that case, use Shoelace formula or similar for polygon area:
          area = 0.5 * abs(sum(x[i]*y[i+1] - x[i+1]*y[i] for i in range(n)))
        - Consider using shapely.geometry.Polygon for robust area calculation
        """
        # Current implementation: axis-aligned box
        # widest_box = (x_min, x_max, y_min, y_max)
        posterior_area = (widest_box[1] - widest_box[0]) * (widest_box[3] - widest_box[2])
        return posterior_area
    
    def _compute_prior_volume_2d(self, pl_module, in_param_idx):
        """
        Compute the area/volume of the prior for a 2D marginal.
        
        Parameters:
        -----------
        pl_module : LightningModule
            The model containing prior information in hparams.
        in_param_idx : tuple
            Indices of the two parameters defining the 2D marginal.
            
        Returns:
        --------
        float
            Area of the prior region.
            
        Notes:
        ------
        **MODIFY THIS METHOD WHEN SWITCHING TO TILTED BOUNDING BOXES**
        
        Current implementation assumes axis-aligned rectangular priors.
        Prior bounds are stored as:
            prior_dict[param_name] = [min_value, max_value]
        
        For tilted/rotated bounding boxes:
        1. Prior specification will change (e.g., vertices, rotation matrix, etc.)
        2. Access prior from: pl_module.hparams["dataset_info"]["conf"]["prior"]
        3. Compute area based on new representation:
           - If vertices: use Shoelace formula or shapely.geometry.Polygon
           - If rotation + bounds: compute area of rotated rectangle
           - Example with vertices:
             ```python
             vertices = prior_dict[marginal_key]  # [(x1,y1), (x2,y2), ...]
             from shapely.geometry import Polygon
             prior_area = Polygon(vertices).area
             ```
        4. Ensure consistency with sampler_init_kwargs format in sampler.py
        
        Potential issues to address:
        - Normalization: If grid evaluation doesn't align with tilted prior,
          posterior normalization may be affected
        - Grid coverage: Axis-aligned grids may inefficiently cover tilted regions
        - Coordinate transforms: May need to transform between rotated and
          canonical coordinate systems
        """
        # Current implementation: axis-aligned rectangular prior
        prior_dict = pl_module.hparams["dataset_info"]["conf"]["prior"]
        
        # Get bounds for each parameter
        param_name_0 = utils._ORDERED_PRIOR_KEYS[in_param_idx[0]]
        param_name_1 = utils._ORDERED_PRIOR_KEYS[in_param_idx[1]]
        
        prior_bounds_0 = prior_dict[param_name_0]
        prior_bounds_1 = prior_dict[param_name_1]
        
        # Compute area as product of widths
        prior_area = (prior_bounds_0[1] - prior_bounds_0[0]) * (prior_bounds_1[1] - prior_bounds_1[0])
        
        return prior_area

    def _compute_posterior_volume_1d(self, widest_interval):
        """Compute the width of the posterior credible interval for a 1D marginal.
        
        Parameters:
        -----------
        widest_interval : list
            [low, high] bounds of the credible interval.
            
        Returns:
        --------
        float
            Width of the posterior interval.
        """
        return widest_interval[1] - widest_interval[0]
    
    def _compute_prior_volume_1d(self, pl_module, in_param_idx):
        """Compute the width of the prior for a 1D marginal.
        
        Parameters:
        -----------
        pl_module : LightningModule
            The model containing prior information in hparams.
        in_param_idx : int
            Index of the parameter.
            
        Returns:
        --------
        float
            Width of the prior range.
        """
        prior_dict = pl_module.hparams["dataset_info"]["conf"]["prior"]
        param_name = utils._ORDERED_PRIOR_KEYS[in_param_idx]
        prior_bounds = prior_dict[param_name]
        return prior_bounds[1] - prior_bounds[0]

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.epochs_elapsed == 0: 
            os.makedirs(os.path.join(ROOT_DIR, "plots", TIME_OF_EXECUTION), exist_ok=True)

        self.epochs_elapsed += 1
        if (self.epochs_elapsed-2) % self.call_every_n_epochs == 0:
            #print("plotting posteriors on observed data")
            train_time = datetime.now() - self.init_time
            td_trunc = train_time - timedelta(microseconds=train_time.microseconds)
            title_plot = f"training time={td_trunc}s"
            # plot the posterior on the observed data , using the current model
            for i in range(self.n_marginals):
                in_param_idx = self.input_idx_list[i]
                out_param_idx = self.output_idx_list[i]

                # Initialize widest_boxes dict if not present
                if not hasattr(pl_module, 'widest_boxes'):
                    pl_module.widest_boxes = {}
                marginal_key = tuple(in_param_idx)

                if len(in_param_idx) == 1:
                    # Handle 1D marginals
                    param_idx = in_param_idx[0]
                    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
                    fig.suptitle(
                        f"Round {self.round_idx} - Epoch {trainer.current_epoch} - {title_plot}",
                        fontsize=10,
                    )
                    
                    try:
                        widest_interval, norm1d, grid, inj_params = get_widest_interval_1d(
                            pl_module,
                            self.obs_loader,
                            in_param_idx=param_idx,
                            out_param_idx=out_param_idx
                        )
                        
                        # Plot
                        ax.plot(grid.flatten(), norm1d, 'b-', linewidth=1.5)
                        ax.axvline(inj_params[0], color='r', linestyle='--', label='Injection')
                        ax.axvline(widest_interval[0], color='g', linestyle=':', label='99.7% CI')
                        ax.axvline(widest_interval[1], color='g', linestyle=':')
                        ax.fill_between(grid.flatten(), 0, norm1d, 
                                       where=(grid.flatten() >= widest_interval[0]) & (grid.flatten() <= widest_interval[1]),
                                       alpha=0.3, color='green')
                        ax.set_xlabel(utils._ORDERED_PRIOR_KEYS[param_idx])
                        ax.set_ylabel('Posterior density')
                        ax.legend()
                        
                        # Store the widest interval
                        pl_module.widest_boxes[marginal_key] = widest_interval
                        
                        # Compute posterior-to-prior volume (width) ratio for 1D marginal
                        posterior_width = self._compute_posterior_volume_1d(widest_interval)
                        prior_width = self._compute_prior_volume_1d(pl_module, param_idx)
                        volume_ratio = posterior_width / prior_width
                        
                        # Store and log the volume ratio
                        if marginal_key not in self.volume_ratios:
                            self.volume_ratios[marginal_key] = []
                        self.volume_ratios[marginal_key].append({
                            'epoch': trainer.current_epoch,
                            'ratio': volume_ratio,
                            'posterior_width': posterior_width,
                            'prior_width': prior_width
                        })
                        
                        # Log metric to tensorboard if logger exists
                        if trainer.logger is not None:
                            metric_name = f"volume_ratio/{utils._ORDERED_PRIOR_KEYS[param_idx]}"
                            trainer.logger.log_metrics({metric_name: volume_ratio}, step=trainer.current_epoch)
                        
                        # Print diagnostic
                        param_name = utils._ORDERED_PRIOR_KEYS[param_idx]
                        print(f"Round {self.round_idx}, Epoch {trainer.current_epoch}, {param_name}: "
                              f"Volume ratio (posterior/prior) = {volume_ratio:.6f} "
                              f"(posterior width: {posterior_width:.6e}, prior width: {prior_width:.6e})")
                        
                        out = os.path.join(ROOT_DIR, "plots", self.timestamp, 
                                          f"posterior_round_{self.round_idx}_epoch_{trainer.current_epoch}_{utils._ORDERED_PRIOR_KEYS[param_idx]}.pdf")
                        fig.savefig(out, bbox_inches="tight")
                    except Exception as e:
                        print(f"Error plotting 1D marginal for {utils._ORDERED_PRIOR_KEYS[param_idx]}: {e}")
                    finally:
                        plt.close(fig)

                elif len(in_param_idx) == 2:
                    # Handle 2D marginals
                    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
                    fig.tight_layout()
                    fig.suptitle(
                        f"Round {self.round_idx} - Epoch {trainer.current_epoch} - {title_plot}",
                        fontsize=10,
                    )

                    try:
                        widest_box, inj_params = get_widest_box_2d(
                            pl_module,
                            self.obs_loader,
                            in_param_idx=in_param_idx,
                            out_param_idx=out_param_idx,
                            ax_buffer=ax,
                            do_plot=True
                        )

                        # Store widest_box keyed by the marginal (tuple of input parameter indices)
                        pl_module.widest_boxes[marginal_key] = widest_box
                        
                        # Compute posterior-to-prior volume ratio for 2D marginal
                        posterior_area = self._compute_posterior_volume_2d(widest_box)
                        prior_area = self._compute_prior_volume_2d(pl_module, in_param_idx)
                        volume_ratio = posterior_area / prior_area
                        
                        # Store and log the volume ratio
                        if marginal_key not in self.volume_ratios:
                            self.volume_ratios[marginal_key] = []
                        self.volume_ratios[marginal_key].append({
                            'epoch': trainer.current_epoch,
                            'ratio': volume_ratio,
                            'posterior_area': posterior_area,
                            'prior_area': prior_area
                        })
                        
                        # Log metric to tensorboard if logger exists
                        if trainer.logger is not None:
                            metric_name = f"volume_ratio/{utils._ORDERED_PRIOR_KEYS[in_param_idx[0]]}_{utils._ORDERED_PRIOR_KEYS[in_param_idx[1]]}"
                            trainer.logger.log_metrics({metric_name: volume_ratio}, step=trainer.current_epoch)
                        
                        # Print diagnostic
                        param_names = f"{utils._ORDERED_PRIOR_KEYS[in_param_idx[0]]}-{utils._ORDERED_PRIOR_KEYS[in_param_idx[1]]}"
                        print(f"Round {self.round_idx}, Epoch {trainer.current_epoch}, {param_names}: "
                              f"Volume ratio (posterior/prior) = {volume_ratio:.6f} "
                              f"(posterior area: {posterior_area:.6e}, prior area: {prior_area:.6e})")
                        
                        out = os.path.join(ROOT_DIR, "plots", self.timestamp,
                                          f"posterior_round_{self.round_idx}_epoch_{trainer.current_epoch}_{utils._ORDERED_PRIOR_KEYS[in_param_idx[0]]}_{utils._ORDERED_PRIOR_KEYS[in_param_idx[1]]}.pdf")
                        fig.savefig(out, bbox_inches="tight")
                    except ValueError as ve:
                        print(f"caught ValueError: {ve} during contour plotting, skipping this plot")
                    finally:
                        plt.close(fig)

    def on_train_end(self, trainer, pl_module):
        self.on_validation_epoch_end(trainer, pl_module)
        print(f"Total training time: {datetime.now() - self.init_time}")
class SequentialTrainer:
    def __init__(self, train_conf, datagen_conf, dataset_obs_path):
        self.train_conf = train_conf
        self.datagen_conf = datagen_conf
        self.dataset_obs_path = dataset_obs_path
        
        # Validate that no parameter index appears in multiple marginals
        validate_marginals(train_conf["marginals"])
        
        # Subset is there because utils.mbhb_collate_fn expects a Subset, it will access its dataset attribute
        self.dataset_observation = Subset(MBHBDataset(dataset_obs_path, cache_in_memory=True), indices=[0])
        #self.dataset_observation = Subset(MBHBDataset(dataset_obs_path, cache_in_memory=True),indices =[0])
        self.dataloader_obs = DataLoader(self.dataset_observation, batch_size=train_conf["batch_size"], shuffle=False, collate_fn=lambda b: mbhb_collate_fn(b, self.dataset_observation, noise_shuffling=False, noise_factor=self.train_conf["noise_factor"]))
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
                            in_param_idx=marginal[0], out_param_idx=out_idx
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
        self.data_module.setup(stage="fit")
        self.test_dataloader = self.data_module.test_dataloader()

    def _train_rom ( self, round_idx) : 
        print("Training ROM...")
        rom = ReducedOrderModel(tolerance=1e-9, device="cuda", batch_size=5000)
        filename = os.path.join(DATA_ROOT_DIR, TIME_OF_EXECUTION, f"rom_round_{round_idx}.pt")
        
        # Get ROM training options from config (defaults optimize for multi-session usage)
        use_pinned_memory = self.train_conf.get("rom_use_pinned_memory", False)
        prefetch_batches = self.train_conf.get("rom_prefetch_batches", 1)
        
        if not os.path.exists(filename):
            train_dl = self.data_module.train_dataloader(num_workers=0, pin_memory=use_pinned_memory)
            rom.train(train_dl, use_pinned_memory=use_pinned_memory, prefetch_batches=prefetch_batches)
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
                rom.train(train_dl, use_pinned_memory=use_pinned_memory, prefetch_batches=prefetch_batches)
                rom.to_file(filename)
                print(f"Retrained ROM and saved to {filename}")
            else:
                print(f"Using existing ROM at {filename}")

        self.data_summary = ROMWrapper(filename=filename, device=self.train_conf["device"])
        
        # Free cached data and switch to disk-based loading to reduce memory
        self.data_module.full_dataset.clear_cache()

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
            call_every_n_epochs=5)

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
        if hasattr(self.model, 'widest_boxes'):
            print(f"Widest boxes after training: {self.model.widest_boxes}")
        if copy_bounds_file: 
            shutil.copy(self.data_fname_yaml, logger.log_dir)


    def round(self, idx, sampler_init_kwargs):
        self._generate_data(round_idx=idx, sampler_init_kwargs=sampler_init_kwargs)
        data_summary_type = self.train_conf["architecture"]["data_summary"]["type"]
        
        if data_summary_type == "ROM":
            self._train_rom(round_idx=idx)
            self._train_inference_network(round_idx=idx, data_summary=self.data_summary)
        elif data_summary_type == "Autoencoder":
            self._load_autoencoder(round_idx=idx)
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
            self.round(
                idx=i,
                sampler_init_kwargs={"prior_bounds": self.datagen_conf["prior"]},
            )
            # model = InferenceNetwork.load_from_checkpoint("/u/g/gpuleo/pembhb/logs/20251111_100948_round_0/version_0/checkpoints/epoch=13-step=2380.ckpt")
            out_idx = 0
            for key, marginal_list in self.train_conf["marginals"].items():
                for marginal in marginal_list:
                    marginal_key = tuple(marginal)
                    
                    if len(marginal) == 1:
                        # 1D marginal
                        inj_idx = marginal[0]
                        param_name = utils._ORDERED_PRIOR_KEYS[inj_idx]
                        utils.pp_plot(self.test_dataloader, self.model, in_param_idx=inj_idx, name=f"round_{i}_{key}", out_param_idx=out_idx)
                        
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
                        #utils.pp_plot_2d(self.test_dataloader, self.model, in_param_idx=marginal, out_idx=out_idx,
                        #           name=f"{ROOT_DIR}/plots/{TIME_OF_EXECUTION}/round_{i}_{utils._ORDERED_PRIOR_KEYS[inj1]}_{utils._ORDERED_PRIOR_KEYS[inj2]}")
            
                        # Get widest_box for this specific marginal from the dictionary
                        if hasattr(self.model, 'widest_boxes') and marginal_key in self.model.widest_boxes:
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
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--config", type=str, required=True)
    # args = parser.parse_args()
    train_config_filename = "train_config.yaml"
    datagen_config_filename = "datagen_config.yaml"
    
    train_config   = utils.read_config(os.path.join(ROOT_DIR, "configs", train_config_filename))
    datagen_config = utils.read_config(os.path.join(ROOT_DIR, "configs", datagen_config_filename))
                               
    trainer = SequentialTrainer(train_conf=train_config, datagen_conf=datagen_config, dataset_obs_path="/data/gpuleo/mbhb/observation_skyloc.h5")
    
    # run with low noise: 
    # trainer = SequentialTrainer(train_conf=train_config, datagen_conf=datagen_config, dataset_obs_path=os.path.join(ROOT_DIR, "/data/gpuleo/mbhb/observation_low_noise.h5"))
    trainer.run(n_rounds=4)

    #round(conf, sampler_init_kwargs={'low': 0.5, 'high': 1.0} , lr=conf["training"]["learning_rate"], idx=0)