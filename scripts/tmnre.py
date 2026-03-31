import os, shutil
import numpy as np
import torch
from torch.utils.data import DataLoader , Subset
import copy
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint, Callback


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
from pembhb.utils import validate_marginals, resolve_marginals_for_round, transfer_classifier_weights, get_widest_interval_1d, get_widest_box_2d
from pembhb.callbacks import PlotPosteriorCallback, VolumeRatioEarlyStopping

def get_timestamp():
    return datetime.now().strftime("%Y%m%d")

TIME_OF_EXECUTION = get_timestamp() + "_sequential_volratio_earlystop"

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

def get_widest_interval_1d(model, dataloader, in_param_idx, out_param_idx, eps=0.0001):
    """Get the widest credible interval for a 1D marginal posterior.
    
    :param model: trained inference model
    :param dataloader: dataloader containing the observation
    :param in_param_idx: index of the input parameter
    :param out_param_idx: index of the output (logratio)
    :param eps: credible level (default 0.0001 for 99.99% interval)
    :return: (widest_interval, norm1d, grid, inj_params) where widest_interval is [low, high]
    """
    logratios, inj_params, grid = utils.get_logratios_grid(
        dataloader,
        model,
        ngrid_points=100,
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
    dp1 = gx[0, 1] - gx[0, 0]  # param_0 spacing (x varies along columns with xy indexing)
    dp2 = gy[1, 0] - gy[0, 0]  # param_1 spacing (y varies along rows with xy indexing)
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
        # Use the actual sampling prior (sampler_init_kwargs) as the
        # authoritative source.  Fall back to conf["prior"] for backward compat.
        _sik = pl_module.hparams["dataset_info"].get("sampler_init_kwargs", {})
        if "prior_bounds" in _sik:
            prior_dict = _sik["prior_bounds"]
        else:
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
        # Use the actual sampling prior (sampler_init_kwargs) as the
        # authoritative source.  Fall back to conf["prior"] for backward compat.
        _sik = pl_module.hparams["dataset_info"].get("sampler_init_kwargs", {})
        if "prior_bounds" in _sik:    
            prior_dict = _sik["prior_bounds"]
        else:
            prior_dict = pl_module.hparams["dataset_info"]["conf"]["prior"]
        param_name = utils._ORDERED_PRIOR_KEYS[in_param_idx]
        prior_bounds = prior_dict[param_name]
        return prior_bounds[1] - prior_bounds[0]

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.epochs_elapsed == 0: 
            os.makedirs(os.path.join(ROOT_DIR, "plots", self.timestamp), exist_ok=True)

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
                        epsilon_value = 1e-4
                        widest_interval, norm1d, grid, inj_params = get_widest_interval_1d(
                            pl_module,
                            self.obs_loader,
                            in_param_idx=param_idx,
                            out_param_idx=out_param_idx,
                            eps=epsilon_value
                        )
                        
                        # Plot
                        ax.plot(grid.flatten(), norm1d, 'b-', linewidth=1.5)
                        ax.axvline(inj_params[0], color='r', linestyle='--', label='Injection')
                        ax.axvline(widest_interval[0], color='g', linestyle=':', label=f'{100*(1-epsilon_value):.2f}% CI')
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
                        raise e
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
        
        #self.dataset_observation = Subset(MBHBDataset(dataset_obs_path, cache_in_memory=True), indices=[2])
        self.dataset_observation = Subset(MBHBDataset(dataset_obs_path, cache_in_memory=True),indices =[0])
        obs_noise_scale = self.dataset_observation.dataset.noise_scale
        obs_td_params = self.dataset_observation.dataset.td_params
        self.dataloader_obs = DataLoader(self.dataset_observation, batch_size=train_conf["batch_size"], shuffle=False, collate_fn=lambda b: mbhb_collate_fn(b, obs_noise_scale, noise_factor=self.train_conf["noise_factor"], noise_shuffling=False, td_params=obs_td_params))
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
            enable_progress_bar=True,
            callbacks=[checkpoint_cb, early_stop_cb, ae_diag_cb],
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
            enable_progress_bar=True,
            callbacks=[checkpoint_cb, early_stop_cb],
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
            call_every_n_epochs=2)

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

        trainer = Trainer(
            logger=logger,
            max_epochs=self.train_conf["epochs"],
            accelerator=self.train_conf["device"],
            devices=1,
            enable_progress_bar=True,
            callbacks=callbacks_list,
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
                        if marginal_key == (7, 8):  # sky marginal
                            from pembhb.sky_truncation import truncate_sky_prior
                            _, sky_info = truncate_sky_prior(
                                self.model, self.test_dataloader, out_param_idx=out_idx,
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
    parser.add_argument("--train-config", default="train_config.yaml",
                        help="Train config filename inside configs/")
    args = parser.parse_args()

    train_config_filename = args.train_config
    datagen_config_filename = "datagen_config.yaml"

    train_config   = utils.read_config(os.path.join(ROOT_DIR, "configs", train_config_filename))
    datagen_config = utils.read_config(os.path.join(ROOT_DIR, "configs", datagen_config_filename))

    # Activate the configured precision (default: float32)
    set_precision(train_config.get("precision", "float32"))

    # Dynamic timestamp incorporating the data summary type
    ds_type = train_config["architecture"]["data_summary"]["type"].lower()
    TIME_OF_EXECUTION = get_timestamp() + f"_{ds_type}_sequential_v3nonres"

    trainer = SequentialTrainer(train_conf=train_config, datagen_conf=datagen_config, dataset_obs_path="/data/gpuleo/mbhb/obs_logfreq_q3_t.h5")
    trainer.run(n_rounds=9)

    #round(conf, sampler_init_kwargs={'low': 0.5, 'high': 1.0} , lr=conf["training"]["learning_rate"], idx=0)