from torch.utils.data import Dataset, random_split, DataLoader, Subset 
import lightning as L
from pembhb.utils import mbhb_collate_fn
from pembhb import get_torch_dtype, get_torch_complex_dtype
import torch
import numpy as np
import h5py
import gc
# i want a class / function that can : 
# generate data and store in memory using a sampler and a simulator 
# optionally save and load the data to/from disk


class MBHBDataset(Dataset):

    def __init__(self, filename: str, cache_in_memory: bool = False):
        self.filename = filename
        self.cache_in_memory = cache_in_memory
        self.data_cache = {} if cache_in_memory else None

        with h5py.File(self.filename, "r") as f:
            self.len = f["wave_fd"].shape[0]
            self.has_td = "wave_td" in f
            # Stored noise (e.g. for observation files): when present, the
            # collate fn will use it as-is rather than drawing fresh noise.
            self.has_stored_noise = "noise_fd" in f

            # Load ASD (Amplitude Spectral Density) for noise-weighting.
            # Shape: (n_channels, n_freq).  This is the same for all samples.
            if "asd" in f:
                self.asd = torch.tensor(f["asd"][()], device="cpu", dtype=get_torch_dtype())
            else:
                self.asd = None

            # Pre-compute noise_scale = filtered_asd / sqrt(4 * df) for on-the-fly noise generation.
            # Zero-valued bins (below high-pass cutoff) naturally produce zero noise.
            if "asd" in f and "frequencies" in f:
                asd_np = f["asd"][()]
                freqs_np = f["frequencies"][()]
                # Per-bin df: stored explicitly for non-uniform grids; derived for uniform grids
                T_obs_total = f.attrs["observation_duration_SI"]
                filtered_asd = asd_np.copy()
                filtered_asd[:, freqs_np < 5e-5] = 0.0
                self.noise_scale = torch.tensor(
                    filtered_asd / np.sqrt(4.0 / T_obs_total ), dtype=get_torch_dtype()
                )
            else:
                self.noise_scale = None

            # TD params (dt, n_time) needed for on-the-fly TD noise via IFFT
            if self.has_td and "times_SI" in f:
                times_np = f["times_SI"][()]
                self.td_params = (float(times_np[1] - times_np[0]), len(times_np))
            else:
                self.td_params = None

            if cache_in_memory:
                self.wave_fd = torch.tensor(f["wave_fd"][()], device="cpu", dtype=get_torch_complex_dtype())
                self.source_parameters = torch.tensor(f["source_parameters"][()], device="cpu", dtype=get_torch_dtype())
                if self.has_td:
                    self.wave_td = torch.tensor(f["wave_td"][()], device="cpu", dtype=get_torch_dtype())
                else:
                    self.wave_td = None
                if self.has_stored_noise:
                    self.noise_fd = torch.tensor(f["noise_fd"][()], device="cpu", dtype=get_torch_complex_dtype())
                else:
                    self.noise_fd = None
            else:
                self.wave_fd = self.wave_td = self.parameters = None
                self.noise_fd = None

        if self.cache_in_memory:
            self._load = self._load_from_memory
        else:
            self._load  = self._load_from_disk
        
    def _load_from_memory(self, key, idx):
        data = getattr(self, key)
        return data[idx]
    def _load_from_disk(self, key, idx):
        with h5py.File(self.filename, "r") as f:
            return torch.tensor(f[key][idx], device="cpu")

    def _transform_identity(self, data):
        return data

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        wave_fd = self._load("wave_fd", idx)
        params = self._load("source_parameters", idx)
        out = {
            "idx": idx,
            "wave_fd": wave_fd,
            "params": params,
        }
        if self.has_td:
            out["wave_td"] = self._load("wave_td", idx)
        if self.has_stored_noise:
            out["noise_fd"] = self._load("noise_fd", idx)
        return out

    def to(self, device):
        if self.cache_in_memory:
            keys = ["wave_fd", "source_parameters"]
            if self.has_td:
                keys += ["wave_td"]
            if self.has_stored_noise:
                keys += ["noise_fd"]
            for k in keys:
                tensor = getattr(self, k, None)
                if tensor is not None:
                    setattr(self, k, tensor.to(device))
            if self.noise_scale is not None:
                self.noise_scale = self.noise_scale.to(device)
            self.device = device
        else:
            print("Dataset not cached in memory; cannot move to device.")
            return

    def clear_cache(self):
        """Free cached tensors and switch to disk-based loading."""
        if self.cache_in_memory:
            keys = ["wave_fd", "wave_td", "source_parameters", "noise_fd"]
            for k in keys:
                if hasattr(self, k) and getattr(self, k) is not None:
                    delattr(self, k)
            self.wave_fd = self.wave_td = self.source_parameters = None
            self.noise_fd = None
            self.cache_in_memory = False
            self._load = self._load_from_disk
            gc.collect()
            torch.cuda.empty_cache()
            print("[Dataset] Cache cleared, switched to disk-based loading.")
        else:
            print("[Dataset] Not cached in memory; nothing to clear.")
    #         print("Dataset not cached in memory; nothing to clear.")
    #         return
    #     self.cache_in_memory = False
    #     self._load = self._load_from_disk

class MBHBDataModule( L.LightningDataModule ): 

    def __init__(self, filename: str, batch_size: int, num_workers: int = 15, cache_in_memory: bool = False, shuffle_data: bool = True, noise_factor=1.0, seed: int = 31415):
        """Initialize the data module.

        :param filename: Path to the HDF5 file.
        :type filename: str
        :param batch_size: Batch size for data loading.
        :type batch_size: int
        :param seed: RNG seed for the train/val/test split.
        :type seed: int
        """
        super().__init__()
        self.batch_size = batch_size
        self.generator = torch.Generator().manual_seed(seed)
        self.filename = filename
        self.num_workers = num_workers
        self.cache_in_memory = cache_in_memory
        self.shuffle_data = shuffle_data 
        self.noise_factor = noise_factor

        # read median snr from the dataset for safety checks: 
        with h5py.File(self.filename, "r") as f:
            snr = f["snr"][()]
            self.median_snr = np.median(snr)
            
    def prepare_data(self):

        pass
    
    def setup(self, stage=None):
        """Setup the dataset."""
        if hasattr(self, 'train') and stage == 'fit': 
            # avoid running setup twice if it was already done. 
            return
        if stage == "fit" or stage is None:
            self.full_dataset = MBHBDataset(self.filename, cache_in_memory=self.cache_in_memory)
            self.train, self.val, self.test = random_split(self.full_dataset,  [0.7,0.25, 0.05], generator=self.generator)
            self.train_indices = sorted(self.train.indices)
            self.val_indices = sorted(self.val.indices)
            self.test_indices = sorted(self.test.indices)

        elif stage == "test":
            test_dataset = MBHBDataset(self.filename, cache_in_memory=self.cache_in_memory)
            self.test = Subset(test_dataset, indices=range(len(test_dataset)))

    def get_max_td(self):
        """Get the maximum time-domain value from the training dataset."""
        if not self.full_dataset.has_td:
            return None
        maxtd = self.train.dataset[self.train_indices]["wave_td"].abs().max()
        print("Max td:", maxtd)
        return maxtd
    
    def get_params_mean_std(self):
        """get mean of source parameters from training dataset."""

        params = self.train.dataset[self.train_indices]["params"]
        return params.mean(dim=0), params.std(dim=0)

    # def get_params_std(self):
    #     """get std of source parameters from training dataset."""
    #     params = self.train.dataset[self.train_indices]["params"]
    #     return params.std(dim=0)
    
    def get_asd(self):
        """Return the ASD tensor of shape (n_channels, n_freq) from the dataset."""
        if hasattr(self, 'full_dataset') and self.full_dataset.asd is not None:
            return self.full_dataset.asd
        with h5py.File(self.filename, "r") as f:
            if "asd" in f:
                return torch.tensor(f["asd"][()], dtype=get_torch_dtype())
        return None

    def get_freqs(self):
        """Get the frequency bins from the dataset."""
        with h5py.File(self.filename, "r") as f:
            freqs = f["frequencies"][()]
        return freqs
    
    def get_times(self):
        """Get the time bins from the dataset. Times are in SI units (seconds).
        Returns None for FD-only datasets."""
        with h5py.File(self.filename, "r") as f:
            if "times_SI" not in f:
                return None
            times = f["times_SI"][()]
        return times

    def train_dataloader(self, shuffle=True, num_workers=None, pin_memory=False):
        if num_workers is None:
            num_workers = self.num_workers
        noise_scale = self.full_dataset.noise_scale
        td_params = self.full_dataset.td_params
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=shuffle, num_workers=num_workers,
                          pin_memory=pin_memory,
                          collate_fn=lambda b: mbhb_collate_fn(b, noise_scale, self.noise_factor,
                                                                noise_shuffling=shuffle, td_params=td_params))

    def val_dataloader(self, shuffle=True):
        noise_scale = self.full_dataset.noise_scale
        td_params = self.full_dataset.td_params
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                          collate_fn=lambda b: mbhb_collate_fn(b, noise_scale, self.noise_factor,
                                                                noise_shuffling=False, td_params=td_params))

    def test_dataloader(self):
        noise_scale = self.full_dataset.noise_scale
        td_params = self.full_dataset.td_params
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                          collate_fn=lambda b: mbhb_collate_fn(b, noise_scale, self.noise_factor,
                                                                noise_shuffling=False, td_params=td_params))

class DummyDataset(Dataset):
    def __init__(self, params, data):
        self.params = params
        self.data = data

    def __len__(self):
        return len(self.params)

    def __getitem__(self, idx):
        return {'source_parameters': self.params[idx], 'data_fd': self.data[idx]}




