from torch.utils.data import Dataset, random_split, DataLoader, Subset 
import lightning as L
from pembhb.utils import mbhb_collate_fn
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

            if cache_in_memory:
                self.wave_fd = torch.tensor(f["wave_fd"][()], device="cpu", dtype=torch.complex64)
                self.wave_td = torch.tensor(f["wave_td"][()], device="cpu", dtype=torch.float32)
                self.noise_fd = torch.tensor(f["noise_fd"][()], device="cpu", dtype=torch.complex64)
                self.noise_td = torch.tensor(f["noise_td"][()], device="cpu", dtype=torch.float32)
                self.source_parameters = torch.tensor(f["source_parameters"][()], device="cpu", dtype=torch.float32)
            else:
                self.wave_fd = self.wave_td = self.parameters = None
                # noise_fd / noise_td not loaded; collate_fn will handle sampling

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
        wave_td = self._load("wave_td", idx)
        params = self._load("source_parameters", idx)
        return {
            "idx": idx,
            "wave_fd": wave_fd,
            "wave_td": wave_td,
            "params": params,
        }
    
    def to(self, device):
        if self.cache_in_memory:
            for k in ["wave_fd","wave_td","source_parameters","noise_fd","noise_td"]:

                setattr(self, k, getattr(self, k).to(device))
                # data = torch.tensor(getattr(self, k), device=device)
                # setattr(self, k, data)

            self.device = device

        else: 
            print("Dataset not cached in memory; cannot move to device.")
            return
    # def clear_cache(self): 
    #     if self.cache_in_memory:
    #         del self.wave_fd, self.wave_td, self.source_parameters, self.noise_fd, self.noise_td
    #         gc.collect()
    #         torch.cuda.empty_cache()
    #     else: 
    #         print("Dataset not cached in memory; nothing to clear.")
    #         return
    #     self.cache_in_memory = False
    #     self._load = self._load_from_disk

class MBHBDataModule( L.LightningDataModule ): 

    def __init__(self, filename: str, batch_size: int, num_workers: int = 15, cache_in_memory: bool = False, shuffle_data: bool = True, noise_factor=1.0):
        """Initialize the data module.

        :param filename: Path to the HDF5 file.
        :type filename: str
        :param batch_size: Batch size for data loading.
        :type batch_size: int
        :param transform_fd: Transformation to apply to the frequency domain data.
        :type transform_fd: str
        """
        super().__init__()
        self.batch_size = batch_size
        self.generator = torch.Generator().manual_seed(31415)
        self.filename = filename
        self.num_workers = num_workers
        self.cache_in_memory = cache_in_memory
        self.shuffle_data = shuffle_data 
        self.noise_factor = noise_factor
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
    
    def get_freqs(self):
        """Get the frequency bins from the dataset."""
        with h5py.File(self.filename, "r") as f:
            freqs = f["frequencies"][()]
        return freqs

    def train_dataloader(self, shuffle=True, num_workers=None):
        if num_workers is None:
            num_workers = self.num_workers  

        return DataLoader(self.train, batch_size=self.batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=lambda b: mbhb_collate_fn(b, self.train, self.noise_factor, noise_shuffling=shuffle))

    def val_dataloader(self, shuffle=True):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers, collate_fn=lambda b: mbhb_collate_fn(b, self.val, self.noise_factor, noise_shuffling=shuffle))
    
    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=lambda b: mbhb_collate_fn(b, self.test, self.noise_factor, noise_shuffling=False))

class DummyDataset(Dataset):
    def __init__(self, params, data):
        self.params = params
        self.data = data

    def __len__(self):
        return len(self.params)

    def __getitem__(self, idx):
        return {'source_parameters': self.params[idx], 'data_fd': self.data[idx]}




