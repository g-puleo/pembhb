from torch.utils.data import Dataset, random_split, DataLoader
import lightning as L
import torch
import numpy as np
import h5py
# i want a class / function that can : 
# generate data and store in memory using a sampler and a simulator 
# optionally save and load the data to/from disk


class MBHBDataset(Dataset):
    _TRANSFORMS = {
        "none": "_transform_identity",
        "log": "_transform_log",
    }

    def __init__(self, filename: str, transform_fd: str = "none", device: str = "cpu", cache_in_memory: bool = False):
        self.filename = filename
        self.device = device
        self.transform_fd = getattr(self, self._TRANSFORMS[transform_fd])
        self.cache_in_memory = cache_in_memory

        self.data_cache = {} if cache_in_memory else None

        with h5py.File(self.filename, "r") as f:
            self.len = f["wave_fd"].shape[0]

            if cache_in_memory:
                self.wave_fd = torch.tensor(f["wave_fd"][()], device=self.device, dtype=torch.complex64)
                self.wave_td = torch.tensor(f["wave_td"][()], device=self.device, dtype=torch.float32)
                self.noise_fd = torch.tensor(f["noise_fd"][()], device=self.device, dtype=torch.complex64)
                self.noise_td = torch.tensor(f["noise_td"][()], device=self.device, dtype=torch.float32)
                self.source_parameters = torch.tensor(f["source_parameters"][()], device=self.device, dtype=torch.float32)
            else:
                self.wave_fd = self.wave_td = self.parameters = None
                # noise_fd / noise_td not loaded; collate_fn will handle sampling

    def _load(self, key, idx, dtype):
        if self.cache_in_memory:
            data = getattr(self, key)
            return data[idx]
        else:
            with h5py.File(self.filename, "r") as f:
                return torch.tensor(f[key][idx], device=self.device, dtype=dtype)

    def _transform_identity(self, data):
        return data

    def _transform_log(self, data):
        return torch.log10(data + 1e-33)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        wave_fd = self._load("wave_fd", idx, torch.complex64)
        wave_td = self._load("wave_td", idx, torch.float32)
        params = self._load("source_parameters", idx, torch.float32)

        wave_fd = self.transform_fd(wave_fd)

        return {
            "idx": idx,
            "wave_fd": wave_fd,
            "wave_td": wave_td,
            "params": params,
        }

    

class MBHBDataModule( L.LightningDataModule ): 

    def __init__(self, filename: str, conf: dict):
        """Initialize the data module.

        :param filename: Path to the HDF5 file.
        :type filename: str
        :param conf: Configuration dictionary, used to set the batch size and the transformation to apply to the frequency domain data.
        :type conf: dict
        """
        super().__init__()
        self.batch_size = conf["batch_size"]
        self.generator = torch.Generator().manual_seed(31415)
        self.filename = filename
        self.transform_fd = conf["transform_fd"]
    
    def prepare_data(self):

        pass
    
    def setup(self, stage=None):
        """Setup the dataset."""
        if hasattr(self, 'train') and stage == 'fit': 
            # avoid running setup twice if it was already done. 
            return
        if stage == "fit" or stage is None:
            full_dataset = MBHBDataset(self.filename, transform_fd=self.transform_fd)
            self.train, self.val, self.test = random_split(full_dataset,  [0.7,0.25, 0.05], generator=self.generator)
            
        elif stage == "test":
            self.test = MBHBDataset(self.filename, transform_fd=self.transform_fd)

    def get_max_td(self): 
        """Get the maximum time-domain value from the training dataset."""
        maxtd = self.train.dataset[:]["data_td"].abs().max()
        print("Max td:", maxtd)
        return maxtd

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=15, collate_fn=lambda b: mbhb_collate_fn(b, self.train))
    
    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, num_workers=15, collate_fn=lambda b: mbhb_collate_fn(b, self.val))
    
    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False, num_workers=15, collate_fn=lambda b: mbhb_collate_fn(b, self.test))


class DummyDataset(Dataset):
    def __init__(self, params, data):
        self.params = params
        self.data = data

    def __len__(self):
        return len(self.params)

    def __getitem__(self, idx):
        return {'source_parameters': self.params[idx], 'data_fd': self.data[idx]}

def mbhb_collate_fn(batch, subset: torch.utils.data.Subset, noise_shuffling=True):
    B = len(batch)
    device = subset.dataset.device

    wave_fd = torch.stack([b["wave_fd"] for b in batch])
    wave_td = torch.stack([b["wave_td"] for b in batch])
    params  = torch.stack([b["params"] for b in batch])

    # pick noise indices randomly
    if noise_shuffling:
        subset_idxs = torch.tensor(subset.indices, device=device)
        pick = subset_idxs[torch.randint(0, len(subset_idxs), (B,), device=device)]
    else:
        pick = subset.indices

    # load noise (cached or lazy)
    noise_fd = torch.stack([subset.dataset._load("noise_fd", i, torch.complex64) for i in pick])
    noise_td = torch.stack([subset.dataset._load("noise_td", i, torch.float32) for i in pick])

    # combine waveform + noise
    data_fd_re_im = wave_fd + noise_fd
    data_fd_ampl = subset.dataset.transform_fd(torch.abs(data_fd_re_im))
    data_fd_phase = torch.angle(data_fd_re_im)
    data_fd = torch.cat((data_fd_ampl, data_fd_phase), dim=1)

    data_td = wave_td + noise_td

    return {
        "source_parameters": params,
        "data_fd": data_fd,
        "data_td": data_td,
        "wave_fd": wave_fd,
        "wave_td": wave_td,
        "noise_fd": noise_fd,
        "noise_td": noise_td,
        "noise_index": pick,
    }



