from torch.utils.data import Dataset, random_split, DataLoader
import lightning as L
import torch
import numpy as np
import h5py
# i want a class / function that can : 
# generate data and store in memory using a sampler and a simulator 
# optionally save and load the data to/from disk

class MBHBDataset(Dataset):
    def __init__(self, filename: str):
        """Initialize the dataset.

        :param data: Dictionary containing the data.
        :type data: dict
        :param targets: List of keys to load from the data dictionary.
        :type targets: list[str]
        """
        self.filename = filename
        with h5py.File(self.filename, 'r') as f:
            self.keys = list(f.keys())
            self.len = f[self.keys[0]].shape[0]
    
    def transform(self, data): 
        """
        data: np.array of shape (6, n_pt) where n_pt is the number of points in the frequency domain
        This function transforms the data to log10 scale for the amplitude and keeps the phase.
        
        :param data: Frequency domain data.
        :type data: np.array
        :return: Transformed data with log10 amplitude and phase.
        :rtype: np.array
        """
        # apply log10 to the amplitude of the data (channels from 0 to 2)
        # data_ampl = np.log10(data[:3]+1e-33)
        # data_phase = data[3:]
        # return np.concatenate((data_ampl, data_phase), axis=0)
        return data

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        
        with h5py.File(self.filename, 'r') as f:
            data_fd= self.transform(f["data_fd"][idx])
            dict_out = {
                "data_fd": data_fd,
                "source_parameters": f["source_parameters"][idx],
            }
        return dict_out
            

    

class MBHBDataModule( L.LightningDataModule ): 

    def __init__(self, filename: str, targets: list[str], batch_size: int = 32):
        """Initialize the data module.

        :param filename: Path to the HDF5 file.
        :type filename: str
        :param targets: List of keys to load from the HDF5 file. If None, all keys are loaded.
        :type targets: list[str], optional
        :param batch_size: Batch size for data loading.
        :type batch_size: int
        """
        super().__init__()
        self.batch_size = batch_size
        self.targets = targets
        self.generator = torch.Generator().manual_seed(31415)
        self.filename = filename


    
    def setup(self, stage=None):
        """Setup the dataset."""
        if stage == "fit" or stage is None:
            full_dataset = MBHBDataset(self.filename)
            self.train, self.val = random_split(full_dataset,  [0.9,0.1], generator=self.generator)

        elif stage == "test":
            self.test = MBHBDataset(self.filename)
    
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=50)
    
    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, num_workers=50)
    
    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False)