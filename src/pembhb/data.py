from torch.utils.data import Dataset, random_split, DataLoader
import lightning as L
import torch
import numpy as np
import h5py
# i want a class / function that can : 
# generate data and store in memory using a sampler and a simulator 
# optionally save and load the data to/from disk

class MBHBDataset(Dataset):
    def __init__(self, filename: str, channels: str):
        """Initialize the dataset.

        :param filename: name of the .h5 file where data are stored
        :type filename: str
        :param channels: channels that you want to retrieve from the dataset. 
        :type channels: str
        """

      
        self.channels_amp = [0,1]
        self.channels_phase = [2,3]
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
        #apply log10 to the amplitude of the data 
        #data is of shape (6, n) with channels sorted as AETAET, first half amplitude is , second half is phase
        #this line fetches only the channels in self.channels, in order. 
        data_ampl = np.log10(data[self.channels_amp]+1e-33)
        data_phase = data[self.channels_phase]
        return np.concatenate((data_ampl, data_phase), axis=0)

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        
        with h5py.File(self.filename, 'r') as f:
            dict_out = {
                "data_fd": self.transform(f["white_data_fd"][idx]),
                "source_parameters": f["source_parameters"][idx],
            }
        return dict_out
            

    

class MBHBDataModule( L.LightningDataModule ): 

    def __init__(self, filename: str, conf: dict):
        """Initialize the data module.

        :param filename: Path to the HDF5 file.
        :type filename: str
        :param targets: List of keys to load from the HDF5 file. If None, all keys are loaded.
        :type targets: list[str], optional
        :param batch_size: Batch size for data loading.
        :type batch_size: int
        """
        super().__init__()
        self.batch_size = conf["training"]["batch_size"]
        self.generator = torch.Generator().manual_seed(31415)
        self.filename = filename
        self.channels = conf["waveform_params"]["TDI"]

    def prepare_data(self):

        pass
    
    def setup(self, stage=None):
        """Setup the dataset."""
        if stage == "fit" or stage is None:
            full_dataset = MBHBDataset(self.filename, self.channels)
            self.train, self.val = random_split(full_dataset,  [0.9,0.1], generator=self.generator)

        elif stage == "test":
            self.test = MBHBDataset(self.filename, self.channels)
    
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=15)
    
    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, num_workers=15)
    
    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False)
    


class DummyDataset(Dataset):
    def __init__(self, params, data):
        self.params = params
        self.data = data

    def __len__(self):
        return len(self.params)

    def __getitem__(self, idx):
        return {'source_parameters': self.params[idx], 'data_fd': self.data[idx]}
