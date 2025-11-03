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
        "logwhiten": "_transform_logwhiten",
        "log": "_transform_log",
        "whiten": "_transform_whiten",
        "normalise_max": "_transform_normalise_max"
    }

    def __init__(self, filename: str, transform_fd: str = "none", device: str = "cpu"):
        """Initialize the dataset.

        :param filename: name of the .h5 file where data are stored
        :type filename: str
        """
        print("Loading data from file:", filename)
        self.channels_amp = torch.tensor([0,1], device=device, dtype=torch.int64)  # AE
        self.channels_phase = torch.tensor([2,3], device=device, dtype=torch.int64)  # AE
        self.filename = filename
        self.transform_fd = getattr(self, self._TRANSFORMS[transform_fd])   
        self.device = device
        with h5py.File(self.filename, 'r') as f:
            self.keys = list(f.keys())
            self.len = f[self.keys[0]].shape[0]
            if transform_fd in ["logwhiten", "whiten"]:
                self.PSD = torch.tensor(f["psd"][()], device="self.device", dtype=torch.float32)
            self.data_fd = self.transform_fd(torch.tensor(f["data_fd"][()], device=self.device, dtype=torch.float32))
            self.data_td = torch.tensor(f["data_td"][()], device=self.device, dtype=torch.float32)
            self.parameters =  torch.tensor(f["source_parameters"][()], device=self.device, dtype=torch.float32)
            self.frequencies = torch.tensor(f["frequencies"][()], device=self.device, dtype=torch.float32)
            self.times = torch.tensor(f["times_SI"][()], device=self.device, dtype=torch.float32)
    def _transform_identity(self, data):
        return data
    
    def _transform_logwhiten(self, data): 
        """
        data: np.array of shape (2*n_channels, n_pt) where n_pt is the number of points in the frequency domain
        n_channels is the number of TDI lisa channels used. 
        This function transforms the data to log10 scale for the amplitude and keeps the phase.
        
        :param data: Frequency domain data.
        :type data: np.array
        :return: Transformed data with log10 amplitude and phase.
        :rtype: np.array
        """
        #apply log10 to the amplitude of the data 
        #data is of shape (6, n) with channels sorted as AETAET, first half amplitude is , second half is phase
        #this line fetches only the channels in self.channels, in order. 
        data_ampl = np.log10((data[:,self.channels_amp]+1e-33)/self.PSD[self.channels_amp]) # add a small value to avoid log(0)
        data_phase = data[:,self.channels_phase]
        return torch.concatenate((data_ampl, data_phase), dim=1)

    def _transform_log(self, data): 
        """
        data: np.array of shape (2*n_channels, n_pt) where n_pt is the number of points in the frequency domain
        n_channels is the number of TDI lisa channels used. 
        This function transforms the data to log10 scale for the amplitude and keeps the phase.
        
        :param data: Frequency domain data.
        :type data: np.array
        :return: Transformed data with log10 amplitude and phase.
        :rtype: np.array
        """
        #apply log10 to the amplitude of the data 
        #data is of shape (6, n) with channels sorted as AETAET, first half amplitude is , second half is phase
        #this line fetches only the channels in self.channels, in order. 
        # breakpoint()
        data_ampl = torch.log10(data[:,self.channels_amp]+1e-33) # add a small value to avoid log(0)
        data_phase = data[:,self.channels_phase]
        return torch.concatenate((data_ampl, data_phase), dim=1)
    
    def _transform_whiten(self, data):

        print(self.PSD.shape, data[self.channels_amp].shape)
        raise NotImplementedError("Have to check psd shape.")
        data_ampl = data[:,self.channels_amp]/self.PSD[self.channels_amp]
        data_phase = data[:,self.channels_phase]
        return torch.concatenate((data_ampl, data_phase), dim=1)
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        dict_out = {
            'source_parameters': self.parameters[idx],
            'data_fd': self.data_fd[idx],
            'data_td': self.data_td[idx]
        }
        return dict_out
            

    

class MBHBDataModule( L.LightningDataModule ): 

    def __init__(self, filename: str, conf: dict):
        """Initialize the data module.

        :param filename: Path to the HDF5 file.
        :type filename: str
        :param conf: Configuration dictionary, used to set the batch size and the transformation to apply to the frequency domain data.
        :type conf: dict
        """
        super().__init__()
        self.batch_size = conf["training"]["batch_size"]
        self.generator = torch.Generator().manual_seed(31415)
        self.filename = filename
        self.transform_fd = conf["training"]["transform_fd"]
    def prepare_data(self):

        pass
    
    def setup(self, stage=None):
        """Setup the dataset."""
        if hasattr(self, 'train') and stage == 'fit': 
            # avoid running setup twice if it was already done. 
            return
        if stage == "fit" or stage is None:
            full_dataset = MBHBDataset(self.filename, transform_fd=self.transform_fd)
            self.train, self.val, self.test = random_split(full_dataset,  [0.85,0.095, 0.055], generator=self.generator)
            
        elif stage == "test":
            self.test = MBHBDataset(self.filename, transform_fd=self.transform_fd)

    def get_max_td(self): 
        """Get the maximum time-domain value from the training dataset."""
        maxtd = self.train.dataset[:]["data_td"].abs().max()
        print("Max td:", maxtd)
        return maxtd

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
