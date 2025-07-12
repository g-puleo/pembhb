from torch.utils.data import Dataset, random_split, DataLoader
import lightning as L
import torch
import h5py
# i want a class / function that can : 
# generate data and store in memory using a sampler and a simulator 
# optionally save and load the data to/from disk

class MBHBDataset(Dataset):
    def __init__(self, data: dict):
        """Initialize the dataset.

        :param data: Dictionary containing the data.
        :type data: dict
        :param targets: List of keys to load from the data dictionary.
        :type targets: list[str]
        """
        self.data = data
        self.keys = list(data.keys())
    def __len__(self):
        return self.data[self.keys[0]].shape[0] 
    def __getitem__(self, idx):
        return {key: self.data[key][idx] for key in self.data}


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
        self.data = self._load_data(filename)
        self.generator = torch.Generator().manual_seed(31415)

    def _load_data(self, filename : str):
        with h5py.File(filename, 'r') as f:
            if not self.targets:
                return {key: f[key][:] for key in f.keys()}
            else:
                return {key: f[key][:] for key in self.targets}
        
    
    def setup(self, stage=None):
        """Setup the dataset."""
        if stage == "fit" or stage is None:
            full_dataset = MBHBDataset(self.data)
            self.train, self.val = random_split(full_dataset,  [0.9,0.1], generator=self.generator)

        elif stage == "test":
            self.test = MBHBDataset(self.data)
    
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False)