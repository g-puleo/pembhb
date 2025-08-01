import lightning as L 
import os
from torch.utils.data import  DataLoader
import numpy as np
from pembhb.model import InferenceNetwork
from pembhb.data import MBHBDataset
from pembhb import utils, ROOT_DIR

model = InferenceNetwork.load_from_checkpoint( "/u/g/gpuleo/pembhb/logs_0725/peregrine_norm/version_8/checkpoints/epoch=960-step=43245.ckpt", conf=utils.read_config(os.path.join(ROOT_DIR, "config.yaml")))
data = MBHBDataset("/u/g/gpuleo/pembhb/data/test_data1k.h5", channels="AE")
dataloader = DataLoader(data, batch_size=50, shuffle=False)
trainer = L.Trainer(accelerator="gpu", devices=1, enable_progress_bar=True)
input("Press Enter to continue...")
trainer.test(model, dataloaders=dataloader)
