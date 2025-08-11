import lightning as L 
import os
from torch.utils.data import  DataLoader
import numpy as np
from pembhb.model import InferenceNetwork
from pembhb.data import MBHBDataset
from pembhb import utils, ROOT_DIR
from glob import glob
#candidate good roun1
ckpt_files = glob("/u/g/gpuleo/pembhb/logs/20250806_161531/round_0/version_0/checkpoints/*.ckpt")
print("Found checkpoint files:", ckpt_files)
if len(ckpt_files) > 1: 
    print("Multiple checkpoint files found, using the latest one.")
ckpt_file = sorted(ckpt_files, key=os.path.getmtime)[-1]
print("Using checkpoint file:", ckpt_file)
model = InferenceNetwork.load_from_checkpoint(ckpt_file, conf=utils.read_config(os.path.join(ROOT_DIR, "config.yaml")))
data = MBHBDataset("/u/g/gpuleo/pembhb/data/test_data1k.h5", transform="log")
dataloader = DataLoader(data, batch_size=50, shuffle=False)
trainer = L.Trainer(accelerator="gpu", devices=1, enable_progress_bar=True)
# input("Press Enter to continue...")
trainer.test(model, dataloaders=dataloader)
