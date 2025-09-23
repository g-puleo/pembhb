import h5py
import numpy as np
import argparse

import matplotlib.pyplot as plt
parser = argparse.ArgumentParser(description="Plot samples from HDF5 file")
parser.add_argument("filename", type=str, help="Path to the .h5 file")
args = parser.parse_args()
filename = args.filename
plt.figure(figsize=(4,3))
with h5py.File(filename, 'r') as f:
    snr = f['snr'][:]
    print(f.keys())
    #params = f['source_parameters'][:]
plt.hist(snr, bins=np.logspace(np.log10(snr.min()), np.log10(snr.max()), 50), color='blue', alpha=0.7)
plt.xlabel('SNR')
plt.ylabel('Number of events')
plt.xscale('log')
plt.grid(True)
plt.savefig('snr_histogram.png' , dpi=300, bbox_inches='tight')
