import h5py
import numpy as np
import argparse

import matplotlib.pyplot as plt
parser = argparse.ArgumentParser(description="Plot samples from HDF5 file")
parser.add_argument("filename", type=str, help="Path to the .h5 file")
args = parser.parse_args()
filename = args.filename
with h5py.File(filename, 'r') as f:
    snr = f['snr'][:]
    print(f.keys())
    #params = f['source_parameters'][:]
plt.hist(snr, bins=50, color='blue', alpha=0.7)
plt.xlabel('SNR')
plt.ylabel('Count')
plt.xscale('log')
plt.title('Histogram of SNR')
plt.grid(True)
plt.savefig('snr_histogram.png')
