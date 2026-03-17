import h5py
import numpy as np
import argparse
import os

from scipy import signal 
from pembhb  import ROOT_DIR

import matplotlib.pyplot as plt
parser = argparse.ArgumentParser(description="Plot samples from HDF5 file")
parser.add_argument("filename", type=str, help="Path to the .h5 file")
args = parser.parse_args()
filename = args.filename
plt.figure(figsize=(4,3))
with h5py.File(filename, 'r') as f:
    snr = f['snr'][:]
    print(f.keys())
    params = f['source_parameters'][:]
    ### compute the snr manually: 
    # data = f['wave_fd'][:] + f['noise_fd'][:]
    # freqs = f['frequencies'][:]
    # asd = f['asd'][:]
    # df = freqs[1] - freqs[0]
 
    # high_pass_idx =  (freqs >= 5e-5)
    # data_over_asd = data[..., high_pass_idx] / asd[..., high_pass_idx]
    # data_over_asd_conj = data_over_asd.conj()
    # prod = data_over_asd * data_over_asd_conj
    # weighted = prod * df
    # summed = np.sum(weighted, axis=(1, 2))
    # real_part = summed.real
    # SNR2 = real_part * 4.0
    # SNR = np.sqrt(SNR2)
    # print(f"Manually computed SNR mean : {np.mean(SNR):.6e}")
    # print(f"Manually computed SNR median : {np.median(SNR):.6e}")
    # print(f"Manually computed SNR min : {np.min(SNR):.6e}")
    # print(f"Manually computed SNR max : {np.max(SNR):.6e}")
    # print(f"Manually computed snr std : {np.std(SNR):.6e}")
print(f"SNR mean : {np.mean(snr):.6e}")
print(f"SNR median : {np.median(snr):.6e}")
print(f"SNR min : {np.min(snr):.6e}")
print(f"SNR max : {np.max(snr):.6e}")
print(f"snr std : {np.std(snr):.6e}")
plt.hist(SNR, bins=np.logspace(np.log10(SNR.min()), np.log10(SNR.max()), 50), color='blue', alpha=0.7)
plt.xlabel('SNR')
plt.ylabel('Number of events')
plt.xscale('log')
plt.grid(True)
filename_out = os.path.join(ROOT_DIR, "plots", os.path.basename(filename).replace('.h5', '_snr_histogram.png'))
plt.savefig(filename_out, dpi=300, bbox_inches='tight')
print(f"Saved SNR histogram to {filename_out}")
