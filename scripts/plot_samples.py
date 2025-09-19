import h5py
import numpy as np
import argparse

import matplotlib.pyplot as plt
parser = argparse.ArgumentParser(description="Plot samples from HDF5 file")
parser.add_argument("filename", type=str, help="Path to the .h5 file")
args = parser.parse_args()
DAY_SI = 24*3600
filename = args.filename
with h5py.File(filename, 'r') as f:
    data_fd     = f['data_fd'][:]
    data_td     = f['data_td'][:]
    noise_fd_twosided  = f['noise_fd'][:]
    times       = f['times_SI'][:]
    frequencies = f['frequencies'][:]
    psd = f['psd'][:]
# Plot 3 examples from data_fd
noise_fd_onesided = np.sqrt(2)*noise_fd_twosided
print(noise_fd_twosided.shape, noise_fd_onesided.shape)
fig_fd, ax_fd = plt.subplots(1, 3, figsize=(12, 4))
for i in range(3):
    ax_fd[i].plot(frequencies, data_fd[i,0], label='A')
    ax_fd[i].plot(frequencies, psd[0]**0.5, label='noise ASD', linestyle='--', color='gray')
    ax_fd[i].plot(frequencies, np.abs(noise_fd_onesided[i,0]), label='A noise', linestyle='--', color='green')
    #ax_fd[i].plot(data_fd[i,1], label='E')
    ax_fd[i].set_title(f'data_fd Example {i+1}')
    ax_fd[i].legend()
    ax_fd[i].set_yscale('log')
    ax_fd[i].set_xscale('log')
fig_fd.tight_layout()
fig_fd.savefig('data_fd_examples.png')

# Plot 3 examples from data_td
fig_td, ax_td = plt.subplots(1, 3, figsize=(12, 4))
for i in range(3):
    ax_td[i].plot(times/DAY_SI, data_td[i,0], label='A')
    #ax_td[i].plot(data_td[i,1], label='E')
    ax_td[i].set_title(f'data_td Example {i+1}')
    ax_td[i].legend()
    ax_td[i].set_xlabel('Time (days)')

fig_td.tight_layout()
fig_td.savefig('data_td_examples.png')
