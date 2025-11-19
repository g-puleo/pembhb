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
    wave_fd     = f['wave_fd'][:]
    noise_fd = f['noise_fd'][:]
    wave_td     = f['wave_td'][:]
    noise_td = f['noise_td'][:]
    times       = f['times_SI'][:]
    frequencies = f['frequencies'][:]
    asd = f['asd'][:]
# # Plot 3 examples from data_fd
# noise_fd_onesided = np.sqrt(2)*noise_fd_twosided
# print(noise_fd_twosided.shape, noise_fd_onesided.shape)
# fig_fd, ax_fd = plt.subplots(1, 3, figsize=(12, 4))
# for i in range(3):
##### plot only channel 0 (A)
#     ax_fd[i].plot(frequencies, data_fd[i,0], label='A')
#     ax_fd[i].plot(frequencies, psd[0]**0.5, label='noise ASD', linestyle='--', color='gray')
#     ax_fd[i].plot(frequencies, np.abs(noise_fd_onesided[i,0]), label='A noise', linestyle='--', color='green')
#     #ax_fd[i].plot(data_fd[i,1], label='E')
#     ax_fd[i].set_title(f'data_fd Example {i+1}')
#     ax_fd[i].legend()
#     ax_fd[i].set_yscale('log')
#     ax_fd[i].set_xscale('log')
# fig_fd.tight_layout()
# fig_fd.savefig('data_fd_examples.png')

# # Plot 3 examples from data_td
# fig_td, ax_td = plt.subplots(1, 3, figsize=(12, 4))
# for i in range(3):
#     ax_td[i].plot(times/DAY_SI, data_td[i,0], label='A')
#     #ax_td[i].plot(data_td[i,1], label='E')
#     ax_td[i].set_title(f'data_td Example {i+1}')
#     ax_td[i].legend()
#     ax_td[i].set_xlabel('Time (days)')
fig_td, ax_td = plt.subplots(figsize=(6, 3))

ax_td.plot(times/DAY_SI, wave_td[0,0], label='A (wave)')
#ax_td.plot(times/DAY_SI, wave_td[0,1], label='E (wave)')
ax_td.plot(times/DAY_SI, noise_td[0,0], label='A (noise)', linestyle='--', alpha=0.7)
#ax_td.plot(times/DAY_SI, noise_td[0,1], label='E (noise)', linestyle='--', alpha=0.7)
ax_td.plot(times/DAY_SI, wave_td[0,0] + noise_td[0,0], label='A (wave+noise)', linestyle='-.', color='C3')
#ax_td.plot(times/DAY_SI, wave_td[0,1] + noise_td[0,1], label='E (wave+noise)', linestyle='-.', color='C4')
ax_td.set_xlim(1,1.2)
ax_td.set_ylim(-2e-21, 2e-21)

#ax_td.set_title('data_td Example 1')
ax_td.set_xlabel('Time (days)')
ax_td.set_ylabel('Amplitude')
ax_td.legend()
ax_td.grid(True)

fig_td.tight_layout()
fig_td.savefig('data_td_example_presentation.png')


fig_fd, ax_fd = plt.subplots(figsize=(6, 3))
# frequency-domain plots: magnitudes (fd data are complex)
abs_wave = np.abs(wave_fd[0, 0])
abs_noise = np.abs(noise_fd[0, 0])
abs_sum = np.abs(wave_fd[0, 0] + noise_fd[0, 0])

ax_fd.plot(frequencies, abs_wave, label='wave (|A|)')
ax_fd.plot(frequencies, abs_noise, label='noise (|A|)', linestyle='--')
ax_fd.plot(frequencies, abs_sum, label='wave + noise (|A+N|)', linestyle='-.', color='C3')

ax_fd.set_xlabel('Frequency (Hz)')
ax_fd.set_ylabel('Amplitude')
ax_fd.set_yscale('log')
ax_fd.set_xscale('log')
ax_fd.legend()
ax_fd.grid(True)
fig_fd.tight_layout()
fig_fd.savefig('data_fd_example_presentation.png')