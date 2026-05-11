import h5py
import numpy as np
import argparse
import os

import matplotlib.pyplot as plt
parser = argparse.ArgumentParser(description="Plot samples from HDF5 file")
parser.add_argument("filename", type=str, help="Path to the .h5 file")
parser.add_argument("--sample_idx", default=3, type=int, help="Index of the sample to plot")
args = parser.parse_args()
DAY_SI = 24*3600
filename = args.filename
output_dir = os.path.dirname(filename)
filename_only = os.path.basename(filename).replace('.h5','')
os.makedirs('plots/'+filename_only, exist_ok=True)
with h5py.File(filename, 'r') as f:
    wave_fd     = f['wave_fd'][:]
    frequencies = f['frequencies'][:]
    parameters = f["source_parameters"][:]
    asd = f['asd'][:]
    if 'noise_fd' in f:
        noise_fd = f['noise_fd'][:]
    else:
        # No stored noise — draw a fresh realisation matching mbhb_collate_fn:
        # noise_fd = CN(0, 1) * (asd / sqrt(4 / T_obs)).
        T_obs_total = f.attrs['observation_duration_SI']
        noise_scale = asd / np.sqrt(4.0 / T_obs_total)            # (C, F)
        rng = np.random.default_rng()
        re = rng.standard_normal(wave_fd.shape)
        im = rng.standard_normal(wave_fd.shape)
        noise_fd = (re + 1j * im) * noise_scale[None, :, :]
        print(f"[plot_samples] '{filename}' has no noise_fd — generated noise on the fly.")
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
channel_names = ['A', 'E']
component_funcs = [('Real', np.real), ('Imag', np.imag)]

indices = [args.sample_idx]  # Plot a single specified sample
for i in indices:
    print(f"Plotting sample {i+1}/{len(indices)} from '{filename}'")
    print(f"chirp mass: {parameters[i,0]:.8e} Msun,\nq: {parameters[i,1]:.8e} Mpc")

    fig_fd, ax_fd = plt.subplots(2, 2, figsize=(10, 7), sharex=True)

    for c in range(2):
        for k, (comp_name, comp_func) in enumerate(component_funcs):
            ax = ax_fd[c, k]
            ax.plot(frequencies, comp_func(wave_fd[i, c]), label='wave')
            ax.plot(frequencies, comp_func(noise_fd[i, c]), label='noise', linestyle='--')
            ax.plot(frequencies, comp_func(wave_fd[i, c] + noise_fd[i, c]),
                    label='wave + noise', linestyle='-.', color='C3')

            ax.set_xscale('log')
            ax.set_title(f'Channel {channel_names[c]} — {comp_name}')
            if c == 1:
                ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel(f'{comp_name} part')
            ax.set_xlim(2.5e-3, 5e-3)
            ax.set_ylim(-3e-17, 3e-17)
    ax_fd[0, 0].legend(loc='best', fontsize='small')
    fig_fd.tight_layout()
    fig_fd.savefig(f'plots/{filename_only}/data_fd_event_{i}.png', dpi=600)
    plt.close(fig_fd)
    print(f"Saved plot for sample {i+1} to plots/{filename_only}/data_fd_event_{i}.png")