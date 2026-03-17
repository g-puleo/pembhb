import h5py
import numpy as np
import argparse
import os

import matplotlib.pyplot as plt
parser = argparse.ArgumentParser(description="Plot samples from HDF5 file")
parser.add_argument("filename", type=str, help="Path to the .h5 file")
parser.add_argument("--n_samples", default=3, type=int, help="Number of samples to plot")
args = parser.parse_args()
DAY_SI = 24*3600
filename = args.filename
output_dir = os.path.dirname(filename)
filename_only = os.path.basename(filename).replace('.h5','')
os.makedirs('plots/'+filename_only, exist_ok=True)
with h5py.File(filename, 'r') as f:
    wave_fd     = f['wave_fd'][:]
    noise_fd = f['noise_fd'][:]
    # wave_td     = f['wave_td'][:]
    # noise_td = f['noise_td'][:]
    #times       = f['times_SI'][:]
    frequencies = f['frequencies'][:]
    parameters = f["source_parameters"][:]
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
abs_wave = np.abs(wave_fd)
abs_noise = np.abs(noise_fd)
abs_sum = np.abs(wave_fd + noise_fd)

counts = []
# for j in range(10):
#     aw = np.abs(wave_fd[j, 0])
#     an = np.abs(noise_fd[j,  0])
#     cnt = int(np.count_nonzero(aw > an))
#     counts.append(cnt)
#     print(f"i={j}: {cnt} frequencies where |wave| > |noise|")

indices = np.random.choice(wave_fd.shape[0], size=args.n_samples, replace=False)
for i in indices:
    print(f"Plotting sample {i+1}/{args.n_samples}")
    print(f"chirp mass: {parameters[i,0]:.8e} Msun,\nq: {parameters[i,1]:.8e} Mpc")
    print(f"mean over channel A: {np.mean(wave_fd[i,0]):.3e} ± {np.std(wave_fd[i,0]):.3e}")
    print(f"mean over channel E: {np.mean(wave_fd[i,1]):.3e} ± {np.std(wave_fd[i,1]):.3e}")
    print(f"amp: mean over amplitudes of channel A: {np.mean(abs_wave[i,0]):.3e} ± {np.std(abs_wave[i,0]):.3e}")
    print(f"amp: mean over amplitudes of channel E: {np.mean(abs_wave[i,1]):.3e} ± {np.std(abs_wave[i,1]):.3e}")
    #fig_td, ax_td = plt.subplots(1, 2, figsize=(8, 4))
    fig_fd, ax_fd = plt.subplots(1, 2, figsize=(8, 4))

    for c in range(2): 
    #     ax_td[c].plot(times/DAY_SI, wave_td[i,c], label='A (wave)' if c==0 else 'E (wave)')
    #     ax_td[c].plot(times/DAY_SI, noise_td[i,c], label='A (noise)' if c==0 else 'E (noise)', linestyle='--', alpha=0.7)
    #     ax_td[c].plot(times/DAY_SI, wave_td[i,c] + noise_td[i,c], label='A (wave+noise)' if c==0 else 'E (wave+noise)', linestyle='-.', color='C3')
        
    #     ax_td[c].set_xlim(4.4,4.6)
    #     # ax_td.set_ylim(-2e-21, 2e-21)

    #     ax_td[c].set_title('data_td Example 1')
    #     ax_td[c].set_xlabel('Time (days)')
    #     ax_td[c].set_ylabel('Amplitude')
    #     ax_td[c].legend()
    #     ax_td[c].grid(True)




        counts_array = np.array(counts)
        #print(f"Average number of frequencies where |wave| > |noise|: {np.mean(counts_array)} ± {np.std(counts_array)}")
    
        ax_fd[c].plot(frequencies, abs_wave[i,c], label='wave (|A|)')
        ax_fd[c].plot(frequencies, abs_noise[i,c], label='noise (|A|)', linestyle='--')
        ax_fd[c].plot(frequencies, abs_sum[i,c], label='wave + noise (|A+N|)', linestyle='-.', color='C3')

        ax_fd[c].set_xlabel('Frequency (Hz)')
        ax_fd[c].set_ylabel('Amplitude')
        ax_fd[c].set_yscale('log')
        ax_fd[c].set_xscale('log')
        # ax_fd.legend()
        # ax_fd.grid(True)

    # fig_td.savefig(f'plots/{filename_only}/data_td_event_{i}.png', dpi=600)
    # fig_td.tight_layout()
    fig_fd.tight_layout()
    fig_fd.savefig(f'plots/{filename_only}/data_fd_event_{i}.png', dpi=600)
    print(f"Saved plot for sample {i+1} to plots/{filename_only}/data_fd_event_{i}.png")
#print(f"Saved plot for sample {i+1} to plots/{filename_only}/data_td_event_{i}.png")