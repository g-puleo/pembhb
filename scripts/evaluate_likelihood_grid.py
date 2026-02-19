from bbhx.likelihood import Likelihood
from torch.utils.data import DataLoader, Subset
from pembhb.data import MBHBDataset, mbhb_collate_fn
from pembhb.simulator import MBHBSimulatorFD_TD 
from pembhb.utils import read_config
from pembhb import ROOT_DIR
from grid_evaluation_config import q_width, logmc_width
import matplotlib.pyplot as plt
import numpy as np
import h5py
import os 
from tqdm import tqdm
def load_samples(fname):
    with h5py.File(fname, "r") as f:
        src = f["source_parameters"][:]        # (N,11)
        #bbhx = f["bbhx_parameters"][:]         # (N,12)
        freqs = f["frequencies"][:]            # (n_freqs,)
        times = f["times_SI"][:]               # (n_time,)
        wave_fd = f["wave_fd"][:]              # (N, ch, n_freqs)
        wave_td = f["wave_td"][:]              # (N, ch, n_time)
        noise_fd = f["noise_fd"][:]            # (N, ch, n_freqs)
        noise_td = f["noise_td"][:]            # (N, ch, n_time)
        snr = f["snr"][:]                      # (N,)
        asd = f["asd"][:]                      # (...)
    return {
        "source_parameters": src,
        #"bbhx_parameters": bbhx,
        "frequencies": freqs,
        "times_SI": times,
        "wave_fd": wave_fd,
        "wave_td": wave_td,
        "noise_fd": noise_fd,
        "noise_td": noise_td,
        "snr": snr,
        "asd": asd,
    }
datagen_config = read_config("configs/datagen_config.yaml")
#loaded_dataset = load_samples("/u/g/gpuleo/pembhb/data/testes_newdata_fixall_notmcq.h5")
#loaded_dataset = load_samples("/u/g/gpuleo/pembhb/data/observation_fix_all_notmcq_newdata.h5")
loaded_dataset = load_samples("/data/gpuleo/mbhb/observation_skyloc.h5")
simulator = MBHBSimulatorFD_TD(datagen_config, sampler_init_kwargs={'prior_bounds': datagen_config["prior"]}, seed=42)
sampler = simulator.sampler
waveform_gen = simulator.wfd 
frequencies = simulator.freqs_pos
for i in tqdm(range(10)):
    if i!=0:
        continue
    event_idx = i
    freqs = loaded_dataset["frequencies"]      # shape (n_freqs,)    

    assert np.allclose(freqs, frequencies)
    tmnre_params = loaded_dataset["source_parameters"][event_idx] # shape (11, )
    bbhx_params = sampler.samples_to_bbhx_input(tmnre_params.reshape(-1,1), t_obs_end=simulator.t_obs_end_SI)
    # bbhx_params = loaded_dataset["bbhx_parameters"][1] 
    N_grid_points = 50
    true_mc = tmnre_params[0]
    true_q = tmnre_params[1]
    print( )
    # x_min = true_mc - logmc_width/2
    # x_max = true_mc + logmc_width/2
    # y_min = true_q - q_width/2
    # y_max = true_q + q_width/2
    x_min = 5.25-3e-5
    x_max = 5.25+3e-5
    y_min = 4.678
    y_max = 4.683

    x = np.linspace(x_min, x_max, N_grid_points)   # for param 0
    y = np.linspace(y_min, y_max, N_grid_points)   # for param 1
    X, Y = np.meshgrid(x, y, indexing="xy")

    N = X.size  # total grid points

    # initialize b
    b = np.zeros((11, N))
    # fill varying dims
    b[0, :] = X.reshape(-1)
    b[1, :] = Y.reshape(-1)
    # fill fixed dims
    b[2:, :] = tmnre_params[2:, None]

    bbhx_grid = sampler.samples_to_bbhx_input(b, t_obs_end=simulator.t_obs_end_SI)  # shape (12, N)
    data_fd_complex = (loaded_dataset["wave_fd"]+loaded_dataset["noise_fd"])[event_idx]
    psd_AE = simulator.asd**2 # shape (2, 4096)
    psd_ones_channelT = np.ones(shape=(1, psd_AE.shape[1]))
    psd_AET = np.concatenate([psd_AE, psd_ones_channelT], axis=0) 

    waveform_kwargs = simulator.waveform_kwargs
    data_T_channels = np.zeros(shape=(1, data_fd_complex.shape[1]), dtype=np.complex128)
    data_fd = np.concatenate([data_fd_complex, data_T_channels], axis=0)  # shape (3, n_freqs)
    likelihood = Likelihood(waveform_gen, frequencies, data_fd, psd_AET, force_backend="cpu")
    print("Evaluating likelihood on grid...")
    breakpoint()
    out = likelihood.get_ll(bbhx_grid, **waveform_kwargs)
    print("Done. Saving results...")
    Z = out.reshape((N_grid_points, N_grid_points))
    out_dir = "plots/likelihood_eval"
    out_dir = os.path.join(ROOT_DIR, "plots", "likelihood_evaluation", f"with_sky_event_{event_idx}")

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "loglikelihood-values.npy"), Z )
    np.save(os.path.join(out_dir, "grid_x.npy"), x )
    np.save(os.path.join(out_dir, "grid_y.npy"), y )
    np.save(os.path.join(out_dir, "true_params.npy"), tmnre_params[:2] )
#out = np.load("plots/likelihood_values_finegrid.npy")
# Z-= np.max(Z)  # for numerical stability
# fig, ax = plt.subplots(figsize=(7, 6))


# plot_posterior_2d(X, Y, Z, [x_center, y_center],  ax, [r"$\log_{10}(\mathcal{M}_c/M_{\odot})$", r"$q$"], title="Likelihood Contours", levels=[0.6827, 0.9545, 0.9973])