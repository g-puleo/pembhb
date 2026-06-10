"""
Example: Using vectorized likelihood with emcee for faster MCMC sampling

This script demonstrates how to use the new log_likelihood_vectorized()
method with emcee, which natively supports vectorized likelihood evaluations
for significant speedup.
"""
import emcee
import numpy as np
import corner
import matplotlib.pyplot as plt
import bilby
from bbhx.likelihood import Likelihood
from pembhb.simulator import MBHBSimulatorFD_TD, MBHBSimulatorFD
from pembhb.utils import read_config, _ORDERED_PRIOR_KEYS, compute_fisher_matrix_waveform_deriv
from pembhb import ROOT_DIR
import h5py
import os


def load_observation(fname):
    """Load observation data from an HDF5 file."""
    with h5py.File(fname, "r") as f:
        src = f["source_parameters"][:]        # (N, 11)
        freqs = f["frequencies"][:]            # (n_freqs,)
        wave_fd = f["wave_fd"][:]              # (N, ch, n_freqs)
        noise_fd = f["noise_fd"][:]            # (N, ch, n_freqs)
        snr = f["snr"][:]                      # (N,)
    return {
        "source_parameters": src,
        "frequencies": freqs,
        "wave_fd": wave_fd,
        "noise_fd": noise_fd,
        "snr": snr,
    }


class BBHXLikelihood(bilby.Likelihood):
    """Bilby Likelihood wrapper around the BBHX likelihood.

    Handles the transform from prior space (what the sampler explores) to the
    BBHX input space, and exposes a vectorized evaluation for batched walkers.
    """

    def __init__(self, bbhx_likelihood, sampler, simulator, true_params=None, fixed_params=None):
        super().__init__()
        self.bbhx_likelihood = bbhx_likelihood
        self.sampler = sampler
        self.simulator = simulator
        self.true_params = true_params
        self.fixed_params = fixed_params or {}

    def log_likelihood(self, parameters):
        """Log likelihood for one parameter dict (keys in _ORDERED_PRIOR_KEYS)."""
        tmnre_params = np.array(
            [parameters[key] if key in parameters else self.fixed_params[key]
             for key in _ORDERED_PRIOR_KEYS],
            dtype=np.float64,
        ).reshape(-1, 1)  # (11, 1)
        bbhx_params = self.sampler.samples_to_bbhx_input(
            tmnre_params, t_obs_end=self.simulator.t_obs_end_SI
        )  # (12, 1)
        waveform_kwargs = self.simulator.waveform_kwargs.copy()  # get_ll mutates in place
        log_l = self.bbhx_likelihood.get_ll(bbhx_params, **waveform_kwargs)
        return float(log_l[0])

    def log_likelihood_vectorized(self, parameters_list):
        """Log likelihood for many parameter dicts at once (batched get_ll)."""
        tmnre_params_list = [
            np.array(
                [parameters[key] if key in parameters else self.fixed_params[key]
                 for key in _ORDERED_PRIOR_KEYS],
                dtype=np.float64,
            )
            for parameters in parameters_list
        ]
        tmnre_params_batch = np.stack(tmnre_params_list, axis=1)  # (11, n_walkers)
        bbhx_params_batch = self.sampler.samples_to_bbhx_input(
            tmnre_params_batch, t_obs_end=self.simulator.t_obs_end_SI
        )  # (12, n_walkers)
        waveform_kwargs = self.simulator.waveform_kwargs.copy()
        return self.bbhx_likelihood.get_ll(bbhx_params_batch, **waveform_kwargs)


def main():
    # ---- Load MCMC settings ----
    mcmc_config_file = os.path.join(ROOT_DIR, "configs", "mcmc_config.yaml")
    mcmc_conf = read_config(mcmc_config_file)
    event_idx        = mcmc_conf["event_idx"]
    high_freq_only   = mcmc_conf["high_freq_only"]
    freq_split_idx   = mcmc_conf["freq_split_idx"]
    observation_file = mcmc_conf["observation_file"]
    fisher_conf      = mcmc_conf["fisher"]
    manual_conf      = mcmc_conf.get("manual", {})
    emcee_conf       = mcmc_conf["emcee"]
    config_file = os.path.join(ROOT_DIR, "configs", "datagen_config.yaml")
    
    # Load configuration and data
    print("Loading configuration and data...")
    datagen_config = read_config(config_file)
    

    loaded_dataset = load_observation(observation_file)
    
    # Initialize simulator

    # THESE PRIOR BOUNDS ARE COMPLETELY MEANINGLESS, THEY SERVE THE SIMULATOR AT DATA GENERATION TIME, 
    # BUT MCMC DOES NOT USE THE SIMULATOR'S SAMPLER, SO THIS CODE IS NOT AFFECTED BY THESE PRIOR BOUNDS. 
    prior_bounds_dummy = {
        "logMchirp": [0,0],
        "q": [0,0],
        "chi1": [0.0, 0.0],
        "chi2": [0.0, 0.0],
        "dist": [0, 0],
        "phi": [0.0, 0.0],
        "inc": [0.0, 0.0],
        "lambda": [0.0, 0.0],
        "beta": [0.0, 0.00],
        "psi": [0.0, 0.0],
        "Deltat": [0.0, 0.0],
    }
    print("Initializing simulator...")
    wp = datagen_config["waveform_params"]
    datagen_config["backend"] = "cpu"  # CuPy JIT incompatible with CUDA 12.4
    # Make the simulator grid consistent with the observation, independent of
    # datagen_config's fmin/fmax/downsamplefactor (which may be set for a
    # different experiment). The grid is fully determined by the obs frequencies.
    _obs_freqs = np.asarray(loaded_dataset["frequencies"], dtype=np.float64)
    _T = wp["duration"] * 7 * 86400
    _df_obs = float(_obs_freqs[1] - _obs_freqs[0])
    wp["fmin"] = float(_obs_freqs[0])
    # +0.5*df_obs: np.arange must include the last obs bin without overshooting by one,
    # regardless of FP rounding (full +df sometimes produces n_obs+1 bins).
    wp["fmax"] = float(_obs_freqs[-1]) + 0.5 * _df_obs
    wp["downsamplefactor"] = int(round(_df_obs * _T))
    simulator = MBHBSimulatorFD(
        datagen_config,
        sampler_init_kwargs={'prior_bounds': prior_bounds_dummy},
        seed=42,
        n_freq_bins=wp.get("n_freq_bins", 4096),
        freq_spacing=wp.get("freq_spacing", "log"),
    )
    sampler = simulator.sampler
    frequencies = simulator.freqs
    
    # Extract observation
    print(f"Extracting event {event_idx}...")
    freqs = loaded_dataset["frequencies"]
    assert np.allclose(freqs, frequencies), "Frequency mismatch!"
    
    true_tmnre_params = loaded_dataset["source_parameters"][event_idx]
    data_fd_complex = (loaded_dataset["wave_fd"] + loaded_dataset["noise_fd"])[event_idx]
    
    # Setup PSD and data
    psd_AE = simulator.asd**2
    psd_ones_channelT = np.ones(shape=(1, psd_AE.shape[1]))
    psd_AET = np.concatenate([psd_AE, psd_ones_channelT], axis=0)
    
    data_T_channels = np.zeros(shape=(1, data_fd_complex.shape[1]), dtype=np.complex128)
    data_fd = np.concatenate([data_fd_complex, data_T_channels], axis=0)

    # Optional high-freq-only slicing (mirrors AE/NRE training-time config).
    # Restricts every later bbhx_likelihood.get_ll call to frequencies[freq_split_idx:].
    if high_freq_only:
        n_full = frequencies.shape[0]
        if not (0 < freq_split_idx < n_full):
            raise ValueError(
                f"freq_split_idx={freq_split_idx} out of range for n_freqs={n_full}"
            )
        frequencies = frequencies[freq_split_idx:]
        data_fd     = data_fd[:, freq_split_idx:]
        psd_AET     = psd_AET[:, freq_split_idx:]
        assert frequencies.shape[0] == data_fd.shape[1] == psd_AET.shape[1]
        # Keep simulator.waveform_kwargs in sync. bbhx.Likelihood.get_ll
        # overrides "freqs" with its own data_freqs, but pre-slicing here
        # guards against likelihood classes that don't override.
        simulator.waveform_kwargs["freqs"] = simulator.xp.asarray(frequencies)
        print(f"[mcmc] high_freq_only: keeping bins [{freq_split_idx}:{n_full}] "
              f"= {frequencies.shape[0]} bins (band {frequencies[0]:.3e}–{frequencies[-1]:.3e} Hz)")

    # Create BBHX likelihood
    print("Creating BBHX likelihood...")
    bbhx_likelihood = Likelihood(
        simulator.wfd,
        frequencies,
        data_fd,
        psd_AET,
        force_backend="cpu"
    )
            # Define prior bounds (same as mcmc.py)
    # prior_bounds = {
    #     "logMchirp": [5.25-3e-4, 5.25+3e-4],
    #     "q": [4.6777, 4.683],
    #     "chi1": [0.0, 0.0],
    #     "chi2": [0.0, 0.0],
    #     "dist": [10, 10],
    #     "phi": [0.0, 0.0],
    #     "inc": [0.5, 0.5],
    #     "lambda": [3.13, 3.15],
    #     "beta": [-0.01, 0.01],
    #     "psi": [1.0, 1.0],
    #     "Deltat": [-2.6, -2.4],
    # }

    # Separate fixed and varying parameters
    print("\\nSetting up parameters...")
    varying_params = list(mcmc_conf["varying_params"])
    # Save varying_params to a file for reproducibility

    varying_indices = [
        _ORDERED_PRIOR_KEYS.index(param) for param in varying_params
    ]
    fixed_params = {
        key: val for key, val in zip(_ORDERED_PRIOR_KEYS, true_tmnre_params) if key not in varying_params
    }

    ndim = len(varying_params)
    print(f"\\nTotal varying parameters: {ndim}")
    
    # Wrap in BBHXLikelihood
    likelihood = BBHXLikelihood(
        bbhx_likelihood,
        sampler,
        simulator,
        true_params=true_tmnre_params,
        fixed_params=fixed_params
    )
    
    true_params_dict = {key: true_tmnre_params[i] for i, key in enumerate(_ORDERED_PRIOR_KEYS)
                        if key not in fixed_params}

    # Compute Fisher Information Matrix (optional — sets prior bounds + walker init scale).
    if fisher_conf.get("enabled", True):
        print("\\n=== Computing Fisher Information Matrix ===")
        # Waveform-derivative Fisher. The routine generates on the simulator's
        # own asd/df grid (immune to the in-place slicing above); when
        # high_freq_only is on we pass the matching band mask so the Fisher
        # inner product mirrors the sampling likelihood.
        fisher_freq_mask = None
        if high_freq_only:
            fisher_freq_mask = np.zeros(simulator.freqs.shape[0], dtype=bool)
            fisher_freq_mask[freq_split_idx:] = True
        fisher_matrix, param_uncertainties = compute_fisher_matrix_waveform_deriv(
            simulator,
            true_tmnre_params,
            varying_params,
            freq_mask=fisher_freq_mask,
        )
        n_sigma = fisher_conf.get("prior_n_sigma", 15.0)
        prior_mins = np.array([true_tmnre_params[i] - n_sigma * param_uncertainties[j]
                               for j, i in enumerate(varying_indices)])
        prior_maxs = np.array([true_tmnre_params[i] + n_sigma * param_uncertainties[j]
                               for j, i in enumerate(varying_indices)])
        init_widths = fisher_conf.get("init_widths_factor", 0.1) * param_uncertainties
        print("proposed prior based on fim: ")
        for i, param in enumerate(varying_params):
            print(f"  {param}: [{prior_mins[i]:.6e}, {prior_maxs[i]:.6e}]")
    else:
        print("\\n=== Skipping Fisher matrix; using manual prior bounds ===")
        param_uncertainties = None
        manual_prior = manual_conf.get("prior", {})
        missing_p = [p for p in varying_params if p not in manual_prior]
        if missing_p:
            raise KeyError(f"manual.prior missing entries for: {missing_p}")
        prior_mins  = np.array([manual_prior[p][0] for p in varying_params], dtype=np.float64)
        prior_maxs  = np.array([manual_prior[p][1] for p in varying_params], dtype=np.float64)
        init_widths = manual_conf.get("init_widths_frac", 0.01) * (prior_maxs - prior_mins)

    prior_widths = prior_maxs - prior_mins
    for i, param in enumerate(varying_params):
        print(f"{param}: [{prior_mins[i]:.3e}, {prior_maxs[i]:.3e}] (width: {prior_widths[i]:.3e})")
    # Define log probability for emcee
    def log_prior(theta):
        """Uniform prior"""
        if np.all((theta >= prior_mins) & (theta <= prior_maxs)):
            return 0.0
        return -np.inf
    
    def log_probability(theta):
        """Log posterior = log prior + log likelihood"""
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        
        # Convert theta array to parameter dict
        params_dict = {param: theta[i] for i, param in enumerate(varying_params)}
        ll = likelihood.log_likelihood(params_dict)
        return lp + ll
    
    def log_probability_vectorized(theta_array):
        """Vectorized log posterior for all walkers
        
        theta_array has shape (n_walkers, ndim)
        """
        n_walkers = theta_array.shape[0]
        
        # Check priors for all walkers
        log_priors = np.array([log_prior(theta_array[i]) for i in range(n_walkers)])
        
        # Only evaluate likelihood for walkers with finite prior
        finite_mask = np.isfinite(log_priors)
        if not np.any(finite_mask):
            return log_priors
        
        # Convert to list of parameter dicts
        params_list = [
            {param: theta_array[i, j] for j, param in enumerate(varying_params)}
            for i in range(n_walkers) if finite_mask[i]
        ]
        # Vectorized likelihood evaluation - THE KEY OPTIMIZATION!
        log_likes_finite = likelihood.log_likelihood_vectorized(params_list)
        
        # Combine results
        log_posts = log_priors.copy()
        log_posts[finite_mask] += log_likes_finite
        
        return log_posts
    
    # Initialize walkers
    nwalkers = emcee_conf.get("nwalkers", 32)
    print(f"\\n=== Initializing {nwalkers} walkers ===")

    # Gaussian ball around the true parameters (init_widths set above —
    # FIM-derived or manual depending on fisher_conf.enabled).
    true_theta = np.array([true_params_dict[param] for param in varying_params])

    pos = true_theta + init_widths * np.random.randn(nwalkers, ndim)
    
    # Ensure all walkers start within prior
    pos = np.clip(pos, prior_mins, prior_maxs)
    
    print(f"Initial walker spread (std): {np.std(pos, axis=0)}")
    print(f"Prior widths: {prior_widths}")
    
    # Set up emcee sampler with vectorization
    print("\\n=== Setting up emcee sampler ===")
    sampler_emcee = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        log_probability_vectorized,
        vectorize=True  # Enable vectorized likelihood evaluation!
    )

    # Run MCMC
    nsteps = emcee_conf.get("nsteps", 1000)
    print(f"\\n=== Running MCMC for {nsteps} steps ===")
    state = sampler_emcee.run_mcmc(pos, nsteps, progress=True)
    
    # Get samples
    print("\\n=== Processing results ===")
    samples = sampler_emcee.get_chain()
    log_probs = sampler_emcee.get_log_prob()
    
    # Compute autocorrelation time
    try:
        tau = sampler_emcee.get_autocorr_time()
        print(f"Autocorrelation time: {tau}")
        burnin = int(2 * np.max(tau))
        thin = int(0.5 * np.min(tau))
    except emcee.autocorr.AutocorrError:
        print("Warning: Chain too short for autocorr estimate")
        burnin = nsteps // 4
        thin = 1
    
    # Flatten samples
    flat_samples = sampler_emcee.get_chain(discard=burnin, thin=thin, flat=True)
    loglikelihoods_samples = sampler_emcee.get_log_prob(discard=burnin, thin=thin, flat=True)
    # Save flat samples to a file
    
    print(f"Burned {burnin} steps, thinned by {thin}")
    print(f"Final samples: {flat_samples.shape[0]}")
    
    # Plot results
    print("\\nGenerating plots...")
    
    # Corner plot
    fig = corner.corner(
        flat_samples,
        labels=varying_params,
        truths=true_theta,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 12}
    )
    if "output_name" not in mcmc_conf or not mcmc_conf["output_name"]:
        raise KeyError("mcmc_config.yaml: 'output_name' is required (refusing to "
                       "default — would risk overwriting a previous run).")
    name = mcmc_conf["output_name"]
    outdir = os.path.join(ROOT_DIR, "mc_results_emcee_vec", name)
    os.makedirs(outdir, exist_ok=True)
    npy_file = os.path.join(outdir, "flat_samples.npy")
    npy_logprobs_file = os.path.join(outdir, "loglikelihoods_samples.npy")
    np.save(npy_file, flat_samples)
    np.save(npy_logprobs_file, loglikelihoods_samples)
    print(f"Saved flat samples to {npy_file}")
    print(f"Saved log-likelihoods of samples to {npy_logprobs_file}")

    # Also write samples in copparoni HDF5 format (per-parameter datasets) so
    # that scripts/visualise_truncation_rounds.py can consume them via
    # load_mcmc_samples().  Keys + units must match the convention in
    # mcmc_coppa/logf_samples_5D_copparoni.h5.
    duration_sec = datagen_config["waveform_params"]["duration"] * 7 * 86400.0
    _INTERNAL_TO_MCMC = {
        "logMchirp": ("logMchirp", lambda v: v),
        "q":         ("q",         lambda v: v),
        "lambda":    ("lambda",    lambda v: v),
        "beta":      ("sinbeta",   lambda v: v),
        "inc":       ("cosinc",    lambda v: v),
        "dist":      ("dist_Gpc",  lambda v: v),
        "Deltat":    ("tref",      lambda v: duration_sec + v * 86400.0),
        "chi1":      ("chi1",      lambda v: v),
        "chi2":      ("chi2",      lambda v: v),
        "phi":       ("phi",       lambda v: v),
        "psi":       ("psi",       lambda v: v),
    }
    samples_h5 = {}
    for col_idx, internal_name in enumerate(varying_params):
        if internal_name not in _INTERNAL_TO_MCMC:
            raise KeyError(
                f"No copparoni-key mapping for varying parameter '{internal_name}'"
            )
        mcmc_key, transform = _INTERNAL_TO_MCMC[internal_name]
        samples_h5[mcmc_key] = transform(flat_samples[:, col_idx])

    h5_file = os.path.join(outdir, "flat_samples.h5")
    with h5py.File(h5_file, "w") as f:
        for k, v in samples_h5.items():
            f.create_dataset(k, data=np.asarray(v, dtype=np.float64))
    print(f"Saved flat samples (copparoni format) to {h5_file}")
    varying_params_file = os.path.join(outdir, "varying_params.txt")
    with open(varying_params_file, "w") as f:
        for param in varying_params:
            f.write(param + "\n")
    print(f"Saved varying_params to {varying_params_file}")
    fig.savefig(os.path.join(outdir, "emcee_vectorized_corner.png"), dpi=150)
    print(f"Saved corner plot to {outdir}/emcee_vectorized_corner.png")
    
    # Chain plot
    fig, axes = plt.subplots(ndim, figsize=(10, 2*ndim), sharex=True)
    for i in range(ndim):
        ax = axes[i] if ndim > 1 else axes
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.axhline(true_theta[i], color="r", linestyle="--", label="True")
        ax.set_ylabel(varying_params[i])
        if i == 0:
            ax.legend()
    axes[-1].set_xlabel("Step")
    fig.savefig(os.path.join(outdir, "emcee_vectorized_chains.png"), dpi=150)
    print(f"Saved chain plot to {outdir}/emcee_vectorized_chains.png")
    
    # Print summary statistics
    print("\\n=== Summary Statistics ===")
    for i, param in enumerate(varying_params):
        mcmc_median = np.median(flat_samples[:, i])
        mcmc_std = np.std(flat_samples[:, i])
        true_val = true_theta[i]
        if param_uncertainties is not None:
            fim_std = param_uncertainties[i] if np.isfinite(param_uncertainties[i]) else np.nan
            fim_str = f"{fim_std:.6e}"
        else:
            fim_str = "N/A (fisher disabled)"

        print(f"{param}:")
        print(f"  True: {true_val:.6e}")
        print(f"  MCMC: {mcmc_median:.6e} ± {mcmc_std:.6e}")
        print(f"  FIM σ: {fim_str}")
        print(f"  Bias: {(mcmc_median - true_val)/true_val * 100:.2f}%")
    
    print("\\nDone!")


if __name__ == "__main__":
    main()
