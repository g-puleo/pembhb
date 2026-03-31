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
from bbhx.likelihood import Likelihood
from pembhb.simulator import MBHBSimulatorFD_TD, MBHBSimulatorFD
from pembhb.utils import read_config, _ORDERED_PRIOR_KEYS
from pembhb import ROOT_DIR
import h5py
import os
from mcmc import BBHXLikelihood, load_observation, compute_fisher_information_matrix


def main():
    # Configuration
    event_idx = 0  # Which event to analyze
    # 5d
    #observation_file = "/data/gpuleo/mbhb/observation_skyloc_tc_mass.h5"
    # 2d 
    observation_file = "/data/gpuleo/mbhb/obs_logspace_freqonly_q3.h5"
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
    varying_params = [ #declare explicitly which ones : 
        "logMchirp",
        "q", 
        "Deltat",
        "lambda",
        "beta"
    ]
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
    
    # Compute Fisher Information Matrix
    print("\\n=== Computing Fisher Information Matrix ===")
    true_params_dict = {key: true_tmnre_params[i] for i, key in enumerate(_ORDERED_PRIOR_KEYS) 
                        if key not in fixed_params}
    
    fisher_matrix, param_uncertainties = compute_fisher_information_matrix(
        likelihood,
        true_params_dict,
        varying_params,
        delta_frac=1e-6
    )

    
    
    # Define prior bounds for varying params
    prior_mins = np.array([true_tmnre_params[i]-15*param_uncertainties[j] for j, i in enumerate(varying_indices)])
    prior_maxs = np.array([true_tmnre_params[i]+15*param_uncertainties[j] for j, i in enumerate(varying_indices)])
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
    nwalkers = 32
    print(f"\\n=== Initializing {nwalkers} walkers ===")
    
    # Initialize in small ball around true parameters using FIM uncertainties
    true_theta = np.array([true_params_dict[param] for param in varying_params])
    
    # Use FIM uncertainties if available, otherwise use 1% of prior range
    init_widths = 0.1 * param_uncertainties  # 10% of 1σ uncertainty
    
    
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
    nsteps = 1000
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
    name="5d_qwide"
    outdir = os.path.join(ROOT_DIR, "mc_results_emcee_vec", name)
    os.makedirs(outdir, exist_ok=True)
    output_file = os.path.join(outdir, "flat_samples.npy")
    np.save(output_file, flat_samples)
    print(f"Saved flat samples to {output_file}")
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
        fim_std = param_uncertainties[i] if np.isfinite(param_uncertainties[i]) else np.nan
        
        print(f"{param}:")
        print(f"  True: {true_val:.6e}")
        print(f"  MCMC: {mcmc_median:.6e} ± {mcmc_std:.6e}")
        print(f"  FIM σ: {fim_std:.6e}")
        print(f"  Bias: {(mcmc_median - true_val)/true_val * 100:.2f}%")
    
    print("\\nDone!")


if __name__ == "__main__":
    main()
