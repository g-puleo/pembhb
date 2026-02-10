"""
MCMC/Nested Sampling parameter estimation using bilby for MBHB data
"""
import bilby
from bilby.bilby_mcmc.proposals import ProposalCycle, AdaptiveGaussianProposal
from bilby.core.prior.dict import PriorDict
from bilby.bilby_mcmc.sampler import Bilby_MCMC
import matplotlib.pyplot as plt
import numpy as np
from bbhx.likelihood import Likelihood
from pembhb.simulator import MBHBSimulatorFD_TD
from pembhb.utils import read_config, _ORDERED_PRIOR_KEYS
from pembhb import ROOT_DIR
import h5py
import os


def load_observation(fname):
    """Load observation data from HDF5 file"""
    with h5py.File(fname, "r") as f:
        src = f["source_parameters"][:]        # (N,11)
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
    """
    Bilby Likelihood wrapper for BBHX likelihood
    
    This class wraps the bbhx.Likelihood and handles the parameter
    transformation from the prior space (what bilby samples) to the
    bbhx input space (what bbhx.likelihood expects).
    """
    
    def __init__(self, bbhx_likelihood, sampler, simulator, true_params=None, fixed_params=None):
        """
        Initialize the BBHX likelihood wrapper
        
        Parameters
        ----------
        bbhx_likelihood : bbhx.Likelihood
            The BBHX likelihood object
        sampler : UniformSampler
            The sampler object that handles parameter transformations
        simulator : MBHBSimulatorFD_TD
            The simulator object containing observation time info
        true_params : np.array, optional
            True parameters (for reference), shape (11,)
        fixed_params : dict, optional
            Parameters to hold fixed (e.g., for zero-width priors)
        """
        super().__init__()
        self.bbhx_likelihood = bbhx_likelihood
        self.sampler = sampler
        self.simulator = simulator
        self.true_params = true_params
        self.fixed_params = fixed_params or {}
        
    def log_likelihood(self, parameters):
        """
        Compute log likelihood given parameters sampled from prior
        
        Parameters
        ----------
        parameters : dict
            Dictionary with parameter names as keys (following _ORDERED_PRIOR_KEYS)
            
        Returns
        -------
        float
            Log likelihood value
        """
        # Convert bilby parameters dict to array in the correct order
        # For each parameter, use the sampled value if varying, otherwise use the fixed value
        tmnre_params = np.array(
            [parameters[key] if key in parameters else self.fixed_params[key] 
             for key in _ORDERED_PRIOR_KEYS],
                         dtype=np.float64
        )
        tmnre_params = tmnre_params.reshape(-1, 1)  # shape (11, 1)
        
        # Transform to bbhx input space
        bbhx_params = self.sampler.samples_to_bbhx_input(
            tmnre_params, 
            t_obs_end=self.simulator.t_obs_end_SI
        )  # shape (12, 1)
        
        # Get waveform kwargs (copy because get_ll mutates the dict in-place)
        waveform_kwargs = self.simulator.waveform_kwargs.copy()
        
        # Evaluate likelihood
        log_l = self.bbhx_likelihood.get_ll(bbhx_params, **waveform_kwargs)
        return float(log_l[0])


def main():
    # Configuration
    event_idx = 0  # Which event to analyze
    observation_file = "/data/gpuleo/mbhb/observation_skyloc.h5"
    config_file = os.path.join(ROOT_DIR, "configs", "datagen_config.yaml")
    
    # Output settings
    label = f"bbhx_event_{event_idx}_bilbymcmc"
    outdir = os.path.join(ROOT_DIR, "mc_results")
    os.makedirs(outdir, exist_ok=True)
    
    # Load configuration and data
    print("Loading configuration and data...")
    datagen_config = read_config(config_file)
    # Override priors with hardcoded bounds
    # NOTE: dist is in Gpc (will be cubed in UniformSampler.__init__)
    prior_bounds = {
        "logMchirp": [5.25-3e-5, 5.25+3e-5],
        "q": [4.678, 4.683],
        "chi1": [0.0, 0.0],
        "chi2": [0.0, 0.0],
        "dist": [10, 10],  # 10 Gpc 
        "phi": [0.0, 0.0],
        "inc": [0.5, 0.5],
        "lambda": [3.13, 3.15],
        "beta": [0.498, 0.5025],
        "psi": [1.0, 1.0],
        "Deltat": [-2.0, -2.0],
    }
    loaded_dataset = load_observation(observation_file)
    
    # Initialize simulator and sampler
    print("Initializing simulator...")
    simulator = MBHBSimulatorFD_TD(
        datagen_config, 
        sampler_init_kwargs={'prior_bounds': prior_bounds}, 
        seed=42
    )
    sampler = simulator.sampler
    waveform_gen = simulator.wfd
    frequencies = simulator.freqs_pos
    
    # Extract observation data for the event
    print(f"Extracting event {event_idx}...")
    freqs = loaded_dataset["frequencies"]
    assert np.allclose(freqs, frequencies), "Frequency mismatch!"
    
    true_tmnre_params = loaded_dataset["source_parameters"][event_idx]  # shape (11,)
    data_fd_complex = (loaded_dataset["wave_fd"] + loaded_dataset["noise_fd"])[event_idx]  # shape (2, n_freqs)
    
    # Setup PSD
    psd_AE = simulator.asd**2  # shape (2, n_freqs)
    psd_ones_channelT = np.ones(shape=(1, psd_AE.shape[1]))
    psd_AET = np.concatenate([psd_AE, psd_ones_channelT], axis=0)  # shape (3, n_freqs)
    
    # Prepare data with T channel
    data_T_channels = np.zeros(shape=(1, data_fd_complex.shape[1]), dtype=np.complex128)
    data_fd = np.concatenate([data_fd_complex, data_T_channels], axis=0)  # shape (3, n_freqs)
    
    # Create BBHX likelihood
    print("Creating BBHX likelihood...")
    bbhx_likelihood = Likelihood(
        waveform_gen, 
        frequencies, 
        data_fd, 
        psd_AET, 
        force_backend="cpu"
    )
    

    print("Setting up priors...")
    priors = {}
    fixed_params = {}
    for key in _ORDERED_PRIOR_KEYS:
        bounds = prior_bounds[key]
        if bounds[0] == bounds[1]:
            fixed_params[key] = sampler.prior_bounds[key][0]  # use already-cubed value for dist
            print(f"fixed {key} at {fixed_params[key]:.6f} (prior_bounds: {bounds[0]:.6f})")
        else:
            priors[key] = bilby.core.prior.Uniform(
                minimum=bounds[0],
                maximum=bounds[1],
                name=key
            )
            print(f"uniform prior for {key} in [{bounds[0]:.6f}, {bounds[1]:.6f}]")

    prior_dict = PriorDict(priors)
    
    # Use smaller step size for narrow priors
    # sigma controls step size: step = prior_width * sigma * random_normal
    proposal_cycle = ProposalCycle([
        AdaptiveGaussianProposal(prior_dict, sigma=0.1)  # Reduce from default 1.0
    ])
    
    # Wrap in bilby likelihood
    likelihood = BBHXLikelihood(
        bbhx_likelihood,
        sampler,
        simulator,
        true_params=true_tmnre_params,
        fixed_params=fixed_params
    )
    
    # DIAGNOSTIC: Grid search to verify likelihood peak
    print("\n=== DIAGNOSTIC: Grid Search Around True Parameters ===")
    test_params_base = {key: true_tmnre_params[i] for i, key in enumerate(_ORDERED_PRIOR_KEYS) if key not in fixed_params}
    
    # Evaluate at true params first
    true_logl = likelihood.log_likelihood(test_params_base)
    print(f"Log-likelihood at TRUE parameters: {true_logl:.1f}")
    
    best_logl = true_logl
    best_params = {}
    
    for key in ['logMchirp', 'q', 'lambda', 'beta']:
        if key not in test_params_base:
            continue
        print(f"\nScanning {key}:")
        bounds = prior_bounds[key]
        grid = np.linspace(bounds[0], bounds[1], 11)
        
        test_params = test_params_base.copy()
        logls = []
        for val in grid:
            test_params[key] = val
            logl = likelihood.log_likelihood(test_params)
            logls.append(logl)
            marker = " <-- TRUE" if abs(val - test_params_base[key]) < 1e-10 else ""
            if logl > best_logl:
                marker += " *** NEW MAX ***"
                best_logl = logl
                best_params[key] = val
            print(f"  {key}={val:.6f}: log_L = {logl:.1f}{marker}")
        
        max_idx = np.argmax(logls)
        print(f"  --> Max at grid point {max_idx}: {key}={grid[max_idx]:.6f}, log_L={logls[max_idx]:.1f}")
    
    if best_params:
        print(f"\n=== Best found in grid search ===")
        print(f"Best log_L = {best_logl:.1f} (improvement: {best_logl - true_logl:.1f})")
        for key, val in best_params.items():
            diff = val - test_params_base[key]
            print(f"  {key}: {val:.6f} (shift: {diff:+.6f})")
    print("="*50)
    
    # Run sampler
    print("\n=== Starting MCMC Sampler ===")

    sampler = Bilby_MCMC(
        likelihood=likelihood,
        priors=prior_dict, 
        outdir=outdir,
        label=label,
        use_ratio=False,  # We are using the likelihood directly, not a ratio
        check_point_plot=True,
        check_point_delta_t=60,
        diagnostic=True,
        resume=False,
        nsamples=1000,
        burn_in_nact=100,
        fixed_discard=0,
        thin_by_nact=0.2,
        printdt=30,
        proposal_cycle=proposal_cycle,
        initial_sample_method="prior"  # Try "prior" but watch if it starts at bad location
    )
    result = sampler.run_sampler()
    # Generate corner plot
    print("Generating corner plot...")
    result.plot_corner()
    
    print(f"\nResults saved live=1000,to {outdir}")
    print("Done!")


if __name__ == "__main__":
    main()
