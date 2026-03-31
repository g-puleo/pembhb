### a class that will take prior bounds as input and return samples from uniform prior. 

### should be extendable to support different ways to sample, e.g. rejection sampling based on criteria defined in the future

### 
import numpy as np
import copy
from bbhx.utils.constants import PC_SI
from pembhb.utils import _ORDERED_PRIOR_KEYS
DAY_SI = 24 * 3600  # seconds in a day

def lMcq_m1m2(x: np.array):
    """Return m1, m2 from log10(chirp mass) and q

    :param x: x[0] is log10(chirp mass) and x[1] is q
    :type x: :class:`np.array`
    :return: m1 and m2
    :rtype: np.array
    """
    lMc = x[0]
    q = x[1]
    Mc = 10 ** lMc
    M = Mc * (q/(1+q)**2)**(-0.6)
    return np.stack([M * q / (1. + q), M / (1. + q)], axis=0)

class UniformSampler ():

    def __init__(self, prior_bounds: dict = None ):
        """Initialise sampler with given prior bounds. 

        :param prior_bounds: dict of prior bounds
        :type prior_bounds: dict
        """
        print("init of uniform sampler")
        self.prior_bounds = copy.deepcopy(prior_bounds)
        ## value is in Gpc^3
        self.prior_bounds["dist"][0]   = self.prior_bounds["dist"][0]**3
        self.prior_bounds["dist"][1]   = self.prior_bounds["dist"][1]**3
        self.lower_bounds = np.array([self.prior_bounds[key][0] for key in _ORDERED_PRIOR_KEYS]).reshape(-1,1)
        self.upper_bounds = np.array([self.prior_bounds[key][1] for key in _ORDERED_PRIOR_KEYS]).reshape(-1,1)
        self.n_params = self.lower_bounds.shape[0]
    
    def sample(self, n_samples: int, t_obs_end: float) -> np.array:
        """ Generate samples from the uniform prior.

        :param n_samples: number of samples to generate
        :type n_samples: int
        :param t_obs_end: end of observation time in seconds (used to offset the t_ref)
        :type t_obs_end: float
        :return: samples in bbhx input format, samples for tmnre
        :rtype: list[np.array]
        
        """

        unif_samples = np.random.uniform(0, 1, size=(self.n_params, n_samples))
        tmnre_input = unif_samples * (self.upper_bounds - self.lower_bounds) + self.lower_bounds

        is_monotonic = self.lower_bounds <= self.upper_bounds
        if not np.all(is_monotonic):
            # find which parameters have non-monotonic bounds and raise an error
            idxs = np.argwhere(~is_monotonic)
            raise ValueError(f"All upper bounds must be greater than lower bounds, but this was violated by params at positions {idxs.flatten()}")
        # take cube root of tmnre input for distance to get back to Gpc units
        tmnre_input[4] = np.cbrt(tmnre_input[4])
        #NB IT IS VERY IMPORTANT TO USE .copy() OTHERWISE THE OPERATIONS WILL BE PERFORMED IN-PLACE
        bbhx_input = self.samples_to_bbhx_input(tmnre_input.copy(), t_obs_end)
        ## insert f_ref=0
        return bbhx_input , tmnre_input

    def samples_to_bbhx_input(self, samples: np.array, t_obs_end: float) -> np.array:
        """ Convert the sampler output to the bbhx input format : 
    
        :param samples: MBHB parameters in the following order: log10(chirp mass), q, chi1, chi2, dist, phi, cos(inc), lambda, sin(beta), psi, Deltat
        :type samples: np.array
        :param t_obs_end: observation time in seconds (used to offset the t_ref)
        :type t_obs_end: float
        :return: MBHB parameters in the following order: m1, m2, chi1, chi2, distance, phase, inclination, lambda, beta, psi, Deltat
        :rtype: np.array
        """
        n_samples = samples.shape[1]
        samples_ = samples.copy()
        samples_[0:2] = lMcq_m1m2(samples_[0:2]) # log(Mc), q --> m1, m2
        samples_[4] = samples_[4]* 1e9 * PC_SI # d^3 -->distance
        # 5: phase is already in 0,2pi
        samples_[6] = np.arccos(samples_[6]) # cos(inclination)-->inclination in [0,pi]
        # 7: lambda is already in 0,2pi
        samples_[8] = np.arcsin(samples_[8]) # sin(beta)--> beta in [-pi/2, pi/2] (ecliptic latitude)
        # 9: psi is already in 0,pi
        # 10: Deltat is already in seconds
        samples_[10] = samples_[10]*DAY_SI + t_obs_end # offset t_ref by the observation time
        samples_ = np.insert(samples_, 6, np.zeros(n_samples), axis=0)

        return samples_


class MaskRejectSampler:
    """Uniform sampler with sky-mask rejection for (lambda, beta).

    All 11 parameters are drawn uniformly within rectangular prior bounds
    (identical to ``UniformSampler``), but samples whose (lambda, sin_beta)
    falls outside a precomputed boolean mask are rejected and redrawn.

    This is the recommended sampler when the sky posterior has irregular
    shape, multiple modes, or wraps across the lambda = 0 / 2pi boundary.

    Parameters
    ----------
    prior_bounds : dict
        Same format as ``UniformSampler``.
    sky_mask : np.ndarray, shape (n_beta, n_lam)
        Boolean acceptance mask on the (lambda, sin_beta) grid.
    grid_lam : np.ndarray, shape (n_beta, n_lam)
        Lambda meshgrid (``indexing='xy'``).
    grid_beta : np.ndarray, shape (n_beta, n_lam)
        sin(beta) meshgrid (``indexing='xy'``).
    """

    def __init__(self, prior_bounds: dict, sky_mask: np.ndarray,
                 grid_lam: np.ndarray, grid_beta: np.ndarray):
        print("init of MaskRejectSampler")
        self.base_sampler = UniformSampler(prior_bounds)
        self.sky_mask = sky_mask
        self.grid_lam = grid_lam
        self.grid_beta = grid_beta

        # Precompute grid spacings for fast index lookup
        self._lam_min = grid_lam[0, 0]
        self._dlam = grid_lam[0, 1] - grid_lam[0, 0]
        self._beta_min = grid_beta[0, 0]
        self._dbeta = grid_beta[1, 0] - grid_beta[0, 0]
        self._n_beta, self._n_lam = sky_mask.shape

        frac = sky_mask.sum() / sky_mask.size
        print(f"  sky mask covers {100*frac:.1f}% of the rectangular prior")

    def _accept_sky(self, lam, sin_beta):
        """Return boolean mask: True where (lam, sin_beta) falls inside the sky mask."""
        col = np.clip(((lam - self._lam_min) / self._dlam).astype(int), 0, self._n_lam - 1)
        row = np.clip(((sin_beta - self._beta_min) / self._dbeta).astype(int), 0, self._n_beta - 1)
        return self.sky_mask[row, col]

    def sample(self, n_samples: int, t_obs_end: float):
        """Generate samples, rejecting those outside the sky mask.

        Returns the same (bbhx_input, tmnre_input) tuple as ``UniformSampler.sample``.
        """
        # Oversample to reduce the number of rejection iterations
        oversample = max(int(n_samples / max(self.sky_mask.mean(), 0.01)), n_samples * 2)

        bbhx_parts = []
        tmnre_parts = []
        collected = 0

        while collected < n_samples:
            bbhx_batch, tmnre_batch = self.base_sampler.sample(oversample, t_obs_end)
            # tmnre_batch has shape (11, oversample)
            # lambda = index 7, sin(beta) = index 8
            accept = self._accept_sky(tmnre_batch[7], tmnre_batch[8])
            n_acc = accept.sum()
            if n_acc == 0:
                oversample *= 4  # very low acceptance, increase batch
                continue

            need = min(n_acc, n_samples - collected)
            idx = np.where(accept)[0][:need]
            bbhx_parts.append(bbhx_batch[:, idx])
            tmnre_parts.append(tmnre_batch[:, idx])
            collected += need

        bbhx_input = np.concatenate(bbhx_parts, axis=1)
        tmnre_input = np.concatenate(tmnre_parts, axis=1)
        return bbhx_input, tmnre_input

    def samples_to_bbhx_input(self, samples, t_obs_end):
        return self.base_sampler.samples_to_bbhx_input(samples, t_obs_end)
