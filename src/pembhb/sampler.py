### a class that will take prior bounds as input and return samples from uniform prior. 

### should be extendable to support different ways to sample, e.g. rejection sampling based on criteria defined in the future

### 
import numpy as np
import copy
from bbhx.utils.constants import PC_SI
from pembhb.utils import _ORDERED_PRIOR_KEYS, ordered_prior_keys
DAY_SI = 24 * 3600  # seconds in a day


def chieff_chidiff_to_chi12(q, chi_eff, chi_diff):
    """Invert (chi_eff, chi_diff) -> (chi1, chi2) given mass ratio q = m1/m2.

    chi_eff = (q*chi1 + chi2)/(1+q), chi_diff = (chi1 - chi2)/2
        =>  chi1 = chi_eff + 2*chi_diff/(1+q)
            chi2 = chi_eff - 2*q*chi_diff/(1+q)
    Single source of truth for both the rejection test and bbhx-input transform.
    """
    chi1 = chi_eff + 2.0 * chi_diff / (1.0 + q)
    chi2 = chi_eff - 2.0 * q * chi_diff / (1.0 + q)
    return chi1, chi2


def chi12_to_chieff_chidiff(q, chi1, chi2):
    """Forward (chi1, chi2) -> (chi_eff, chi_diff) given mass ratio q = m1/m2.

    Used to compare external (chi1, chi2) samples — e.g. MCMC chains — against
    an NRE trained on the (chi_eff, chi_diff) basis.
    """
    chi_eff = (q * chi1 + chi2) / (1.0 + q)
    chi_diff = (chi1 - chi2) / 2.0
    return chi_eff, chi_diff


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

    def __init__(self, prior_bounds: dict = None, rng: np.random.Generator = None,
                 dist_uniform_in_volume: bool = True,
                 spin_param_basis: str = "chi1chi2"):
        """Initialise sampler with given prior bounds.

        :param prior_bounds: dict of prior bounds
        :type prior_bounds: dict
        :param rng: NumPy random Generator. If None, uses the global np.random state.
        :type rng: np.random.Generator or None
        :param dist_uniform_in_volume: if True (default), sample distance uniformly in d^3
            (i.e. uniform-in-volume); if False, sample distance uniformly in d.
        :type dist_uniform_in_volume: bool
        :param spin_param_basis: "chi1chi2" (default/legacy) samples slots 2,3 as
            the individual aligned spins. "chieff_chidiff" instead samples
            chi_eff=(m1*chi1+m2*chi2)/(m1+m2) and chi_diff=(chi1-chi2)/2, inverts
            to (chi1,chi2) at bbhx-input time, and rejects draws with |chi|>1.
        :type spin_param_basis: str
        """
        print(f"init of uniform sampler (dist_uniform_in_volume={dist_uniform_in_volume}, "
              f"spin_param_basis={spin_param_basis})")
        self.rng = rng
        self.spin_param_basis = spin_param_basis
        # Names for prior-bound lookup follow the spin basis (slots 2,3).
        self._prior_keys = ordered_prior_keys(spin_param_basis)
        self.prior_bounds = copy.deepcopy(prior_bounds)
        self.dist_uniform_in_volume = dist_uniform_in_volume
        if self.dist_uniform_in_volume:
            ## value is in Gpc^3
            self.prior_bounds["dist"][0]   = self.prior_bounds["dist"][0]**3
            self.prior_bounds["dist"][1]   = self.prior_bounds["dist"][1]**3
        self.lower_bounds = np.array([self.prior_bounds[key][0] for key in self._prior_keys]).reshape(-1,1)
        self.upper_bounds = np.array([self.prior_bounds[key][1] for key in self._prior_keys]).reshape(-1,1)
        self.n_params = self.lower_bounds.shape[0]
        # Acceptance ratio of the most recent sample() call (1.0 when no
        # rejection is performed, i.e. the chi1chi2 basis).
        self.last_acceptance_ratio = 1.0

    def _draw_tmnre(self, n_samples: int) -> np.array:
        """Draw ``n_samples`` raw tmnre-space vectors uniformly in the prior box.

        Shape (n_params, n_samples). No spin inversion or rejection — slots 2,3
        are the basis coordinates as drawn. Distance is returned in Gpc.
        """
        _rng = self.rng if self.rng is not None else np.random
        is_monotonic = self.lower_bounds <= self.upper_bounds
        if not np.all(is_monotonic):
            idxs = np.argwhere(~is_monotonic)
            raise ValueError(f"All upper bounds must be greater than lower bounds, but this was violated by params at positions {idxs.flatten()}")
        unif_samples = _rng.uniform(0, 1, size=(self.n_params, n_samples))
        tmnre_input = unif_samples * (self.upper_bounds - self.lower_bounds) + self.lower_bounds
        if self.dist_uniform_in_volume:
            # take cube root of tmnre input for distance to get back to Gpc units
            tmnre_input[4] = np.cbrt(tmnre_input[4])
        return tmnre_input

    def _accept_spin(self, tmnre_input: np.array) -> np.array:
        """Boolean mask: rows whose inverted spins satisfy |chi1|<=1 & |chi2|<=1.

        Only valid in the chieff_chidiff basis (the caller already branches on
        it); raises otherwise so a misuse fails loudly instead of silently
        accepting everything.
        """
        if self.spin_param_basis != "chieff_chidiff":
            raise RuntimeError(
                "_accept_spin called outside the chieff_chidiff basis "
                f"(spin_param_basis={self.spin_param_basis!r})")
        q, chi_eff, chi_diff = tmnre_input[1], tmnre_input[2], tmnre_input[3]
        chi1, chi2 = chieff_chidiff_to_chi12(q, chi_eff, chi_diff)
        return (np.abs(chi1) <= 1.0) & (np.abs(chi2) <= 1.0)

    def sample(self, n_samples: int, t_obs_end: float) -> np.array:
        """ Generate samples from the uniform prior.

        :param n_samples: number of samples to generate
        :type n_samples: int
        :param t_obs_end: end of observation time in seconds (used to offset the t_ref)
        :type t_obs_end: float
        :return: samples in bbhx input format, samples for tmnre
        :rtype: list[np.array]

        In the chieff_chidiff basis, draws are rejected when the inverted spins
        leave the physical region (|chi|>1); the accepted fraction is recorded in
        ``self.last_acceptance_ratio``.
        """
        if self.spin_param_basis != "chieff_chidiff":
            tmnre_input = self._draw_tmnre(n_samples)
            self.last_acceptance_ratio = 1.0
        else:
            collected = []
            n_drawn = 0
            n_acc_total = 0
            while sum(c.shape[1] for c in collected) < n_samples:
                batch = self._draw_tmnre(n_samples)
                accept = self._accept_spin(batch)
                n_drawn += batch.shape[1]
                n_acc_total += int(accept.sum())
                if accept.any():
                    collected.append(batch[:, accept])
            tmnre_input = np.concatenate(collected, axis=1)[:, :n_samples]
            self.last_acceptance_ratio = n_acc_total / max(n_drawn, 1)
            print(f"  spin rejection (chieff_chidiff): acceptance ratio "
                  f"{self.last_acceptance_ratio:.3f}")

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
        if self.spin_param_basis == "chieff_chidiff":
            # slots 2,3 hold (chi_eff, chi_diff); invert to (chi1, chi2) using q
            # (slot 1, still the raw mass ratio before the lMcq->m1m2 transform).
            chi1, chi2 = chieff_chidiff_to_chi12(samples_[1], samples_[2], samples_[3])
            samples_[2] = chi1
            samples_[3] = chi2
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
                 grid_lam: np.ndarray, grid_beta: np.ndarray,
                 rng: np.random.Generator = None,
                 dist_uniform_in_volume: bool = True,
                 spin_param_basis: str = "chi1chi2"):
        print("init of MaskRejectSampler")
        self.base_sampler = UniformSampler(prior_bounds, rng=rng,
                                           dist_uniform_in_volume=dist_uniform_in_volume,
                                           spin_param_basis=spin_param_basis)
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
