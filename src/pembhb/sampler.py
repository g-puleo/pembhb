### a class that will take prior bounds as input and return samples from uniform prior. 

### should be extendable to support different ways to sample, e.g. rejection sampling based on criteria defined in the future

### 
import numpy as np
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
    Mc = 10 ** x[0]
    m1 = Mc * x[1]**1.4 * (1. + x[1])**0.2
    m2 = Mc * x[1]**0.4 * (1. + x[1])**(-0.6)
    return np.stack([m1, m2], axis=0)


class UniformSampler ():

    def __init__(self, prior_bounds: dict = None ):
        """_summary_

        :param prior_bounds: _description_
        :type prior_bounds: dict
        """
        self.prior_bounds = prior_bounds.copy()
        ## value is in Gpc^3
        self.prior_bounds["dist"][0]   = self.prior_bounds["dist"][0]**3
        self.prior_bounds["dist"][1]   = self.prior_bounds["dist"][1]**3
        self.lower_bounds = np.array([self.prior_bounds[key][0] for key in _ORDERED_PRIOR_KEYS]).reshape(-1,1)
        self.upper_bounds = np.array([self.prior_bounds[key][1] for key in _ORDERED_PRIOR_KEYS]).reshape(-1,1)
        self.n_params = self.lower_bounds.shape[0]
    
    def sample(self, n_samples: int) -> np.array:
        """ Generate samples from the uniform prior.

        :param n_samples: number of samples to generate
        :type n_samples: int
        :return: samples in bbhx input format, samples for tmnre
        :rtype: list[np.array]
        """

        unif_samples = np.random.uniform(0, 1, size=(self.n_params, n_samples))
        tmnre_input = unif_samples * (self.upper_bounds - self.lower_bounds) + self.lower_bounds
        #NB IT IS VERY IMPORTANT TO USE .copy() OTHERWISE THE OPERATIONS WILL BE PERFORMED IN-PLACE
        bbhx_input = self.samples_to_bbhx_input(tmnre_input.copy())
        return bbhx_input , tmnre_input

    def samples_to_bbhx_input(self, samples: np.array) -> np.array:
        """convert the sampler to the bbhx input format : 
        

        :param samples: MBHB parameters in the following order: log10(chirp mass), q, chi1, chi2, dist, phi, cos(inc), lambda, sin(beta), psi, Deltat
        :type samples: np.array
        :return: MBHB parameters in the following order: m1, m2, chi1, chi2, distance, phase, inclination, lambda, beta, psi, Deltat
        :rtype: np.array
        """
    
        samples[0:2] = lMcq_m1m2(samples[0:2]) # log(Mc), q --> m1, m2
        samples[4] = np.cbrt(samples[4]) * 1e9 * PC_SI # d^3 -->distance
        # 5: phase is already in 0,2pi
        samples[6] = np.arccos(samples[6]) # cos(inclination)-->inclination in [0,pi]
        # 7: lambda is already in 0,2pi
        samples[8] = np.arcsin(samples[8]) # sin(beta)--> beta in [-pi/2, pi/2] (ecliptic latitude)
        # 9: psi is already in 0,pi
        # 10: Deltat is already in seconds   
        samples[10] = samples[10]*DAY_SI
        return samples
    




# def hpd_thresholder(grid: np.array, fun: callable, hpd: float = 0.9) -> int:
#     f = fun(grid)
#     sorted_pixels = np.sort(f)
#     cumsum = np.cumsum(sorted_pixels)
#     total_sum = cumsum[-1]
#     threshold = hpd * total_sum
#     idx = np.searchsorted(cumsum, threshold)
#     return sorted_pixels[idx]