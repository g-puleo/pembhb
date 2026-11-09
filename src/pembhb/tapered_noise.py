"""FD noise of a tapered time-domain cut.

Stationary noise on the circle of the cut window (diagonal on its full rfft
grid) -> irfft -> Tukey window -> rfft -> kept bins. This equals the tapered cut
of a long noise stream (notebooks/cut_noise_toy.ipynb, test 3a + taper); the
taper leaves a known, frequency-independent correlation between neighbouring bins.
"""
import numpy as np
import torch

# Batches built in DataLoader workers carry a key, not the model: pickling it would copy its arrays
# every batch (and its cached CUDA tensors cannot leave a forked worker).
_REGISTRY = {}


def lookup(key):
    return _REGISTRY[key]


def tukey(n, n_ramp, sides="both", n_ramp_end=None):
    """Cosine ramps at the start (n_ramp samples), the end (n_ramp_end, default n_ramp) or both;
    a ramp of 0 samples is a box edge."""
    if sides not in ("both", "start", "end"):
        raise ValueError(f"taper sides must be 'both', 'start' or 'end', not {sides!r}")
    n_end = n_ramp if n_ramp_end is None else n_ramp_end
    ramp = lambda m: 0.5 * (1 - np.cos(np.pi * np.arange(m) / m))
    win = np.ones(n)
    if n_ramp and sides != "end":
        win[:n_ramp] = ramp(n_ramp)
    if n_end and sides != "start":
        win[-n_end:] = ramp(n_end)[::-1]
    return win


class TaperedNoise:
    def __init__(self, sigma_full, window, keep_idx, dt, t_obs_start_SI):
        """
        :param sigma_full: (C, n//2 + 1) per-bin std of Re and Im, ASD / sqrt(4 df), on the full rfft grid (DC = 0)
        :param window: (n,) taper
        :param keep_idx: indices of the output bins in the full rfft grid
        :param dt: sampling interval [s]
        :param t_obs_start_SI: absolute start time of the window [s]; the output carries exp(-2 pi i f t0) like cut_to_fd
        """
        self.sigma_full = torch.as_tensor(np.asarray(sigma_full), dtype=torch.float64)
        self.window = torch.as_tensor(np.asarray(window), dtype=torch.float64)
        self.keep_idx = torch.as_tensor(np.asarray(keep_idx), dtype=torch.long)
        self.dt = float(dt)
        self.t_obs_start_SI = float(t_obs_start_SI)
        self.n = self.window.numel()
        T = self.n * self.dt
        # f t0 = k t0 / T with integer k, so t0 mod T gives the same phase
        self.phase = torch.exp(-2j * torch.pi * self.keep_idx.double() * ((self.t_obs_start_SI % T) / T))
        self._cache = {}
        self.key = id(self)
        _REGISTRY[self.key] = self

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_cache"] = {}
        return state

    def _on(self, device, dtype):
        key = (str(device), dtype)
        if key not in self._cache:
            cdtype = torch.complex64 if dtype == torch.float32 else torch.complex128
            self._cache[key] = (self.sigma_full.to(device, dtype), self.window.to(device, dtype),
                                self.keep_idx.to(device), self.phase.to(device, cdtype))
        return self._cache[key]

    def draw(self, B, device="cpu", dtype=torch.float32, generator=None):
        """(B, C, n_keep) complex noise, same scaling as ``(re + j im) * noise_scale`` on the circle.
        Return: 
        - noise in the frequency domain, after tapering in time domain x(f)--> x(t) --> x(t)*w(t) ---> x_tapered(f)
        """
        sigma, w, keep, phase = self._on(device, dtype)
        shape = (B,) + tuple(sigma.shape)
        re = torch.randn(shape, device=device, dtype=dtype, generator=generator)
        im = torch.randn(shape, device=device, dtype=dtype, generator=generator)
        x = torch.fft.irfft(torch.complex(re, im) * sigma, n=self.n, dim=-1) / self.dt
        return (self.dt * torch.fft.rfft(x * w, dim=-1))[..., keep] * phase

    def to_h5(self, f):
        f.create_dataset("noise_sigma_full", data=self.sigma_full.numpy())
        f.create_dataset("noise_window", data=self.window.numpy())
        f.create_dataset("noise_keep_idx", data=self.keep_idx.numpy())
        f.attrs["noise_dt"] = self.dt
        f.attrs["noise_t_obs_start_SI"] = self.t_obs_start_SI

    @classmethod
    def from_h5(cls, f):
        if "noise_window" not in f:
            return None
        return cls(f["noise_sigma_full"][()], f["noise_window"][()], f["noise_keep_idx"][()], f.attrs["noise_dt"],
                   f.attrs["noise_t_obs_start_SI"])

