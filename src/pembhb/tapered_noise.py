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


def tukey(n, n_ramp):
    """Cosine ramps of n_ramp samples at both ends; n_ramp = 0 is a box."""
    win = np.ones(n)
    if n_ramp:
        ramp = 0.5 * (1 - np.cos(np.pi * np.arange(n_ramp) / n_ramp))
        win[:n_ramp], win[-n_ramp:] = ramp, ramp[::-1]
    return win


class TaperedNoise:
    def __init__(self, sigma_full, window, keep_idx, dt):
        """
        :param sigma_full: (C, n//2 + 1) per-bin std of Re and Im, ASD / sqrt(4 df), on the full rfft grid (DC = 0)
        :param window: (n,) taper
        :param keep_idx: indices of the output bins in the full rfft grid
        :param dt: sampling interval [s]
        """
        self.sigma_full = torch.as_tensor(np.asarray(sigma_full), dtype=torch.float64)
        self.window = torch.as_tensor(np.asarray(window), dtype=torch.float64)
        self.keep_idx = torch.as_tensor(np.asarray(keep_idx), dtype=torch.long)
        self.dt = float(dt)
        self.n = self.window.numel()
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
            self._cache[key] = (self.sigma_full.to(device, dtype), self.window.to(device, dtype),
                                self.keep_idx.to(device))
        return self._cache[key]

    def draw(self, B, device="cpu", dtype=torch.float32, generator=None):
        """(B, C, n_keep) complex noise, same scaling as ``(re + j im) * noise_scale`` on the circle."""
        sigma, w, keep = self._on(device, dtype)
        shape = (B,) + tuple(sigma.shape)
        re = torch.randn(shape, device=device, dtype=dtype, generator=generator)
        im = torch.randn(shape, device=device, dtype=dtype, generator=generator)
        x = torch.fft.irfft(torch.complex(re, im) * sigma, n=self.n, dim=-1) / self.dt
        return (self.dt * torch.fft.rfft(x * w, dim=-1))[..., keep]

    def to_h5(self, f):
        f.create_dataset("noise_sigma_full", data=self.sigma_full.numpy())
        f.create_dataset("noise_window", data=self.window.numpy())
        f.create_dataset("noise_keep_idx", data=self.keep_idx.numpy())
        f.attrs["noise_dt"] = self.dt

    @classmethod
    def from_h5(cls, f):
        if "noise_window" not in f:
            return None
        return cls(f["noise_sigma_full"][()], f["noise_window"][()], f["noise_keep_idx"][()], f.attrs["noise_dt"])
