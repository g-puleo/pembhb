"""Relative binning (heterodyne) of FD data on bins of equal drift of h / h_fid.

Per channel, with w_k = 4 df_k / S_k (effective PSD), c_k = conj(h_fid,k) w_k and
B_b = sum_{k in b} |h_fid,k|^2 w_k:

    r_bar_b(d) = sum_{k in b} c_k d_k / B_b

r_bar_b(h_fid) = 1; diagonal noise (Re, Im each of variance S / (4 df)) gives
Var(Re r_bar_b) = Var(Im r_bar_b) = 1 / B_b.
The binned grid is published as centroid freqs, widths df_b and the effective ASD
sqrt(4 df_b / B_b), so noise_scale = asd / sqrt(4 df) = 1 / sqrt(B_b).
"""
import numpy as np
import torch

from pembhb.tapered_noise import _REGISTRY, TaperedNoise

_H5_KEYS = ("het_edges", "het_c", "het_B", "het_freqs", "het_df", "het_fine_freqs")


def support_start(h_fid):
    """First fine bin from which h_fid is nonzero in every channel."""
    support = np.all(np.abs(h_fid) > 0, axis=0)
    i0 = int(np.argmax(support))
    if not support[i0:].all():
        raise ValueError("h_fid has zeros above its low-f edge")
    return i0


def drift(h_of, theta_fid, box, h_fid, i0):
    """sum_p max_{2 box edges, channels} |ln(r_{k+1} / r_k)|, r = h(theta_fid with p at an edge) / h_fid.

    Summing over parameters bounds the drift at the box corners if r is linear in theta.
    :return: (F - i0 - 1,) drift between neighbouring fine bins from i0 on
    """
    d = np.zeros(h_fid.shape[1] - i0 - 1)
    for p, edges in box.items():
        steps = []
        for e in edges:
            r = h_of(dict(theta_fid, **{p: float(e)}))[:, i0:] / h_fid[:, i0:]
            steps.append(np.abs(np.log(r[:, 1:] / r[:, :-1])).max(axis=0))
        d += np.max(steps, axis=0)
    return d


def bins_from_drift(dlnr, n_bins):
    """Bin edges (relative to i0, last = len(dlnr) + 1, exclusive) with equal summed drift eps."""
    if n_bins > len(dlnr):
        raise ValueError(f"{n_bins} bins from {len(dlnr) + 1} fine bins")
    D = np.concatenate([[0.0], np.cumsum(dlnr)])
    eps_lo, eps_hi = 0.0, D[-1]
    for _ in range(60):  # several edges can land in one step: bisect eps for exactly n_bins bins
        eps = 0.5 * (eps_lo + eps_hi)
        edges = np.unique(np.searchsorted(D, eps * np.arange(1, int(D[-1] / eps) + 1), side="left"))
        edges = np.unique(np.concatenate([[0], edges[edges < D.size - 1], [D.size - 1]]))
        if len(edges) - 1 > n_bins:
            eps_lo = eps
        elif len(edges) - 1 < n_bins:
            eps_hi = eps
        else:
            break
    if len(edges) - 1 != n_bins:
        raise RuntimeError(f"bisection gave {len(edges) - 1} bins, not {n_bins}")
    edges[-1] = D.size  # the last bin includes the last fine bin
    return edges, eps


def bin_sum(x, edges):
    """Sums of x[..., edges[j]:edges[j+1]] (numpy)."""
    return np.add.reduceat(x[..., edges[0]:edges[-1]], edges[:-1] - edges[0], axis=-1)


class Heterodyne:
    def __init__(self, edges, c, B, freqs, df, fine_freqs, eps=float("nan")):
        """
        :param edges: (N + 1,) indices into the fine kept grid, edges[-1] exclusive
        :param c: (C, edges[-1] - edges[0]) conj(h_fid) w on the binned span
        :param B: (C, N) fiducial SNR^2 per bin
        :param freqs: (N,) bin centroids (fiducial SNR^2-weighted, summed over channels)
        :param df: (N,) bin widths
        :param fine_freqs: (F,) the fine kept grid the edges index
        """
        self.edges = np.asarray(edges, dtype=np.int64)
        self.c = np.asarray(c, dtype=np.complex128)
        self.B = np.asarray(B, dtype=np.float64)
        self.freqs = np.asarray(freqs, dtype=np.float64)
        self.df = np.asarray(df, dtype=np.float64)
        self.fine_freqs = np.asarray(fine_freqs, dtype=np.float64)
        self.eps = float(eps)
        self.n_bins = self.B.shape[-1]
        self._cache = {}

    @classmethod
    def from_fiducial(cls, edges, h_fid, w, fine_freqs, fine_df, eps=float("nan")):
        """Bins ``edges`` of the fine grid around h_fid (C, F); w (C, F) = 4 df / S_eff."""
        sl = slice(edges[0], edges[-1])
        p = np.abs(h_fid) ** 2 * w
        B = bin_sum(p, edges)
        freqs = bin_sum(p.sum(0) * fine_freqs, edges) / B.sum(0)
        return cls(edges, (h_fid.conj() * w)[:, sl], B, freqs, bin_sum(fine_df, edges), fine_freqs, eps)

    @property
    def asd(self):
        """Effective ASD of the binned grid, so asd / sqrt(4 df) = 1 / sqrt(B)."""
        return np.sqrt(4 * self.df / self.B)

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_cache"] = {}
        return state

    def _on(self, x):
        if isinstance(x, torch.Tensor):
            key = str(x.device)
            if key not in self._cache:
                self._cache[key] = (torch.as_tensor(self.c, device=x.device),
                                    torch.as_tensor(self.edges - self.edges[0], device=x.device),
                                    torch.as_tensor(self.B, device=x.device))
        else:
            key = type(x).__module__.split(".")[0]
            if key not in self._cache:
                xp = np if key == "numpy" else __import__(key)
                self._cache[key] = (xp.asarray(self.c), xp.asarray(self.edges - self.edges[0]),
                                    xp.asarray(self.B))
        return self._cache[key]

    def bin(self, x):
        """(..., C, F) on the fine grid -> (..., C, N) r_bar, same array type, device and dtype as x.

        Cumulative sum in complex128, differenced at the edges (deterministic, no atomics).
        """
        c, idx, B = self._on(x)
        lo, hi = self.edges[0], self.edges[-1]
        if isinstance(x, torch.Tensor):
            y = torch.cumsum(x[..., lo:hi].to(torch.complex128) * c, dim=-1)
            y = torch.cat([torch.zeros_like(y[..., :1]), y], dim=-1)
            return ((y[..., idx[1:]] - y[..., idx[:-1]]) / B).to(x.dtype)
        xp = np if isinstance(x, np.ndarray) else __import__(type(x).__module__.split(".")[0])
        y = xp.cumsum(x[..., lo:hi].astype(np.complex128) * c, axis=-1)
        y = xp.concatenate([xp.zeros_like(y[..., :1]), y], axis=-1)
        return ((y[..., idx[1:]] - y[..., idx[:-1]]) / B).astype(x.dtype)

    def _arrays(self):
        return dict(zip(_H5_KEYS, (self.edges, self.c, self.B, self.freqs, self.df, self.fine_freqs)))

    def save(self, path):
        np.savez(path, eps=self.eps, **self._arrays())

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(*(z[k] for k in _H5_KEYS), eps=float(z["eps"]))

    def to_h5(self, f):
        for k, v in self._arrays().items():
            f.create_dataset(k, data=v)
        f.attrs["het_eps"] = self.eps

    @classmethod
    def from_h5(cls, f):
        if "het_edges" not in f:
            return None
        return cls(*(f[k][()] for k in _H5_KEYS), eps=float(f.attrs["het_eps"]))


class HeterodynedNoise:
    """Tapered noise drawn on the fine grid, then relative-binned. Same interface as TaperedNoise."""

    def __init__(self, fine, het, chunk=16):
        self.fine, self.het, self.chunk = fine, het, int(chunk)
        self.key = id(self)
        _REGISTRY[self.key] = self

    def draw(self, B, device="cpu", dtype=torch.float32, generator=None):
        """(B, C, N) binned noise; fine draws of ``chunk`` samples at a time bound the memory."""
        return torch.cat([self.het.bin(self.fine.draw(min(self.chunk, B - i), device, dtype, generator))
                          for i in range(0, B, self.chunk)])

    def to_h5(self, f):
        self.fine.to_h5(f)
        self.het.to_h5(f)

    @classmethod
    def from_h5(cls, f):
        fine, het = TaperedNoise.from_h5(f), Heterodyne.from_h5(f)
        return None if fine is None or het is None else cls(fine, het)


def load_noise_model(f):
    """HeterodynedNoise, TaperedNoise or None, from what the HDF5 file stores."""
    return HeterodynedNoise.from_h5(f) or TaperedNoise.from_h5(f)
