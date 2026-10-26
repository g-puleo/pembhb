import h5py
import numpy as np
import pytest
import torch

from pembhb.heterodyne import Heterodyne, HeterodynedNoise, bins_from_drift, load_noise_model
from pembhb.tapered_noise import TaperedNoise, lookup, tukey

N_T = 256
N_F = N_T // 2 + 1


def _setup(n_bins=12, seed=0):
    rng = np.random.default_rng(seed)
    keep = np.arange(5, N_F - 1)
    f = keep / N_T
    sigma = np.zeros((2, N_F))
    sigma[:, 1:] = 1 + rng.random((2, N_F - 1))
    h = (1 + rng.random((2, keep.size))) * np.exp(2j * np.pi * rng.random((2, keep.size)))
    i0 = 3
    edges, eps = bins_from_drift(rng.random(keep.size - i0 - 1), n_bins)
    het = Heterodyne.from_fiducial(edges + i0, h, 1 / sigma[:, keep] ** 2, f, np.full(f.size, 1 / N_T), eps)
    return het, h, sigma, keep


def test_bins_from_drift_count_and_span():
    rng = np.random.default_rng(1)
    d = rng.random(500)
    edges, eps = bins_from_drift(d, 40)
    assert len(edges) == 41 and edges[0] == 0 and edges[-1] == d.size + 1
    assert np.all(np.diff(edges) > 0)


def test_bin_fiducial_is_one_and_linear():
    het, h, _, _ = _setup()
    np.testing.assert_allclose(het.bin(h), 1, atol=1e-12)
    rng = np.random.default_rng(2)
    a, b = (rng.normal(size=(3,) + h.shape) + 1j * rng.normal(size=(3,) + h.shape) for _ in range(2))
    np.testing.assert_allclose(het.bin(a + 2 * b), het.bin(a) + 2 * het.bin(b), atol=1e-12)
    t = het.bin(torch.as_tensor(a))
    assert isinstance(t, torch.Tensor) and t.dtype == torch.complex128
    np.testing.assert_allclose(t.numpy(), het.bin(a), atol=1e-12)
    assert het.bin(a.astype(np.complex64)).dtype == np.complex64
    np.testing.assert_allclose(het.asd / np.sqrt(4 * het.df), 1 / np.sqrt(het.B))


def test_npz_and_h5_round_trip(tmp_path):
    het, h, _, _ = _setup()
    het.save(tmp_path / "het.npz")
    with h5py.File(tmp_path / "het.h5", "w") as f:
        het.to_h5(f)
    with h5py.File(tmp_path / "het.h5") as f:
        loaded = [Heterodyne.load(tmp_path / "het.npz"), Heterodyne.from_h5(f)]
    for other in loaded:
        np.testing.assert_array_equal(other.edges, het.edges)
        np.testing.assert_allclose(other.bin(h), het.bin(h))
        assert other.eps == het.eps


def test_binned_noise_variance_without_taper():
    # box window: fine noise is diagonal with Var(Re n_k) = sigma_k^2 and w = 1 / sigma^2, so Var(Re r_bar) B = 1
    het, _, sigma, keep = _setup()
    fine = TaperedNoise(sigma, tukey(N_T, 0), keep, dt=1.0)
    noise = HeterodynedNoise(fine, het, chunk=7)
    n = noise.draw(20000, dtype=torch.float64, generator=torch.Generator().manual_seed(0)).numpy()
    assert n.shape == (20000, 2, het.n_bins)
    z = n * np.sqrt(het.B)
    np.testing.assert_allclose(z.real.var(0), 1, atol=0.06)
    np.testing.assert_allclose(z.imag.var(0), 1, atol=0.06)
    assert lookup(noise.key) is noise


def test_load_noise_model_dispatch(tmp_path):
    het, _, sigma, keep = _setup()
    fine = TaperedNoise(sigma, tukey(N_T, 16), keep, dt=1.0)
    with h5py.File(tmp_path / "a.h5", "w") as f:
        HeterodynedNoise(fine, het).to_h5(f)
    with h5py.File(tmp_path / "b.h5", "w") as f:
        fine.to_h5(f)
    with h5py.File(tmp_path / "c.h5", "w") as f:
        f.create_dataset("asd", data=np.ones(3))
    with h5py.File(tmp_path / "a.h5") as f:
        assert isinstance(load_noise_model(f), HeterodynedNoise)
    with h5py.File(tmp_path / "b.h5") as f:
        assert isinstance(load_noise_model(f), TaperedNoise)
    with h5py.File(tmp_path / "c.h5") as f:
        assert load_noise_model(f) is None


def test_too_many_bins_is_loud():
    with pytest.raises(ValueError):
        bins_from_drift(np.ones(5), 10)
