"""CPU smoke tests for MBHBSimulatorFD on the small grid in test/configs/datagen_test.yaml."""
import copy
import os

import h5py
import numpy as np
import pytest
import yaml

from pembhb import ROOT_DIR
from pembhb.data import MBHBDataset
from pembhb.simulator import MBHBSimulatorFD
from pembhb.utils import read_config

CONF_PATH = os.path.join(ROOT_DIR, "test", "configs", "datagen_test.yaml")


def _build(conf, prior=None, seed=0):
    wp = conf["waveform_params"]
    return MBHBSimulatorFD(
        conf,
        sampler_init_kwargs={"prior_bounds": prior or conf["prior"],
                             "spin_param_basis": conf["spin_param_basis"]},
        seed=seed,
        n_freq_bins=wp["n_freq_bins"],
        freq_spacing=wp["freq_spacing"],
    )


@pytest.fixture(scope="module")
def conf():
    return read_config(CONF_PATH)


@pytest.fixture(scope="module")
def sim(conf):
    return _build(conf)


def test_frequency_grid(sim, conf):
    wp = conf["waveform_params"]
    f = sim.freqs
    assert f[0] >= wp["fmin"] and f[-1] < wp["fmax"]
    assert np.all(np.diff(f) > 0)
    assert sim.df.shape == f.shape and np.all(sim.df > 0)
    assert sim.asd.shape == (len(wp["channels"]), len(f))
    assert np.all(np.isfinite(sim.asd)) and np.all(sim.asd > 0)


def test_sample_shapes(sim):
    n = 3
    out = sim.sample(n)
    assert out["parameters"].shape == (11, n)
    assert out["wave_fd"].shape == (n, sim.n_channels, sim.n_freq_bins)
    assert np.iscomplexobj(out["wave_fd"])
    assert np.all(np.isfinite(out["wave_fd"]))
    assert np.all(np.abs(out["wave_fd"]).max(axis=(1, 2)) > 0)


def test_injection_is_pinned(conf):
    inj = conf["injection"]
    prior = {k: [float(inj[k]), float(inj[k])] for k in conf["prior"]}
    out = _build(conf, prior=prior).sample(2)
    expected = np.array([inj[k] for k in conf["prior"]], dtype=float)
    np.testing.assert_allclose(out["parameters"][:, 0], expected, rtol=1e-6)
    np.testing.assert_allclose(out["wave_fd"][0], out["wave_fd"][1])


def test_fmax_required(conf):
    c = copy.deepcopy(conf)
    del c["waveform_params"]["fmax"]
    with pytest.raises(KeyError):
        _build(c)


def test_fmax_above_veto_grid_rejected(conf):
    c = copy.deepcopy(conf)
    c["waveform_params"]["fmax"] = 0.2
    with pytest.raises(ValueError):
        _build(c)


def test_downsamplefactor_rejected(conf):
    c = copy.deepcopy(conf)
    c["waveform_params"]["downsamplefactor"] = 3
    with pytest.raises(NotImplementedError):
        _build(c)


def test_sample_and_store_with_noise(sim, tmp_path):
    fname = str(tmp_path / "obs.h5")
    sim.sample_and_store(fname, N=2, batch_size=2, store_noise=True, noise_seed=1)
    with h5py.File(fname, "r") as f:
        for k in ("source_parameters", "frequencies", "df", "wave_fd", "noise_fd", "asd", "snr"):
            assert k in f, k
        assert f["noise_fd"].shape == f["wave_fd"].shape
        noise = f["noise_fd"][()]
    assert os.path.exists(fname[:-3] + ".yaml")

    # Stored noise is coloured by ASD / sqrt(4 df): whitened |z|^2 has mean 2.
    scale = sim.filtered_asd / np.sqrt(4 * sim.df)
    white = noise / scale[None]
    assert abs(np.mean(np.abs(white) ** 2) - 2.0) < 0.1


def test_training_noise_scale_matches_stored(sim, tmp_path):
    # On-the-fly training noise must use the per-bin df, as the stored noise does.
    fname = str(tmp_path / "train.h5")
    sim.sample_and_store(fname, N=2, batch_size=2)
    ds = MBHBDataset(fname)
    np.testing.assert_allclose(ds.noise_scale.numpy(),
                               sim.filtered_asd / np.sqrt(4 * sim.df), rtol=1e-5)


# --- td block: DFT grid, round trip, tapered noise ------------------------------------

def _td(conf, t_obs_start=0.0, **td):
    c = copy.deepcopy(conf)
    wp = c["waveform_params"]
    del wp["fmax"]
    wp["t_obs_start"] = t_obs_start
    wp["td"] = dict(dict(dt=100.0, round_trip=False, bbhx_length=1024), **td)
    return c


def _pinned(conf):
    inj = conf["injection"]
    return {k: [float(inj[k]), float(inj[k])] for k in conf["prior"]}


def test_td_grid(conf):
    s = _build(_td(conf))
    f = np.fft.rfftfreq(s.n_time, s.dt)
    np.testing.assert_array_equal(s.freqs, f[s.keep_idx])
    assert s.fmax == 0.5 / 100.0 and s.freqs[-1] < s.fmax and s.freqs[0] >= conf["waveform_params"]["fmin"]
    np.testing.assert_allclose(s.df, 1 / s.obs_length)


def test_td_rejects_fmax_and_taper_without_round_trip(conf):
    c = _td(conf)
    c["waveform_params"]["fmax"] = 3e-3
    with pytest.raises(ValueError):
        _build(c)
    with pytest.raises(ValueError):
        _build(_td(conf, taper_days=1.0))
    with pytest.raises(ValueError):           # generation window before t = 0
        _build(_td(conf, round_trip=True, lookback_days=7, taper_days=0))


def test_round_trip_identity(conf):
    # lookback 0, no taper: irfft -> rfft on the same grid is the identity
    plain = _build(_td(conf), prior=_pinned(conf)).sample(1)["wave_fd"]
    rt = _build(_td(conf, round_trip=True, lookback_days=0, taper_days=0), prior=_pinned(conf)).sample(1)["wave_fd"]
    assert np.max(np.abs(rt - plain)) < 1e-6 * np.max(np.abs(plain))


def test_tapered_noise_statistics():
    from pembhb.tapered_noise import TaperedNoise, tukey
    import torch
    n, ramp = 512, 64
    w = tukey(n, ramp)
    keep = np.arange(20, 200)
    tn = TaperedNoise(np.ones((1, n // 2 + 1)), w, keep, dt=1.0)
    z = tn.draw(4000, dtype=torch.float64, generator=torch.Generator().manual_seed(0)).numpy()[:, 0]
    W2 = np.fft.fft(w ** 2)
    assert abs(np.mean(z.real ** 2) - np.mean(w ** 2)) < 0.03
    adj = np.mean(z.real[:, :-1] * z.real[:, 1:] + z.imag[:, :-1] * z.imag[:, 1:]) / 2 / np.mean(w ** 2)
    assert abs(adj - (W2[1] / W2[0]).real) < 0.02
    z2 = tn.draw(4000, dtype=torch.float64, generator=torch.Generator().manual_seed(0)).numpy()[:, 0]
    np.testing.assert_array_equal(z, z2)


def test_round_trip_taper_store_and_collate(conf, tmp_path):
    import torch
    from pembhb.utils import mbhb_collate_fn, materialize_gpu_noise
    s = _build(_td(conf, t_obs_start=7.0, round_trip=True, lookback_days=7, taper_days=0.5))
    assert s.noise_model is not None and 0.5 < s.taper_w2 < 1
    fname = str(tmp_path / "rt.h5")
    s.sample_and_store(fname, N=2, batch_size=2, store_noise=True, noise_seed=1)
    ds = MBHBDataset(fname)
    assert ds.noise_model is not None
    np.testing.assert_allclose(ds.noise_scale.numpy(), s.asd / np.sqrt(4 * s.df), rtol=1e-5)
    items = [{"wave_fd": torch.zeros(s.n_channels, s.n_freq_bins, dtype=torch.complex64),
              "params": torch.zeros(11)} for _ in range(64)]
    ns = ds.noise_scale.float()
    for gpu_noise in (False, True):
        b = materialize_gpu_noise(mbhb_collate_fn(items, ns, 1.0, gpu_noise=gpu_noise, noise_model=ds.noise_model))
        white = b["noise_fd"] / ns
        assert white.shape == (64, s.n_channels, s.n_freq_bins)
        assert abs(white.real.pow(2).mean().item() - 1) < 0.1     # unit variance after the effective ASD
