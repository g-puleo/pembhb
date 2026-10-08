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
