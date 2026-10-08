import copy
import os

import h5py
import numpy as np
import yaml
from tqdm import tqdm

from bbhx.waveformbuild import BBHWaveformFD
from bbhx.utils.constants import MTSUN_SI, YRSID_SI
import lisatools.sensitivity as lisasens
from lisatools.detector import EqualArmlengthOrbits
from lisatools.sensitivity import get_sensitivity

from pembhb import ROOT_DIR, FMIN_FLOOR
from pembhb import get_numpy_dtype, get_numpy_complex_dtype
from pembhb.sampler import UniformSampler
from pembhb.psd_veto import bin_widths, bands_to_mask, reference_veto_bands, REF_FMAX


WEEK_SI = 7 * 24 * 3600
DAY_SI = 24 * 3600


# Ajith+ 2008 (arXiv:0710.2335) phenomenological IMR coefficients for the
# non-spinning amplitude transition frequencies (Eqs. 4.18--4.19).
# Each frequency is f = (a*eta^2 + b*eta + c) / (pi * G M / c^3).
_AJITH_COEFFS = {
    "f_merger": (2.9740e-1, 4.4810e-2, 9.5560e-2),
    "f_ring":   (5.9411e-1, 8.9794e-2, 1.9111e-1),
    "sigma":    (5.0801e-1, 7.7515e-2, 2.2369e-2),
    "f_cut":    (8.4845e-1, 1.2848e-1, 2.7299e-1),
}


def ajith_transition_frequencies(m1_msun, m2_msun):
    """Return f_merger, f_ring, sigma, f_cut [Hz] for total mass M = m1+m2 [Msun]
    and symmetric mass ratio eta = m1*m2 / M^2. Vectorised over the inputs."""
    m1 = np.asarray(m1_msun)
    m2 = np.asarray(m2_msun)
    M = m1 + m2
    eta = (m1 * m2) / (M * M)
    denom = np.pi * MTSUN_SI * M  # = pi * G * M / c^3, in seconds
    out = {}
    for name, (a, b, c) in _AJITH_COEFFS.items():
        out[name] = (a * eta**2 + b * eta + c) / denom
    return out

# =========================================================================
# Shared helper functions
# =========================================================================

_SENS_MAP = {
    "A": lisasens.A2TDISens,
    "E": lisasens.E2TDISens,
    "T": lisasens.T2TDISens,
}


def build_asd(freqs, channels, noise_model):
    """Build ASD array for given frequency grid and TDI channels."""
    asd = np.zeros((len(channels), len(freqs)))
    psd_kwargs = {"model": noise_model, "return_type": "ASD"}
    for i, ch in enumerate(channels):
        asd[i] = get_sensitivity(freqs, sens_fn=_SENS_MAP[ch], **psd_kwargs)
    return asd


def generate_noise_fd(rng, asd, df, n_obs):
    """Generate coloured Gaussian noise in FD with per-bin df.

    :param rng: NumPy random generator
    :param asd: ASD array, shape (n_channels, n_freqs)
    :param df: frequency bin widths — scalar or array of shape (n_freqs,)
    :param n_obs: number of observations (batch size)
    :return: complex noise array, shape (n_obs, n_channels, n_freqs)
    """
    n_channels, n_freqs = asd.shape
    z = (rng.normal(size=(n_obs, n_channels, n_freqs))
         + 1j * rng.normal(size=(n_obs, n_channels, n_freqs)))
    # df can be scalar (uniform grid) or 1-D array (non-uniform grid)
    return z * (asd / np.sqrt(4 * df))[None, :, :]


def compute_snr_fd(signal, freqs, asd, df, fmin_highpass=FMIN_FLOOR):
    """Compute FD SNR with per-bin df.

    :param signal: complex FD data, shape (n_obs, n_channels, n_freqs)
    :param freqs: frequency grid, shape (n_freqs,)
    :param asd: ASD array, shape (n_channels, n_freqs)
    :param df: bin widths — scalar or array of shape (n_freqs,)
    :param fmin_highpass: high-pass cutoff frequency
    :return: SNR values, shape (n_obs,)
    """
    mask = freqs >= fmin_highpass
    data_over_asd = signal[..., mask] / asd[..., mask]
    prod = data_over_asd * data_over_asd.conj()
    # df may be scalar or 1-D; broadcast over (n_obs, n_channels)
    df_masked = df[mask] if np.ndim(df) > 0 else df
    weighted = prod * df_masked
    SNR2 = 4.0 * np.sum(weighted, axis=(1, 2)).real
    return np.sqrt(SNR2)


def setup_bbhx(backend):
    """Initialise BBHWaveformFD and orbits for a given backend."""
    orbits = EqualArmlengthOrbits(force_backend=backend)
    orbits.configure(linear_interp_setup=True)
    resp_kwargs = {
        "TDItag": "AET",
        "rescaled": False,
        "orbits": orbits,
        "tdi2": True
    }
    wfd = BBHWaveformFD(
        amp_phase_kwargs=dict(run_phenomd=False),
        response_kwargs=resp_kwargs,
        force_backend=backend,
    )
    return wfd


# =========================================================================
# FD-only simulator with custom frequency grids
# =========================================================================

class MBHBSimulatorFD:
    """Frequency-domain-only MBHB simulator with linear or log frequency grids.

    Supports uniform (linear) or logarithmic frequency spacing; no IFFT is
    ever computed.
    """

    def __init__(self, conf, sampler_init_kwargs, seed=0,
                 n_freq_bins=4096, freq_spacing="linear", sampler=None):
        """
        :param conf: datagen config dict (see configs/datagen_config.yaml)
        :param sampler_init_kwargs: dict with 'prior_bounds' key
        :param seed: RNG seed
        :param n_freq_bins: number of frequency bins (default 4096)
        :param freq_spacing: 'linear' or 'log'
        :param sampler: optional pre-built sampler (overrides sampler_init_kwargs)
        """
        self.rng = np.random.default_rng(seed)
        self.sampler = sampler if sampler is not None else UniformSampler(**sampler_init_kwargs, rng=self.rng)
        self.backend_name = conf.get("backend", "cpu")

        self.channels = conf["waveform_params"]["channels"]
        self.channel_map = {ch: i for i, ch in enumerate(self.channels)}
        self.channels_idx = [self.channel_map[ch] for ch in self.channels]
        self.n_channels = len(self.channels)
        self.modes = conf["waveform_params"]["modes"]

        self.t_obs_start_SI = 0
        self.t_obs_end_SI = conf["waveform_params"]["duration"] * WEEK_SI
        self.obs_length = self.t_obs_end_SI - self.t_obs_start_SI

        # Frequency grid — free from FFT constraints. The requested fmin
        # (from waveform_params.fmin, defaulting to FMIN_FLOOR) competes with
        # 1/T_obs; whichever is *larger* wins, so the grid never extends below
        # max(requested_fmin, 1/T_obs) and there are no PSD-masked dead bins
        # to worry about downstream.
        self.fmax = conf["waveform_params"]["fmax"]
        if self.fmax > REF_FMAX:
            raise ValueError(
                f"fmax={self.fmax:g} exceeds the PSD-veto reference grid "
                f"({REF_FMAX:g} Hz); TDI nulls above it would go undetected."
            )
        fmin_request = conf["waveform_params"].get("fmin", FMIN_FLOOR)
        self.fmin = max(fmin_request, 1.0 / self.obs_length)
        self.freq_spacing = freq_spacing

        if freq_spacing == "linear":
            # df = 1/T_obs is the natural FFT spacing; downsamplefactor lets
            # the user thin the grid when df would otherwise produce too many
            # bins. n_freq_bins becomes a consequence of the grid, not an input.
            downsamplefactor = conf["waveform_params"].get("downsamplefactor", 1)
            if downsamplefactor != 1:
                raise NotImplementedError(
                    f"downsamplefactor={downsamplefactor}: only 1 is supported "
                    "(coarser grids mis-scale the noise by sqrt(downsamplefactor))."
                )
            df = 1.0 / self.obs_length
            step = downsamplefactor * df
            self.freqs = np.arange(self.fmin, self.fmax, step)
            self.n_freq_bins = len(self.freqs)
        elif freq_spacing == "log":
            self.n_freq_bins = n_freq_bins
            self.freqs = np.logspace(
                np.log10(self.fmin), np.log10(self.fmax), n_freq_bins
            )
        else:
            raise ValueError(f"freq_spacing must be 'linear' or 'log', got '{freq_spacing}'")

        # Drop bins inside the TDI nulls: the PSD collapses there but the
        # splined response does not, so 1/S diverges. Deleting them leaves a
        # gap that bin_widths bridges, so no interpolation is needed.
        self.psd_veto_bands = reference_veto_bands(
            self.channels, conf["waveform_params"]["noise"]
        )
        vetoed = bands_to_mask(self.freqs, self.psd_veto_bands)
        if vetoed.any():
            self.freqs = self.freqs[~vetoed]
            self.n_freq_bins = len(self.freqs)
            print(f"[MBHBSimulatorFD] PSD veto: dropped {vetoed.sum()} bins "
                  f"across {len(self.psd_veto_bands)} TDI-null bands.")

        # Per-bin frequency widths for inner products and noise colouring.
        self.df = bin_widths(self.freqs)

        # ASD. The grid starts at fmin >= FMIN_FLOOR by construction, so no
        # masking is needed; ``filtered_asd`` is kept as an alias for the
        # raw ``asd`` for backward compatibility with downstream consumers.
        noise_model = conf["waveform_params"]["noise"]
        self.asd = build_asd(self.freqs, self.channels, noise_model)
        self.filtered_asd = self.asd.copy()
        self.psd_fmin_mask = conf["waveform_params"].get("psd_fmin_mask", None)
        if self.psd_fmin_mask is not None:
            n_masked = int((self.freqs < self.psd_fmin_mask).sum())
            self.filtered_asd[:, self.freqs < self.psd_fmin_mask] = 0
            print(
                f"[MBHBSimulatorFD] PSD mask active: zeroed ASD in "
                f"{n_masked} bins below {self.psd_fmin_mask:.3e} Hz "
                f"(out of {len(self.freqs)} total)."
            )

        # BBHx waveform generator
        self.wfd = setup_bbhx(self.backend_name)
        self.xp = self.wfd.xp

        t0 = self.t_obs_start_SI / YRSID_SI
        t1 = self.t_obs_end_SI / YRSID_SI
        freqs_backend = self.xp.asarray(self.freqs)
        self.waveform_kwargs = {
            "t_obs_start": t0,
            "t_obs_end": t1,
            "freqs": freqs_backend,
            "modes": self.modes,
            "direct": False,
            "fill": True,
            "compress": True,
            "squeeze": False,
            "length": 1024,
        }

        self.info = {
            "backend": self.backend_name,
            "seed": seed,
            "conf": conf,
            "sampler_init_kwargs": sampler_init_kwargs,
            "channels": list(self.channels),
            "n_channels": self.n_channels,
            "freq_spacing": freq_spacing,
            "n_freq_bins": self.n_freq_bins,
            "fmin": float(self.fmin),
            "fmax": float(self.fmax),
        }
        

    # -----------------------------------------
    def generate(self, inj, keep_on_gpu=False, host_out=None):
        """Generate FD waveform for a batch of injections.

        :param inj: injection parameters, shape (n_params, n_obs)
        :param keep_on_gpu: if True, skip the device->host copy and return a
            torch CUDA tensor (zero-copy from the cupy result via dlpack).
            Requires a CUDA backend. Used by the streaming producer.
        :param host_out: optional pinned host array of shape
            ``(>= n_obs, n_channels, n_freq_bins)`` in the complex dtype. On a CUDA
            backend the cast and channel slice run on the GPU and the result is
            copied straight into ``host_out[:n_obs]``, which is returned (a view,
            overwritten by the next call). Avoids pageable host allocations,
            which dominate generation time when host memory is under pressure.
        :return: wave_fd — shape (n_obs, n_channels, n_freq_bins); numpy array
            (``keep_on_gpu=False``) or torch CUDA tensor (``keep_on_gpu=True``).
        """
        inj = inj.copy()
        n_obs = inj.shape[1]

        wave = self.wfd(*inj, **self.waveform_kwargs)

        if host_out is not None and hasattr(wave, "get"):
            dev = wave[:, self.channels_idx, :].astype(get_numpy_complex_dtype())
            out = host_out[:n_obs]
            dev.get(out=out)
            return out

        if keep_on_gpu:
            if not hasattr(wave, "get"):
                raise RuntimeError(
                    "keep_on_gpu=True requires a CUDA backend, but waveform "
                    f"generator on backend '{self.backend_name}' returned a host array."
                )
            import torch
            from pembhb import get_torch_complex_dtype
            wave = torch.from_dlpack(wave)  # zero-copy cupy -> torch (stays on GPU)
            wave = wave.to(get_torch_complex_dtype())
            return wave[:, self.channels_idx, :]

        if hasattr(wave, "get"):
            wave = wave.get()
        wave = wave.astype(get_numpy_complex_dtype())
        wave = wave[:, self.channels_idx, :]

        return wave

    # -----------------------------------------
    def sample(self, N, keep_on_gpu=False, host_out=None):
        """Draw N samples from the prior and simulate FD data.

        :param N: number of samples
        :param keep_on_gpu: forwarded to :meth:`generate`; when True the
            returned ``wave_fd`` is a torch CUDA tensor.
        :param host_out: forwarded to :meth:`generate` (pinned output buffer).
        :return: dict with keys 'parameters', 'bbhx_parameters', 'wave_fd'
        """
        z, inj = self.sampler.sample(N, self.t_obs_end_SI)
        wave_fd = self.generate(z, keep_on_gpu=keep_on_gpu, host_out=host_out)
        return {
            "parameters": inj,
            "bbhx_parameters": z,
            "wave_fd": wave_fd,
        }

    # -----------------------------------------
    def get_SNR_FD(self, signal):
        return compute_snr_fd(signal, self.freqs, self.asd, self.df)

    # -----------------------------------------
    def sample_and_store(self, filename: str, N: int, batch_size=None,
                         store_noise: bool = False, noise_seed: int = 0):
        """Sample N waveforms and store FD-only data to HDF5.

        :param filename: output HDF5 path
        :param N: total number of samples
        :param batch_size: samples per batch (default N/10)
        :param store_noise: if True, also draw and persist a fixed noise
            realisation per sample under the ``noise_fd`` HDF5 dataset.  The
            draw uses an independent RNG seeded by ``noise_seed`` so signal
            and noise sampling are decoupled.
        :param noise_seed: seed for the noise RNG when ``store_noise=True``.
        """
        if batch_size is None:
            batch_size = max(1, int(N / 10.0))

        _np_real = get_numpy_dtype()
        _np_complex = get_numpy_complex_dtype()

        noise_rng = np.random.default_rng(noise_seed) if store_noise else None
        # Use the same colouring formula as mbhb_collate_fn so stored noise
        # is statistically identical to on-the-fly noise.
        noise_scale_np = self.filtered_asd / np.sqrt(4 * self.df)

        with h5py.File(filename, "a") as f:
            source_params = f.create_dataset("source_parameters", shape=(N, 11), dtype=_np_real)
            bbhx_params = f.create_dataset("bbhx_parameters", shape=(N, 12), dtype=_np_real)
            f.create_dataset("frequencies", data=self.freqs, dtype=_np_real)
            f.create_dataset("df", data=self.df, dtype=_np_real)
            wave_fd = f.create_dataset("wave_fd", shape=(N, self.n_channels, self.n_freq_bins), dtype=_np_complex)
            snr = f.create_dataset("snr", shape=(N,), dtype=_np_real)
            f_isco = f.create_dataset("f_ISCO", shape=(N,), dtype=_np_real)
            f_merger_ds = f.create_dataset("f_merger", shape=(N,), dtype=_np_real)
            f_ring_ds = f.create_dataset("f_ring", shape=(N,), dtype=_np_real)
            sigma_ds = f.create_dataset("sigma", shape=(N,), dtype=_np_real)
            f_cut_ds = f.create_dataset("f_cut", shape=(N,), dtype=_np_real)
            f.create_dataset("asd", data=self.asd, dtype=_np_real)
            
            if store_noise:
                noise_fd_ds = f.create_dataset(
                    "noise_fd",
                    shape=(N, self.n_channels, self.n_freq_bins),
                    dtype=_np_complex,
                )
                f.attrs["noise_seed"] = int(noise_seed)

            # Store metadata as HDF5 attributes
            f.attrs["freq_spacing"] = self.freq_spacing
            f.attrs["n_freq_bins"] = self.n_freq_bins
            f.attrs["fmin"] = self.fmin
            f.attrs["fmax"] = self.fmax
            f.attrs["psd_fmin_mask"] = self.psd_fmin_mask if self.psd_fmin_mask is not None else 0.0
            f.attrs["observation_duration_SI"] = self.obs_length
            # Record the spin sampling basis so the meaning of source_parameters
            # slots 2,3 is self-describing (chi1/chi2 vs chi_eff/chi_diff).
            # MaskRejectSampler wraps the real sampler in .base_sampler.
            _spin_sampler = getattr(self.sampler, "base_sampler", self.sampler)
            f.attrs["spin_param_basis"] = getattr(_spin_sampler, "spin_param_basis", "chi1chi2")
            print("Sampling and storing FD-only simulations to", filename)
            for i in tqdm(range(0, N, batch_size)):
                batch_end = min(i + batch_size, N)
                batch_size_actual = batch_end - i
                out = self.sample(batch_size_actual)
                bbhx_params_batch = out["bbhx_parameters"].T
                source_params[i:batch_end] = out["parameters"].T
                bbhx_params[i:batch_end] = bbhx_params_batch
                wave_fd[i:batch_end] = out["wave_fd"]
                # SNR is matched-filter (waveform-only) rather than noisy-data SNR
                snr[i:batch_end] = self.get_SNR_FD(out["wave_fd"])
                M_tot = bbhx_params_batch[:, 0] + bbhx_params_batch[:, 1]
                f_isco[i:batch_end] = (1.0 / np.pi) * np.sqrt(1.0 / 216.0) * 203025.44672808357 / M_tot
                trans = ajith_transition_frequencies(
                    bbhx_params_batch[:, 0], bbhx_params_batch[:, 1]
                )
                f_merger_ds[i:batch_end] = trans["f_merger"]
                f_ring_ds[i:batch_end] = trans["f_ring"]
                sigma_ds[i:batch_end] = trans["sigma"]
                f_cut_ds[i:batch_end] = trans["f_cut"]
                if store_noise:
                    z = (noise_rng.normal(size=(batch_size_actual, self.n_channels, self.n_freq_bins))
                         + 1j * noise_rng.normal(size=(batch_size_actual, self.n_channels, self.n_freq_bins)))
                    noise_fd_ds[i:batch_end] = (z * noise_scale_np[None, :, :]).astype(_np_complex)

            print("HDF5 dataset shapes (current state):")
            for dname in ["source_parameters", "frequencies", "df",
                          "wave_fd", "noise_fd", "snr", "f_ISCO",
                          "f_merger", "f_ring", "sigma", "f_cut", "asd"]:
                if dname in f:
                    ds = f[dname]
                    print(f"  {dname}: shape={tuple(ds.shape)}, dtype={ds.dtype}")

        self.save_info_yaml(filename=filename, overwrite=True)

    # -----------------------------------------
    def save_info_yaml(self, filename: str = None, overwrite: bool = False, indent: int = 2):
        """Save simulation metadata as a YAML sidecar next to ``filename``."""
        if filename is None:
            yamlpath = os.path.join(ROOT_DIR, f"simulator_info_{int(os.times()[4])}.yaml")
        else:
            yamlpath = filename.removesuffix(".h5")
            if not yamlpath.lower().endswith(".yaml"):
                yamlpath = yamlpath + ".yaml"

        if os.path.exists(yamlpath) and not overwrite:
            raise FileExistsError(f"File '{yamlpath}' already exists. Pass overwrite=True to replace it.")

        def _convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer, np.floating, np.bool_)):
                return obj.item()
            if isinstance(obj, complex):
                return {"real": obj.real, "imag": obj.imag}
            if isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_convert(v) for v in obj]
            return obj if isinstance(obj, (str, int, float, bool, type(None))) else str(obj)

        sik = self.info.get("sampler_init_kwargs", {})
        if "prior_bounds" in sik:
            conf_copy = copy.deepcopy(self.info.get("conf", {}))
            conf_copy["prior"] = copy.deepcopy(sik["prior_bounds"])
        else:
            conf_copy = self.info.get("conf", {})

        payload = {
            "conf": _convert(conf_copy),
            "sampler_init_kwargs": _convert(sik),
        }

        with open(yamlpath, "w") as fh:
            yaml.safe_dump(payload, fh, sort_keys=False, default_flow_style=False, indent=indent)

        return yamlpath


class DummySampler:
    def __init__(self, low=0.0, high=1.0):
        self.low = low
        self.high = high

    def sample(self, N):
        """Sample N sets of parameters (mu, sigma) from a uniform prior."""
        mu = np.random.uniform(self.low, self.high, size=(N, 1))
        sigma = np.random.uniform(self.low, self.high, size=(N, 1))
        z_samples = np.hstack((mu, sigma))
        return z_samples, z_samples  # Return z_samples for both parameters and tmnre_input


class DummySimulator:
    def __init__(self, sampler_init_kwargs):
        self.sampler = DummySampler(**sampler_init_kwargs)
        self.n_samples = 10  # Number of data points per line
        self.noise_std = 0.01  # Standard deviation of the fixed noise

    def generate_d_f(self, injection: np.array):
        """Generate data samples from a line with fixed noise.

        :param injection: Parameters (slope, intercept) for the line
        :type injection: np.array
        :return: Simulated data
        :rtype: np.array
        """
        n_examples = injection.shape[0]
        x = np.linspace(0, 1, self.n_samples)
        data_fd = np.zeros((n_examples, self.n_samples))
        for i in range(n_examples):
            slope, intercept = injection[i]
            y = slope * x + intercept
            data_fd[i] = y + np.random.normal(0, self.noise_std, self.n_samples)
        return data_fd

    def _sample(self, N=1):
        """Draw samples from the prior and generate data.

        :param N: Number of samples to generate
        :type N: int
        :return: z_samples, data_fd
        :rtype: dict
        """
        z_samples, tmnre_input = self.sampler.sample(N)
        data_fd = self.generate_d_f(z_samples)
        out_dict = {"parameters": tmnre_input, "data_fd": data_fd}
        return out_dict

    def sample_and_store(self, filename, N, batch_size=1000):
        """Sample N samples and store them in an HDF5 file.

        :param filename: Name of the file to store the samples
        :type filename: str
        :param N: Number of samples to generate
        :type N: int
        :param batch_size: Number of samples to generate in each batch
        :type batch_size: int
        """
        with h5py.File(filename, "a") as f:
            source_params = f.create_dataset("parameters", shape=(N, 2), dtype=np.float32)
            data_fd = f.create_dataset("data_fd", shape=(N, self.n_samples), dtype=np.float32)

            for i in tqdm(range(0, N, batch_size)):
                batch_end = min(i + batch_size, N)
                batch_size_actual = batch_end - i
                out = self._sample(batch_size_actual)
                z_samples = out["parameters"]
                data_fd_batch = out["data_fd"]
                source_params[i:batch_end] = z_samples
                data_fd[i:batch_end] = data_fd_batch
