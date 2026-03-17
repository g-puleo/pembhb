import copy
import os
import sys

import h5py
import numpy as np
import yaml
from scipy.signal.windows import tukey
from tqdm import tqdm

from bbhx.waveformbuild import BBHWaveformFD
from bbhx.utils.constants import MTSUN_SI, PC_SI, YRSID_SI
from bbhx.utils.transform import LISA_to_SSB
import lisatools.sensitivity as lisasens
from lisatools.detector import EqualArmlengthOrbits
from lisatools.sensitivity import get_sensitivity

from pembhb import ROOT_DIR
from pembhb import get_numpy_dtype, get_numpy_complex_dtype
from pembhb.sampler import UniformSampler


WEEK_SI = 7 * 24 * 3600
DAY_SI = 24 * 3600

class MBHBSimulatorFD_TD:

    def __init__(self, conf, sampler_init_kwargs, seed=0):
        self.rng = np.random.default_rng(seed)
        self.sampler = UniformSampler(**sampler_init_kwargs)
        self.backend_name = conf.get("backend", "cpu")

        self.dt = conf["waveform_params"]["dt"] 
        self.channels = conf["waveform_params"]["channels"]
        # maps "AET" to [0,1,2], "TEA" to [2,0,1]: 
        self.channel_map = {ch: i for i, ch in enumerate(self.channels)}
        self.channels_idx = [self.channel_map[ch] for ch in self.channels]
        self.n_channels = len(self.channels)
        self.modes = conf["waveform_params"]["modes"]

        self.t_max = conf["waveform_params"]["t_max"] * 24 * 3600  # user-provided max merger time (convert to seconds)
        self.t_obs_start_SI = 0
        self.t_obs_end_SI = conf["waveform_params"]["duration"] * 7 * 24 * 3600
        self.n_time = int(self.t_obs_end_SI / self.dt)

        # waveform FD grid (TD_wrapper logic)
        n_fft = int(2**np.ceil(np.log2(max(self.n_time, self.t_max/self.dt))))
        self.n_fft = n_fft
        self.df = 1.0 / (n_fft * self.dt)
        self.freqs_pos = np.fft.rfftfreq(n_fft, d=self.dt)[1:]    # positive
        self.n_freqs_pos = len(self.freqs_pos)
        # noise ASD grid
        self.asd = self._build_asd(conf)
        self.filtered_asd = self.asd.copy()
        self.filtered_asd[:, self.freqs_pos < 5e-5] = 0

        self.window = tukey(self.n_time, alpha=0.0005)
        orbits = EqualArmlengthOrbits(force_backend=self.backend_name)
        orbits.configure(linear_interp_setup=True)

        resp_kwargs = {
            "TDItag": "AET",
            "rescaled": False,
            "orbits": orbits
        }

        self.wfd = BBHWaveformFD(
            amp_phase_kwargs=dict(run_phenomd=False),
            response_kwargs=resp_kwargs,
            force_backend=self.backend_name
        )
        self.xp = self.wfd.xp
        self.info = {
            "backend": self.backend_name,
            "seed": seed,
            "conf": conf,
            "sampler_init_kwargs": sampler_init_kwargs,
            "dt": self.dt,
            "channels": list(self.channels),
            "n_channels": len(self.channels),
            "n_time_pt_noise":  self.n_time,
            "df": self.df,
            "f_len": len(self.freqs_pos)
        }
        t0 = self.t_obs_start_SI / YRSID_SI
        t1 = self.t_obs_end_SI / YRSID_SI
        # Convert freqs to appropriate backend (important for GPU)
        freqs_backend = self.xp.asarray(self.freqs_pos)
        self.waveform_kwargs = {
            "t_obs_start": t0,
            "t_obs_end": t1,
            "freqs": freqs_backend,
            "modes": self.modes,
            "direct": False,
            "fill": True,
            "compress": True,
            "squeeze": False,
            "length": 1024
        }
    # -----------------------------------------
    def _build_asd(self, conf):
        asd = np.zeros((len(self.channels), len(self.freqs_pos)))
        psd_kwargs = {"model": conf["waveform_params"]["noise"], "return_type": "ASD"}
        sens_map = {
            "A": lisasens.A1TDISens,
            "E": lisasens.E1TDISens,
            "T": lisasens.T1TDISens,
        }
        for i, ch in enumerate(self.channels):
            asd[i] = get_sensitivity(self.freqs_pos, sens_fn=sens_map[ch], **psd_kwargs)
        return asd

    # -----------------------------------------
    def _noise_pos(self, n_obs):
        z = (self.rng.normal(size=(n_obs, len(self.channels), len(self.freqs_pos)))
             + 1j * self.rng.normal(size=(n_obs, len(self.channels), len(self.freqs_pos))))
        # interpolate ASD onto waveform freq grid
        return z * (self.filtered_asd/ np.sqrt(4 * self.df))[None, :, :]

    # -----------------------------------------
    def _two_sided(self, pos):
        dc = np.zeros(pos.shape[:-1] + (1,), dtype=pos.dtype)
        pos2 = np.concatenate([dc, pos], axis=2)
        neg = np.flip(pos2[..., 1:].conj(), axis=2)
        return np.concatenate([pos2, neg], axis=2)

    # -----------------------------------------
    def _waveform_fd(self, inj):
        # Pass NumPy arrays — BBHx internally handles GPU conversion.
        # Pre-converting to CuPy breaks its isinstance(Tobs, np.ndarray) check.
        return self.wfd(*inj, **self.waveform_kwargs)

    # -----------------------------------------
    def generate(self, inj):
        inj = inj.copy()
        n_obs = inj.shape[1]

        # insert f_ref=0
        #inj = np.insert(inj, 6, np.zeros(n_obs), axis=0)

        wave_pos = self._waveform_fd(inj)
        if hasattr(wave_pos, "get"):
            wave_pos = wave_pos.get()
        wave_pos = wave_pos.astype(np.complex64)
        wave_pos = wave_pos[:, self.channels_idx,:]
        noise_pos = self._noise_pos(n_obs).astype(get_numpy_complex_dtype())

        wave_two = self._two_sided(wave_pos)
        noise_two = self._two_sided(noise_pos)

        wave_td = np.fft.ifft(wave_two, axis=2).real / self.dt
        noise_td = np.fft.ifft(noise_two, axis=2).real / self.dt

        wave_td = wave_td[..., :self.n_time]
        noise_td = noise_td[..., :self.n_time]

        return noise_pos, wave_pos, noise_td, wave_td

    # -----------------------------------------
    def sample(self, N):
        z, inj = self.sampler.sample(N, self.t_obs_end_SI)

        noise_fd, wave_fd, noise_td, wave_td = self.generate(z)
        return {
            "parameters": inj,
            "bbhx_parameters": z,
            "noise_fd": noise_fd,
            "wave_fd": wave_fd,
            "noise_td": noise_td,
            "wave_td": wave_td
        }

    def sample_and_store(self, filename:str, N:int, batch_size=None): 
        """Sample N samples and store them in an HDF5 file.

        :param filename: name of the file to store the samples
        :type filename: str
        :param N: number of samples to generate
        :type N: int
        :param batch_size: number of samples to generate in each batch, defaults to 1000
        :type batch_size: int, optional
        :return: None
        """
        if batch_size is None:
            batch_size = max(1,int(N/10.0))
        
        with h5py.File(filename, "a") as f:
            _np_real = get_numpy_dtype()
            _np_complex = get_numpy_complex_dtype()
            source_params = f.create_dataset("source_parameters", shape=(N, 11), dtype=_np_real)
            bbhx_params = f.create_dataset("bbhx_parameters", shape=(N, 12), dtype=_np_real)
            sample_frequencies = f.create_dataset("frequencies", data=self.freqs_pos, dtype=_np_real)
            sample_times_SI = f.create_dataset("times_SI", data=np.arange(0, self.n_time)*self.dt, dtype=_np_real)
            wave_fd = f.create_dataset("wave_fd", shape=(N, self.n_channels, self.n_freqs_pos), dtype=_np_complex)
            wave_td = f.create_dataset("wave_td", shape=(N, self.n_channels, self.n_time), dtype=_np_real)
            noise_fd = f.create_dataset("noise_fd", shape=(N, self.n_channels, self.n_freqs_pos), dtype=_np_complex)
            noise_td = f.create_dataset("noise_td", shape=(N, self.n_channels, self.n_time), dtype=_np_real)
            snr = f.create_dataset("snr", shape = (N,), dtype=_np_real)
            asd_dataset = f.create_dataset("asd", data=self.asd, dtype=_np_real)
            print("Sampling and storing simulations to ", filename)
            maximum_timedomain = 0
            
            for i in tqdm(range(0, N, batch_size)):
                batch_end = min(i + batch_size, N)
                batch_size_actual = batch_end - i
                out = self.sample(batch_size_actual)
                z_samples = out["parameters"]
                noise_fd_batch = out["noise_fd"]
                wave_fd_batch = out["wave_fd"]
                noise_td_batch = out["noise_td"]
                wave_td_batch = out["wave_td"]
                bbhx_params_batch = out["bbhx_parameters"].T
                current_max_td = np.max( np.abs(wave_td_batch) )
                #for normalisation purpose and easy access during training, store the maximum value across the time domain data
                if current_max_td > maximum_timedomain:
                    maximum_timedomain = current_max_td
                noise_fd_batch = out["noise_fd"]
                snr_batch = self.get_SNR_FD(noise_fd_batch + wave_fd_batch)
                source_params[i:batch_end] = z_samples.T # Reshape to (batch_size, 11) instead of (11, batch_size)
                wave_fd[i:batch_end] = wave_fd_batch
                wave_td[i:batch_end] = wave_td_batch
                noise_fd[i:batch_end] = noise_fd_batch
                noise_td[i:batch_end] = noise_td_batch
                bbhx_params[i:batch_end] = bbhx_params_batch
                snr[i:batch_end] = snr_batch
        # print all shapes
            print("HDF5 dataset shapes (current state):")
            for dname in ["source_parameters", "frequencies", "times_SI",
                        "wave_fd", "wave_td", "noise_fd", "noise_td",
                        "snr", "asd"]:
                if dname in f:
                    ds = f[dname]
                    print(f"  {dname}: shape={tuple(ds.shape)}, dtype={ds.dtype}")
                else:
                    print(f"  {dname}: MISSING")
        self.info["td_max"] = maximum_timedomain
        self.save_info_yaml( filename=filename, overwrite=True )

    def save_info_yaml(self, filename: str = None, overwrite: bool = False, indent: int = 2):
        """
        Save self.info['conf'] and self.info['sampler_init_kwargs'] as a YAML file.
        If filename is None a timestamped file in ROOT_DIR will be created.

        :param filename: target h5 path (used to derive .yaml). If no extension provided it's used directly.
        :param overwrite: allow overwriting existing file
        :param indent: yaml indent level
        :return: path to written yaml file
        """
        if filename is None:
            yamlpath = os.path.join(ROOT_DIR, f"simulator_info_{int(time.time())}.yaml")
        else:
            # derive yaml path from provided filename (strip .h5 if present)
            yamlpath = filename.removesuffix(".h5")
            if not yamlpath.lower().endswith(".yaml"):
                yamlpath = yamlpath + ".yaml"

        if os.path.exists(yamlpath) and not overwrite:
            raise FileExistsError(f"File '{yamlpath}' already exists. Pass overwrite=True to replace it.")

        def _convert(obj):
            # handle common numpy types and containers
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
            # fallback to string for unsupported objects
            return obj if isinstance(obj, (str, int, float, bool, type(None))) else str(obj)

        # Reconcile: ensure conf["prior"] always reflects the actual
        # sampling bounds used to generate the data.  The authoritative
        # source is sampler_init_kwargs["prior_bounds"].
        sik = self.info.get("sampler_init_kwargs", {})
        if "prior_bounds" in sik:
            conf_copy = copy.deepcopy(self.info.get("conf", {}))
            conf_copy["prior"] = copy.deepcopy(sik["prior_bounds"])
        else:
            conf_copy = self.info.get("conf", {})

        # Extract the two fields requested and convert
        payload = {
            "conf": _convert(conf_copy),
            "sampler_init_kwargs": _convert(sik),
            "td_max": _convert(self.info.get("td_max", None))
        }

        # Write YAML
        with open(yamlpath, "w") as fh:
            yaml.safe_dump(payload, fh, sort_keys=False, default_flow_style=False, indent=indent)

        return yamlpath
            
    def get_SNR_FD(self,
        signal
        ):
        """
        Obtain the SNR of a signal in frequency domain.

        :param signal: data in frequency domain, output by bbhx with shape (n_samples, n_channels, n_freqs)
        :type signal: np.array
        :return: SNR values with shape (n_samples,)
        :rtype: np.array
        """
        
        high_pass_idx =  (self.freqs_pos >= 5e-5)
        data_over_asd = signal[..., high_pass_idx] / self.asd[..., high_pass_idx]
        data_over_asd_conj = data_over_asd.conj()
        prod = data_over_asd * data_over_asd_conj
        weighted = prod * self.df 
        summed = np.sum(weighted, axis=(1, 2))
        real_part = summed.real
        SNR2 = real_part * 4.0
        
        return np.sqrt(SNR2)

# =========================================================================
# Shared helper functions (used by MBHBSimulatorFD; MBHBSimulatorFD_TD
# keeps its own inline implementations untouched)
# =========================================================================

_SENS_MAP = {
    "A": lisasens.A1TDISens,
    "E": lisasens.E1TDISens,
    "T": lisasens.T1TDISens,
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


def compute_snr_fd(signal, freqs, asd, df, fmin_highpass=5e-5):
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

    Unlike MBHBSimulatorFD_TD this class never computes an IFFT and supports
    non-uniform (e.g. logarithmic) frequency spacing.
    """

    def __init__(self, conf, sampler_init_kwargs, seed=0,
                 n_freq_bins=4096, freq_spacing="linear"):
        """
        :param conf: datagen config dict (same format as MBHBSimulatorFD_TD)
        :param sampler_init_kwargs: dict with 'prior_bounds' key
        :param seed: RNG seed
        :param n_freq_bins: number of frequency bins (default 4096)
        :param freq_spacing: 'linear' or 'log'
        """
        self.rng = np.random.default_rng(seed)
        self.sampler = UniformSampler(**sampler_init_kwargs)
        self.backend_name = conf.get("backend", "cpu")

        self.channels = conf["waveform_params"]["channels"]
        self.channel_map = {ch: i for i, ch in enumerate(self.channels)}
        self.channels_idx = [self.channel_map[ch] for ch in self.channels]
        self.n_channels = len(self.channels)
        self.modes = conf["waveform_params"]["modes"]

        dt = conf["waveform_params"]["dt"]
        self.t_obs_start_SI = 0
        self.t_obs_end_SI = conf["waveform_params"]["duration"] * WEEK_SI
        obs_length = self.t_obs_end_SI - self.t_obs_start_SI

        # Frequency grid — free from FFT constraints
        self.fmax = 1.0 / (2.0 * dt)
        self.fmin = max(1e-5, 1.0 / obs_length)
        self.n_freq_bins = n_freq_bins
        self.freq_spacing = freq_spacing

        if freq_spacing == "linear":
            self.freqs = np.linspace(self.fmin, self.fmax, n_freq_bins)
        elif freq_spacing == "log":
            self.freqs = np.logspace(
                np.log10(self.fmin), np.log10(self.fmax), n_freq_bins
            )
        else:
            raise ValueError(f"freq_spacing must be 'linear' or 'log', got '{freq_spacing}'")

        # Per-bin frequency widths for inner products and noise colouring.
        # Length n_freq_bins: use midpoint rule so each bin has a width.
        df_diff = np.diff(self.freqs)
        # Assign each bin a width: average of adjacent diffs, endpoints get half-width
        self.df = np.empty(n_freq_bins)
        self.df[0] = df_diff[0]
        self.df[-1] = df_diff[-1]
        self.df[1:-1] = 0.5 * (df_diff[:-1] + df_diff[1:])

        # ASD and filtered ASD
        noise_model = conf["waveform_params"]["noise"]
        self.asd = build_asd(self.freqs, self.channels, noise_model)
        self.filtered_asd = self.asd.copy()
        self.filtered_asd[:, self.freqs < 5e-5] = 0

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
            "n_freq_bins": n_freq_bins,
            "fmin": float(self.fmin),
            "fmax": float(self.fmax),
        }

    # -----------------------------------------
    def generate(self, inj):
        """Generate FD waveform and noise for a batch of injections.

        :param inj: injection parameters, shape (n_params, n_obs)
        :return: (noise_fd, wave_fd) — both shape (n_obs, n_channels, n_freq_bins)
        """
        inj = inj.copy()
        n_obs = inj.shape[1]

        wave = self.wfd(*inj, **self.waveform_kwargs)
        if hasattr(wave, "get"):
            wave = wave.get()
        wave = wave.astype(get_numpy_complex_dtype())
        wave = wave[:, self.channels_idx, :]

        noise = generate_noise_fd(
            self.rng, self.filtered_asd, self.df, n_obs
        ).astype(get_numpy_complex_dtype())

        return noise, wave

    # -----------------------------------------
    def sample(self, N):
        """Draw N samples from the prior and simulate FD data.

        :param N: number of samples
        :return: dict with keys 'parameters', 'bbhx_parameters', 'noise_fd', 'wave_fd'
        """
        z, inj = self.sampler.sample(N, self.t_obs_end_SI)
        noise_fd, wave_fd = self.generate(z)
        return {
            "parameters": inj,
            "bbhx_parameters": z,
            "noise_fd": noise_fd,
            "wave_fd": wave_fd,
        }

    # -----------------------------------------
    def get_SNR_FD(self, signal):
        return compute_snr_fd(signal, self.freqs, self.asd, self.df)

    # -----------------------------------------
    def sample_and_store(self, filename: str, N: int, batch_size=None):
        """Sample N waveforms and store FD-only data to HDF5.

        :param filename: output HDF5 path
        :param N: total number of samples
        :param batch_size: samples per batch (default N/10)
        """
        if batch_size is None:
            batch_size = max(1, int(N / 10.0))

        _np_real = get_numpy_dtype()
        _np_complex = get_numpy_complex_dtype()

        with h5py.File(filename, "a") as f:
            source_params = f.create_dataset("source_parameters", shape=(N, 11), dtype=_np_real)
            bbhx_params = f.create_dataset("bbhx_parameters", shape=(N, 12), dtype=_np_real)
            f.create_dataset("frequencies", data=self.freqs, dtype=_np_real)
            f.create_dataset("df", data=self.df, dtype=_np_real)
            wave_fd = f.create_dataset("wave_fd", shape=(N, self.n_channels, self.n_freq_bins), dtype=_np_complex)
            noise_fd = f.create_dataset("noise_fd", shape=(N, self.n_channels, self.n_freq_bins), dtype=_np_complex)
            snr = f.create_dataset("snr", shape=(N,), dtype=_np_real)
            f.create_dataset("asd", data=self.asd, dtype=_np_real)

            # Store metadata as HDF5 attributes
            f.attrs["freq_spacing"] = self.freq_spacing
            f.attrs["n_freq_bins"] = self.n_freq_bins
            f.attrs["fmin"] = self.fmin
            f.attrs["fmax"] = self.fmax

            print("Sampling and storing FD-only simulations to", filename)
            for i in tqdm(range(0, N, batch_size)):
                batch_end = min(i + batch_size, N)
                batch_size_actual = batch_end - i
                out = self.sample(batch_size_actual)
                source_params[i:batch_end] = out["parameters"].T
                bbhx_params[i:batch_end] = out["bbhx_parameters"].T
                wave_fd[i:batch_end] = out["wave_fd"]
                noise_fd[i:batch_end] = out["noise_fd"]
                snr[i:batch_end] = self.get_SNR_FD(out["noise_fd"] + out["wave_fd"])

            print("HDF5 dataset shapes (current state):")
            for dname in ["source_parameters", "frequencies", "df",
                          "wave_fd", "noise_fd", "snr", "asd"]:
                if dname in f:
                    ds = f[dname]
                    print(f"  {dname}: shape={tuple(ds.shape)}, dtype={ds.dtype}")

        self.save_info_yaml(filename=filename, overwrite=True)

    # -----------------------------------------
    def save_info_yaml(self, filename: str = None, overwrite: bool = False, indent: int = 2):
        """Save simulation metadata as YAML (same format as MBHBSimulatorFD_TD)."""
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




if __name__ == "__main__":
    from pembhb.utils import read_config
    datagen_config_filename = "datagen_config.yaml"
    datagen_config = read_config(os.path.join(ROOT_DIR, datagen_config_filename))
    sampler_init_kwargs={"prior_bounds": datagen_config["prior"]}

    simulator = MBHBSimulatorFD_TD(conf=datagen_config, sampler_init_kwargs=sampler_init_kwargs, seed=42)
    simulator.sample_and_store(filename=os.path.join(ROOT_DIR, "data", "pippo-pertica-palla.h5"), N=1000, batch_size=100)