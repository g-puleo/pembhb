import numpy as np 
import os
import h5py
import yaml
from scipy.signal.windows import tukey
from tqdm import tqdm
from bbhx.waveformbuild import BBHWaveformFD
from bbhx.utils.transform import LISA_to_SSB
from bbhx.utils.constants import YRSID_SI, PC_SI, MTSUN_SI
import lisatools.sensitivity as lisasens
from lisatools.detector import EqualArmlengthOrbits


from pembhb import ROOT_DIR
from pembhb.sampler import UniformSampler
import sys
gpu_available = False ### True does not work on coglians, needs nvcc
if gpu_available: 
    backend='cuda12x'
else:
    backend='cpu'
WEEK_SI = 7 * 24 * 3600  # seconds in a week
DAY_SI = 24 * 3600  # seconds in a dayYRSID_SI = wfb.YRSID_SI


    
import numpy as np
from typing import Optional
from bbhx.waveformbuild import BBHWaveformFD
import bbhx.waveformbuild as wfb  # for constants and helpers
from lisatools.sensitivity import get_sensitivity

class MBHBSimulatorFD_TD:

    def __init__(self, conf, sampler_init_kwargs, seed=0):
        self.rng = np.random.default_rng(seed)
        self.sampler = UniformSampler(**sampler_init_kwargs)

        self.dt = conf["waveform_params"]["dt"]
        self.channels = conf["waveform_params"]["channels"]
        self.n_channels = len(self.channels)
        self.modes = conf["waveform_params"]["modes"]

        self.t_max = conf["waveform_params"]["t_max"] * 24 * 3600  # user-provided max merger time (convert to seconds)
        self.t_obs_start_SI = 0
        self.t_obs_end_SI = conf["waveform_params"]["duration"] * 7 * 24 * 3600
        self.n_time = int(self.t_obs_end_SI / self.dt)

        # waveform FD grid (TD_wrapper logic)
        n_fft = 2**int(np.ceil(np.log2(self.t_max / self.dt)))
        self.n_fft = n_fft
        self.df = 1.0 / (n_fft * self.dt)
        self.freqs = np.fft.rfftfreq(n_fft, d=self.dt)         # positive + zero

        # noise ASD grid
        self.asd = self._build_asd(conf)
        self.window = tukey(self.n_time, alpha=0.0005)
        orbits = EqualArmlengthOrbits(force_backend=backend)
        orbits.configure(linear_interp_setup=True)

        resp_kwargs = {
            "TDItag": "AET",
            "rescaled": False,
            "orbits": orbits
        }

        self.wfd = BBHWaveformFD(
            amp_phase_kwargs=dict(run_phenomd=False),
            response_kwargs=resp_kwargs,
            force_backend=conf["backend"]
        )
        self.xp = self.wfd.xp
        self.info = {
            "backend": backend,
            "seed": seed,
            "conf": conf,
            "sampler_init_kwargs": sampler_init_kwargs,
            "dt": self.dt,
            "channels": list(self.channels),
            "n_channels": len(self.channels),
            "n_time_pt_noise":  self.n_time,
            "df": self.df,
            "f_len": len(self.freqs)
        }
    # -----------------------------------------
    def _build_asd(self, conf):
        asd = np.zeros((len(self.channels), len(self.freqs)))
        psd_kwargs = {"model": conf["waveform_params"]["noise"], "return_type": "ASD"}
        sens_map = {
            "A": lisasens.A1TDISens,
            "E": lisasens.E1TDISens,
            "T": lisasens.T1TDISens,
        }
        for i, ch in enumerate(self.channels):
            asd[i] = get_sensitivity(self.freqs, sens_fn=sens_map[ch], **psd_kwargs)
        return asd

    # -----------------------------------------
    def _noise_pos(self, n_obs):
        z = (self.rng.normal(size=(n_obs, len(self.channels), len(self.freqs)))
             + 1j * self.rng.normal(size=(n_obs, len(self.channels), len(self.freqs))))
        # interpolate ASD onto waveform freq grid
        filtered_asd = self.asd[:, self.freqs < 5e-5]
        filtered_asd = 0.0

        return z * (self.asd / np.sqrt(4 * self.df))[None, :, :]

    # -----------------------------------------
    def _two_sided(self, pos):
        dc = np.zeros(pos.shape[:-1] + (1,), dtype=pos.dtype)
        pos2 = np.concatenate([dc, pos], axis=2)
        neg = np.flip(pos2[..., 1:].conj(), axis=2)
        return np.concatenate([pos2, neg], axis=2)

    # -----------------------------------------
    def _waveform_fd(self, inj):
        t0 = self.t_obs_start_SI / YRSID_SI
        t1 = self.t_obs_end_SI / YRSID_SI

        return self.wfd(*inj,
                        t_obs_start=t0,
                        t_obs_end=t1,
                        freqs=self.freqs,
                        modes=self.modes,
                        direct=False,
                        fill=True,
                        compress=True,
                        squeeze=False,
                        length=1024)

    # -----------------------------------------
    def generate(self, inj):
        inj = inj.copy()
        n_obs = inj.shape[1]

        # insert f_ref=0
        inj = np.insert(inj, 6, np.zeros(n_obs), axis=0)

        wave_fd = self._waveform_fd(inj).astype(np.complex64)
        wave_pos = wave_fd[..., 1:]          # drop DC to match noise

        noise_pos = self._noise_pos(n_obs).astype(np.complex64)

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
        if os.path.exists(filename):
            if os.path.isdir(filename):
                raise IsADirectoryError(f"'{filename}' is a directory.")
            # Only prompt when running interactively
            if sys.stdin.isatty():
                resp = input(f"File '{filename}' already exists. Delete it and continue? [y/N]: ").strip().lower()
                if resp in ("y", "yes"):
                    os.remove(filename)
                    print(f"Removed existing file '{filename}'.")
                else:
                    raise FileExistsError(f"Aborted: '{filename}' already exists.")
            else:
                raise FileExistsError(
                    f"File '{filename}' exists. Run interactively to confirm deletion or remove the file manually."
                )
        with h5py.File(filename, "a") as f:
            source_params = f.create_dataset("source_parameters", shape=(N, 11), dtype=np.float32)
            sample_frequencies = f.create_dataset("frequencies", data=self.freqs, dtype=np.float32)
            sample_times_SI = f.create_dataset("times_SI", data=np.arange(0, self.n_time)*self.dt, dtype=np.float32)
            wave_fd = f.create_dataset("wave_fd", shape=(N, 2*self.n_channels, self.n_time//2), dtype=np.float32)
            wave_td = f.create_dataset("wave_td", shape=(N, self.n_channels, self.n_time), dtype=np.float32)
            noise_fd = f.create_dataset("noise_fd", shape=(N, self.n_channels, self.n_time//2), dtype=np.complex64)
            noise_td = f.create_dataset("noise_td", shape=(N, self.n_channels, self.n_time), dtype=np.float32)
            snr = f.create_dataset("snr", shape = (N,), dtype=np.float32)
            asd_dataset = f.create_dataset("asd", data=self.asd, dtype=np.float32)
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
                snr[i:batch_end] = snr_batch
        
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

        # Extract the two fields requested and convert
        payload = {
            "conf": _convert(self.info.get("conf", {})),
            "sampler_init_kwargs": _convert(self.info.get("sampler_init_kwargs", {})),
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
    
        prod = signal * signal.conj()
        high_pass_idx =  self.grid_freq_noise >= 5e-5
        weighted = prod[..., high_pass_idx] * self.df_noise / self.PSD[...,high_pass_idx]
        summed = np.sum(weighted, axis=(1, 2))
        real_part = summed.real
        SNR2 = real_part * 4.0
        
        return np.sqrt(SNR2)

from bbhx.waveformbuild import BBHWaveformFD
# class TD_wrapper:
#     def __init__(
#         self,
#         waveform_generator_kwargs = None,
#         t_start  = 0.0,
#         t_end    = 86400.0,
#         dt       = 10.0,
#         t_max    = 2*86400.0,  # maximum t_merger in the prior,
#         #out_channel = None,
#         **kwargs
#     ):    
#         self.FD_generator = BBHWaveformFD(**waveform_generator_kwargs)
#         self.dt  = dt
#         # set up frequency 
#         self.xp    = self.FD_generator.xp
#         self.freqs = np.fft.rfftfreq(2**int(np.ceil(np.log2(t_max/dt))), d= dt)
#         self.ind_cut = int(t_end/dt)
#         self.t_start = t_start / YRSID_SI
#         self.t_max   = 2**int(np.ceil(np.log2(t_max/dt)))*dt / YRSID_SI
#         self.df      = 1.0/self.t_max
#         self.BBHx_kwargs = dict( modes=modes, direct=False, fill=True, squeeze=False, length=1024)
#     def __call__(
#         self,
#         *args,
#         **waveform_kwargs
#     ):
#         # setup kwargs properly
#         waveform_kwargs["t_obs_start"] = self.t_start
#         waveform_kwargs["t_obs_end"] = self.t_max
#         waveform_kwargs["freqs"] = self.freqs
#         waveform_kwargs["fill"] = True
#         waveform_kwargs["direct"] = False
#         waveform_kwargs["squeeze"] = False
#         waveform_kwargs["length"] = waveform_kwargs.get("length",1024)
#         out_channel = waveform_kwargs.get("out_channel",None)
#         data_FD = self.FD_generator(*args, **waveform_kwargs)
#         ## TODO: add here the noise generation so that you do ifft only once



#         if out_channel is None:
#             # Turn the object to TD
#             FD_series = self.xp.dstack((data_FD[:,:,:-1], self.xp.flip(data_FD[:,:,1:].conj(),axis=2)))
#             ifftseries = self.xp.fft.ifft(FD_series, axis = 2).real / self.dt
#             return ifftseries[:,:,:self.ind_cut]
#             # cut the time signal 
#         else:
#             # Turn the object to TD
#             if isinstance(self.out_channel,(list,np.ndarray,tuple)):
#                 FD_series =  np.dstack((data_FD[:,out_channel,:-1], np.flip(data_FD[:,out_channel,1:].conj(),axis = 2)))
#                 ifftseries= self.xp.fft.ifft(FD_series, axis = 2).real / self.dt
#                 return ifftseries[:,:,:self.ind_cut]
#             else:
#                 FD_series =  np.dstack((data_FD[:,out_channel,:-1], self.xp.flip(data_FD[:,out_channel,1:].conj(),axis = 1)))
#                 ifftseries= self.xp.fft.ifft(FD_series, axis = 1 ).real / self.dt
#                 return ifftseries[:,:self.ind_cut]
    
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