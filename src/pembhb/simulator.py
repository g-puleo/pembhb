import numpy as np 
import os
import h5py
import yaml
from scipy.signal.windows import tukey
from tqdm import tqdm

from lisatools.detector import EqualArmlengthOrbits
import lisatools.sensitivity as lisasens

from bbhx.waveformbuild import BBHWaveformFD, BBHWaveformTD
from bbhx.response.fastfdresponse import LISATDIResponse
from bbhx.utils.transform import LISA_to_SSB
from bbhx.utils.constants import YRSID_SI, PC_SI

from pembhb import ROOT_DIR
from pembhb.sampler import UniformSampler

gpu_available=False
WEEK_SI = 7 * 24 * 3600  # seconds in a week
DAY_SI = 24 * 3600  # seconds in a day

class LISAMBHBSimulator():

    def __init__(self, conf, sampler_init_kwargs):
        super().__init__()
        # initialise the waveform generator
        orbits = EqualArmlengthOrbits(use_gpu=gpu_available)
        orbits.configure(linear_interp_setup=True)

        response_kwargs = {
            "TDItag": "AET",
            "rescaled": False,
            "orbits": orbits
            }
        
        self.waveform_generator = BBHWaveformFD(
            amp_phase_kwargs = dict(run_phenomd=False),
            response_kwargs = response_kwargs,
            use_gpu = gpu_available
        )


        self.obs_time = int( 24*3600*7*conf["waveform_params"]["duration"])# weeks to seconds
        self.freqs = np.logspace(conf["waveform_params"]["min_freq"], conf["waveform_params"]["max_freq"], conf["waveform_params"]["n_freqs"])
        self.df = np.diff(self.freqs, prepend=self.freqs[0])
        self.n_pt = len(self.freqs)
        self.waveform_kwargs = {
                "modes": conf["waveform_params"]["modes"],
                "t_obs_start": 0.0, 
                "t_obs_end": self.obs_time/ YRSID_SI,
                "freqs": self.freqs,
                "direct": False,
                "squeeze": True, 
                "fill": True,
                "length": 1024
                }
        psd_kwargs = {
            "model": conf["waveform_params"]["noise"],
            "return_type": "ASD"
        }
        self.channels = conf["waveform_params"]["channels"]
        self.n_channels = len(self.channels)
        ASD = np.zeros((self.n_channels, self.freqs.shape[0]))
        lisasens_dict = {   'A': lisasens.A1TDISens, 
                            'E': lisasens.E1TDISens,
                            'T': lisasens.T1TDISens, 
                            'X': lisasens.X1TDISens,
                            'Y': lisasens.Y1TDISens,
                            'Z': lisasens.Z1TDISens
                        }
    
        for i, channel in enumerate(self.channels):
            if channel not in lisasens_dict:
                raise ValueError(f"Channel {channel} is not supported. Supported channels are: {list(lisasens_dict.keys())}")
            ASD[i] = lisasens.get_sensitivity(self.freqs, sens_fn=lisasens_dict[channel], **psd_kwargs)

        self.ASD = ASD
        self.PSD = ASD**2
        self.sampler = UniformSampler(**sampler_init_kwargs)

    def generate_d_f(self, injection: np.array):
        """ Generate simulated data in frequency domain given the injection parameters.

        :param injection: injection parameteres in LISA frame
        :type injection: list[np.array]
        :return: simulated data in frequency domain
        :rtype: np.array
        """
        # adjust sky position in the Lframe:
        injection[-1] = injection[-1] + self.waveform_kwargs["t_obs_end"]*YRSID_SI
        injection[-1], injection[-4], injection[-3],  injection[-2] = LISA_to_SSB(injection[-1], injection[-4], injection[-3], injection[-2])
        f_len = len(self.freqs)
        n_samples = injection.shape[1]
        noise_fft = np.random.normal(loc= 0.0,size = (n_samples,self.n_channels, f_len)) + 1j*np.random.normal(loc= 0.0,size = (n_samples, self.n_channels, f_len))
        noise_fd = noise_fft * self.ASD * np.hanning(self.n_pt)
        # Insert a set of zeros between injection[5] and injection[6]. this is the f_ref parameter , which in bbhx can be set to 0 in order to set f_ref @ t_chirp
        injection = np.insert(injection, 6, np.zeros(injection[5].shape), axis=0) 
        wave_FD = self.waveform_generator(*injection, **self.waveform_kwargs)[:, :2, :]

        simulated_data_fd = (noise_fd + wave_FD)
        # stack real and imaginary parts over channels
        return simulated_data_fd

    def _sample(self, N=1): 
        """Draw one sample from the joint distribution, first sampling parameters from the prior and then generating the data in frequency domain.

        :param N: _description_, defaults to 1
        :type N: int, optional
        :return: out_dict {"parameters": prior samples , frequency domain data)
        :rtype: dict
        """
        z_samples, tmnre_input = self.sampler.sample(N, self.obs_time)
        data_fd = self.generate_d_f(z_samples)
        out_dict = {"parameters": tmnre_input, "data_fd": data_fd}
        return out_dict
    
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
            source_params = f.create_dataset("source_parameters", shape=(N, 11), dtype=np.float32)
            data_fd = f.create_dataset("data_fd", shape=(N, 4, self.n_pt), dtype=np.float32)
            snr = f.create_dataset("snr", shape = (N,), dtype=np.float32)
            psd_dataset = f.create_dataset("psd", data=self.PSD, dtype=np.float32)
            for i in tqdm(range(0, N, batch_size)):
                batch_end = min(i + batch_size, N)
                batch_size_actual = batch_end - i
                out = self._sample(batch_size_actual)

                z_samples = out["parameters"]
                data_fd_batch = out["data_fd"]
                snr_batch = self.get_SNR_FD(data_fd_batch)
                data_fd_amp_phase = np.concatenate((np.abs(data_fd_batch), np.angle(data_fd_batch)), axis=1)
                source_params[i:batch_end] = z_samples.T # Reshape to (batch_size, 11) instead of (11, batch_size)
                data_fd[i:batch_end] = data_fd_amp_phase
                snr[i:batch_end] = snr_batch

    def get_SNR_FD(self,
        signal_FD
        ):
        """
        Obtain the SNR of a signal in frequency domain.

        :param signal: data in frequency domain, output by bbhx with shape (n_samples, 3, n_freqs)
        :type signal: np.array
        :return: SNR values with shape (n_samples,)
        :rtype: np.array
        """
    
        SNR2 =  np.sum(signal_FD*signal_FD.conj()*self.df/self.PSD,axis=(1,2)).real * 4.0 
        return np.sqrt(SNR2)

    

    
# if __name__ == "__main__":

    # Load configuration from config.yaml
    # config_path = os.path.join(ROOT_DIR, "config.yaml")
    # with open(config_path, "r") as file:
    #     conf = yaml.safe_load(file)

    # # Initialize the simulator and the sampler
    # simulator = LISAMBHBSimulator(conf)
    # samples = simulator.sample(  targets=["data_fd"] )
    # Example: Plot the absolute value of the noise and waveform
    # plt.plot(simulator.freqs, np.abs(noise_fd), label='Noise')
    # for i, channel in enumerate(["A", "E", "T"]):
    #     plt.plot(simulator.freqs, np.abs(data_fd[0,i,:]), label=f'Waveform {channel}')
    # plt.legend()
    # plt.xscale("log")
    # plt.yscale("log")
    # plt.xlabel("Frequency (Hz)")
    # plt.ylabel("Amplitude")
    # plt.title( "Generated Waveform + Noise" )
    # plt.grid()
    # plt.show()


class LISAMBHBSimulatorTD(): 

    def __init__(self, conf, sampler_init_kwargs):
        super().__init__()
        # initialise the waveform generator
        orbits = EqualArmlengthOrbits(use_gpu=gpu_available)
        orbits.configure(linear_interp_setup=True)
        response_kwargs = {
            "TDItag": "AET", 
            "rescaled": False,
            "orbits": orbits
            }
        
        self.waveform_generator = BBHWaveformTD(
            amp_phase_kwargs = dict(run_phenomd=False),
            response_kwargs = response_kwargs,
            use_gpu = gpu_available
        )

        #### SET UP FREQUENCY GRID from OBSERVATION TIME AND SAMPLING RATE ####
        self.obs_time_SI = int( 24*3600*7*conf["waveform_params"]["duration"])# weeks to seconds
        self.t_obs_start_SI = 0.0 # seconds
        self.t_obs_end_SI = self.t_obs_start_SI + self.obs_time_SI # seconds
        self.dt = conf['waveform_params']['dt']
        self.channels = conf["waveform_params"]["channels"]
        self.n_channels = len(self.channels)
        # pad the time to power of 2 for the noise
        self.n_time_pt_noise = int(self.obs_time_SI / self.dt)
        self.df_noise = 1./self.n_time_pt_noise/self.dt
        self.grid_freq_noise = np.arange(1,self.n_time_pt_noise//2 +1 ) * self.df_noise
        self.f_len_noise = len(self.grid_freq_noise)
        self.window_td = tukey(self.n_time_pt_noise, alpha=0.0005)
        self.window_fd_noise = np.hanning(self.n_time_pt_noise//2 +1)
        #### set up the noise model
        psd_kwargs = {
            "model": conf["waveform_params"]["noise"],
            "return_type": "ASD"
        }
        ASD = np.zeros((self.n_channels, self.grid_freq_noise.shape[0]))
        lisasens_dict = {   'A': lisasens.A1TDISens, 
                            'E': lisasens.E1TDISens,
                            'T': lisasens.T1TDISens, 
                            'X': lisasens.X1TDISens,
                            'Y': lisasens.Y1TDISens,
                            'Z': lisasens.Z1TDISens
                        }
    
        for i, channel in enumerate(self.channels):
            if channel not in lisasens_dict:
                raise ValueError(f"Channel {channel} is not supported. Supported channels are: {list(lisasens_dict.keys())}")
            ASD[i] = lisasens.get_sensitivity(self.grid_freq_noise, sens_fn=lisasens_dict[channel], **psd_kwargs)

        ASD[:,self.grid_freq_noise < 5e-5] =0.0 
        self.ASD = ASD

        self.PSD = self.ASD**2
        # self.noise_factor will undergo ifft: ifft expects the output format of np.fft. 
        # the output of fft is such that out[0] is the DC component, out[:n//2] is the positive frequencies and out[n//2:] is the negative frequencies
        # look at https://numpy.org/doc/stable/reference/routines.fft.html#module-numpy.fft . 
        self.noise_factor = np.concatenate(([[0.0]]*self.n_channels,self.ASD[:,1:], self.ASD[:,::-1].conj()), axis=1)/np.sqrt(4*self.df_noise)
        self.noise_rng = np.random.default_rng(seed=0)

        self.channels = conf["waveform_params"]["channels"]
        channel_idx = [ i for i, c in enumerate(["A", "E", "T"]) if c in self.channels ]
        self.waveform_kwargs = {
            #"freqs": self.grid_freq,
            "modes": conf["waveform_params"]["modes"],
            "out_channel": channel_idx,
            "length": 1024,
            "t_obs_start": self.t_obs_start_SI/ YRSID_SI, 
            "t_obs_end": self.t_obs_end_SI/ YRSID_SI,
            "dt": self.dt
            }
        self.n_channels = len(self.channels)

        self.sampler = UniformSampler(**sampler_init_kwargs)
    def generate_noise_td( self, n_observations):
        noise_fd = self.noise_rng.normal(loc= 0.0,size = (n_observations , self.n_channels, self.f_len_noise)) + 1j*self.noise_rng.normal(loc= 0.0,size = (n_observations , self.n_channels, self.f_len_noise))
        #noise_fd *= self.window_fd_noise
        two_sided_noise = self.noise_factor * np.concatenate((noise_fd,noise_fd[::-1].conj()), axis=-1)
        two_sided_noise[..., 0] = 0.0 + 1j *0.0 #Force the DC component to be 0
        noise_td_cmplx = np.fft.ifft(two_sided_noise)/self.dt
        noise_td = noise_td_cmplx[:self.n_time_pt_noise].real
        return noise_td, two_sided_noise[...,1:self.f_len_noise+1]
    def generate_signal_td( self, injection: np.array):
        signal_td = self.waveform_generator(*injection, **self.waveform_kwargs) 
        return signal_td

    def generate_d_f(self, injection: np.array):
        """ Generate simulated data in frequency domain given the injection parameters.

        :param injection: injection parameteres in LISA frame
        :type injection: list[np.array]
        :return: simulated data in frequency domain
        :rtype: np.array
        """
        injection[-1], injection[-4], injection[-3], injection[-2] = LISA_to_SSB(injection[-1], injection[-4], injection[-3], injection[-2])
        # generate wave in TD
        # insert a set of zeros between injection[5] and injection[6]. this is the f_ref parameter , which in bbhx can be set to 0 in order to set f_ref @ t_chirp
        injection = np.insert(injection, 6, np.zeros(injection[5].shape), axis=0) 
        n_observations = injection.shape[1]
 # add noise 
        noise_td , noise_fd = self.generate_noise_td(n_observations)
        signal_td = self.generate_signal_td(injection)
        wave_TD = signal_td + noise_td
        # Apply window to time-domain signal
        wave_TD_windowed = wave_TD * self.window_td
        # Compute the real FFT along the time axis (assume wave_TD shape is (n_observations, n_time_pt_noise))
        wave_FD_raw = np.fft.rfft(wave_TD_windowed)
        # Remove the DC component (first frequency bin)
        wave_FD_no_dc = wave_FD_raw[...,1:]
        # Convert to complex64
        wave_FD_complex = wave_FD_no_dc.astype(np.complex64)
        # Multiply by dt and ASD
        wave_FD = wave_FD_complex * self.dt #/ self.ASD
        return (wave_TD.astype(np.float32), wave_FD, noise_fd)
    
    def _sample(self, N=1): 
        """Draw one sample from the joint distribution, first sampling parameters from the prior and then generating the data in frequency domain.

        :param N: _description_, defaults to 1
        :type N: int, optional
        :return: out_dict {"parameters": prior samples , frequency domain data)
        :rtype: dict
        """
        z_samples, tmnre_input = self.sampler.sample(N, self.t_obs_end_SI)
        data_td, data_fd, noise_fd = self.generate_d_f(z_samples)
        out_dict = {"parameters": tmnre_input, "data_td": data_td, "data_fd": data_fd, "noise_fd": noise_fd}
        return out_dict
    
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
            source_params = f.create_dataset("source_parameters", shape=(N, 11), dtype=np.float32)
            sample_frequencies = f.create_dataset("frequencies", data=self.grid_freq_noise, dtype=np.float32)
            sample_times_SI = f.create_dataset("times_SI", data=np.arange(0, self.n_time_pt_noise)*self.dt, dtype=np.float32)
            data_fd = f.create_dataset("data_fd", shape=(N, 2*self.n_channels, self.n_time_pt_noise//2), dtype=np.float32)
            data_td = f.create_dataset("data_td", shape=(N, self.n_channels, self.n_time_pt_noise), dtype=np.float32)
            noise_fd = f.create_dataset("noise_fd", shape=(N, self.n_channels, self.n_time_pt_noise//2), dtype=np.complex64)
            snr = f.create_dataset("snr", shape = (N,), dtype=np.float32)
            psd_dataset = f.create_dataset("psd", data=self.PSD, dtype=np.float32)
            print("Sampling and storing simulations to ", filename)
            for i in tqdm(range(0, N, batch_size)):
                batch_end = min(i + batch_size, N)
                batch_size_actual = batch_end - i
                out = self._sample(batch_size_actual)
                z_samples = out["parameters"]
                data_fd_batch = out["data_fd"]
                data_td_batch = out["data_td"]
                noise_fd_batch = out["noise_fd"]
                snr_batch = self.get_SNR_FD(data_fd_batch)
                data_fd_amp_phase = np.concatenate((np.abs(data_fd_batch), np.angle(data_fd_batch)), axis=1)
                source_params[i:batch_end] = z_samples.T # Reshape to (batch_size, 11) instead of (11, batch_size)
                data_fd[i:batch_end] = data_fd_amp_phase
                data_td[i:batch_end] = data_td_batch
                noise_fd[i:batch_end] = noise_fd_batch
                snr[i:batch_end] = snr_batch
    
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



