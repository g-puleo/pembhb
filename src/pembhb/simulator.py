from bbhx.waveformbuild import BBHWaveformFD
from lisatools.detector import EqualArmlengthOrbits
import lisatools.sensitivity as lisasens
import numpy as np 
from bbhx.utils.transform import LISA_to_SSB
import yaml
from bbhx.utils.constants import YRSID_SI, PC_SI
import os
from pembhb import ROOT_DIR
from pembhb.sampler import UniformSampler
import h5py
from tqdm import tqdm

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
            "TDItag": conf["waveform_params"]["TDI"],
            "rescaled": False,
            "orbits": orbits
            }
        
        self.waveform_generator = BBHWaveformFD(
            amp_phase_kwargs = dict(run_phenomd=False),
            response_kwargs = response_kwargs,
            use_gpu = gpu_available
        )


        self.obs_time = int( 24*3600*7*conf["waveform_params"]["duration"])# weeks to seconds
        self.freqs = np.logspace(-4, -2, 10000)
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
        ASD = np.zeros((3, self.freqs.shape[0]))
        self.channels = conf["waveform_params"]["TDI"]
        if self.channels == "AET":
            ASD[0] = lisasens.get_sensitivity(self.freqs, sens_fn = lisasens.A1TDISens, **psd_kwargs)
            ASD[1] = lisasens.get_sensitivity(self.freqs, sens_fn = lisasens.E1TDISens, **psd_kwargs)
            ASD[2] = lisasens.get_sensitivity(self.freqs, sens_fn = lisasens.T1TDISens, **psd_kwargs)
            # ASD[self.freqs<1e-5] =0.0 
        elif self.channels == "XYZ":
            ASD[0] = lisasens.get_sensitivity(self.freqs, sens_fn = lisasens.X1TDISens, **psd_kwargs)
            ASD[1] = lisasens.get_sensitivity(self.freqs, sens_fn = lisasens.Y1TDISens, **psd_kwargs)
            ASD[2] = lisasens.get_sensitivity(self.freqs, sens_fn = lisasens.Z1TDISens, **psd_kwargs)
        else:
            raise ValueError("conf['waveform_params']['TDI'] must be either XYZ or AET. ")
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
        noise_fft = np.random.normal(loc= 0.0,size = (n_samples,3, f_len)) + 1j*np.random.normal(loc= 0.0,size = ( n_samples, 3, f_len))
        noise_fd = noise_fft * self.ASD * np.hanning(self.n_pt)
        # Insert a set of zeros between injection[5] and injection[6]. this is the f_ref parameter , which in bbhx can be set to 0 in order to set f_ref @ t_chirp
        injection = np.insert(injection, 6, np.zeros(injection[5].shape), axis=0) 
        wave_FD = self.waveform_generator(*injection, **self.waveform_kwargs) 
        simulated_data_fd = (noise_fd + wave_FD)
        # stack real and imaginary parts over channels
        #breakpoint()
        return simulated_data_fd

    def _sample(self, N=1): 
        """Draw one sample from the joint distribution, first sampling parameters from the prior and then generating the data in frequency domain.

        :param N: _description_, defaults to 1
        :type N: int, optional
        :return: out_dict {"parameters": prior samples , frequency domain data)
        :rtype: dict
        """
        z_samples, tmnre_input = self.sampler.sample(N)
        data_fd = self.generate_d_f(z_samples)
        out_dict = {"parameters": tmnre_input, "data_fd": data_fd}
        return out_dict
    
    def sample_and_store(self, filename, N, batch_size=1000): 
        """Sample N samples and store them in an HDF5 file.

        :param filename: name of the file to store the samples
        :type filename: str
        :param N: number of samples to generate
        :type N: int
        :param batch_size: number of samples to generate in each batch, defaults to 1000
        :type batch_size: int, optional
        """

        with h5py.File(filename, "a") as f:
            source_params = f.create_dataset("source_parameters", shape=(N, 11), dtype=np.float32)
            data_fd = f.create_dataset("data_fd", shape=(N, 6, self.n_pt), dtype=np.float32)
            snr = f.create_dataset("snr", shape = (N,), dtype=np.float32)
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
        signal
        ):
        """
        Obtain the SNR of a signal in frequency domain.

        :param signal: data in frequency domain, output by bbhx with shape (n_samples, 3, n_freqs)
        :type signal: np.array
        :return: SNR values with shape (n_samples,)
        :rtype: np.array
        """
    
        SNR2 =  np.sum(signal*signal.conj()*self.df/self.PSD,axis=(1,2)).real * 4.0 
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