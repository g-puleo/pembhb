from bbhx.waveformbuild import BBHWaveformFD
from lisatools.detector import EqualArmlengthOrbits
import lisatools.sensitivity as lisasens
import numpy as np 
from bbhx.utils.transform import LISA_to_SSB
import yaml
from bbhx.utils.constants import YRSID_SI, PC_SI
import os
import matplotlib.pyplot as plt
from pembhb import ROOT_DIR
from pembhb.sampler import UniformSampler
import swyft
import torch
gpu_available = torch.cuda.is_available() 
WEEK_SI = 7 * 24 * 3600  # seconds in a week
DAY_SI = 24 * 3600  # seconds in a day

class LISAMBHBSimulator(swyft.Simulator):

    def __init__(self, conf):
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
        self.freqs = np.logspace(-4, 0, 10000)
        self.n_pt = len(self.freqs)
        self.waveform_kwargs =     {
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
        
        if conf["waveform_params"]["TDI"]=="AET":
            ASD[0] = lisasens.get_sensitivity(self.freqs, sens_fn = lisasens.A1TDISens, **psd_kwargs)
            ASD[1] = lisasens.get_sensitivity(self.freqs, sens_fn = lisasens.E1TDISens, **psd_kwargs)
            ASD[2] = lisasens.get_sensitivity(self.freqs, sens_fn = lisasens.T1TDISens, **psd_kwargs)
            # ASD[self.freqs<1e-5] =0.0 
        elif conf["waveform_params"]["TDI"]=="XYZ":
            ASD[0] = lisasens.get_sensitivity(self.freqs, sens_fn = lisasens.X1TDISens, **psd_kwargs)
            ASD[1] = lisasens.get_sensitivity(self.freqs, sens_fn = lisasens.Y1TDISens, **psd_kwargs)
            ASD[2] = lisasens.get_sensitivity(self.freqs, sens_fn = lisasens.Z1TDISens, **psd_kwargs)
        else:
            raise ValueError("conf['waveform_params']['TDI'] must be either XYZ or AET. ")
        self.ASD = ASD

        # initialise sampler 
        self.sampler = UniformSampler(conf["prior"])

    def generate_d_f(self, injection: np.array):
        """_summary_

        :param injection: injection parameteres in LISA frame
        :type injection: list[np.array]
        :return: simulated data in frequency domain
        :rtype: _type_
        """
        # adjust sky position in the Lframe:
        injection[-1] = injection[-1] + self.waveform_kwargs["t_obs_end"]*YRSID_SI
        injection[-1], injection[-4], injection[-3],  injection[-2] = LISA_to_SSB(injection[-1], injection[-4], injection[-3], injection[-2])
        f_len = len(self.freqs)
        noise_fft = np.random.normal(loc= 0.0,size = (1,3, f_len)) + 1j*np.random.normal(loc= 0.0,size = ( 1, 3, f_len))
        noise_fd = noise_fft * self.ASD * np.hanning(self.n_pt)
        # Insert a set of zeros between injection[5] and injection[6]
        injection = np.insert(injection, 6, np.zeros(injection[5].shape), axis=0) 
        wave_FD = self.waveform_generator(*injection, **self.waveform_kwargs) 
        simulated_data_fd = (noise_fd + wave_FD)[0,:,:]
        # stack real and imaginary parts over channels
        simulated_data_fd = np.concatenate((simulated_data_fd.real, simulated_data_fd.imag), axis=0)
        return simulated_data_fd

    def build(self, graph) : 
        # generate the source parameters from prior
        z_tot = graph.node("z_tot", self.sampler.sample, 1)
        # generate the waveform and noise
        data_fd = graph.node("data_fd", self.generate_d_f, z_tot)




if __name__ == "__main__":

    from pembhb.sampler import UniformSampler


    # Load configuration from config.yaml
    config_path = os.path.join(ROOT_DIR, "config.yaml")
    with open(config_path, "r") as file:
        conf = yaml.safe_load(file)

    # Initialize the simulator and the sampler
    simulator = LISAMBHBSimulator(conf)
    samples = simulator.sample(3, targets=["data_fd"])
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



