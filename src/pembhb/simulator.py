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
gpu_available=False
WEEK_SI = 7 * 24 * 3600  # seconds in a week
DAY_SI = 24 * 3600  # seconds in a day

class LISAMBHBSimulator():

    def __init__(self, conf):


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
                # "t_obs_start": 0.0, 
                # "t_obs_end": self.obs_time/ YRSID_SI,
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
        if conf["waveform_params"]["TDI"]=="AET":
            ASD = lisasens.get_sensitivity(self.freqs, sens_fn = lisasens.A1TDISens, **psd_kwargs)
            ASD[self.freqs<1e-5] =0.0 
        elif conf["waveform_params"]["TDI"]=="XYZ":
            ASD = lisasens.get_sensitivity(self.freqs, sens_fn = lisasens.X1TDISens, **psd_kwargs)
        else:
            raise ValueError("conf['waveform_params']['TDI'] must be either XYZ or AET. ")
        self.ASD = ASD

    def generate_d_f(self, injection):
        # adjust sky position in the Lframe:
        injection[-1], injection[-4], injection[-3], injection[-2] = LISA_to_SSB(injection[-1], injection[-4], injection[-3], injection[-2])
        f_len = len(self.freqs)
        print(f"f_len {f_len}")
        noise_fft = np.random.normal(loc= 0.0,size = f_len) + 1j*np.random.normal(loc= 0.0,size = f_len)
        print(f"noise_fft {noise_fft.shape}")
        #apply a filter 
        noise_fd = noise_fft * self.ASD * np.hanning(self.n_pt)
        print(f"noise_fd {noise_fd.shape}")
        wave_FD = self.waveform_generator(*injection,**self.waveform_kwargs) 
        #print(f"wave_FD {wave_FD.shape}")
        return noise_fd, wave_FD


if __name__ == "__main__":

    # set parameters
    f_ref = 0.0  # let phenom codes set f_ref -> fmax = max(f^2A(f))
    phi_ref = 0.0 # phase at f_ref
    m1 = 1e6
    m2 = 5e5
    a1 = 0.2
    a2 = 0.4
    dist = 18e3  * PC_SI * 1e6 # 3e3 in Mpc
    inc = np.pi/3.
    beta = np.pi/4.  # ecliptic latitude
    lam = np.pi/5.  # ecliptic longitude
    psi = np.pi/6.  # polarization angle
    t_ref = 0.5 * YRSID_SI  # t_ref  (in the SSB reference frame)


    # Load configuration from config.yaml
    config_path = os.path.join(ROOT_DIR, "config.yaml")
    with open(config_path, "r") as file:
        conf = yaml.safe_load(file)

    # Initialize the simulator with the configuration
    simulator = LISAMBHBSimulator(conf)

    # Generate the waveform and noise
    injection_params = [m1, m2, a1, a2,
                          dist, phi_ref, f_ref, inc, lam,
                          beta, psi, t_ref]
    noise_fd, wave_fd = simulator.generate_d_f(injection_params)
    print(np.allclose(np.abs(wave_fd), 0))
    # Example: Plot the absolute value of the noise and waveform
    plt.plot(simulator.freqs, np.abs(noise_fd), label='Noise')
    for i, channel in enumerate(["A", "E", "T"]):
        plt.plot(simulator.freqs, np.abs(wave_fd[0,i,:]), label=f'Waveform {channel}')
    plt.legend()
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.title("Generated Waveform + Noise")
    plt.grid()
    plt.show()

