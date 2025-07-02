import pytest
import numpy as np
from pembhb.simulator import LISAMBHBSimulator
import yaml
DAY_SI = 24*3600
WEEK_SI = 7*DAY_SI
from bbhx.utils.constants import PC_SI, YRSID_SI
@pytest.fixture
def example_config():
    """Fixture to provide example configuration."""
    return {
        "waveform_params": {
            "TDI": "XYZ",
            "modes": [[2, 2], [2, 1], [3, 3], [3, 2], [4, 4], [4, 3]],
            "noise": "sangria",
            "duration": 1,  # weeks
            "dt": 10,  # seconds
        }
    }



@pytest.fixture
def example_injection():
    """Fixture to provide example injection parameters."""
    return [
        5e6, #m1
        3e4, #m2
        0.0, #chi1
        0.0, #chi2
        10.0 * 1e9 * PC_SI,  # distance
        0, #f_ref 
        0.5, #inc
        0.5, #lambda
        0.5, # beta
        0.5, # psi
        0.5 * YRSID_SI 
    ]


def test_simulator_initialization(example_config):
    """Test initialization of the simulator."""
    simulator = LISAMBHBSimulator(conf=example_config)
    assert simulator.obs_time == 24 * 3600 * 7 * example_config["waveform_params"]["duration"]
    assert simulator.dt == example_config["waveform_params"]["dt"]
    assert simulator.n_pt > 0
    assert simulator.ASD is not None

def test_generate_d_f(example_config, example_injection):
    """Test the frequency domain data generation."""
    simulator = LISAMBHBSimulator(conf=example_config)
    frequency_data = simulator.generate_d_f(injection=example_injection)
    assert isinstance(frequency_data, np.ndarray)
    assert frequency_data.shape[0] > 0
    assert np.iscomplexobj(frequency_data)

