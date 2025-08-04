import pytest
import numpy as np
import os
import yaml
from pembhb.simulator import LISAMBHBSimulator
from pembhb.utils import read_config
from pembhb.sampler import UniformSampler
from pembhb import ROOT_DIR
DAY_SI = 24*3600
WEEK_SI = 7*DAY_SI
from bbhx.utils.constants import PC_SI, YRSID_SI
@pytest.fixture
def simulator():
    """Fixture to provide example configuration."""
    conf = read_config(os.path.join(ROOT_DIR, "config.yaml"))
    return LISAMBHBSimulator(conf, sampler_init_kwargs={"prior_bounds": conf["prior"]})

def test_sample(simulator): 
    outdict = simulator._sample(N=10)
    frequency_data = outdict["data_fd"]
    parameters = outdict["parameters"]
    
    assert isinstance(frequency_data, np.ndarray)
    assert frequency_data.shape[0] == 10
    assert frequency_data.shape[1] == 2
    assert frequency_data.shape[2] == 10000
    assert isinstance(parameters, np.ndarray)
    assert parameters.shape[0] == 11
    assert parameters.shape[1] == 10
    assert np.iscomplexobj(frequency_data)



def test_sample_and_store(simulator):
    fname = os.path.join(ROOT_DIR, "test", "test_sample_and_store.h5")
    simulator.sample_and_store(fname, N=10, batch_size=5)
    assert os.path.exists(fname)
    os.remove(fname)
    assert not os.path.exists(fname)        

