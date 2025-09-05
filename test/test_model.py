# init an untrained model and test the forward pass 
from pembhb.model import InferenceNetwork
import os
from pembhb import utils, ROOT_DIR
import torch
WEEK_SI = 7*24*3600  # seconds in a week

def test_model_forward_pass():
    conf = utils.read_config(os.path.join(ROOT_DIR, "config_td.yaml"))
    model = InferenceNetwork(conf=conf)
    n_marginals = len(conf["tmnre"]["marginals"])
    # Create dummy input data
    batch_size = 4
    n_channels = len(conf["waveform_params"]["channels"])
    n_timesteps = int(conf["waveform_params"]["duration"]*WEEK_SI/conf["waveform_params"]["dt"])
    n_freqs = n_timesteps // 2 
    
    dummy_input_f = torch.randn(batch_size, 2*n_channels, n_freqs)  # Frequency domain input
    dummy_input_t = torch.randn(batch_size, n_channels, n_timesteps)  # Time domain input
    dummy_parameters = torch.randn(batch_size, 11)  # Assuming 6 parameters to infer
    # Forward pass
    output = model(dummy_input_f, dummy_input_t, dummy_parameters)

    
    # Check output shape
    assert output.shape == (batch_size,n_marginals)  # Assuming 6 parameters to infer