from pembhb.utils import read_config
from pembhb.simulator import LISAMBHBSimulatorTD
from pembhb.data import MBHBDataset
from torch.utils.data import DataLoader
#### 1) generate data, store them on disk 
conf = read_config("config_td.yaml") # defines the prior and waveform options
sampler_init_kwargs = {'prior_bounds': conf["prior"]} # prior to initialise the random sampler
simulator = LISAMBHBSimulatorTD(conf, sampler_init_kwargs=sampler_init_kwargs, seed=314)
data = simulator.sample_and_store(filename="my_dataset.h5", N=100)

#### 2) load data from disk into a torch dataset
dataset = MBHBDataset("my_dataset.h5",
                      #do not provide transform_fd and transform_td if you do not want any transform on top of the data
                      transform_fd='log', # takes log of fourier domain amplitudes (phases are untouched)
                      transform_td='normalise_max',# divides time series by maximum over the whole dataset and all time points
                      device='cuda'
                )

#### use the data 
dataloader = DataLoader(dataset, batch_size=50, shuffle=True)
for batch in dataloader:
    data_fd = batch['data_fd']
    data_td = batch['data_td']
    source_params = batch['source_parameters'] #they are in the same format as the prior keys defined in config_td.yaml

    ## do whatever you want 
