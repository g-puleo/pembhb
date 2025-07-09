import swyft
from pembhb.simulator import LISAMBHBSimulator
from pembhb.model import InferenceNetwork

def round(conf:dict):

    sim = LISAMBHBSimulator(conf)
    
    print("Sampling from the simulator...")
    samples = sim.sample(1000)
    dm = swyft.SwyftDataModule(samples, fractions=(0.75,0.05,0.2), batch_size = 64)
    trainer = swyft.SwyftTrainer(accelerator = "cuda", precision = 64, max_epochs=1)
    model = InferenceNetwork(num_features = 10000, hlayersizes=(500,20), lr=1e-3)
    trainer.fit(model, dm)
    prior_samples = sim.sample(N = 10000, targets = ['z_tot'])
    obs = sim.sample(N = 1 , targets = ['data_fd'])
    predictions = trainer.infer( model , obs , prior_samples)
    new_bounds = swyft.collect_rect_bounds(predictions[0], 'z_tot', (11,), threshold = 1e-5)
    print("New bounds:", new_bounds)
    return predictions, new_bounds, samples


if __name__ == "__main__":
    import yaml
    import os
    from pembhb import ROOT_DIR

    # Load configuration from config.yaml
    config_path = os.path.join(ROOT_DIR, "config.yaml")
    with open(config_path, "r") as file:
        conf = yaml.safe_load(file)

    predictions, new_bounds, samples = round(conf)