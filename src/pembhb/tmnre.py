import swyft
import os
import torch
from pembhb import ROOT_DIR
from pytorch_lightning.loggers import TensorBoardLogger
from pembhb.simulator import LISAMBHBSimulator
from pembhb.model import InferenceNetwork
ACC = "cuda" if torch.cuda.is_available() else "cpu"
def round(sim, obs):
    print("Sampling from the simulator...")
    samples = sim.sample(100)
    dm = swyft.SwyftDataModule(samples, batch_size = 64, fractions=(0.9,0.1,0.0))
    logger = TensorBoardLogger(ROOT_DIR)
    trainer = swyft.SwyftTrainer(accelerator = ACC, precision = 64, max_epochs=1, logger=logger)
    model = InferenceNetwork(num_features = 10000, hlayersizes=(500,20), lr=1e-3)
    print("Training the model...")
    trainer.fit(model, dm)
    trainer.save_checkpoint(os.path.join(ROOT_DIR, "checkpoints", "tmnre.ckpt"))
    prior_samples = sim.sample(N = 10000, targets = ['z_tot'])
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
    sim = LISAMBHBSimulator(conf)

    observation = sim.sample(N = 1 , targets = ['data_fd'])
    predictions, new_bounds, samples = round(sim, observation)  
