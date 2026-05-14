#!/usr/bin/env python
"""Train a DenoisingAutoencoder on frequency-domain MBHB data.

Settings come from a YAML config file (default ``configs/train_config.yaml``);
only the dataset path is taken on the CLI.

Usage
-----
    /data/gpuleo/envs/lisa_pip/bin/python scripts/train_autoencoder.py \\
        --dataset /path/to/data.h5 \\
        [--train-config train_config.yaml]

Config layout (relevant keys):
  top-level:
    batch_size, noise_factor, precision, device
  architecture.data_summary.Autoencoder:
    architecture, representation, n_channels, n_freqs,
    bottleneck_dim, hidden_channels, kernel_size, stride, dropout,
    residual, decoder_post_fc_bn,
    high_freq_only, freq_split_idx, amplitude_normalise,
    lr, weight_decay, epochs, scheduler_patience, scheduler_factor,
    early_stop_patience, log_name, checkpoint_every_n_epochs

The prior bounds in the corresponding ``<dataset>.yaml`` file are stored on
the autoencoder checkpoint for provenance.
"""

import argparse
import os

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

from pembhb import ROOT_DIR, DATA_ROOT_DIR, set_precision
from pembhb import utils
from pembhb.data import MBHBDataModule
from pembhb.autoencoder import DenoisingAutoencoder


def load_prior_bounds(dataset_path: str) -> dict:
    """Load prior bounds from the .yaml sidecar of the HDF5 dataset."""
    yaml_path = dataset_path.replace(".h5", ".yaml")
    if not os.path.exists(yaml_path):
        print(f"[train_autoencoder] Warning: No .yaml file found at {yaml_path}")
        return None
    try:
        config = utils.read_config(yaml_path)
        prior_bounds = config.get("prior", None)
        if prior_bounds is not None:
            print(f"[train_autoencoder] Loaded prior bounds from {yaml_path}")
        return prior_bounds
    except Exception as e:
        print(f"[train_autoencoder] Warning: Failed to read {yaml_path}: {e}")
        return None


def parse_args():
    p = argparse.ArgumentParser(description="Train a denoising autoencoder.")
    p.add_argument("--dataset", type=str, required=True,
                   help="Path to the HDF5 dataset (required).")
    p.add_argument("--train-config", default="train_config.yaml",
                   help="Filename inside configs/ (default: train_config.yaml).")
    return p.parse_args()


def main():
    args = parse_args()

    train_config = utils.read_config(os.path.join(ROOT_DIR, "configs", args.train_config))
    set_precision(train_config.get("precision", "float32"))

    ae_conf = train_config["architecture"]["data_summary"]["Autoencoder"]
    device = ae_conf.get("device", train_config.get("device", "cuda"))
    batch_size = train_config.get("batch_size", 250)
    noise_factor = train_config.get("noise_factor", 1.0)

    dataset_path = args.dataset
    print(f"[train_autoencoder] config       : {args.train_config}")
    print(f"[train_autoencoder] dataset      : {dataset_path}")
    print(f"[train_autoencoder] architecture : {ae_conf['architecture']}")
    print(f"[train_autoencoder] representation : {ae_conf['representation']}")
    print(f"[train_autoencoder] device       : {device}")

    prior_bounds = load_prior_bounds(dataset_path)

    data_module = MBHBDataModule(
        filename=dataset_path,
        batch_size=batch_size,
        num_workers=ae_conf.get("num_workers", 4),
        cache_in_memory=ae_conf.get("cache_in_memory", True),
        noise_factor=noise_factor,
    )
    data_module.setup(stage="fit")

    hidden_channels = tuple(ae_conf.get("hidden_channels", [32, 64, 128, 256, 256]))
    sizes = tuple(ae_conf.get("sizes", [16, 32, 64, 128, 256]))
    down_sampling = tuple(ae_conf.get("down_sampling", [4, 8, 8, 8]))

    model = DenoisingAutoencoder(
        n_channels=ae_conf.get("n_channels", 2),
        n_freqs=ae_conf.get("n_freqs", 4096),
        architecture=ae_conf.get("architecture", "conv"),
        bottleneck_dim=ae_conf.get("bottleneck_dim", 128),
        hidden_channels=hidden_channels,
        kernel_size=ae_conf.get("kernel_size", 4),
        stride=ae_conf.get("stride", 2),
        dropout=ae_conf.get("dropout", 0.0),
        residual=ae_conf.get("residual", False),
        decoder_post_fc_bn=ae_conf.get("decoder_post_fc_bn", True),
        sizes=sizes,
        down_sampling=down_sampling,
        lr=ae_conf.get("lr", 1e-3),
        weight_decay=ae_conf.get("weight_decay", 1e-5),
        scheduler_patience=ae_conf.get("scheduler_patience", 10),
        scheduler_factor=ae_conf.get("scheduler_factor", 0.3),
        representation=ae_conf.get("representation", "amp_phase"),
        high_freq_only=ae_conf.get("high_freq_only", False),
        freq_split_idx=ae_conf.get("freq_split_idx", 2048),
        idx_lowerbound=ae_conf.get("idx_lowerbound", None),
        idx_upperbound=ae_conf.get("idx_upperbound", None),
        amplitude_normalise=ae_conf.get("amplitude_normalise", True),
        prior_bounds=prior_bounds,
        whiten=ae_conf.get("whiten", True),
        subtract_mean_whitened=ae_conf.get("subtract_mean_whitened", False)
    )
    model = model.to(device)

    if ae_conf.get("whiten", True):
        model.set_whitening(data_module.get_noise_scale())
        if model.amplitude_normalise:
            norm_loader = data_module.train_dataloader(shuffle=False, num_workers=0)
            model.fit_white_normalisation(norm_loader)
    else:
        norm_loader = data_module.train_dataloader(shuffle=False, num_workers=4)
        model.fit_normalisation(norm_loader)

    checkpoint_cb = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=2,
        filename="ae-{epoch:03d}-{val_loss:.4e}",
    )
    periodic_checkpoint_cb = ModelCheckpoint(
        every_n_epochs=ae_conf.get("checkpoint_every_n_epochs", 10),
        save_top_k=-1,
        save_on_train_epoch_end=True,
        filename="ae-periodic-{epoch:03d}",
    )
    early_stop_cb = EarlyStopping(
        monitor="val_loss",
        patience=ae_conf.get("early_stop_patience", 50),
        mode="min",
    )
    lr_monitor_cb = LearningRateMonitor(logging_interval="epoch")

    logger = TensorBoardLogger(
        save_dir=os.path.join(DATA_ROOT_DIR, "logs"),
        name=ae_conf.get("log_name", "autoencoder"),
    )

    trainer = Trainer(
        logger=logger,
        max_epochs=ae_conf.get("epochs", train_config.get("epochs", 500)),
        accelerator=device,
        devices=1,
        enable_progress_bar=True,
        callbacks=[checkpoint_cb, periodic_checkpoint_cb, early_stop_cb, lr_monitor_cb],
        gradient_clip_val=ae_conf.get("gradient_clip_val", None),
    )

    trainer.fit(model, data_module)

    print(f"\n[train_autoencoder] best checkpoint : {checkpoint_cb.best_model_path}")
    print(f"[train_autoencoder] best val_loss   : {checkpoint_cb.best_model_score:.6e}")


if __name__ == "__main__":
    main()
