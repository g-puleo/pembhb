#!/usr/bin/env python
"""
Train a DenoisingAutoencoder on frequency-domain MBHB data.

Usage
-----
    python scripts/train_autoencoder.py --dataset /path/to/data.h5   # required
    python scripts/train_autoencoder.py --representation real_imag   # real/imag channels
    python scripts/train_autoencoder.py --bottleneck_dim 256         # larger bottleneck
    python scripts/train_autoencoder.py --dropout 0.1                # add regularization
    python scripts/train_autoencoder.py --high_freq_only             # reconstruct only high freqs
    python scripts/train_autoencoder.py --architecture unet          # legacy unet with skip connections

Prior bounds from the corresponding .yaml file are automatically stored in the checkpoint.
"""

import argparse
import os

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from pembhb import ROOT_DIR, DATA_ROOT_DIR
from pembhb import utils
from pembhb.data import MBHBDataModule
from pembhb.autoencoder import DenoisingAutoencoder

torch.set_float32_matmul_precision("medium")


def load_prior_bounds(dataset_path: str) -> dict:
    """Load prior bounds from the .yaml file corresponding to the dataset.
    
    :param dataset_path: Path to the HDF5 dataset file.
    :return: Dictionary of prior bounds, or None if not found.
    """
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

    # ---------- data ---------------------------------------------------------
    p.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to the HDF5 dataset (required).",
    )
    p.add_argument("--batch_size", type=int, default=250)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--noise_f actor", type=float, default=1.0)
    p.add_argument("--cache_in_memory", action="store_true", default=True)

    # ---------- architecture -------------------------------------------------
    p.add_argument(
        "--architecture",
        type=str,
        default="conv",
        choices=DenoisingAutoencoder.VALID_ARCHITECTURES,
        help="Architecture type: 'conv' (pure conv, no skip connections) or 'unet' (with skip connections).",
    )
    p.add_argument("--n_channels", type=int, default=2,
                   help="Number of TDI channels (AE→2, AET→3).")
    p.add_argument("--n_freqs", type=int, default=4096,
                   help="Number of frequency bins per channel.")
    
    # Conv architecture params
    p.add_argument("--bottleneck_dim", type=int, default=128,
                   help="Bottleneck dimensionality (conv architecture only).")
    p.add_argument(
        "--hidden_channels",
        type=int,
        nargs="+",
        default=[32, 64, 128, 256, 256],
        help="Channel sizes for conv layers (conv architecture only).",
    )
    p.add_argument("--kernel_size", type=int, default=4,
                   help="Kernel size for conv layers (conv architecture only).")
    p.add_argument("--stride", type=int, default=2,
                   help="Stride for conv layers (conv architecture only).")
    p.add_argument("--dropout", type=float, default=0.0,
                   help="Dropout rate for regularization (0.0 to 0.5 recommended).")
    
    # Unet architecture params (legacy)
    p.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[16, 32, 64, 128, 256],
        help="Channel sizes for the Unet levels (unet architecture only).",
    )
    p.add_argument(
        "--down_sampling",
        type=int,
        nargs="+",
        default=[4, 8, 8, 8],
        help="Down-sampling factors for each encoder stage (unet architecture only).",
    )
    
    p.add_argument(
        "--representation",
        type=str,
        default="amp_phase",
        choices=DenoisingAutoencoder.VALID_REPRESENTATIONS,
        help="How to convert complex→real: 'amp_phase' or 'real_imag'.",
    )
    
    # High-frequency only mode (simpler task)
    p.add_argument("--high_freq_only", action="store_true", default=False,
                   help="Reconstruct only the high-frequency half (bins 2048:4096).")
    p.add_argument("--freq_split_idx", type=int, default=2048,
                   help="Frequency index to split at for high_freq_only mode.")

    # ---------- training hyper-params ----------------------------------------
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5,
                   help="Weight decay (L2 regularization) for AdamW.")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--scheduler_patience", type=int, default=10)
    p.add_argument("--scheduler_factor", type=float, default=0.3)
    p.add_argument("--early_stop_patience", type=int, default=50)
    p.add_argument("--gradient_clip_val", type=float, default=None,
                   help="Gradient clipping value (None to disable).")

    # ---------- misc ---------------------------------------------------------
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--log_name", type=str, default="autoencoder",
                   help="Name sub-folder under the TensorBoard log dir.")

    return p.parse_args()


def main():
    args = parse_args()

    # ---- resolve dataset path ------------------------------------------------
    dataset_path = args.dataset
    print(f"[train_autoencoder] Dataset : {dataset_path}")
    print(f"[train_autoencoder] Architecture : {args.architecture}")
    print(f"[train_autoencoder] Representation : {args.representation}")
    if args.architecture == "conv":
        print(f"[train_autoencoder] Bottleneck dim : {args.bottleneck_dim}")
        print(f"[train_autoencoder] Dropout : {args.dropout}")
    print(f"[train_autoencoder] High-freq only : {args.high_freq_only}")
    print(f"[train_autoencoder] Weight decay : {args.weight_decay}")
    print(f"[train_autoencoder] Device : {args.device}")

    # ---- load prior bounds from .yaml file -----------------------------------
    prior_bounds = load_prior_bounds(dataset_path)

    # ---- data module ---------------------------------------------------------
    data_module = MBHBDataModule(
        filename=dataset_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cache_in_memory=args.cache_in_memory,
        noise_factor=args.noise_factor,
    )
    data_module.setup(stage="fit")

    # ---- model ---------------------------------------------------------------
    model = DenoisingAutoencoder(
        n_channels=args.n_channels,
        n_freqs=args.n_freqs,
        architecture=args.architecture,
        # Conv architecture params
        bottleneck_dim=args.bottleneck_dim,
        hidden_channels=tuple(args.hidden_channels),
        kernel_size=args.kernel_size,
        stride=args.stride,
        dropout=args.dropout,
        # Unet architecture params
        sizes=tuple(args.sizes),
        down_sampling=tuple(args.down_sampling),
        # Training params
        lr=args.lr,
        weight_decay=args.weight_decay,
        scheduler_patience=args.scheduler_patience,
        scheduler_factor=args.scheduler_factor,
        representation=args.representation,
        # High-freq only mode
        high_freq_only=args.high_freq_only,
        freq_split_idx=args.freq_split_idx,
        # Prior bounds (for provenance tracking)
        prior_bounds=prior_bounds,
    )

    # Move to device for normalisation fitting
    model = model.to(args.device)

    # ---- fit normalisation from training split (clean signals) ---------------
    print("[train_autoencoder] Fitting normalisation statistics …")
    norm_loader = data_module.train_dataloader(shuffle=False, num_workers=0)
    model.fit_normalisation(norm_loader)

    # ---- callbacks -----------------------------------------------------------
    checkpoint_cb = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=2,
        filename="ae-{epoch:03d}-{val_loss:.4e}",
    )
    early_stop_cb = EarlyStopping(
        monitor="val_loss",
        patience=args.early_stop_patience,
        mode="min",
    )

    # ---- logger --------------------------------------------------------------
    logger = TensorBoardLogger(
        save_dir=os.path.join(DATA_ROOT_DIR, "logs"),
        name=args.log_name,
    )

    # ---- trainer -------------------------------------------------------------
    trainer = Trainer(
        logger=logger,
        max_epochs=args.epochs,
        accelerator=args.device,
        devices=1,
        enable_progress_bar=True,
        callbacks=[checkpoint_cb, early_stop_cb],
        gradient_clip_val=args.gradient_clip_val,
    )

    # ---- train ---------------------------------------------------------------
    trainer.fit(model, data_module)

    # ---- report best checkpoint ----------------------------------------------
    print(f"\n[train_autoencoder] Best checkpoint : {checkpoint_cb.best_model_path}")
    print(f"[train_autoencoder] Best val_loss   : {checkpoint_cb.best_model_score:.6e}")


if __name__ == "__main__":
    main()
