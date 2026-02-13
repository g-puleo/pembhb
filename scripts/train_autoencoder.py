#!/usr/bin/env python
"""
Train a DenoisingAutoencoder on frequency-domain MBHB data.

Usage
-----
    python scripts/train_autoencoder.py                              # defaults
    python scripts/train_autoencoder.py --dataset /path/to/data.h5   # custom dataset
    python scripts/train_autoencoder.py --representation real_imag   # real/imag channels
    python scripts/train_autoencoder.py --epochs 200 --lr 5e-4       # override hyper-params

All arguments are optional and have sensible defaults drawn from the existing
project configuration files.
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


def parse_args():
    p = argparse.ArgumentParser(description="Train a Unet denoising autoencoder.")

    # ---------- data ---------------------------------------------------------
    p.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to the HDF5 dataset.  If not given, uses the first "
             "simulation_round_*.h5 found under DATA_ROOT_DIR.",
    )
    p.add_argument("--batch_size", type=int, default=250)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--noise_factor", type=float, default=1.0)
    p.add_argument("--cache_in_memory", action="store_true", default=True)

    # ---------- architecture -------------------------------------------------
    p.add_argument("--n_channels", type=int, default=2,
                   help="Number of TDI channels (AE→2, AET→3).")
    p.add_argument("--n_freqs", type=int, default=4096,
                   help="Number of frequency bins per channel.")
    p.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[16, 32, 64, 128, 256],
        help="Channel sizes for the Unet levels.",
    )
    p.add_argument(
        "--down_sampling",
        type=int,
        nargs="+",
        default=[4, 8, 8, 8],
        help="Down-sampling factors for each encoder stage.",
    )
    p.add_argument(
        "--representation",
        type=str,
        default="amp_phase",
        choices=DenoisingAutoencoder.VALID_REPRESENTATIONS,
        help="How to convert complex→real: 'amp_phase' or 'real_imag'.",
    )

    # ---------- training hyper-params ----------------------------------------
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--scheduler_patience", type=int, default=10)
    p.add_argument("--scheduler_factor", type=float, default=0.3)
    p.add_argument("--early_stop_patience", type=int, default=50)

    # ---------- misc ---------------------------------------------------------
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--log_name", type=str, default="autoencoder",
                   help="Name sub-folder under the TensorBoard log dir.")

    return p.parse_args()


def find_default_dataset() -> str:
    """Look for a simulation HDF5 file under DATA_ROOT_DIR."""
    from glob import glob

    candidates = sorted(glob(os.path.join(DATA_ROOT_DIR, "**", "simulation_round_*.h5"), recursive=True))
    if candidates:
        return candidates[0]
    # Fall back to any .h5 file that starts with 'observation'
    candidates = sorted(glob(os.path.join(DATA_ROOT_DIR, "observation*.h5")))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(
        f"No HDF5 dataset found under {DATA_ROOT_DIR}.  "
        "Pass --dataset explicitly."
    )


def main():
    args = parse_args()

    # ---- resolve dataset path ------------------------------------------------
    dataset_path = args.dataset or find_default_dataset()
    print(f"[train_autoencoder] Dataset : {dataset_path}")
    print(f"[train_autoencoder] Representation : {args.representation}")
    print(f"[train_autoencoder] Device : {args.device}")

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
        sizes=tuple(args.sizes),
        down_sampling=tuple(args.down_sampling),
        lr=args.lr,
        scheduler_patience=args.scheduler_patience,
        scheduler_factor=args.scheduler_factor,
        representation=args.representation,
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
    )

    # ---- train ---------------------------------------------------------------
    trainer.fit(model, data_module)

    # ---- report best checkpoint ----------------------------------------------
    print(f"\n[train_autoencoder] Best checkpoint : {checkpoint_cb.best_model_path}")
    print(f"[train_autoencoder] Best val_loss   : {checkpoint_cb.best_model_score:.6e}")


if __name__ == "__main__":
    main()
