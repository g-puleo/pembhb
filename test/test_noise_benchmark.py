"""Benchmark: loading pre-computed noise from disk vs. generating noise on the fly.

Run directly to get a printed report:

    python test/test_noise_benchmark.py

Or via pytest (skips the full report, just checks the pipeline runs):

    pytest test/test_noise_benchmark.py
"""

import os
import time
import tempfile

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from pembhb.data import MBHBDataset
from pembhb.utils import mbhb_collate_fn


# ---------------------------------------------------------------------------
# Helpers to build synthetic HDF5 fixtures
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(0)

# Small realistic-ish shapes
N_SAMPLES  = 500
N_CHANNELS = 2
N_FREQS    = 4096
N_PARAMS   = 11
FMIN       = 1e-4     # Hz  (above the 5e-5 Hz high-pass cutoff)
FMAX       = 0.1      # Hz


def _make_freqs():
    return np.linspace(FMIN, FMAX, N_FREQS).astype(np.float32)


def _make_asd(freqs):
    """Toy ASD that increases with frequency (shape: n_channels × n_freqs)."""
    base = 1e-20 * (freqs / freqs[0]) ** 0.5
    return np.tile(base, (N_CHANNELS, 1)).astype(np.float32)


def _make_df(freqs):
    return float(freqs[1] - freqs[0])


def _compute_noise_scale(asd, freqs, df):
    """filtered_asd / sqrt(4 * df)  — the noise colouring factor."""
    filtered = asd.copy()
    filtered[:, freqs < 5e-5] = 0.0
    return filtered / np.sqrt(4.0 * df)


def _generate_noise(asd, freqs, df, n):
    noise_scale = _compute_noise_scale(asd, freqs, df)
    z = (_RNG.standard_normal((n, N_CHANNELS, N_FREQS))
         + 1j * _RNG.standard_normal((n, N_CHANNELS, N_FREQS)))
    return (z * noise_scale[None]).astype(np.complex64)


def build_hdf5_with_noise(path):
    """Old-style HDF5: waveforms + pre-generated noise."""
    freqs = _make_freqs()
    asd   = _make_asd(freqs)
    df    = _make_df(freqs)

    with h5py.File(path, "w") as f:
        f.create_dataset("frequencies", data=freqs)
        f.create_dataset("df",          data=np.float32(df))
        f.create_dataset("asd",         data=asd)
        f.create_dataset("source_parameters",
                         data=_RNG.standard_normal((N_SAMPLES, N_PARAMS)).astype(np.float32))
        f.create_dataset("snr",
                         data=_RNG.uniform(5, 100, N_SAMPLES).astype(np.float32))

        wave_ds  = f.create_dataset("wave_fd",  shape=(N_SAMPLES, N_CHANNELS, N_FREQS), dtype=np.complex64)
        noise_ds = f.create_dataset("noise_fd", shape=(N_SAMPLES, N_CHANNELS, N_FREQS), dtype=np.complex64)
        for start in range(0, N_SAMPLES, 100):
            end = min(start + 100, N_SAMPLES)
            wave_ds[start:end]  = _RNG.standard_normal((end-start, N_CHANNELS, N_FREQS)).astype(np.complex64)
            noise_ds[start:end] = _generate_noise(asd, freqs, df, end-start)


def build_hdf5_without_noise(path):
    """New-style HDF5: waveforms only, no noise stored."""
    freqs = _make_freqs()
    asd   = _make_asd(freqs)
    df    = _make_df(freqs)

    with h5py.File(path, "w") as f:
        f.create_dataset("frequencies", data=freqs)
        f.create_dataset("df",          data=np.float32(df))
        f.create_dataset("asd",         data=asd)
        f.create_dataset("source_parameters",
                         data=_RNG.standard_normal((N_SAMPLES, N_PARAMS)).astype(np.float32))
        f.create_dataset("snr",
                         data=_RNG.uniform(5, 100, N_SAMPLES).astype(np.float32))

        wave_ds = f.create_dataset("wave_fd", shape=(N_SAMPLES, N_CHANNELS, N_FREQS), dtype=np.complex64)
        for start in range(0, N_SAMPLES, 100):
            end = min(start + 100, N_SAMPLES)
            wave_ds[start:end] = _RNG.standard_normal((end-start, N_CHANNELS, N_FREQS)).astype(np.complex64)


# ---------------------------------------------------------------------------
# Old-style collate: loads noise from disk  (replicates original behaviour)
# ---------------------------------------------------------------------------

def _old_collate_fn(batch, subset, noise_factor=1.0, noise_shuffling=True):
    """Original collate that reads pre-stored noise from disk."""
    B = len(batch)
    wave_fd = torch.stack([b["wave_fd"] for b in batch])
    params  = torch.stack([b["params"]  for b in batch])

    if noise_shuffling:
        subset_idxs = torch.tensor(subset.indices)
        pick = subset_idxs[torch.randint(0, len(subset_idxs), (B,))]
    else:
        pick = torch.tensor([b["idx"] for b in batch])

    noise_fd = noise_factor * torch.stack([subset.dataset._load("noise_fd", int(i)) for i in pick])
    return {"source_parameters": params, "wave_fd": wave_fd, "noise_fd": noise_fd}


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def _time_dataloader(loader, n_batches=20):
    """Drain up to n_batches and return elapsed wall-clock seconds."""
    t0 = time.perf_counter()
    for k, _ in enumerate(loader):
        if k + 1 >= n_batches:
            break
    return time.perf_counter() - t0


def run_benchmark(batch_size=64, n_batches=20, num_workers=0):
    with tempfile.TemporaryDirectory() as tmpdir:
        path_with    = os.path.join(tmpdir, "with_noise.h5")
        path_without = os.path.join(tmpdir, "without_noise.h5")

        print("Building HDF5 fixtures …")
        build_hdf5_with_noise(path_with)
        build_hdf5_without_noise(path_without)

        # ── Dataset A: noise on disk ────────────────────────────────────────
        ds_disk = MBHBDataset(path_with, cache_in_memory=False)
        subset_disk = Subset(ds_disk, list(range(len(ds_disk))))

        loader_disk = DataLoader(
            subset_disk,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=lambda b: _old_collate_fn(b, subset_disk, noise_shuffling=True),
        )

        # ── Dataset B: on-the-fly noise (no noise in file) ─────────────────
        ds_otf = MBHBDataset(path_without, cache_in_memory=False)
        subset_otf = Subset(ds_otf, list(range(len(ds_otf))))
        noise_scale = ds_otf.noise_scale  # pre-computed (n_channels, n_freqs)

        loader_otf = DataLoader(
            subset_otf,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=lambda b: mbhb_collate_fn(b, noise_scale, 1.0),
        )

        # ── Warm-up: one pass each ──────────────────────────────────────────
        for _ in loader_disk:
            break
        for _ in loader_otf:
            break

        # ── Timed passes ───────────────────────────────────────────────────
        t_disk = _time_dataloader(loader_disk, n_batches)
        t_otf  = _time_dataloader(loader_otf,  n_batches)

        speedup = t_disk / t_otf if t_otf > 0 else float("inf")

        print()
        print("=" * 60)
        print("  Noise loading benchmark")
        print(f"  Samples: {N_SAMPLES}  |  Channels: {N_CHANNELS}  |  Freqs: {N_FREQS}")
        print(f"  Batch size: {batch_size}  |  Batches timed: {n_batches}  |  Workers: {num_workers}")
        print("=" * 60)
        print(f"  Disk (pre-computed noise)  : {t_disk:.3f} s  ({t_disk/n_batches*1000:.1f} ms/batch)")
        print(f"  On-the-fly (runtime noise) : {t_otf:.3f} s  ({t_otf/n_batches*1000:.1f} ms/batch)")
        print(f"  Speed-up (disk / on-the-fly): {speedup:.2f}x")
        print("=" * 60)

        return {"disk_s": t_disk, "otf_s": t_otf, "speedup": speedup}


# ---------------------------------------------------------------------------
# pytest entry-points
# ---------------------------------------------------------------------------

def test_on_the_fly_noise_shape():
    """Smoke-test: on-the-fly noise has the right shape and dtype."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.h5")
        build_hdf5_without_noise(path)
        ds = MBHBDataset(path, cache_in_memory=False)
        assert ds.noise_scale is not None
        assert ds.noise_scale.shape == (N_CHANNELS, N_FREQS)

        subset = Subset(ds, list(range(min(32, len(ds)))))
        loader = DataLoader(
            subset, batch_size=16, shuffle=False,
            collate_fn=lambda b: mbhb_collate_fn(b, ds.noise_scale, 1.0),
        )
        batch = next(iter(loader))
        assert batch["noise_fd"].shape == (16, N_CHANNELS, N_FREQS)
        assert batch["noise_fd"].dtype == torch.complex64


def test_noise_statistics_match():
    """On-the-fly noise variance should match the expected ASD-coloured distribution."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.h5")
        build_hdf5_without_noise(path)
        ds = MBHBDataset(path, cache_in_memory=False)
        noise_scale = ds.noise_scale  # (C, F)

        # Generate a large batch to get stable statistics
        subset = Subset(ds, list(range(len(ds))))
        loader = DataLoader(
            subset, batch_size=len(ds), shuffle=False,
            collate_fn=lambda b: mbhb_collate_fn(b, noise_scale, 1.0),
        )
        batch = next(iter(loader))
        noise = batch["noise_fd"]  # (N, C, F)

        # Expected std of real part per bin: noise_scale[c, f]
        # (since Re[z * s] ~ N(0, s^2) for unit complex Gaussian z)
        std_real = noise.real.std(dim=0)           # (C, F)
        std_expected = noise_scale                 # (C, F)

        # Allow 10 % relative tolerance (finite sample noise)
        nonzero = std_expected > 0
        rel_err = ((std_real[nonzero] - std_expected[nonzero]).abs()
                   / std_expected[nonzero])
        assert rel_err.mean().item() < 0.10, (
            f"Mean relative error in noise std: {rel_err.mean().item():.3f} (expected < 0.10)"
        )


def test_benchmark_runs():
    """Benchmark completes without error and on-the-fly is not catastrophically slower."""
    result = run_benchmark(batch_size=32, n_batches=5, num_workers=0)
    # On-the-fly should not be more than 10× slower than reading from disk
    assert result["speedup"] > 0.1, (
        f"On-the-fly noise is unexpectedly much slower: speedup={result['speedup']:.2f}"
    )


if __name__ == "__main__":
    run_benchmark(batch_size=64, n_batches=30, num_workers=0)
