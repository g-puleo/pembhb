# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Dialogue with the user

You should, whenever possible, guide the user in implementing new functionalities by themselves, without doing too much work without their full understanding. 
You should be socratic and have a dialogue which lets them understand fully what is being modified and why, especially if the changes are substantial, conceptual and not just minor details (such as plots, animations, changes of one line). This means that you should probably have a plan of what to do in mind, and yield it to the user one-step-at-the-time: long output paragraphs require long revision, and beg lots of questions, and the user might feel overwhelmed and tempted to skim through it without attention. 
If necessary, you should try to remind the users of the risks of letting an agent do the whole thing for you. For example, doing so might lead the code to be out of control and the user to be not aware about the inner workings. 

## Project Overview

`pembhb` is a **Simulation-Based Inference (SBI)** framework for Bayesian parameter estimation of Massive Black Hole Binaries (MBHBs) observed by the LISA space detector. It implements **TMNRE** (Truncated Marginal Neural Ratio Estimation), a likelihood-free inference method that trains neural classifiers to estimate posterior marginals.

`bbhx` and `lisaanalysistools` appear as git submodules but are installed via pip — ignore the submodule directories entirely.

## Commands

```bash
# Install
pip install -e .

# Run all tests
pytest

# Run a single test file
pytest test/test_simulator.py

# Run a single test
pytest test/test_simulator.py::test_name
```

## SBI Workflow Architecture

The pipeline follows: **Simulator → Data → Model → TMNRE**

### 1. Simulator (`src/pembhb/simulator.py`)

`MBHBSimulatorFD_TD` generates synthetic MBHB gravitational wave signals. It takes the 11 physical parameters (log chirp mass, mass ratio, two spins, distance, phase, inclination, ecliptic longitude/latitude, polarization, merger time offset `Deltat`), calls `bbhx` for frequency-domain waveforms, adds realistic LISA noise using the Sangria noise model, and returns both frequency-domain (FD) and time-domain (TD) representations across TDI channels (AET).

Key methods: `sample()` draws from the prior and generates one datum; `sample_and_store()` batches this to HDF5.

### 2. Data (`src/pembhb/data.py`)

`MBHBDataset` / `MBHBDataModule` wrap HDF5 files for PyTorch Lightning. The collate function (`mbhb_collate_fn`) supports **noise shuffling** at training time (each sample gets a randomly drawn noise instance from the batch rather than the fixed noise it was generated with), which acts as data augmentation and decorrelates the network from specific noise realizations.

### 3. Data Summarizer / Compressor (`src/pembhb/model.py`, `src/pembhb/rom.py`, `src/pembhb/autoencoder.py`)

The inference network requires a `data_summarizer` to compress high-dimensional waveform data before passing it to the classifier heads. Available options (registered in `DATA_SUMMARY_REGISTRY`):

| Key | Class | Description |
|-----|-------|-------------|
| `"ROM"` | `ROMWrapper` | Reduced Order Model — greedy Gram-Schmidt basis (~50 elements), whitened inner product `⟨a\|b⟩ = Re[Σ a*b·4Δf]`. Basis built offline from training set. |
| `"Autoencoder"` | `AutoencoderWrapper` | Convolutional denoising autoencoder. Encoder frozen at inference time; bottleneck (~256-dim) is the summary. Architectures: `"conv"` (plain) or `"unet"` (with skip connections). |
| `"BrutalCompression"` | `BrutalCompression` | Simple frequency-bin selection. |
| `"PeregrineModel"` | `PeregrineModel` | Peregrine-style compressor. |

### 4. Inference Network (`src/pembhb/model.py`)

`InferenceNetwork` is a PyTorch Lightning module implementing TMNRE:

- **Input:** compressed data summary + parameter values
- **Training trick:** within each batch, parameters are "rolled" (permuted) to create a 50/50 split of true joint samples (label=1) and marginal product samples (label=0). Loss is `BCEWithLogitsLoss`.
- **Output heads:** `MarginalClassifierHead` — one shallow MLP per marginal (1D or 2D). Each head only receives the relevant parameter subset concatenated with the data summary.
- **Marginals config:** defined in `train_config.yaml` as a dict mapping domain (`f`, `t`, `ft`) to lists of 1D/2D marginals.
- Optional **GradNorm** for multi-task loss balancing.

`JointAEInferenceNetwork` trains the autoencoder encoder and the NRE heads simultaneously (with an optional AE-only warmup phase for round 1).

### 5. TMNRE Scripts

**`scripts/tmnre.py`** — Sequential pipeline (separate AE + NRE training):
```
for each round:
    _generate_data()           # simulator → HDF5
    _train_autoencoder()       # fit DenoisingAutoencoder
    _train_inference_network() # fit InferenceNetwork with frozen AE encoder
    _plot_updated_prior_bounds()
```

**`scripts/tmnre_joint.py`** — Joint training (AE + NRE in single trainer, the primary pipeline). See [CLI section](#tmnre_jointpy-cli) below.

Both scripts implement **prior truncation**: after each round the prior bounds are narrowed using the estimated posterior, focusing subsequent simulation on the high-probability region.

## `tmnre_joint.py` CLI

The primary training entry point. Run from the repo root:

```bash
/data/gpuleo/envs/lisa_pip/bin/python scripts/tmnre_joint.py \
    [--train-config FILENAME] \   # default: train_config.yaml (inside configs/)
    [--n_rounds N]            \   # default: 10
    --obs-path /path/to/obs_*_withnoise.h5 \   # MUST contain stored noise_fd
    NAME                          # unique run name (e.g. joint_v1)
```

### Observation file: stored noise is **mandatory**

`--obs-path` must point to an HDF5 with a stored `noise_fd` dataset. The
script **aborts immediately** if it doesn't (assertion in
`SequentialTrainerJoint.__init__`). The reason: when `noise_fd` is absent
the collate function (`mbhb_collate_fn`, `utils.py:1260-1262`) draws a
fresh `torch.randn` realisation on **every** forward pass, including
training, posterior eval, the PP-KS test set, and the round-end truncation
read. **Truncation under fresh noise is non-reproducible** — the proposal
window drifts with each evaluation, defeating the purpose of TMNRE.

To produce a usable obs file:

```bash
python scripts/add_noise_to_obs.py /path/to/obs_*.h5
# → writes obs_*_withnoise.h5 (same waveform, with stored noise_fd)
```

The obs path used for each run is auto-logged to
`{DATA_ROOT_DIR}/{TIME_OF_EXECUTION}/observation_used.yaml` so it can be
recovered later without parsing stdout.

To **resume** a previous run (auto-detects last completed round from checkpoints):

```bash
/data/gpuleo/envs/lisa_pip/bin/python scripts/tmnre_joint.py \
    --resume 20260331_autoencoder_joint_v1 \
    [--n_rounds N] \
    NAME   # ignored when --resume is used
```

### Run Naming (`TIME_OF_EXECUTION`)

Every fresh run builds a path-style tag:

```
YYYY/MM/DD/{ds_type}_{name}
```

where `ds_type` = `architecture.data_summary.type` from `train_config.yaml` (lowercased, e.g. `autoencoder`).
Example: `2026/03/31/autoencoder_joint_v1`

This tag (with literal `/` separators) is the directory key for all outputs (data, logs, plots) — it expands to a nested `YYYY/MM/DD/...` directory tree on disk.

**Legacy layout.** Older runs used a flat tag `YYYYMMDD_{ds_type}_{name}` (e.g. `20260331_autoencoder_joint_v1`) and a flat per-round log dir `{name}_round_{i}/`. Tools that read existing runs should support both conventions — see `find_round_dirs` in `scripts/visualise_truncation_rounds.py` for the canonical fallback pattern.

## Configuration

Two YAML config files control everything:

- `configs/datagen_config.yaml` — waveform channels, noise model, duration/dt, prior bounds for all 11 parameters, hardware backend (`cuda12x` or `cpu`)
- `configs/train_config.yaml` — batch size, epochs, learning rate, precision (`float32`/`float64`), data summary type and architecture, marginals specification, Fisher prior, early stopping

## Training Output & Nomenclature

All outputs are keyed by `TIME_OF_EXECUTION`. Two root directories:
- `DATA_ROOT_DIR` = `/data/gpuleo/mbhb/`
- `ROOT_DIR` = repo root (`/u/g/gpuleo/pembhb/`)

### Simulation data — `DATA_ROOT_DIR/{TIME_OF_EXECUTION}/`

| File | Contents |
|------|----------|
| `simulation_round_{i}.h5` | 50 000-sample HDF5 dataset for round `i` |
| `simulation_round_{i}.yaml` | Sidecar: prior bounds + datagen settings used to generate it |
| `prior_after_round_{i}.yaml` | Truncated prior saved after round `i` (read by `--resume`) |
| `fisher_prior_round_1.yaml` | Fisher-matrix prior for round 1 (only if Fisher prior enabled) |

### TensorBoard logs — `DATA_ROOT_DIR/logs/{TIME_OF_EXECUTION}/round_{i}/version_{N}/`

Nested layout: the round index is a sub-directory of the run tag.
Legacy layout (old runs): `DATA_ROOT_DIR/logs/{TIME_OF_EXECUTION}_round_{i}/version_{N}/` (flat).

Lightning creates a new `version_N` subdirectory each time the trainer starts. The **last** `version_*` is always the relevant one. `tmnre_joint.py` also writes a final `truncation.ckpt` one level above `checkpoints/` — prefer it when present.

| Path | Contents |
|------|----------|
| `events.out.tfevents.*` | Scalars: `train_loss`, `val_loss`, `val_accuracy`, per-head losses |
| `hparams.yaml` | Hyperparameter snapshot |
| `checkpoints/epoch=*.ckpt` | Best checkpoint (monitored: `val_loss`) — loaded by `--resume` |
| `simulation_round_{i}.yaml` | Copy of the datagen sidecar |

### Plots — `ROOT_DIR/plots/{TIME_OF_EXECUTION}/`

| File pattern | Contents |
|--------------|----------|
| `posterior_round_{i}_epoch_{e}_{param}.pdf` | 1D marginal posterior for `param`, saved every 2 epochs |
| `posterior_round_{i}_epoch_{e}_{p1}_{p2}.pdf` | 2D marginal posterior for parameter pair, every 2 epochs |
| `prior_bounds_iteration_{N}.png` | logMchirp & q prior-bound evolution after round `N` |
| `round_{i}_{key}_pp_plot.png` | 1D PP (coverage) plot; `key` ∈ {`f`, `t`, `ft`}, produced after each round |
| `round_{i}_{key}_pp_plot_2d.png` | 2D PP plot |

## Precision System

A global registry in `src/pembhb/__init__.py` controls numerical precision throughout:
```python
from pembhb import set_precision, get_torch_dtype, get_numpy_dtype
set_precision("float64")  # or "float32"
```
All modules query this on instantiation.

## Key Conventions

- **HDF5 layout:** waveforms stored at positive frequencies only; full two-sided FFT is reconstructed on-the-fly for TD synthesis.
- **Noise weighting:** GW inner product `⟨a|b⟩ = Re[Σ aₖ* bₖ · 4Δfₖ / Sₙ(fₖ)]` used in ROM; ASD stored per-sample in HDF5.
- **Parameter ordering:** `_ORDERED_PRIOR_KEYS` in `simulator.py` defines the canonical 11-parameter order used throughout.
- **Train/val/test split:** 70/25/5 (hardcoded in `MBHBDataModule`).
- Data files and logs are written to `/data/gpuleo/mbhb/` (external path configured in scripts).
