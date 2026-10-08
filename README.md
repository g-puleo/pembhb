# pembhb

Simulation-based inference for **massive black hole binaries (MBHBs) observed
by LISA**. `pembhb` estimates the posterior marginals of a single observed
event with **TMNRE** (Truncated Marginal Neural Ratio Estimation):

1. simulate signal + noise from a prior,
2. train a classifier per 1D / 2D marginal to tell true pairs (θ, d) from shuffled ones,
3. use the learned ratio at the observation to cut the prior to its high-posterior region,
4. repeat from step 1 inside the narrower prior.

Waveforms are frequency-domain IMRPhenomHM signals (with higher harmonics) from
[`bbhx`](https://github.com/mikekatz04/BBHx), projected onto the TDI A/E
channels. Noise is drawn from the LISA Sangria sensitivity in
[`lisaanalysistools`](https://github.com/mikekatz04/LISAanalysistools).

**Install:** see [INSTALL.md](INSTALL.md). It includes a required one-file patch to bbhx.

---

## Quick start

```bash
export PEMBHB_DATA_DIR=/scratch/$USER/pembhb     # simulations, logs, checkpoints
export PEMBHB_PLOTS_DIR=/scratch/$USER/pembhb/plots

# 1. one noisy observation at the injection in configs/datagen_config.yaml
python scripts/simulate_data.py --n 1 --injection --store-noise \
    --fname $PEMBHB_DATA_DIR/obs.h5 --seed 0

# 2. TMNRE, 10 rounds
python scripts/tmnre_joint.py --obs-path $PEMBHB_DATA_DIR/obs.h5 --n_rounds 10 my_run

# 3. inspect (RUN is printed at start-up, e.g. 2026/10/08/channelizedmlp_my_run)
./scripts/plot_posterior.py --run-name $RUN --proposal      # last round's posterior
python scripts/visualise_truncation_rounds.py $RUN          # evolution across rounds
python scripts/visualise_sky_truncation.py    $RUN
python scripts/visualise_entropy_evolution.py $RUN
python scripts/visualise_volume_from_masks.py $RUN
```

---

## 1. Choose the frequency grid and the injection: `configs/datagen_config.yaml`

| block | what it sets |
|---|---|
| `waveform_params` | TDI channels, harmonics, noise model, observation length (`duration`, weeks), frequency band `fmin`–`fmax`, grid spacing (`linear`, or `log` with `n_freq_bins`) |
| `prior` | the round-1 training prior: a uniform box over the 11 parameters (distance uniform in volume) |
| `injection` | the true parameters of the observation |
| `spin_param_basis` | `chieff_chidiff` (sample χ_eff, χ_diff) or `chi1chi2` |
| `backend` | `cuda12x` or `cpu` |

The 11 parameters, in canonical order: `logMchirp` (log₁₀ M_c / M☉), `q`
(m₁/m₂ ≥ 1), two spins, `dist` (Gpc), `phi`, `inc` (cos ι), `lambda`
(ecliptic longitude), `beta` (sin of ecliptic latitude), `psi`, `Deltat`
(merger time relative to the end of the observation, days).

**The grid is linear by default.** Bin spacing is `1 / T_obs` (`downsamplefactor` must stay 1: coarser grids are not supported).
Bins falling inside the TDI transfer-function nulls (where the PSD collapses
to ≈0) are **removed from the grid** automatically, and per-bin widths
`df` bridge the gaps. With `fmax ≤ 0.02 Hz` no bin is removed. `fmax` must not
exceed 0.1 Hz.

The observation and the training data **must share `waveform_params`**.
`tmnre_joint.py` compares them against the observation's YAML sidecar and
aborts on any mismatch.

## 2. Make the observation: `scripts/simulate_data.py`

```bash
python scripts/simulate_data.py --n 1 --injection --store-noise --fname obs.h5 --seed 0
```

- `--injection` pins every parameter to the `injection` block. Without it the script samples `prior`.
- `--store-noise` writes one fixed noise realisation (`noise_fd`, seeded by `--noise-seed`).
  **This is mandatory for TMNRE.** Without stored noise every forward pass would
  see a fresh noise draw, so the truncation would not be reproducible.
- The script also writes a sidecar `obs.yaml` holding the full config used.

`scripts/add_noise_to_obs.py --input obs.h5 --output obs_withnoise.h5` adds a
seeded `noise_fd` to a file simulated without `--store-noise`.

## 3. Configure the network and training: `configs/train_config.yaml`

| block | what it sets |
|---|---|
| `marginals.f` | one classifier head per entry, by parameter name: `[logMchirp]` is a 1D marginal, `[lambda, beta]` the 2D sky marginal. A parameter may appear in only one marginal |
| `architecture.data_summary` | compressor of the whitened FD data shared by all heads (default `ChannelizedMLP`; alternatives `Autoencoder`, `ROM`, …) |
| `batch_size`, `epochs`, `joint_training`, `classifier_*` | optimisation |
| `n_train_noise_realisations` | noise realisations drawn per waveform per batch. Noise is regenerated on the fly, so each epoch sees new noise |
| `fisher_prior` | optionally replace the round-1 prior by a Fisher-matrix box around the injection |
| `pp_ks_early_stop`, `volume_ratio_early_stop` | per-round stopping criteria (coverage test; posterior volume stopped shrinking) |
| `streaming` | simulate **during** training (below) |
| `truncation` | how the prior is cut between rounds (below) |
| `precision` | `float32` or `float64` |

### Streaming data generation

With `streaming.enabled: true`, a background producer keeps simulating with
bbhx while the network trains, so training never runs on a fixed dataset.

- `storage: gpu`: a ring of `n_buffers` chunks held in VRAM. This is the fastest option and needs GPU memory.
- `storage: disk`: a ring of `n_buffers` HDF5 files of `buffer_size` waveforms,
  stored under `PEMBHB_DATA_DIR/<run>/stream_buffers` (set `buffer_dir` to override).
  An epoch reads `samples_per_epoch / buffer_size` whole files. A file is only
  overwritten after it has been read `reuse_threshold` times. Disk storage uses
  almost no VRAM.

In both modes the producer samples from the **current truncated region** (the
mask, not its bounding box). The validation pool is generated once per round, in
batches of `gen_batch_size`.

With `streaming.enabled: false`, each round simulates a fixed 50 000-sample HDF5
file and splits it 70/25/5 into train/validation/test.

### Truncation

`truncation.mode: mask` (recommended) works like this. The NRE posterior at the
observation is evaluated on a grid for each marginal. The highest-posterior-density
region at `credible_level_1d` / `credible_level_2d` is extracted, then split
into connected modes; sky periodicity in λ is handled. The next round samples
the union of those modes by rejection.

- `refine: true` re-evaluates every mode on its own subgrid (`ngrid_*_refined`),
  so once a mode shrinks to a few coarse pixels the resolution no longer limits it.
- `mode_tree: true` (experimental) takes the mode structure from the previous
  round and splits it recursively. It requires `refine: true`.
- `zero_outside_prev_mask` only trusts the network where it was trained, i.e.
  inside the previous round's region.

`mode: rectangle` keeps the legacy behaviour, which shrinks a box per parameter.

## 4. Train: `scripts/tmnre_joint.py`

```bash
python scripts/tmnre_joint.py --obs-path obs.h5 [--train-config train_config.yaml] \
    [--datagen-config datagen_config.yaml] [--n_rounds 10] [--seed 42] NAME
```

Config names are resolved inside `configs/`. Each run is tagged
`YYYY/MM/DD/<data_summary type>_<NAME>`, and every output is keyed by that tag:

| path | contents |
|---|---|
| `$PEMBHB_DATA_DIR/<tag>/prior_after_round_<i>.yaml` | truncated prior (box + 1D intervals) after round `i` |
| `$PEMBHB_DATA_DIR/<tag>/truncation_round_<i>.npz` | the accepted masks / per-mode subgrids |
| `$PEMBHB_DATA_DIR/<tag>/observation_used.yaml` | which observation file was used |
| `$PEMBHB_DATA_DIR/logs/<tag>/round_<i>/version_<N>/` | TensorBoard scalars, `hparams.yaml`, checkpoints |
| `$PEMBHB_PLOTS_DIR/<tag>/` | posterior snapshots during training, PP / coverage plots |

To resume an interrupted run from its last completed round:

```bash
python scripts/tmnre_joint.py --obs-path obs.h5 --resume <tag> --n_rounds 5 ignored
```

If a later round started but did not finish, move its
`logs/<tag>/round_<i>` directory out of the way first.

## 5. Inspect the results

All scripts take the run tag, read the observation recorded in
`observation_used.yaml`, and write to `$PEMBHB_PLOTS_DIR/<tag>/`. Passing a
different observation (`--obs-path` / `--data-path`) raises an error.

**One round: `plot_posterior.py`** (≈20 s on a GPU)

```bash
./scripts/plot_posterior.py --run-name <tag> [--round N] [--mcmc-file mcmc.h5] [--proposal]
```

The figures go to `posterior_round_<N>/` (default: the last round):

- `posterior_1d`: every 1D marginal, optionally against MCMC;
- `sky_zoom` (+ `_lonlat`): the sky posterior re-evaluated on a fine grid over its credible region;
- `modes_<param>`: written when the round's truncation found several modes and
  `truncation.refine` was on. Each mode is plotted on its own refined grid;
  1D panels give each mode's posterior mass.

`--proposal` overlays what this posterior proposed for the next round: interval
bounds in 1D, and the accepted-region contour in 2D.

**Evolution across rounds**

| script | figure |
|---|---|
| `visualise_truncation_rounds.py <tag> [--mcmc-file …]` | 1D credible bands of every marginal vs round (uses `plot_posterior.py`'s evaluation) |
| `visualise_sky_truncation.py <tag> [--mcmc-file …]` | sky posterior and truncation masks per round (Mollweide), 90% sky area vs round |
| `visualise_entropy_evolution.py <tag>` | differential entropy of each marginal vs round (and vs training time) |
| `visualise_volume_from_masks.py <tag>` | prior volume of each marginal vs round, measured exactly on every mode's own grid from the stored truncation masks |

`--last-round N` restricts the evolution scripts to rounds ≤ N.

## Tests

```bash
pytest
```

The suite runs on the CPU with no data files. It covers the simulator and the
noise colouring, the collate and noise augmentation, the region / mask /
refinement geometry, the volume ratios, GPU and disk streaming, and the
LR schedules.

## Layout

```
configs/      datagen_config.yaml, train_config.yaml
scripts/      simulate_data.py, add_noise_to_obs.py, tmnre_joint.py, plot_posterior.py, visualise_*.py
src/pembhb/   simulator, sampler, data, model, autoencoder, callbacks,
              regions / mask_truncation / sky_truncation, streaming(_disk), psd_veto, utils
patches/      bbhx compatibility patch
test/         pytest suite (+ test/configs/datagen_test.yaml)
```
