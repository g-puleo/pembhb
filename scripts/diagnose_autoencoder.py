"""Diagnostics for a trained DenoisingAutoencoder.

Two modes:

* ``--checkpoint <path>.ckpt`` — single shot. Works for both
  ``train_autoencoder.py`` checkpoints (a bare ``DenoisingAutoencoder``)
  and ``tmnre_joint.py`` checkpoints (a ``JointAEInferenceNetwork`` —
  only its ``encoder_model`` is used; the NRE classifier is ignored).
* ``--run-name NAME --round N`` — locate a ``tmnre_joint.py`` run by
  its ``TIME_OF_EXECUTION`` tag and pull round ``N``. Both layouts are
  supported:
    - new nested:  ``{DATA_ROOT_DIR}/logs/{NAME}/round_{N}/version_*/``
                   (``NAME`` = ``"YYYY/MM/DD/{ds_type}_{run}"``)
    - legacy flat: ``{DATA_ROOT_DIR}/logs/{NAME}_round_{N}/version_*/``
                   (``NAME`` = ``"YYYYMMDD_{ds_type}_{run}"``)
  The latest ``version_*`` is used. ``truncation.ckpt`` is preferred
  when present; otherwise the last ``checkpoints/*.ckpt``.

Produces three plots in
``plots/{run_name}/diagnose_ae_round_{N}/`` (nested) or
``plots/diagnose_ae_{run_name}_round_{N}/`` (legacy flat) or
``plots/diagnose_ae_{ckpt_basename}/`` (``--checkpoint``):

  1. ``1_overlay_amplitude.png``
     |target| vs |reconstruction|, two channels × two amplitude spaces:
       * network input/output space (= ``ae.preprocess`` output; its absolute
         scale depends on ``whiten`` / ``amplitude_normalise``)
       * physical Hz⁻¹ amplitude (``preprocess`` fully inverted) — invariant
         across runs that differ only in the choice of normalisation flags.

  2. ``2_residual_histogram.png``
     Histograms of ``data - waveform`` and ``denoised - waveform``, pooled
     over events/channels/Re/Im/freq, compared to N(0, 1). Whitening is
     applied explicitly using the dataset's ``noise_scale = ASD·√(T/4)``
     so the N(0, 1) comparison is valid for **both** ``whiten=True`` and
     ``whiten=False`` runs.

  3. ``3_per_frequency_loss.png``
     Per-frequency MSE distribution across the evaluated samples, with
     median, mean, p5–p95 and p25–p75 bands. Computed on the network-space
     tensors (matches the training MSE).

Run:

    /data/gpuleo/envs/lisa_pip/bin/python scripts/diagnose_autoencoder.py \\
        --checkpoint <path>.ckpt \\
        --dataset    <path>.h5 \\
        [--event-idx 0] [--n-batches 5] [--batch-size 200] [--out-dir DIR]

Or, to load from a tmnre_joint.py run::

    /data/gpuleo/envs/lisa_pip/bin/python scripts/diagnose_autoencoder.py \\
        --run-name 2026/04/30/autoencoder_joint_v1 --round 3
"""

import argparse
import glob
import os
import re

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

from pembhb import ROOT_DIR, DATA_ROOT_DIR
from pembhb.autoencoder import DenoisingAutoencoder
from pembhb.data import MBHBDataModule
from pembhb.model import JointAEInferenceNetwork, load_inference_network

CHANNEL_NAMES = ["A", "E"]


def _resolve_round_version_dir(run_name: str, round_idx: int) -> tuple[str, str]:
    """Find the latest ``version_*`` dir for ``run_name`` round ``round_idx``.

    Tries the new nested layout first
    (``{DATA_ROOT_DIR}/logs/{run_name}/round_{N}/version_*``), falls back
    to the legacy flat layout
    (``{DATA_ROOT_DIR}/logs/{run_name}_round_{N}/version_*``).

    Returns ``(version_dir, layout)`` where ``layout`` is ``"nested"`` or
    ``"flat"``.
    """
    log_root = os.path.join(DATA_ROOT_DIR, "logs")

    nested = os.path.join(log_root, run_name, f"round_{round_idx}")
    flat = os.path.join(log_root, f"{run_name}_round_{round_idx}")

    for round_dir, layout in ((nested, "nested"), (flat, "flat")):
        if not os.path.isdir(round_dir):
            continue
        version_nums = sorted(
            int(m.group(1))
            for entry in os.listdir(round_dir)
            for m in [re.match(r"^version_(\d+)$", entry)]
            if m
        )
        if version_nums:
            return os.path.join(round_dir, f"version_{version_nums[-1]}"), layout

    raise FileNotFoundError(
        f"No log directory for round {round_idx} of run '{run_name}'. "
        f"Tried nested {nested}/version_* and flat {flat}/version_*."
    )


def _pick_ckpt_in_version(version_dir: str) -> str:
    """Prefer ``truncation.ckpt`` (one level up from ``checkpoints/``);
    otherwise the latest ``checkpoints/*.ckpt`` (sorted lexicographically,
    matching Lightning's ``epoch=NNN-step=…`` pattern)."""
    trunc = os.path.join(version_dir, "truncation.ckpt")
    if os.path.isfile(trunc):
        return trunc
    ckpts = sorted(glob.glob(os.path.join(version_dir, "checkpoints", "*.ckpt")))
    if not ckpts:
        raise FileNotFoundError(
            f"No checkpoint under {version_dir} (no truncation.ckpt and "
            f"no checkpoints/*.ckpt)."
        )
    return ckpts[-1]


def resolve_checkpoint(args) -> tuple[str, str, str]:
    """Return ``(ckpt_path, run_tag, layout)``.

    ``run_tag`` is used to derive the output directory and the default
    dataset path. ``layout`` is ``"checkpoint"``, ``"nested"`` or
    ``"flat"``.
    """
    if args.checkpoint is not None:
        tag = os.path.basename(args.checkpoint).removesuffix(".ckpt")
        return args.checkpoint, tag, "checkpoint"

    version_dir, layout = _resolve_round_version_dir(args.run_name, args.round)
    return _pick_ckpt_in_version(version_dir), args.run_name, layout


def extract_autoencoder(ckpt_path: str, device: str) -> DenoisingAutoencoder:
    """Load ``ckpt_path`` and return the underlying ``DenoisingAutoencoder``.

    Handles two cases:

    * Plain ``DenoisingAutoencoder`` checkpoint (from ``train_autoencoder.py``)
      — loaded directly.
    * ``JointAEInferenceNetwork`` checkpoint (from ``tmnre_joint.py``)
      — its ``encoder_model`` attribute is the AE; the NRE classifier
      is ignored. Marginal-encoder runs (where ``encoder_model`` is a
      ``MarginalEncoderTrainer``) are rejected because this diagnostic
      assumes a single global denoising AE.
    """
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = raw.get("hyper_parameters", {}) if isinstance(raw, dict) else {}
    is_joint = "ae_warmup_epochs" in hp

    if is_joint:
        joint = load_inference_network(ckpt_path, device=device)
        assert isinstance(joint, JointAEInferenceNetwork), (
            f"Expected JointAEInferenceNetwork, got {type(joint).__name__}"
        )
        ae = joint.encoder_model
        if not isinstance(ae, DenoisingAutoencoder):
            raise NotImplementedError(
                f"Joint checkpoint's encoder_model is {type(ae).__name__}, "
                f"not DenoisingAutoencoder. This diagnostic only supports "
                f"the global denoising-AE encoder."
            )
        return ae.to(device).eval()

    return DenoisingAutoencoder.load_from_checkpoint(ckpt_path, map_location=device)


def _slice_to_target(t: torch.Tensor, ae: DenoisingAutoencoder) -> torch.Tensor:
    """Slice the last dim of a (..., F) tensor to the AE's reconstruction band."""
    lo = getattr(ae, "idx_lowerbound", 0)
    hi = getattr(ae, "idx_upperbound", t.shape[-1])
    return t[..., lo:hi]


def _to_physical(x_real: torch.Tensor, ae: DenoisingAutoencoder) -> torch.Tensor:
    """Inverse of ``ae.preprocess`` — real (B, 2C, F_t) → complex physical (B, C, F_t).

    Dispatches on ``ae.whiten`` (the two preprocess branches are not
    composable identities of each other):

      ``whiten=True``  : preprocess does
        ``((z/whitening) as real_imag − mean_whitened) / amplitude_scale``,
        so we undo amp_scale, add mean_whitened back, real→complex,
        multiply by whitening.

      ``whiten=False`` : preprocess does
        ``(real_imag(z) − mean_vec) / global_scale_factor``,
        so we apply ``_denormalize`` (``* global_scale + mean_vec``) then
        real→complex.

    Output is in original Hz⁻¹ amplitude units regardless of flag combo.
    """
    if ae.whiten:
        if ae.amplitude_normalise:
            x_real = x_real * ae.amplitude_scale
            if ae.subtract_mean_whitened:
                mean_w = ae._get_mean_whitened_for(x_real.shape[-1])
                x_real = x_real + mean_w
        z = ae._real_to_complex(x_real)
        return z * ae._get_whitening_for(z.shape[-1])

    # whiten=False path — _denormalize does x * global_scale_factor + mean_vec
    x_real = ae._denormalize(x_real)
    return ae._real_to_complex(x_real)


def collect_reconstructions(ae: DenoisingAutoencoder,
                            data_module: MBHBDataModule,
                            n_batches: int,
                            device: str):
    """Run the AE on ``n_batches`` of training data; return tensors on CPU.

    Each of (target, noisy, reconstruction) is returned in three spaces so
    plots are meaningful regardless of whether the AE was trained with
    ``whiten=True`` or ``whiten=False``:

      ``*_norm``  (B, 2C, F_t) real — what the AE sees / outputs
                  (= ``ae.preprocess`` output, sliced to the target band).
                  This is exactly the space the MSE loss is computed in.
      ``*_phys``  (B, C, F_t) complex — original Hz⁻¹ amplitude units,
                  obtained by inverting ``ae.preprocess`` end-to-end via
                  ``_to_physical`` (handles both ``whiten=True/False``).
      ``*_w``     (B, 2C, F_t) real — physical complex divided by the
                  **dataset** noise_scale = ASD·√(T/4), then split into
                  Re/Im channels. For ``noise_factor=1`` each entry of the
                  noise is ~ N(0, 1) per Re/Im. Independent of the AE flags.

    Also returns ``valid_freq_mask`` — bool (F_t,), True where
    ``noise_scale > 0`` after slicing. Bins below ``psd_fmin_mask`` are
    False and should be excluded from plot 2 (division-by-zero source).
    """
    ae.eval()
    noise_scale_full = data_module.get_noise_scale()
    if noise_scale_full is None:
        raise RuntimeError(
            "data_module.get_noise_scale() returned None — the dataset has no "
            "stored ASD; cannot compute whitened residuals for plot 2."
        )
    noise_scale = _slice_to_target(noise_scale_full, ae).to(device)  # (C, F_t)

    # psd_fmin_mask sets bins below the cutoff to zero in noise_scale; those
    # bins must be excluded before dividing or NaN/inf will swamp plot 2.
    valid_freq_mask = (noise_scale > 0).all(dim=0)  # (F_t,)

    keys = ("target_norm", "noisy_norm", "rec_norm",
            "target_phys", "noisy_phys", "rec_phys",
            "target_w",    "noisy_w",    "rec_w")
    out = {k: [] for k in keys}

    # Safe whitening: replace zeros with 1 only for the division (we mask
    # those bins afterwards anyway via valid_freq_mask).
    noise_scale_safe = noise_scale.clone()
    noise_scale_safe[noise_scale == 0] = 1.0

    dl = data_module.train_dataloader(shuffle=False, num_workers=0)
    with torch.no_grad():
        for i, batch in enumerate(dl):
            if i >= n_batches:
                break
            wave_fd = batch["wave_fd"].to(device)
            noise_fd = batch["noise_fd"].to(device)

            noisy_norm_full = ae.preprocess(wave_fd + noise_fd)
            clean_norm_full = ae.preprocess(wave_fd)
            rec_norm    = ae(noisy_norm_full)
            target_norm = ae._get_target(clean_norm_full)
            noisy_norm  = ae._get_target(noisy_norm_full)

            target_phys = _to_physical(target_norm, ae)
            noisy_phys  = _to_physical(noisy_norm,  ae)
            rec_phys    = _to_physical(rec_norm,    ae)

            target_w = ae._complex_to_real(target_phys / noise_scale_safe)
            noisy_w  = ae._complex_to_real(noisy_phys  / noise_scale_safe)
            rec_w    = ae._complex_to_real(rec_phys    / noise_scale_safe)

            out["target_norm"].append(target_norm.cpu())
            out["noisy_norm"].append(noisy_norm.cpu())
            out["rec_norm"].append(rec_norm.cpu())
            out["target_phys"].append(target_phys.cpu())
            out["noisy_phys"].append(noisy_phys.cpu())
            out["rec_phys"].append(rec_phys.cpu())
            out["target_w"].append(target_w.cpu())
            out["noisy_w"].append(noisy_w.cpu())
            out["rec_w"].append(rec_w.cpu())
    bundle = {k: torch.cat(v, dim=0) for k, v in out.items()}
    bundle["valid_freq_mask"] = valid_freq_mask.cpu()
    return bundle


def _amp_complex(real: torch.Tensor, ae: DenoisingAutoencoder) -> torch.Tensor:
    """Convert (N, 2C, F) real → (N, C, F) complex magnitude.

    Only meaningful for representation='real_imag'. The plotting paths only
    call this when that's the case.
    """
    z = ae._real_to_complex(real)
    return torch.abs(z)


def freqs_for_target(ae: DenoisingAutoencoder, freqs: np.ndarray) -> np.ndarray:
    """Frequency axis matching the AE target tensor (sliced to active band mask)."""
    lo = getattr(ae, "idx_lowerbound", 0)
    hi = getattr(ae, "idx_upperbound", len(freqs))
    return freqs[lo:hi]


def plot_overlay(freqs_t, target_norm, noisy_norm, rec_norm,
                 target_phys, noisy_phys, rec_phys,
                 ae, fname, out_dir, idx=0):
    """Two rows (network space vs physical Hz⁻¹), two cols (A, E).

    Row 0 shows what the AE actually sees / outputs (`ae.preprocess` output);
    its absolute scale depends on the active normalisation flags
    (`whiten`, `amplitude_normalise`). Row 1 shows the same signals after
    inverting `preprocess` end-to-end — i.e. in original Hz⁻¹ amplitude
    units. Row 1 should be visually identical across runs that differ only
    in the choice of normalisation flags.
    """
    n_ch = min(target_norm.shape[1] // 2, len(CHANNEL_NAMES))
    amp_target_norm = _amp_complex(target_norm, ae)
    amp_noisy_norm = _amp_complex(noisy_norm, ae)
    amp_rec_norm = _amp_complex(rec_norm, ae)
    amp_target_phys = target_phys.abs()
    amp_noisy_phys = noisy_phys.abs()
    amp_rec_phys = rec_phys.abs()

    fig, axes = plt.subplots(2, n_ch, figsize=(6 * n_ch, 8))
    if n_ch == 1:
        axes = axes.reshape(2, 1)

    row0_label = (
        f"Network space  (whiten={ae.whiten}, "
        f"amplitude_normalise={ae.amplitude_normalise}, "
        f"subtract_mean_whitened={getattr(ae, 'subtract_mean_whitened', False)})"
    )

    for c in range(n_ch):
        # Row 0: network input/output space
        ax = axes[0, c]
        ax.plot(freqs_t, amp_noisy_norm[idx, c].numpy(),
                label="noisy (target+noise)", color="grey",
                alpha=0.25, lw=0.6, zorder=0)
        ax.plot(freqs_t, amp_target_norm[idx, c].numpy(),
                label="target", alpha=0.85, lw=0.9, zorder=2)
        ax.plot(freqs_t, amp_rec_norm[idx, c].numpy(),
                label="reconstruction", alpha=0.85, lw=0.9, ls="--", zorder=3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_title(f"{row0_label} — Ch {CHANNEL_NAMES[c]}")
        ax.grid(True, which="both", ls="--", lw=0.3)
        ax.legend(fontsize=8)

        # Row 1: physical Hz⁻¹ amplitude (preprocess fully inverted)
        ax = axes[1, c]
        ax.plot(freqs_t, amp_noisy_phys[idx, c].numpy(),
                label="noisy (target+noise)", color="grey",
                alpha=0.25, lw=0.6, zorder=0)
        ax.plot(freqs_t, amp_target_phys[idx, c].numpy(),
                label="target", alpha=0.85, lw=0.9, zorder=2)
        ax.plot(freqs_t, amp_rec_phys[idx, c].numpy(),
                label="reconstruction", alpha=0.85, lw=0.9, ls="--", zorder=3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_title(f"Physical |h(f)|  (Hz⁻¹) — Ch {CHANNEL_NAMES[c]}")
        ax.grid(True, which="both", ls="--", lw=0.3)
        ax.legend(fontsize=8)

    fig.suptitle(f"Event {idx} — overlay |target| vs |reconstruction|", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, fname), dpi=150)
    plt.close(fig)
    print(f"  saved {fname}")


def plot_residual_hist(target_w, noisy_w, rec_w, valid_freq_mask, fname, out_dir):
    """Residual distributions in the dataset-noise-whitened space.

    All tensors are real-valued (B, 2C, F_t) and live in the explicitly
    whitened space (physical / noise_scale), so each entry should be
    ~ N(0, 1) component-wise for the noise. ``valid_freq_mask`` (F_t,)
    excludes bins where noise_scale was zeroed by ``psd_fmin_mask`` —
    those bins are pure-zero artefacts and would otherwise dominate.
    Distributions are pooled across (events, channels, Re/Im, freq).

    Two panels:
      * Left  — ``data - waveform`` (≈ pure whitened noise) and
                ``data - denoised`` (what the AE removed) overlaid.
      * Right — ``denoised - waveform`` (reconstruction error).

    Each panel shows the standard normal N(0, 1) for reference.
    """
    # Slice along the last (frequency) dimension before flattening.
    m = valid_freq_mask
    data_minus_wave = (noisy_w - target_w)[..., m].numpy().ravel()
    data_minus_rec = (noisy_w - rec_w)[..., m].numpy().ravel()
    rec_minus_wave = (rec_w - target_w)[..., m].numpy().ravel()

    # Common bin edges so all histograms are directly comparable.
    lo = float(min(data_minus_wave.min(), data_minus_rec.min(), rec_minus_wave.min()))
    hi = float(max(data_minus_wave.max(), data_minus_rec.max(), rec_minus_wave.max()))
    bins = np.linspace(lo, hi, 121)
    centers = 0.5 * (bins[:-1] + bins[1:])
    gauss_pdf = np.exp(-0.5 * centers ** 2) / np.sqrt(2 * np.pi)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left panel: data - waveform  &  data - denoised, with N(0,1)
    ax = axes[0]
    ax.hist(data_minus_wave, bins=bins, density=True, histtype="step",
            color="C0", lw=1.4,
            label=f"data − waveform  (μ={data_minus_wave.mean():.2e}, "
                  f"σ={data_minus_wave.std():.3f})")
    ax.hist(data_minus_rec, bins=bins, density=True, histtype="step",
            color="C1", lw=1.4,
            label=f"data − denoised  (μ={data_minus_rec.mean():.2e}, "
                  f"σ={data_minus_rec.std():.3f})")
    ax.plot(centers, gauss_pdf, color="black", ls="--", lw=1.0,
            label="N(0, 1)")
    ax.set_xlabel("residual (whitened units)")
    ax.set_ylabel("density")
    ax.set_yscale("log")
    ax.set_title("Pre-AE residuals — pooled (events, channels, Re/Im, freq)")
    ax.grid(True, ls="--", lw=0.3)
    ax.legend(fontsize=8)

    # Right panel: denoised - waveform, with N(0,1)
    ax = axes[1]
    ax.hist(rec_minus_wave, bins=bins, density=True, histtype="step",
            color="C2", lw=1.4,
            label=f"denoised − waveform  (μ={rec_minus_wave.mean():.2e}, "
                  f"σ={rec_minus_wave.std():.3f})")
    ax.plot(centers, gauss_pdf, color="black", ls="--", lw=1.0,
            label="N(0, 1)")
    ax.set_xlabel("residual (whitened units)")
    ax.set_ylabel("density")
    ax.set_yscale("log")
    ax.set_title("Reconstruction error — pooled (events, channels, Re/Im, freq)")
    ax.grid(True, ls="--", lw=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, fname), dpi=150)
    plt.close(fig)
    print(f"  saved {fname}")


def per_freq_loss_per_channel(target_norm: torch.Tensor,
                              rec_norm: torch.Tensor) -> list[np.ndarray]:
    """Per-channel per-bin loss array ``loss[c] = 0.5·(Re² + Im²)``.

    Returns a list of ``(N, F_t)`` arrays, one per TDI channel.
    """
    sq = (rec_norm - target_norm) ** 2  # (N, 2C, F_t)
    n_real = target_norm.shape[1]
    n_ch = min(n_real // 2, len(CHANNEL_NAMES))
    out = []
    for c in range(n_ch):
        re_c = c
        im_c = c + (n_real // 2)
        out.append((0.5 * (sq[:, re_c, :] + sq[:, im_c, :])).numpy())
    return out


def render_per_freq_loss_panel(ax, freqs_t, loss_np, channel_name,
                               show_legend=True):
    """Draw a single per-channel per-frequency loss panel on ``ax``.

    ``loss_np`` is ``(N, F_t)``: per-event per-bin MSE. Matches the inner
    body of :func:`plot_per_freq_loss` and is reused by the training
    animation script.
    """
    median = np.median(loss_np, axis=0)
    mean = loss_np.mean(axis=0)
    p5, p25, p75, p95 = np.percentile(loss_np, [5, 25, 75, 95], axis=0)

    ax.fill_between(freqs_t, p5, p95, alpha=0.18, color="C0", label="5–95%")
    ax.fill_between(freqs_t, p25, p75, alpha=0.32, color="C0", label="25–75%")
    ax.plot(freqs_t, median, color="C0", lw=1.4, label="median")
    ax.plot(freqs_t, mean, color="green", lw=1.0, ls="--", label="mean")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("per-bin MSE  ½·(Re² + Im²)")
    ax.set_title(f"Per-frequency loss — Ch {channel_name}")
    ax.grid(True, which="both", ls="--", lw=0.3)
    if show_legend:
        ax.legend(fontsize=8)


def plot_per_freq_loss(freqs_t, target_norm, rec_norm, ae,
                       fname, out_dir):
    """Per-frequency MSE distribution over the evaluated samples.

    Loss matches the training MSE: squared error on the
    (whitened + amplitude-scaled) real representation.
    Real and imaginary channels are averaged within each TDI channel
    so the plot has one subplot per (A, E).
    """
    losses = per_freq_loss_per_channel(target_norm, rec_norm)
    n_ch = len(losses)
    fig, axes = plt.subplots(1, n_ch, figsize=(6 * n_ch, 4.5))
    if n_ch == 1:
        axes = [axes]
    for c in range(n_ch):
        render_per_freq_loss_panel(axes[c], freqs_t, losses[c], CHANNEL_NAMES[c])
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, fname), dpi=150)
    plt.close(fig)
    print(f"  saved {fname}")


def plot_mse_space(freqs_t, target_norm, noisy_norm, rec_norm,
                   ae, fname, out_dir, idx=0, xlim=None):
    """Show target / reconstruction / noisy **in the exact space the MSE
    loss is computed in** — i.e. the AE's preprocess output, sliced to the
    target band. No further normalisation, no abs(), so signs/oscillations
    are visible.

    2×2 channel-major layout::

        [ Re A | Im A ]
        [ Re E | Im E ]

    For ``representation='real_imag'`` (the only one supported), the
    channel layout in ``target_norm`` (shape (B, 2C, F_t)) is
    ``[Re_ch0, Re_ch1, ..., Im_ch0, Im_ch1, ...]``, so Re of channel c is
    at index c and Im of channel c is at index c + n_channels.

    :param xlim: optional ``(fmin, fmax)`` tuple to zoom the x-axis.
    """
    n_real = target_norm.shape[1]
    n_ch = min(n_real // 2, len(CHANNEL_NAMES))
    if n_ch < 2:
        # Degenerate: still draw what we have, but the layout will only be 1×2.
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes = np.array(axes).reshape(1, 2)
    else:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

    for c in range(min(n_ch, 2)):
        re_idx = c
        im_idx = c + (n_real // 2)
        for col, (idx_real, comp) in enumerate(((re_idx, "Re"), (im_idx, "Im"))):
            ax = axes[c, col]
            ax.plot(freqs_t, noisy_norm[idx, idx_real].numpy(),
                    label="noisy (target+noise)", color="grey",
                    alpha=0.35, lw=0.6, zorder=0)
            ax.plot(freqs_t, target_norm[idx, idx_real].numpy(),
                    label="target", color="C0", alpha=0.9, lw=0.9, zorder=2)
            ax.plot(freqs_t, rec_norm[idx, idx_real].numpy(),
                    label="reconstruction", color="C1", alpha=0.9, lw=0.9,
                    ls="--", zorder=3)
            ax.set_xscale("log")
            ax.axhline(0.0, color="black", lw=0.4, alpha=0.4, zorder=1)
            ax.set_xlabel("Frequency (Hz)")
            ax.set_title(f"{comp} — Ch {CHANNEL_NAMES[c]}")
            ax.grid(True, which="both", ls="--", lw=0.3)
            if (c, col) == (0, 0):
                ax.legend(fontsize=8, loc="best")
            if xlim is not None:
                ax.set_xlim(xlim)

    flags = (f"whiten={ae.whiten}, amplitude_normalise={ae.amplitude_normalise}, "
             f"subtract_mean_whitened={getattr(ae, 'subtract_mean_whitened', False)}")
    zoom_tag = "" if xlim is None else f"  (zoom: {xlim[0]:.1e}–{xlim[1]:.1e} Hz)"
    fig.suptitle(
        f"Event {idx} — MSE-loss space (preprocess output){zoom_tag}\n{flags}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, fname), dpi=150)
    plt.close(fig)
    print(f"  saved {fname}")


def render_overlay_panel(ax, x_axis, data_curve, wf_curve, rec_curve,
                         show_legend=False):
    """Draw the (data, wf, rec) overlay used by :func:`plot_two_space_zoom`.

    ``data`` is light/shaded behind, ``wf`` dashed, ``rec`` solid.
    Returns the three ``Line2D`` objects so animators can call
    ``rec_line.set_ydata(...)`` to update a single frame in place.
    """
    line_data, = ax.plot(x_axis, data_curve, color="grey", alpha=0.35, lw=0.7,
                         label="data = wf + noise", zorder=0)
    line_wf, = ax.plot(x_axis, wf_curve, color="C0", lw=0.9, ls="--",
                      label="wf (target)", alpha=0.9, zorder=2)
    line_rec, = ax.plot(x_axis, rec_curve, color="C1", lw=0.9, ls="-",
                       label="reconstruction", alpha=0.9, zorder=3)
    ax.axhline(0.0, color="black", lw=0.4, alpha=0.4, zorder=1)
    ax.set_xlabel("Frequency (Hz)")
    ax.grid(True, which="both", ls="--", lw=0.3)
    if show_legend:
        ax.legend(fontsize=8, loc="best")
    return line_data, line_wf, line_rec


def plot_two_space_zoom(
    freqs_t,
    target_w, noisy_w, rec_w,
    target_norm, noisy_norm, rec_norm,
    out_dir, channel_idx, channel_name, part_name,
    fname, suptitle,
    idx=0, zoom_band=(2.7e-3, 2.9e-3),
):
    """One (channel, Re/Im) → 2 rows × 2 cols figure.

    Columns:
      0. **whitened space** — physical reconstruction divided by ASD·√(T/4).
         This is the AE's output un-doing the loss-space transform
         (``rec_norm · amp_scale + mean_whitened`` for the canonical
         ``whiten=True`` path).
      1. **residual / amp_scale space** — exactly what enters the MSE
         loss: ``(signal − mean_whitened) / amp_scale``.

    Each panel overlays three curves for event ``idx``:
      * ``data = wf + noise`` — light, low-alpha, drawn behind
      * ``wf``                — dashed
      * ``reconstruction``    — solid

    Rows:
      row 0: full frequency band, log-x. Vertical lines and grey shading
             mark the zoom band.
      row 1: same quantities, linear-x, ``xlim = zoom_band``. Dashed
             connectors link the row-0 zoom-band markers to the top
             corners of the corresponding row-1 axis.

    :param target_w / noisy_w / rec_w: (N, 2C, F_t) tensors in the
        **true whitened space** (Re/Im each ~ N(0, 1) for unit noise).
    :param target_norm / noisy_norm / rec_norm: (N, 2C, F_t) tensors in
        the loss space (preprocess output).
    :param part_name: ``"Re"`` selects channel index ``channel_idx``,
                      ``"Im"`` selects ``channel_idx + C``.
    """
    from matplotlib.patches import ConnectionPatch
    from matplotlib.transforms import blended_transform_factory

    n_real = target_w.shape[1]
    C = n_real // 2
    if part_name == "Re":
        re_im_idx = channel_idx
    elif part_name == "Im":
        re_im_idx = C + channel_idx
    else:
        raise ValueError(f"part_name must be 'Re' or 'Im', got {part_name!r}")

    f_np = np.asarray(freqs_t)
    zfmin, zfmax = zoom_band
    zoom_mask = (f_np >= zfmin) & (f_np <= zfmax)
    f_zoom = f_np[zoom_mask]
    zoom_label = f"[{zfmin*1e3:.2f}, {zfmax*1e3:.2f}] mHz"

    data_w_curve = noisy_w[idx, re_im_idx].numpy()
    wf_w_curve = target_w[idx, re_im_idx].numpy()
    rec_w_curve = rec_w[idx, re_im_idx].numpy()

    data_n_curve = noisy_norm[idx, re_im_idx].numpy()
    wf_n_curve = target_norm[idx, re_im_idx].numpy()
    rec_n_curve = rec_norm[idx, re_im_idx].numpy()

    columns = [
        (0, "whitened space", data_w_curve, wf_w_curve, rec_w_curve),
        (1, "residual / amp_scale  (MSE-loss space)",
         data_n_curve, wf_n_curve, rec_n_curve),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    for col, title, data_curve, wf_curve, rec_curve in columns:
        ax0 = axes[0, col]
        ax1 = axes[1, col]

        render_overlay_panel(ax0, f_np, data_curve, wf_curve, rec_curve,
                             show_legend=(col == 0))
        render_overlay_panel(ax1, f_zoom,
                             data_curve[zoom_mask], wf_curve[zoom_mask],
                             rec_curve[zoom_mask])

        ax0.set_xscale("log")
        ax0.set_title(f"{title} — full band")
        ax0.axvspan(zfmin, zfmax, color="gray", alpha=0.15, zorder=0)
        ax0.axvline(zfmin, color="gray", lw=0.8, ls="--", alpha=0.8)
        ax0.axvline(zfmax, color="gray", lw=0.8, ls="--", alpha=0.8)

        ax1.set_xlim(zfmin, zfmax)
        ax1.set_title(f"zoom {zoom_label}")
        ax1.ticklabel_format(axis="x", style="sci", scilimits=(-3, -3),
                             useMathText=True)

    axes[0, 0].set_ylabel(f"{part_name}( whitened )")
    axes[1, 0].set_ylabel(f"{part_name}( whitened )")
    axes[0, 1].set_ylabel(f"{part_name}( normalised )")
    axes[1, 1].set_ylabel(f"{part_name}( normalised )")

    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()

    # Connector lines from the row-0 zoom-band edges to the row-1 corners.
    for col in range(2):
        ax0 = axes[0, col]
        ax1 = axes[1, col]
        trans0 = blended_transform_factory(ax0.transData, ax0.transAxes)
        for x_a, x_b in [(zfmin, 0.0), (zfmax, 1.0)]:
            con = ConnectionPatch(
                xyA=(x_a, 0.0), coordsA=trans0,
                xyB=(x_b, 1.0), coordsB=ax1.transAxes,
                color="gray", lw=0.8, alpha=0.7, ls="--",
            )
            fig.add_artist(con)

    fig.savefig(os.path.join(out_dir, fname), dpi=150)
    plt.close(fig)
    print(f"  saved {fname}")


def _default_out_dir(run_tag: str, layout: str, round_idx: int | None) -> str:
    """Pick the output directory tag.

    * ``--checkpoint``       → ``plots/diagnose_ae_{ckpt_basename}/``
    * nested run-name layout → ``plots/{run_name}/diagnose_ae_round_{N}/``
                               (e.g. ``plots/2026/04/30/{tag}/diagnose_ae_round_3/``)
    * legacy flat layout     → ``plots/diagnose_ae_{run_name}_round_{N}/``
    """
    if layout == "checkpoint":
        return os.path.join(ROOT_DIR, "plots", f"diagnose_ae_{run_tag}")
    if layout == "nested":
        return os.path.join(ROOT_DIR, "plots", run_tag, f"diagnose_ae_round_{round_idx}")
    return os.path.join(ROOT_DIR, "plots", f"diagnose_ae_{run_tag}_round_{round_idx}")


def _default_dataset_path(run_name: str, round_idx: int) -> str:
    return os.path.join(DATA_ROOT_DIR, run_name, f"simulation_round_{round_idx}.h5")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", type=str,
                     help="Direct path to a .ckpt file.")
    src.add_argument("--run-name", type=str,
                     help="tmnre_joint.py TIME_OF_EXECUTION tag, e.g. "
                          "'2026/04/30/autoencoder_joint_v1' (new) or "
                          "'20260430_autoencoder_joint_v1' (legacy).")
    parser.add_argument("--round", type=int, default=None,
                        help="Round index to load (required with --run-name).")
    parser.add_argument("--dataset", default=None,
                        help="HDF5 dataset to evaluate on. With --run-name, "
                             "defaults to "
                             "{DATA_ROOT_DIR}/{run_name}/simulation_round_{round}.h5.")
    parser.add_argument("--event-idx", type=int, default=0,
                        help="Event index for the overlay and residual histogram.")
    parser.add_argument("--n-batches", type=int, default=5,
                        help="Number of batches to evaluate for the per-frequency stats.")
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--out-dir", default=None,
                        help="Output directory (default: see module docstring).")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.run_name is not None and args.round is None:
        parser.error("--round is required with --run-name.")

    ckpt_path, run_tag, layout = resolve_checkpoint(args)

    if args.dataset is None:
        if layout == "checkpoint":
            parser.error("--dataset is required with --checkpoint.")
        args.dataset = _default_dataset_path(args.run_name, args.round)

    print(f"[diagnose_autoencoder] checkpoint : {ckpt_path}")
    print(f"[diagnose_autoencoder] dataset    : {args.dataset}")
    print(f"[diagnose_autoencoder] layout     : {layout}")

    if args.out_dir is None:
        args.out_dir = _default_out_dir(run_tag, layout, args.round)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[diagnose_autoencoder] out_dir    : {args.out_dir}")

    data_module = MBHBDataModule(filename=args.dataset, batch_size=args.batch_size,
                                 num_workers=0, cache_in_memory=False)
    data_module.setup(stage="fit")
    with h5py.File(args.dataset, "r") as f:
        freqs = f["frequencies"][()]
        psd_fmin_mask = float(f.attrs.get("psd_fmin_mask", 0.0))
    if psd_fmin_mask > 0:
        print(f"[diagnose_autoencoder] psd_fmin_mask = {psd_fmin_mask:.3e} Hz "
              f"(noise zeroed below this cutoff; those bins are excluded from plot 2)")

    ae = extract_autoencoder(ckpt_path, args.device)
    if ae.representation != "real_imag":
        raise NotImplementedError(
            "diagnose_autoencoder.py currently assumes representation='real_imag' "
            "for converting reconstructions back to complex amplitude. "
            f"Checkpoint has representation='{ae.representation}'."
        )

    freqs_t = freqs_for_target(ae, freqs)
    bundle = collect_reconstructions(ae, data_module, args.n_batches, args.device)
    print(f"  N samples evaluated = {bundle['target_norm'].shape[0]}, "
          f"target shape = {tuple(bundle['target_norm'].shape)}")

    plot_overlay(freqs_t,
                 bundle["target_norm"], bundle["noisy_norm"], bundle["rec_norm"],
                 bundle["target_phys"], bundle["noisy_phys"], bundle["rec_phys"], ae,
                 "1_overlay_amplitude.png", args.out_dir, idx=args.event_idx)
    plot_residual_hist(bundle["target_w"], bundle["noisy_w"], bundle["rec_w"],
                       bundle["valid_freq_mask"],
                       "2_residual_histogram.png", args.out_dir)
    plot_per_freq_loss(freqs_t, bundle["target_norm"], bundle["rec_norm"], ae,
                       "3_per_frequency_loss.png", args.out_dir)
    plot_mse_space(freqs_t, bundle["target_norm"], bundle["noisy_norm"], bundle["rec_norm"],
                   ae, "4_mse_space.png", args.out_dir, idx=args.event_idx)
    plot_mse_space(freqs_t, bundle["target_norm"], bundle["noisy_norm"], bundle["rec_norm"],
                   ae, "5_mse_space_zoom.png", args.out_dir, idx=args.event_idx,
                   xlim=(3e-3, 3.2e-3))

    # 6_*: per (channel, Re/Im) two-space overlay with full-band + zoom rows.
    n_real = bundle["target_w"].shape[1]
    n_ch = min(n_real // 2, len(CHANNEL_NAMES))
    flags = (f"whiten={ae.whiten}, amplitude_normalise={ae.amplitude_normalise}, "
             f"subtract_mean_whitened={getattr(ae, 'subtract_mean_whitened', False)}")
    for c in range(n_ch):
        ch_name = CHANNEL_NAMES[c]
        for part in ("Re", "Im"):
            plot_two_space_zoom(
                freqs_t,
                bundle["target_w"], bundle["noisy_w"], bundle["rec_w"],
                bundle["target_norm"], bundle["noisy_norm"], bundle["rec_norm"],
                args.out_dir, channel_idx=c, channel_name=ch_name, part_name=part,
                fname=f"6_two_space_ch{ch_name}_{part}.png",
                suptitle=(f"Event {args.event_idx} — Ch {ch_name} ({part}) — "
                          f"whitened vs MSE-loss space\n{flags}"),
                idx=args.event_idx,
            )

    print(f"\n[diagnose_autoencoder] done → {args.out_dir}")


if __name__ == "__main__":
    main()
