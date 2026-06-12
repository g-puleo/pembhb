"""Diagnostic plots for the frequency-domain whitening transformation.

Whitening:  ``d_w(f) = d(f) / [ASD(f) * sqrt(T_obs / 4)]``

Convention used throughout this codebase: the *real* and *imaginary*
parts of the whitened LISA noise each have **unit variance** and are
independent across frequency bins.  Equivalently, the whitened noise is
a complex normal with per-quadrature variance 1 (``z = Re + j Im`` with
``Re, Im ~ N(0, 1)`` i.i.d.; this is ``CN(0, 2)`` in the "total-variance"
notation).

Loads an HDF5 simulation file and produces, in
``plots/{tag}/whitening/``:

  1. ``1_original_signal.png``         |h(f)|
  2. ``2_whitened_signal.png``         |h_w(f)|
  3. ``3_noise_and_signal_plus_noise.png``  |n(f)| and |d(f)| overlaid
  4. ``4_whitened_signal_plus_noise.png``   |d_w(f)|
  5. ``5_whitened_noise_distribution.png``  Re/Im histogram of n_w
                                             vs N(0, 1)
  6. ``6_whitened_signal_plus_noise_distribution.png``  same for d_w
  7. ``7_meansub_norm_wave_ch{NAME}_{Re|Im}.png``  per-(channel, part)
                                                   effect of the autoencoder's
                                                   ``subtract_mean_whitened`` +
                                                   ``amplitude_normalise`` on
                                                   the *clean waveform*.
  8. ``8_meansub_norm_data_ch{NAME}_{Re|Im}.png``  same for *signal + noise*.

Plots 7-8 reproduce the AE preprocessing (`autoencoder.py::preprocess` with
``representation='real_imag'``).  Each figure is one (channel, part) pair
laid out as 2 rows × 4 columns:

  columns: ``(signals + mean) → (residual) → (residual / amp_scale)
            → (signal / amp_scale_no_mean_sub)``
  row 0 :  full band, log-x, with a shaded zoom band marked by dashed
           vertical lines.
  row 1 :  same quantities, linear-x, ``xlim`` = the zoom band; dashed
           connectors link the bottoms of the row-0 vertical lines to the
           top corners of the corresponding row-1 axis.

Stats (mean and ``amplitude_scale``) are computed on the *same* ``--hdf5``
file used for the plots — exactly as
``DenoisingAutoencoder.fit_amplitude_normalisation`` does on the training
set in ``tmnre_joint.py``.

Run:

    /data/gpuleo/envs/lisa_pip/bin/python scripts/diagnose_whitening.py \\
        --hdf5 /data/gpuleo/mbhb/RUN/simulation_round_1.h5 \\
        [--event-idx 0] [--n-samples 2000] [--n-overlay 30] [--out-dir DIR]
"""

import argparse
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

from pembhb import ROOT_DIR, DATA_ROOT_DIR, FMIN_FLOOR

CHANNEL_NAMES = ["A", "E"]


def load_data(hdf5_path: str, n_samples: int):
    """Return wave_fd, noise_fd (complex), freqs, whitening (real), T_obs.

    If the HDF5 has no stored noise realisation, draws one with the same
    rule as ``mbhb_collate_fn``.
    """
    with h5py.File(hdf5_path, "r") as f:
        N = min(n_samples, f["wave_fd"].shape[0])
        wave_fd_np = f["wave_fd"][:N]
        freqs = f["frequencies"][()]
        asd = f["asd"][()]
        T_obs = float(f.attrs["observation_duration_SI"])
        has_noise = "noise_fd" in f
        if has_noise:
            noise_fd_np = f["noise_fd"][:N]

    wave_fd = torch.tensor(wave_fd_np)

    # noise_scale = ASD * sqrt(T_obs / 4), with bins below FMIN_FLOOR zeroed
    filtered_asd = asd.copy()
    filtered_asd[:, freqs < FMIN_FLOOR] = 0.0
    noise_scale_np = filtered_asd / np.sqrt(4.0 / T_obs)
    noise_scale = torch.tensor(noise_scale_np, dtype=torch.float32)

    if has_noise:
        noise_fd = torch.tensor(noise_fd_np)
    else:
        # Same generation as mbhb_collate_fn: Re, Im ~ N(0, 1) i.i.d.,
        # then scaled bin-wise by noise_scale = ASD * sqrt(T_obs / 4).
        re = torch.randn(N, *noise_scale.shape)
        im = torch.randn(N, *noise_scale.shape)
        noise_fd = (re + 1j * im) * noise_scale.to(torch.complex64)

    whitening = noise_scale.clone()
    whitening[noise_scale == 0] = float("inf")

    return wave_fd, noise_fd, freqs, whitening, T_obs


def whiten(z: torch.Tensor, whitening: torch.Tensor) -> torch.Tensor:
    """``z: (N, C, F) complex`` divided bin-wise by ``whitening: (C, F) real``."""
    return z / whitening


def _setup_fd_axes(ax, title):
    ax.set_xlabel("Frequency (Hz)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.grid(True, which="both", ls="--", lw=0.3)


def plot_fd_amp(freqs, z, title, fname, out_dir, idx=0):
    """One event, two panels (A, E). Plots ``|z(f)|`` per channel."""
    n_ch = min(z.shape[1], len(CHANNEL_NAMES))
    fig, axes = plt.subplots(1, n_ch, figsize=(6 * n_ch, 4))
    if n_ch == 1:
        axes = [axes]
    for c in range(n_ch):
        ax = axes[c]
        ax.plot(freqs, torch.abs(z[idx, c]).numpy(), lw=0.8)
        _setup_fd_axes(ax, f"{title} — Channel {CHANNEL_NAMES[c]}")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, fname), dpi=150)
    plt.close(fig)
    print(f"  saved {fname}")


def plot_fd_amp_overlay(freqs, zs, labels, title, fname, out_dir, idx=0):
    n_ch = min(zs[0].shape[1], len(CHANNEL_NAMES))
    fig, axes = plt.subplots(1, n_ch, figsize=(6 * n_ch, 4))
    if n_ch == 1:
        axes = [axes]
    for c in range(n_ch):
        ax = axes[c]
        for z, label in zip(zs, labels):
            ax.plot(freqs, torch.abs(z[idx, c]).numpy(), lw=0.8, alpha=0.7, label=label)
        _setup_fd_axes(ax, f"{title} — Channel {CHANNEL_NAMES[c]}")
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, fname), dpi=150)
    plt.close(fig)
    print(f"  saved {fname}")


def plot_distribution(z, freqs, title, fname, out_dir, hist_range=(-5, 5)):
    """Histogram Re and Im of ``z (N, C, F)`` per channel, masked to
    ``f >= FMIN_FLOOR``. Overlays a unit-variance N(0, 1) reference,
    matching the convention that whitened noise has Re and Im each ~ N(0, 1)."""
    mask = freqs >= FMIN_FLOOR
    n_ch = min(z.shape[1], len(CHANNEL_NAMES))
    fig, axes = plt.subplots(1, n_ch, figsize=(6 * n_ch, 4))
    if n_ch == 1:
        axes = [axes]

    bins = np.linspace(hist_range[0], hist_range[1], 100)
    sigma = 1.0  # Re, Im of whitened noise each have unit variance.
    xs = np.linspace(hist_range[0], hist_range[1], 400)
    ref_pdf = np.exp(-xs ** 2 / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))

    for c in range(n_ch):
        ax = axes[c]
        sub = z[:, c, :][:, mask]
        re = sub.real.flatten().numpy()
        im = sub.imag.flatten().numpy()
        re_var = float(re.var())
        im_var = float(im.var())
        ax.hist(re, bins=bins, density=True, alpha=0.45,
                label=f"Re (var={re_var:.3f})")
        ax.hist(im, bins=bins, density=True, alpha=0.45,
                label=f"Im (var={im_var:.3f})")
        ax.plot(xs, ref_pdf, "k--", lw=1.2, label="N(0, 1)")
        ax.set_xlabel("value")
        ax.set_ylabel("density")
        ax.set_title(f"{title} — Channel {CHANNEL_NAMES[c]}")
        ax.legend(fontsize=8)
        ax.grid(True, ls="--", lw=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, fname), dpi=150)
    plt.close(fig)
    print(f"  saved {fname}")


def complex_to_real_imag(z: torch.Tensor) -> torch.Tensor:
    """Match ``DenoisingAutoencoder._complex_to_real`` with
    ``representation='real_imag'``: complex (B, C, F) → real (B, 2C, F)
    laid out as ``[Re_ch0, …, Re_chC-1, Im_ch0, …, Im_chC-1]``.
    """
    return torch.cat([z.real, z.imag], dim=1)


def compute_norm_stats(real_w: torch.Tensor):
    """Reproduce ``DenoisingAutoencoder.fit_amplitude_normalisation`` on a
    whitened, real_imag tensor.

    :param real_w: (N, 2C, F) whitened clean signal in real_imag layout.
    :return: (mean (2C, F), amp_scale_with_sub, amp_scale_no_sub).
             ``amp_scale_with_sub``  = max|real_w - mean|        (used when
                                       ``subtract_mean_whitened=True``).
             ``amp_scale_no_sub``    = max|real_w|               (used when
                                       ``subtract_mean_whitened=False``).
    """
    mean = real_w.mean(dim=0)
    amp_scale_with_sub = float((real_w - mean).abs().max())
    amp_scale_no_sub = float(real_w.abs().max())
    return mean, amp_scale_with_sub, amp_scale_no_sub


def plot_meansub_norm(
    real_w, mean, amp_scale_sub, amp_scale_no_sub, freqs,
    title_prefix, fname, out_dir, channel_idx, channel_name, part_name,
    n_overlay=30, zoom_band=(2e-3, 2.2e-3),
):
    """One channel, one part (Re or Im) → 2 rows × 4 columns figure.

    Columns (stages of the AE preprocessing):
      0. overlaid signals + training-set mean (black)
      1. residual = signal − mean
      2. (signal − mean) / amp_scale_with_sub
      3. signal / amp_scale_no_sub                 (the no-mean-sub branch)

    Row layout:
      row 0:  full frequency band, log-x. Two vertical lines + light grey
              shading mark the ``zoom_band``.
      row 1:  same quantities, linear-x, ``xlim = zoom_band``.
              Dashed connectors link the bottom of each row-0 vertical
              line to the corresponding top corner of the row-1 axis.

    :param real_w: (N, 2C, F) whitened real_imag tensor (already on CPU).
    :param mean:   (2C, F) mean computed on the training set.
    :param part_name: ``"Re"`` selects channel ``channel_idx``;
                      ``"Im"`` selects ``channel_idx + C``.
    """
    from matplotlib.patches import ConnectionPatch
    from matplotlib.transforms import blended_transform_factory

    N, twoC, F = real_w.shape
    C = twoC // 2
    if part_name == "Re":
        idx = channel_idx
    elif part_name == "Im":
        idx = C + channel_idx
    else:
        raise ValueError(f"part_name must be 'Re' or 'Im', got {part_name!r}")
    n_overlay = min(n_overlay, N)

    f_np = np.asarray(freqs)
    zfmin, zfmax = zoom_band
    zoom_mask = (f_np >= zfmin) & (f_np <= zfmax)
    f_zoom = f_np[zoom_mask]
    zoom_label = f"[{zfmin*1e3:.2f}, {zfmax*1e3:.2f}] mHz"

    # Pre-extract numpy data for the requested part.
    samples = real_w[:n_overlay, idx, :].numpy()                 # (n_overlay, F)
    mean_curve = mean[idx].numpy()                                # (F,)
    residuals = samples - mean_curve[None, :]
    residuals_norm = residuals / amp_scale_sub
    samples_norm = samples / amp_scale_no_sub

    # (column index, label, per-sample data, optional mean overlay, ylim_norm)
    stages = [
        (0, "signals + mean",                  samples,        mean_curve, None),
        (1, "residual = signal − mean",         residuals,      None,        None),
        (2, f"(residual) / amp_scale\n"
            f"amp_scale_with_sub = {amp_scale_sub:.3e}",
                                                residuals_norm, None,        (-1.1, 1.1)),
        (3, f"signal / amp_scale (no mean sub)\n"
            f"amp_scale_no_sub = {amp_scale_no_sub:.3e}",
                                                samples_norm,   None,        (-1.1, 1.1)),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(22, 9))

    for col, title, y_per_sample, mean_overlay, ylim_norm in stages:
        ax0 = axes[0, col]
        ax1 = axes[1, col]

        # ---------- row 0: full band ----------
        for k in range(n_overlay):
            ax0.plot(f_np, y_per_sample[k], color="C0", alpha=0.25, lw=0.5)
        if mean_overlay is not None:
            ax0.plot(f_np, mean_overlay, color="black", lw=1.5,
                     label="training-set mean")
            ax0.legend(fontsize=7, loc="upper left")
        else:
            ax0.axhline(0.0, color="black", lw=0.6, ls="--")
        ax0.set_xscale("log")
        ax0.set_xlabel("f (Hz)")
        ax0.set_title(f"Ch {channel_name} {part_name} — {title}")
        ax0.grid(True, ls="--", lw=0.3)
        if ylim_norm is not None:
            ax0.set_ylim(*ylim_norm)
        # Mark the zoom band on row 0.
        ax0.axvspan(zfmin, zfmax, color="gray", alpha=0.15, zorder=0)
        ax0.axvline(zfmin, color="gray", lw=0.8, ls="--", alpha=0.8)
        ax0.axvline(zfmax, color="gray", lw=0.8, ls="--", alpha=0.8)

        # ---------- row 1: zoomed band ----------
        for k in range(n_overlay):
            ax1.plot(f_zoom, y_per_sample[k, zoom_mask],
                     color="C0", alpha=0.45, lw=0.7)
        if mean_overlay is not None:
            ax1.plot(f_zoom, mean_overlay[zoom_mask], color="black", lw=1.2)
        else:
            ax1.axhline(0.0, color="black", lw=0.5, ls="--")
        ax1.set_xlim(zfmin, zfmax)
        ax1.set_xlabel("f (Hz)")
        ax1.set_title(f"zoom {zoom_label}")
        ax1.grid(True, ls=":", lw=0.3)
        ax1.ticklabel_format(axis="x", style="sci", scilimits=(-3, -3),
                              useMathText=True)
        if ylim_norm is not None:
            ax1.set_ylim(*ylim_norm)

    axes[0, 0].set_ylabel(f"{part_name}( whitened )")
    axes[1, 0].set_ylabel(f"{part_name}( whitened )")

    fig.suptitle(title_prefix, fontsize=12)
    fig.tight_layout()

    # ---------- Connectors (added after tight_layout so positions stick) ----
    # For each column: connect (zfmin, bottom of ax0) → top-left of ax1
    #                  and    (zfmax, bottom of ax0) → top-right of ax1
    # Blended transform: x in data, y in axes-fraction → bottom of ax0.
    for col in range(4):
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5", required=True,
                        help="Path to an HDF5 simulation file with wave_fd / asd / "
                             "frequencies. Used for both the plotted samples and "
                             "the AE mean / amplitude_scale stats.")
    parser.add_argument("--event-idx", type=int, default=0,
                        help="Index of the event to plot in the FD amplitude figures")
    parser.add_argument("--n-samples", type=int, default=2000,
                        help="Samples to load for distribution histograms and "
                             "mean-subtraction / normalisation plots")
    parser.add_argument("--n-overlay", type=int, default=1,
                        help="Number of overlaid signals in the mean-sub/norm plots")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory (default: plots/{tag}/whitening/)")
    args = parser.parse_args()

    if args.out_dir is None:
        # Build a tag that mirrors the run-path convention
        # YYYY/MM/DD/{ds_type}_{name}/simulation_round_i, when --hdf5 lives
        # under DATA_ROOT_DIR; otherwise fall back to just the file stem.
        hdf5_abs = os.path.abspath(args.hdf5)
        data_root_abs = os.path.abspath(DATA_ROOT_DIR)
        stem = os.path.splitext(os.path.basename(hdf5_abs))[0]
        parent = os.path.dirname(hdf5_abs)
        if parent.startswith(data_root_abs + os.sep):
            rel_parent = os.path.relpath(parent, data_root_abs)
            tag = os.path.join(rel_parent, stem)
        else:
            tag = stem
        args.out_dir = os.path.join(ROOT_DIR, "plots", tag, "whitening")
    os.makedirs(args.out_dir, exist_ok=True)

    wave_fd, noise_fd, freqs, whitening, T_obs = load_data(args.hdf5, args.n_samples)
    print(f"[diagnose_whitening] N={wave_fd.shape[0]} samples, "
          f"shape={tuple(wave_fd.shape)}, T_obs={T_obs:.3e} s")
    print(f"[diagnose_whitening] FMIN_FLOOR={FMIN_FLOOR} Hz")
    print(f"[diagnose_whitening] out_dir={args.out_dir}")

    data_fd = wave_fd + noise_fd
    wave_w = whiten(wave_fd, whitening)
    data_w = whiten(data_fd, whitening)
    noise_w = whiten(noise_fd, whitening)

    idx = args.event_idx
    plot_fd_amp(freqs, wave_fd, "Original signal |h(f)|",
                "1_original_signal.png", args.out_dir, idx)
    plot_fd_amp(freqs, wave_w, "Whitened signal |h_w(f)|",
                "2_whitened_signal.png", args.out_dir, idx)
    plot_fd_amp_overlay(freqs, [noise_fd, data_fd],
                        ["pure noise", "signal + noise"],
                        "Pure noise vs signal+noise",
                        "3_noise_and_signal_plus_noise.png", args.out_dir, idx)
    plot_fd_amp(freqs, data_w, "Whitened signal+noise |d_w(f)|",
                "4_whitened_signal_plus_noise.png", args.out_dir, idx)
    plot_distribution(noise_w, freqs,
                      "Whitened noise — expect N(0, 1) per quadrature",
                      "5_whitened_noise_distribution.png", args.out_dir)
    plot_distribution(data_w, freqs,
                      "Whitened signal+noise (signal tail breaks Gaussian)",
                      "6_whitened_signal_plus_noise_distribution.png", args.out_dir)

    # ------------------------------------------------------------------
    # Plots 7-8: AE preprocessing — mean subtraction + amplitude norm.
    # Stats are fitted on the same clean waveforms we then plot, mirroring
    # DenoisingAutoencoder.fit_amplitude_normalisation on the training set.
    # ------------------------------------------------------------------
    real_wave = complex_to_real_imag(wave_w)
    real_data = complex_to_real_imag(data_w)
    mean_w, amp_scale_sub, amp_scale_no_sub = compute_norm_stats(real_wave)
    print(f"[diagnose_whitening] AE stats fitted on N={real_wave.shape[0]}: "
          f"mean range=[{mean_w.min().item():.3e}, {mean_w.max().item():.3e}], "
          f"amp_scale_with_sub={amp_scale_sub:.3e}, "
          f"amp_scale_no_sub={amp_scale_no_sub:.3e}")

    n_channels = wave_fd.shape[1]
    for c in range(n_channels):
        ch_name = CHANNEL_NAMES[c] if c < len(CHANNEL_NAMES) else str(c)
        for part in ("Re", "Im"):
            plot_meansub_norm(
                real_wave, mean_w, amp_scale_sub, amp_scale_no_sub, freqs,
                f"AE preprocessing on clean waveform — channel {ch_name} ({part})",
                f"7_meansub_norm_wave_ch{ch_name}_{part}.png", args.out_dir,
                channel_idx=c, channel_name=ch_name, part_name=part,
                n_overlay=args.n_overlay,
            )
            plot_meansub_norm(
                real_data, mean_w, amp_scale_sub, amp_scale_no_sub, freqs,
                f"AE preprocessing on signal+noise — channel {ch_name} ({part})",
                f"8_meansub_norm_data_ch{ch_name}_{part}.png", args.out_dir,
                channel_idx=c, channel_name=ch_name, part_name=part,
                n_overlay=args.n_overlay,
            )

    print(f"[diagnose_whitening] done → {args.out_dir}")


if __name__ == "__main__":
    main()
