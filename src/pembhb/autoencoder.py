"""
Convolutional denoising autoencoder for frequency-domain MBHB data.

The encoder compresses noisy input (wave_fd + noise_fd) to a low-dimensional
bottleneck, and the decoder reconstructs the clean signal (wave_fd).
The bottleneck representation can be extracted separately for use as a
data summarizer in InferenceNetwork.

Two architectures are supported:
- "conv": Pure convolutional autoencoder (no skip connections) - RECOMMENDED
- "unet": Unet-based autoencoder with skip connections (reconstruction aided by encoder features)
"""

import warnings

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from lightning import LightningModule
from torch.utils.data import DataLoader

from pembhb.model import DoubleConv, Down, Up, OutConv
from pembhb import get_torch_dtype
from pembhb.utils import materialize_gpu_noise, GPUNoiseMixin


def resolve_loss_band(freqs, ae_conf):
    """Resolve the autoencoder loss-mask band from an autoencoder config dict.

    Reads ``fmin_loss`` / ``fmax_loss`` (Hz) from ``ae_conf`` and converts
    them to bin indices via the supplied ``freqs`` grid. Falls back to the
    legacy ``idx_lowerbound`` / ``idx_upperbound`` (raw indices) when the
    Hz keys are absent. Warns if both forms are present.

    :param freqs: 1D numpy array of frequencies (the simulator's grid).
    :param ae_conf: autoencoder sub-config dict from train_config.yaml.
    :return: ``(idx_lo, idx_hi)`` — either may be ``None`` to mean no bound.
    """
    fmin_loss = ae_conf.get("fmin_loss", None)
    fmax_loss = ae_conf.get("fmax_loss", None)
    idx_lo_legacy = ae_conf.get("idx_lowerbound", None)
    idx_hi_legacy = ae_conf.get("idx_upperbound", None)

    has_hz = fmin_loss is not None or fmax_loss is not None
    has_legacy = idx_lo_legacy is not None or idx_hi_legacy is not None

    if has_hz:
        if has_legacy:
            warnings.warn(
                "ae_conf has both fmin_loss/fmax_loss (Hz) and "
                "idx_lowerbound/idx_upperbound (bin idx); using fmin_loss/fmax_loss."
            )
        idx_lo = int(np.searchsorted(freqs, fmin_loss)) if fmin_loss is not None else None
        idx_hi = int(np.searchsorted(freqs, fmax_loss)) if fmax_loss is not None else None
        return idx_lo, idx_hi

    return idx_lo_legacy, idx_hi_legacy


# ---------------------------------------------------------------------------
# Pure Convolutional Encoder / Decoder (no skip connections)
# ---------------------------------------------------------------------------

class ResidualConvBlock(nn.Module):
    """Single strided conv layer with a residual shortcut.

    ``output = LeakyReLU( BN(Conv(x)) + shortcut(x) )``

    A 1×1 convolution is used as the shortcut whenever the channel count or
    spatial dimension changes (i.e. ``in_ch != out_ch`` or ``stride != 1``).
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int, dropout: float = 0.0):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, stride, padding=kernel_size // 2),
            nn.BatchNorm1d(out_ch),
        )
        if in_ch != out_ch or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.dropout(self.activation(self.main(x) + self.shortcut(x)))


class ConvEncoder(nn.Module):
    """Pure convolutional encoder.

    Compresses input to a fixed-size bottleneck vector.

    Architecture:
        Input (B, C_in, L)
        → [Conv1d + BN + LeakyReLU + Dropout] × N_layers (with stride=2 downsampling)
        → Flatten
        → Linear → bottleneck (B, bottleneck_dim)

    When ``residual=True`` each conv layer is wrapped in a
    :class:`ResidualConvBlock` that adds a 1×1 shortcut, improving
    gradient flow through deep encoders (≥ 5 layers).
    """

    def __init__(
        self,
        n_in_channels: int,
        n_freqs: int,
        bottleneck_dim: int = 128,
        hidden_channels: tuple = (32, 64, 128, 256, 256),
        kernel_size: int = 4,
        stride: int = 2,
        dropout: float = 0.0,
        residual: bool = False,
    ):
        super().__init__()
        self.bottleneck_dim = bottleneck_dim
        self.n_freqs = n_freqs

        # Build conv layers with strided downsampling
        in_ch = n_in_channels
        if residual:
            blocks = []
            for out_ch in hidden_channels:
                blocks.append(ResidualConvBlock(in_ch, out_ch, kernel_size, stride, dropout))
                in_ch = out_ch
            self.conv = nn.Sequential(*blocks)
        else:
            layers = []
            for out_ch in hidden_channels:
                layers.extend([
                    nn.Conv1d(in_ch, out_ch, kernel_size, stride, padding=kernel_size // 2),
                    nn.BatchNorm1d(out_ch),
                    nn.LeakyReLU(0.2, inplace=True),
                ])
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                in_ch = out_ch
            self.conv = nn.Sequential(*layers)

        # Compute flattened size after convolutions
        with torch.no_grad():
            dummy = torch.zeros(1, n_in_channels, n_freqs)
            out = self.conv(dummy)
            self.pre_fc_channels = out.shape[1]
            self.pre_fc_length = out.shape[2]
            self.flat_size = out.shape[1] * out.shape[2]

        # Linear projection to bottleneck
        self.fc_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(self.flat_size, bottleneck_dim)

    def forward(self, x):
        """Encode input to bottleneck vector.

        :param x: (B, C_in, L) input tensor
        :return: (B, bottleneck_dim) bottleneck vector
        """
        x = self.conv(x)
        x = x.flatten(1)  # (B, C * L)
        x = self.fc_dropout(x)
        return self.fc(x)  # (B, bottleneck_dim)


class ConvDecoder(nn.Module):
    """Pure convolutional decoder without skip connections.
    
    Reconstructs signal from bottleneck vector.
    
    Architecture:
        bottleneck (B, bottleneck_dim)
        → Linear → Reshape (B, C, L)
        → [ConvTranspose1d + BN + LeakyReLU] × (N_layers - 1)
        → ConvTranspose1d → Output (B, C_out, L_out)
    """
    
    def __init__(
        self,
        n_out_channels: int,
        n_freqs: int,
        bottleneck_dim: int = 128,
        hidden_channels: tuple = (32, 64, 128, 256, 256),
        kernel_size: int = 4,
        stride: int = 2,
        pre_fc_channels: int = None,
        pre_fc_length: int = None,
        dropout: float = 0.0,
        decoder_post_fc_bn: bool = True,
    ):
        super().__init__()
        self.n_freqs = n_freqs
        self.pre_fc_channels = pre_fc_channels
        self.pre_fc_length = pre_fc_length

        # Linear from bottleneck to pre-conv shape
        self.fc = nn.Linear(bottleneck_dim, pre_fc_channels * pre_fc_length)
        self.fc_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # Optional BatchNorm right after the bottleneck → conv-shape projection.
        # Set False to recover the architecture used at commit cb948642.
        self.post_fc_norm = (
            nn.BatchNorm1d(pre_fc_channels) if decoder_post_fc_bn else nn.Identity()
        )

        # Build conv transpose layers (reverse order of encoder)
        reversed_ch = list(reversed(hidden_channels))
        layers = []
        
        for i in range(len(reversed_ch) - 1):
            in_ch = reversed_ch[i]
            out_ch = reversed_ch[i + 1]
            layers.extend([
                nn.ConvTranspose1d(
                    in_ch, out_ch, kernel_size, stride,
                    padding=kernel_size // 2, output_padding=stride - 1
                ),
                nn.BatchNorm1d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
            ])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        
        # Final layer to output channels (no activation - linear output)
        layers.append(
            nn.ConvTranspose1d(
                reversed_ch[-1], n_out_channels, kernel_size, stride,
                padding=kernel_size // 2, output_padding=stride - 1
            )
        )
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, bottleneck):
        """Decode bottleneck to reconstructed signal.
        
        :param bottleneck: (B, bottleneck_dim) bottleneck vector
        :return: (B, C_out, L_out) reconstructed signal
        """
        x = self.fc(bottleneck)
        x = self.fc_dropout(x)
        x = x.view(-1, self.pre_fc_channels, self.pre_fc_length)
        x = self.post_fc_norm(x)
        x = self.conv(x)
        
        # Crop or pad to exact output size if needed
        if x.shape[2] != self.n_freqs:
            if x.shape[2] > self.n_freqs:
                x = x[:, :, :self.n_freqs]
            else:
                pad_size = self.n_freqs - x.shape[2]
                x = F.pad(x, (0, pad_size))
        return x



class FreqBinCompressor(nn.Module): 

    def __init__(self, n_window) : 
        super().__init__() 
        self.n_window = n_window
        
    def forward(self, x):

        B, C, N = x.shape
        n_in_freqs = N
        n_out_freqs = N // self.n_window + (int(N % self.n_window != 0))  # ceil division
        W = self.n_window
        # Pad so N is a multiple of W
        remainder = N % W
        if remainder != 0:
            x = F.pad(x, (0, W - remainder))  # zero-pad last dim
        N_padded = x.shape[-1]
        x = x.view(B, C, N_padded // W, W)   # (B, C, n_out, W)
        mean = x.mean(dim=-1)                  # (B, C, n_out)
        std  = x.std(dim=-1, correction=1)     # (B, C, n_out)
        return torch.cat([mean, std], dim=1)   # (B, 2*C, n_out)


# ---------------------------------------------------------------------------
# Unet-based Encoder / Decoder (with skip connections) - LEGACY
# ---------------------------------------------------------------------------

class UnetEncoder(nn.Module):
    """Encoder (contracting) path of a 1-D Unet.

    Produces skip-connection feature maps *and* the bottleneck tensor.
    """

    def __init__(
        self,
        n_in_channels: int,
        sizes: tuple = (16, 32, 64, 128, 256),
        down_sampling: tuple = (2, 2, 2, 2),
    ):
        super().__init__()
        self.inc = DoubleConv(n_in_channels, sizes[0])
        self.down1 = Down(sizes[0], sizes[1], down_sampling[0])
        self.down2 = Down(sizes[1], sizes[2], down_sampling[1])
        self.down3 = Down(sizes[2], sizes[3], down_sampling[2])
        self.down4 = Down(sizes[3], sizes[4], down_sampling[3])

    def forward(self, x):
        """Return (bottleneck, [skip1, skip2, skip3, skip4])."""
        x1 = self.inc(x)       # (B, sizes[0], L)
        x2 = self.down1(x1)    # (B, sizes[1], L/d0)
        x3 = self.down2(x2)    # (B, sizes[2], L/d0/d1)
        x4 = self.down3(x3)    # (B, sizes[3], L/d0/d1/d2)
        x5 = self.down4(x4)    # (B, sizes[4], L/d0/d1/d2/d3)  ← bottleneck
        return x5, [x1, x2, x3, x4]


class UnetDecoder(nn.Module):
    """Decoder (expanding) path of a 1-D Unet."""

    def __init__(
        self,
        n_out_channels: int,
        sizes: tuple = (16, 32, 64, 128, 256),
    ):
        super().__init__()
        self.up1 = Up(sizes[4], sizes[3])
        self.up2 = Up(sizes[3], sizes[2])
        self.up3 = Up(sizes[2], sizes[1])
        self.up4 = Up(sizes[1], sizes[0])
        self.outc = OutConv(sizes[0], n_out_channels)

    def forward(self, bottleneck, skips):
        """Reconstruct from bottleneck + skip connections.

        :param bottleneck: tensor of shape (B, sizes[-1], L_bottleneck)
        :param skips: list [x1, x2, x3, x4] from encoder
        :return: reconstructed tensor of shape (B, n_out_channels, L_input)
        """
        x1, x2, x3, x4 = skips
        x = self.up1(bottleneck, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


# ---------------------------------------------------------------------------
# Denoising Autoencoder (LightningModule)
# ---------------------------------------------------------------------------

class DenoisingAutoencoder(GPUNoiseMixin, LightningModule):
    """Convolutional denoising autoencoder for frequency-domain gravitational-wave data.

    Training
    --------
    * **Input** : ``batch["wave_fd"] + batch["noise_fd"]``  (noisy complex signal)
    * **Target**: ``batch["wave_fd"]``                       (clean complex signal)
    * Both are normalised before being fed to the network:
        1. Convert complex → real by stacking two channels per TDI channel
           along the channel axis (``C`` → ``2*C``).
           The representation is controlled by ``representation``:
           - ``"amp_phase"``: ``[log|amplitude|, phase]``
           - ``"real_imag"``: ``[real, imag]``
        2. Center by subtracting a running mean.
        3. Scale by dividing by a running global max.
    * The loss is MSE between the output and the normalised-scaled clean signal.

    Architecture
    ------------
    Two architectures are supported via the ``architecture`` parameter:
    
    * ``"conv"`` (default, recommended): Pure convolutional autoencoder.
      The decoder reconstructs **only from the bottleneck** (no skip connections).
      This is a true compression-decompression architecture.
      
    * ``"unet"``: Unet-based autoencoder with skip connections.
      The decoder uses both the bottleneck AND encoder feature maps.
      WARNING: This means reconstruction is aided by encoder features,
      not purely from the compressed bottleneck representation.

    Inference / data-summary mode
    -----------------------------
    After training, wrap this module in :class:`AutoencoderWrapper` and pass
    it to ``InferenceNetwork`` as ``data_summarizer``.  Calling
    ``AutoencoderWrapper.forward(d_f, d_t)`` returns the **flattened
    bottleneck** (not the reconstructed signal).
    """

    VALID_REPRESENTATIONS = ("amp_phase", "real_imag")
    VALID_ARCHITECTURES = ("conv", "unet")

    def __init__(
        self,
        n_channels: int = 2,
        n_freqs: int = None,
        # --- Conv architecture params ---
        architecture: str = "conv",
        bottleneck_dim: int = 128,
        hidden_channels: tuple = (32, 64, 128, 256, 256),
        kernel_size: int = 4,
        stride: int = 2,
        dropout: float = 0.0,
        residual: bool = False,
        decoder_post_fc_bn: bool = True,
        # --- Unet architecture params (legacy) ---
        sizes: tuple = (16, 32, 64, 128, 256),
        down_sampling: tuple = (2, 2, 2, 2),
        # --- Training params ---
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        scheduler_patience: int = 10,
        scheduler_factor: float = 0.5,
        representation: str = "real_imag",
        # --- Reconstruction band masking ---
        # Encoder always sees the full input; only the reconstruction
        # target / decoder output is restricted to bins
        # [idx_lowerbound : idx_upperbound]. ``None`` on either side means
        # "no cut on that side"; both ``None`` reconstructs the full band.
        idx_lowerbound: int | None = None,
        idx_upperbound: int | None = None,
        # --- DEPRECATED: old high-freq-only API, kept for back-compat ---
        # Maps to idx_lowerbound=freq_split_idx, idx_upperbound=None.
        high_freq_only: bool = False,
        freq_split_idx: int = 2048,
        # --- Optional global amplitude normalisation on top of whitening ---
        amplitude_normalise: bool = True,
        subtract_mean_whitened: bool = True,
        whiten: bool = True,
        # --- Frequency-bin compression ---
        # Window width for non-overlapping block average+std before the encoder.
        # None = disabled (backward compatible). When set, band masking is ignored.
        compressor_window: int = None,
        # When compression is active the encoder always sees both the per-block
        # mean and std channels. ``reconstruct_std=False`` makes the decoder
        # output (and the MSE target) only the mean channels, so the std is
        # summarised by the bottleneck but never reconstructed. Requires
        # compressor_window to be set.
        reconstruct_std: bool = True,
        # --- Prior bounds (for provenance tracking) ---
        prior_bounds: dict = None,
    ):
        super().__init__()
        if representation not in self.VALID_REPRESENTATIONS:
            raise ValueError(
                f"representation must be one of {self.VALID_REPRESENTATIONS}, "
                f"got '{representation}'"
            )
        if architecture not in self.VALID_ARCHITECTURES:
            raise ValueError(
                f"architecture must be one of {self.VALID_ARCHITECTURES}, "
                f"got '{architecture}'"
            )
        if (amplitude_normalise or subtract_mean_whitened) and representation != "real_imag":
            raise NotImplementedError(
                "amplitude_normalise / subtract_mean_whitened are only "
                "supported with representation='real_imag'."
            )
        self.save_hyperparameters()

        self.n_channels = n_channels
        self.n_freqs = n_freqs
        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler_patience = scheduler_patience
        self.scheduler_factor = scheduler_factor
        self.representation = representation
        self.architecture = architecture
        self.bottleneck_dim = bottleneck_dim
        self.dropout = dropout
        self.amplitude_normalise = amplitude_normalise
        self.whiten = whiten
        self.subtract_mean_whitened = subtract_mean_whitened
        self.reconstruct_std = reconstruct_std
        if not reconstruct_std and compressor_window is None:
            raise ValueError(
                "reconstruct_std=False requires compressor_window to be set "
                "(there are no std channels to drop without compression)."
            )
        # Complex → real representation doubles the channels
        n_real_channels = n_channels * 2
        self.n_real_channels = n_real_channels

        if compressor_window is not None:
            self.compressor = FreqBinCompressor(compressor_window)
            n_encoder_channels = n_real_channels * 2  # mean + std doubles channels
            # Decoder reconstructs all 4C channels, or only the 2C mean channels
            # when reconstruct_std=False.
            n_decoder_channels = n_encoder_channels if reconstruct_std else n_real_channels
            n_encoder_freqs = int(np.ceil(n_freqs / compressor_window))
            n_freqs_target = n_encoder_freqs
            self.idx_lowerbound = 0
            self.idx_upperbound = n_encoder_freqs
            self._mask_active = False
            self.high_freq_only = False
            self.freq_split_idx = 0
            print(
                f"[AutoEncoder] FreqBinCompressor active: {n_freqs} → "
                f"{n_encoder_freqs} bins (window={compressor_window}), "
                f"{n_real_channels} → {n_encoder_channels} channels; "
                f"decoder reconstructs {n_decoder_channels} channels "
                f"(reconstruct_std={reconstruct_std})"
            )
        else:
            self.compressor = None
            n_encoder_channels = n_real_channels
            n_decoder_channels = n_real_channels
            n_encoder_freqs = n_freqs
            # Resolve old (high_freq_only/freq_split_idx) and new
            # (idx_lowerbound/idx_upperbound) APIs into a single internal
            # representation. Old takes precedence only if explicitly enabled.
            if high_freq_only:
                if idx_lowerbound is not None or idx_upperbound is not None:
                    raise ValueError(
                        "Cannot mix deprecated high_freq_only with new "
                        "idx_lowerbound/idx_upperbound; use only the new API."
                    )
                warnings.warn(
                    "high_freq_only/freq_split_idx are deprecated; "
                    "use idx_lowerbound/idx_upperbound instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                idx_lo = freq_split_idx
                idx_hi = n_freqs
            else:
                idx_lo = 0 if idx_lowerbound is None else int(idx_lowerbound)
                idx_hi = n_freqs if idx_upperbound is None else int(idx_upperbound)

            if not (0 <= idx_lo < idx_hi <= n_freqs):
                raise ValueError(
                    f"Invalid band mask: idx_lowerbound={idx_lo}, "
                    f"idx_upperbound={idx_hi}, n_freqs={n_freqs}. "
                    f"Required: 0 <= lo < hi <= n_freqs."
                )

            self.idx_lowerbound = idx_lo
            self.idx_upperbound = idx_hi
            self._mask_active = (idx_lo > 0) or (idx_hi < n_freqs)
            self.high_freq_only = self._mask_active
            self.freq_split_idx = idx_lo

            n_freqs_target = idx_hi - idx_lo
            if self._mask_active:
                print(
                    f"[AutoEncoder] Band mask active: reconstructing bins "
                    f"[{idx_lo}:{idx_hi}] ({n_freqs_target} bins out of {n_freqs})"
                )

        if architecture == "conv":
            # Pure convolutional autoencoder (no skip connections)
            self.encoder = ConvEncoder(
                n_in_channels=n_encoder_channels,
                n_freqs=n_encoder_freqs,
                bottleneck_dim=bottleneck_dim,
                hidden_channels=hidden_channels,
                kernel_size=kernel_size,
                stride=stride,
                dropout=dropout,
                residual=residual,
            )
            self.decoder = ConvDecoder(
                n_out_channels=n_decoder_channels,
                n_freqs=n_freqs_target,
                bottleneck_dim=bottleneck_dim,
                hidden_channels=hidden_channels,
                kernel_size=kernel_size,
                stride=stride,
                pre_fc_channels=self.encoder.pre_fc_channels,
                pre_fc_length=self.encoder.pre_fc_length,
                dropout=dropout,
                decoder_post_fc_bn=decoder_post_fc_bn,
            )
        else:  # unet
            # Unet-based autoencoder with skip connections (legacy)
            self.sizes = sizes
            self.encoder = UnetEncoder(
                n_in_channels=n_real_channels,
                sizes=sizes,
                down_sampling=down_sampling,
            )
            self.decoder = UnetDecoder(
                n_out_channels=n_real_channels,
                sizes=sizes,
            )

        # Whitening scale: noise_scale = ASD * sqrt(T_obs/4), shape (C, F).
        # Set via set_whitening() before training; persisted in state_dict so
        # resume picks it up automatically. Initialised to 1 (identity) so
        # forward passes are well-defined before set_whitening is called.
        self.register_buffer("whitening", torch.ones(n_channels, n_freqs, dtype=get_torch_dtype()))

        # Optional global amplitude scale: scalar max over (samples, channels,
        # freqs) of the whitened clean training signal. Populated by
        # fit_white_normalisation(). Initialised to 1 (identity).
        # When compression is active, amplitude_scale covers the mean channels
        # and amplitude_scale_std covers the std channels independently.
        self.register_buffer("amplitude_scale",     torch.tensor(1.0, dtype=get_torch_dtype()))
        self.register_buffer("amplitude_scale_std", torch.tensor(1.0, dtype=get_torch_dtype()))
        self.register_buffer("mean_whitened", torch.zeros(n_channels * 2, n_freqs, dtype=get_torch_dtype()))

    # ------------------------------------------------------------------
    # Complex → real conversion
    # ------------------------------------------------------------------
    def _complex_to_real(self, z: torch.Tensor) -> torch.Tensor:
        """Convert a complex tensor (B, C, F) → real tensor (B, 2C, F).

        The channel layout depends on ``self.representation``:

        * ``"amp_phase"``:  [log|amp_ch0|, …, phase_ch0, …]
        * ``"real_imag"``:  [re_ch0, …, im_ch0, …]
        """
        if self.representation == "amp_phase":
            amplitude = torch.abs(z)
            phase = torch.angle(z)
            log_amplitude = torch.log(amplitude + 1e-33)
            return torch.cat([log_amplitude, phase], dim=1)
        else:  # real_imag
            return torch.cat([z.real, z.imag], dim=1)

    def _real_to_complex(self, x_real: torch.Tensor) -> torch.Tensor:
        """Convert a real tensor (B, 2C, F) → complex tensor (B, C, F).

        Inverse of :meth:`_complex_to_real`.
        """
        C = self.n_channels
        if self.representation == "amp_phase":
            log_amp = x_real[:, :C, :]
            phase = x_real[:, C:, :]
            return torch.exp(log_amp) * torch.exp(1j * phase)
        else:  # real_imag
            re = x_real[:, :C, :]
            im = x_real[:, C:, :]
            return torch.complex(re, im)

    # ------------------------------------------------------------------
    # Whitening
    # ------------------------------------------------------------------

    def set_whitening(self, noise_scale: torch.Tensor) -> None:
        """Store ``noise_scale = ASD * sqrt(T_obs/4)`` as the whitening scale.

        Zero bins (e.g. below the high-pass cutoff) are mapped to ``inf`` so
        that those frequencies whiten to 0.

        :param noise_scale: real tensor of shape ``(n_channels, n_freqs)``.
        """
        whitening_safe = noise_scale.to(self.whitening.dtype).clone()
        whitening_safe[noise_scale == 0] = float("inf")
        self.whitening.copy_(whitening_safe)
        nonzero = noise_scale[noise_scale > 0]
        print(
            f"[AutoEncoder] whitening set (shape={tuple(self.whitening.shape)}, "
            f"range=[{nonzero.min().item():.4e}, {nonzero.max().item():.4e}])"
        )

    def _get_whitening_for(self, n_freq: int) -> torch.Tensor:
        """Return the slice of ``whitening`` matching the decoder output size."""
        if n_freq == self.n_freqs:
            return self.whitening
        masked_size = self.idx_upperbound - self.idx_lowerbound
        if self._mask_active and n_freq == masked_size:
            return self.whitening[:, self.idx_lowerbound:self.idx_upperbound]
        raise ValueError(
            f"Frequency dimension {n_freq} does not match full ({self.n_freqs}) "
            f"or masked ({masked_size}) expected size."
        )
    def _prewhiten_real(self, z: torch.Tensor) -> torch.Tensor:
        """Complex FD (B, C, F) → real channels (B, 2C, F), whitened iff
        ``self.whiten``. This is the common representation that both
        ``preprocess`` and the amplitude/mean fit operate on, so the two
        stay in sync for both the whitened and unwhitened paths."""
        return self._complex_to_real(z / self.whitening if self.whiten else z)

    def _get_mean_whitened_for(self, n_freq: int) -> torch.Tensor:
        """Slice ``mean_whitened`` to match the decoder output size (handles band mask)."""
        if n_freq == self.n_freqs:
            return self.mean_whitened
        masked_size = self.idx_upperbound - self.idx_lowerbound
        if self._mask_active and n_freq == masked_size:
            return self.mean_whitened[:, self.idx_lowerbound:self.idx_upperbound]
        raise ValueError(
            f"Frequency dimension {n_freq} does not match full ({self.n_freqs}) "
            f"or masked ({masked_size}) expected size."
        )
    # ------------------------------------------------------------------
    # Optional amplitude normalisation (real_imag only)
    # ------------------------------------------------------------------

    def fit_white_normalisation(self, dataloader: DataLoader) -> None:
        """Fit the mean and/or global amplitude scale used by ``preprocess``.

        Operates on the same real representation ``preprocess`` sees
        (``_prewhiten_real``): whitened clean signals when ``whiten`` is set,
        raw real channels otherwise. ``mean_whitened`` is fitted when
        ``subtract_mean_whitened`` is set; ``amplitude_scale`` is fitted when
        ``amplitude_normalise`` is set. The two flags are independent.
        """
        if not (self.amplitude_normalise or self.subtract_mean_whitened):
            raise RuntimeError(
                "fit_white_normalisation called but neither amplitude_normalise "
                "nor subtract_mean_whitened is set."
            )
        if self.representation != "real_imag":
            raise NotImplementedError(
                "amplitude_normalise / subtract_mean_whitened are only "
                "supported with representation='real_imag'."
            )
        device = next(self.parameters()).device
        n_samples = 0
        running_sum = torch.zeros(
            self.n_channels * 2, self.n_freqs, device=device,
            dtype=self.mean_whitened.dtype,
        )

        # --- Pass 1: accumulate mean on the full grid ---
        with torch.no_grad():
            for batch in dataloader:
                batch = materialize_gpu_noise(batch)
                wave_fd = batch["wave_fd"].to(device)
                real = self._prewhiten_real(wave_fd)  # (B, 2C, F)
                n_samples += real.shape[0]
                if self.subtract_mean_whitened:
                    running_sum += real.sum(dim=0)

        if self.subtract_mean_whitened:
            self.mean_whitened.copy_(running_sum / n_samples)
            print(
                f"[AutoEncoder] mean_whitened fitted on {n_samples} whitened "
                f"clean samples: mean_whitened range=[{self.mean_whitened.min().item():.4e}, "
                f"{self.mean_whitened.max().item():.4e}]"
            )

        if not self.amplitude_normalise:
            return

        # --- Pass 2: compute amplitude scale(s) after mean-subtraction and compression ---
        max_mean = 0.0
        max_std  = 0.0
        with torch.no_grad():
            for batch in dataloader:
                batch = materialize_gpu_noise(batch)
                wave_fd = batch["wave_fd"].to(device)
                real = self._prewhiten_real(wave_fd)  # (B, 2C, F)
                if self.subtract_mean_whitened:
                    real = real - self.mean_whitened
                if self.compressor is not None:
                    real = self.compressor(real)        # (B, 4C, n_compressed)
                    C = self.n_real_channels
                    max_mean = max(max_mean, real[:, :C, :].abs().max().item())
                    max_std  = max(max_std,  real[:, C:, :].abs().max().item())
                else:
                    max_mean = max(max_mean, real.abs().max().item())

        self.amplitude_scale.fill_(max_mean)
        self.amplitude_scale_std.fill_(max_std if self.compressor is not None else 1.0)
        print(
            f"[AutoEncoder] amplitude scales fitted on {n_samples} clean samples: "
            f"amplitude_scale={max_mean:.4e}"
            + (f", amplitude_scale_std={max_std:.4e}" if self.compressor is not None else "")
        )

    def _denormalize_amplitude(self, x_real: torch.Tensor) -> torch.Tensor:
        """Inverse of the amplitude scaling. For visualisation / inverse
        passes — the loss is computed in the doubly-normalised space."""
        if self.amplitude_normalise:
            return x_real * self.amplitude_scale
        return x_real

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def encode(self, x_norm: torch.Tensor):
        """Run the encoder on **already-normalised** real input.

        :param x_norm: (B, 2C, F) normalised real tensor.
        :return: For "conv" architecture: bottleneck (B, bottleneck_dim)
                 For "unet" architecture: (bottleneck, skips)
        """
        return self.encoder(x_norm)

    def decode(self, bottleneck: torch.Tensor, skips=None) -> torch.Tensor:
        """Run the decoder.

        :param bottleneck: For "conv": (B, bottleneck_dim) vector
                          For "unet": (B, sizes[-1], L_bottleneck) tensor
        :param skips: For "conv": ignored (None)
                      For "unet": list of skip-connection tensors from encoder
        :return: (B, 2C, F) reconstructed normalised signal
        """
        if self.architecture == "conv":
            return self.decoder(bottleneck)
        else:  # unet
            return self.decoder(bottleneck, skips)

    def forward(self, x_norm: torch.Tensor) -> torch.Tensor:
        """Full autoencoder pass (encode → decode).

        :param x_norm: (B, 2C, F) normalised real tensor.
        :return: (B, 2C, F) reconstructed normalised signal.
        """
        if self.architecture == "conv":
            bottleneck = self.encode(x_norm)
            return self.decode(bottleneck)
        else:  # unet
            bottleneck, skips = self.encode(x_norm)
            return self.decode(bottleneck, skips)

    # ------------------------------------------------------------------
    # Preprocessing helper (raw complex batch → normalised real tensor)
    # ------------------------------------------------------------------

    def preprocess(self, z: torch.Tensor) -> torch.Tensor:
        """Complex FD data → normalised real tensor. Each step is independently
        gated: optional whitening, optional mean subtraction, optional
        compression, optional global amplitude scaling.

        :param z: complex tensor (B, C, F).
        :return: real tensor (B, 2C, F) — channel layout per ``representation``.
        """
        real = self._prewhiten_real(z)
        if self.subtract_mean_whitened:
            real = real - self._get_mean_whitened_for(z.shape[-1])
        if self.compressor is not None:
            real = self.compressor(real)   # (B, 2*n_real_channels, n_compressed)
        if self.amplitude_normalise:
            if self.compressor is not None:
                C = self.n_real_channels
                real = torch.cat([
                    real[:, :C, :] / self.amplitude_scale,
                    real[:, C:, :] / self.amplitude_scale_std,
                ], dim=1)
            else:
                real = real / self.amplitude_scale
        return real
    # ------------------------------------------------------------------
    # Lightning training / validation steps
    # ------------------------------------------------------------------

    def _compute_extra_metrics(self, reconstructed: torch.Tensor, target: torch.Tensor):
        """Compute absolute error (MAE), relative error, and max relative error.

        :param reconstructed: model output (B, 2C, F)
        :param target: normalised clean signal (B, 2C, F)
        :return: (mae, relative_error, max_relative_error) scalar tensors
        """
        abs_error = (reconstructed - target).abs().mean()
        target_norm = target.norm(dim=(1, 2)).mean()  # mean L2 norm across batch
        relative_error = abs_error / (target_norm + 1e-30)
        
        # Compute per-sample relative error and take the maximum
        per_sample_error_norm = (reconstructed - target).norm(dim=(1, 2))  # (B,)
        per_sample_target_norm = target.norm(dim=(1, 2))  # (B,)
        per_sample_rel_error = per_sample_error_norm / (per_sample_target_norm + 1e-30)  # (B,)
        max_relative_error = per_sample_rel_error.max()
        
        return abs_error, relative_error, max_relative_error

    def _get_target(self, clean_norm: torch.Tensor) -> torch.Tensor:
        """Get the reconstruction target, sliced to the active band mask.

        :param clean_norm: (B, 2C, F) full normalised clean signal
        :return: (B, 2C, F) or (B, 2C, F_target) depending on the mask
        """
        if self._mask_active:
            return clean_norm[:, :, self.idx_lowerbound:self.idx_upperbound]
        if self.compressor is not None and not self.reconstruct_std:
            # Drop the std channels (last n_real_channels) from the target so
            # the MSE only penalises the per-block mean reconstruction.
            return clean_norm[:, :self.n_real_channels, :]
        return clean_norm

    def _step(self, batch, prefix: str):
        noisy = batch["wave_fd"] + batch["noise_fd"]
        clean = batch["wave_fd"]

        # Whiten then convert to real channels.
        noisy_norm = self.preprocess(noisy)
        clean_norm = self.preprocess(clean)

        reconstructed = self(noisy_norm)
        target = self._get_target(clean_norm)

        # MSE on whitened representations equals 4-side noise-weighted MSE.
        loss = F.mse_loss(reconstructed, target)
        mae, rel_err, max_rel_err = self._compute_extra_metrics(reconstructed, target)

        self.log(f"{prefix}_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{prefix}_mae", mae, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log(f"{prefix}_rel_err", rel_err, on_step=True, on_epoch=True, prog_bar=(prefix == "val"), logger=True)
        self.log(f"{prefix}_max_rel_err", max_rel_err, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.scheduler_factor,
            patience=self.scheduler_patience,
            min_lr=1e-7,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }


# ---------------------------------------------------------------------------
# Diagnostic plot for the block-compression autoencoder
# ---------------------------------------------------------------------------

@torch.no_grad()
def plot_compression_reconstruction(
    ae: "DenoisingAutoencoder",
    batch: dict,
    sample_idx: int = 0,
    freqs: np.ndarray = None,
    save_path: str = None,
):
    """Overlay original / target / reconstruction for a block-compression AE.

    Tailored to ``compressor_window`` runs (and especially
    ``reconstruct_std=False``). Everything is shown in the **mean** channels,
    in the exact normalised, block-averaged space the MSE loss is computed in
    — i.e. ``preprocess`` output, no inversion to physical units:

    * **original**       — encoder input ``preprocess(wave + noise)``, mean channels.
                           This is the whitened, mean-subtracted, block-averaged
                           signal that is fed to the encoder.
    * **target**         — ``_get_target(preprocess(wave))``, mean channels
                           (the clean signal the decoder is trained to match).
    * **reconstruction** — decoder output ``ae(encoder_input)``, mean channels.

    The std channels are not plotted: with ``reconstruct_std=False`` they are
    not reconstructed, and with ``reconstruct_std=True`` they live in the
    second half of the channel axis and are summarised separately.

    :param ae: a trained :class:`DenoisingAutoencoder` with ``compressor`` active.
    :param batch: dict with ``wave_fd`` and ``noise_fd`` complex tensors (B, C, F).
    :param sample_idx: which sample in the batch to plot.
    :param freqs: optional full positive-frequency array (length ``n_freqs``);
        if given, the x-axis is the per-block mean frequency [Hz], else block index.
    :param save_path: if given, the figure is written here (format inferred from
        the extension) at dpi=150.
    :return: the matplotlib Figure.
    """
    import matplotlib.pyplot as plt  # lazy: keep core import light

    if ae.compressor is None:
        raise ValueError(
            "plot_compression_reconstruction requires an AE with an active "
            "compressor (compressor_window set); got compressor=None."
        )

    ae.eval()
    device = next(ae.parameters()).device
    wave_fd = batch["wave_fd"].to(device)
    noise_fd = batch["noise_fd"].to(device)

    enc_in = ae.preprocess(wave_fd + noise_fd)      # (B, 4C, n_blk) — encoder input
    target = ae._get_target(ae.preprocess(wave_fd))  # (B, 2C or 4C, n_blk)
    rec = ae(enc_in)                                 # (B, 2C or 4C, n_blk)

    n_mean = ae.n_real_channels                      # # of mean channels (= 2 * n_channels)
    orig = enc_in[sample_idx, :n_mean].cpu().numpy()    # (n_mean, n_blk)
    tgt = target[sample_idx, :n_mean].cpu().numpy()
    recon = rec[sample_idx, :n_mean].cpu().numpy()
    n_blk = orig.shape[-1]

    # x-axis: per-block mean frequency if a freq grid is supplied, else block index.
    if freqs is not None:
        W = ae.compressor.n_window
        freqs = np.asarray(freqs)
        x = np.array([
            freqs[j * W:(j + 1) * W].mean() if j * W < len(freqs) else np.nan
            for j in range(n_blk)
        ])
        xlabel = "per-block mean frequency [Hz]"
    else:
        x = np.arange(n_blk)
        xlabel = "block index"

    # Channel layout: mean channels are [comp0_ch0..comp0_ch{nc-1}, comp1_ch0..].
    n_phys = ae.n_channels
    comp_names = ("Re", "Im") if ae.representation == "real_imag" else ("logAmp", "Phase")
    phys_names = ["A", "E", "T"][:n_phys] if n_phys <= 3 else [f"ch{c}" for c in range(n_phys)]

    fig, axes = plt.subplots(
        2, n_phys, figsize=(5 * n_phys, 7), sharex=True, squeeze=False
    )
    for comp in range(2):           # 0 -> Re/logAmp, 1 -> Im/Phase
        for c in range(n_phys):     # physical channel A/E/(T)
            ch = comp * n_phys + c  # index into the mean-channel block
            ax = axes[comp][c]
            ax.plot(x, orig[ch], color="0.6", lw=1.0, label="original (encoder input)")
            ax.plot(x, tgt[ch], color="tab:blue", lw=1.2, label="target")
            ax.plot(x, recon[ch], color="tab:red", lw=1.0, ls="--", label="reconstruction")
            ax.set_title(f"{comp_names[comp]}[{phys_names[c]}] (mean)")
            ax.grid(alpha=0.3)
            if comp == 1:
                ax.set_xlabel(xlabel)
            if c == 0:
                ax.set_ylabel("normalised amplitude")
    axes[0][0].legend(fontsize=8, loc="best")
    fig.suptitle(
        f"Block-compression AE — sample {sample_idx} "
        f"(reconstruct_std={ae.reconstruct_std}, window={ae.compressor.n_window})",
        fontsize=11,
    )
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    return fig


# ---------------------------------------------------------------------------
# Wrapper for InferenceNetwork.data_summary
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Regression Head for parameter prediction (replaces decoder in per-marginal arch)
# ---------------------------------------------------------------------------

class RegressionHead(nn.Module):
    """MLP that predicts physical parameter(s) from a bottleneck vector.

    Used as the "decoder" in the per-marginal encoder architecture: each
    encoder is trained to regress its associated parameter(s) rather than
    reconstruct the input signal.

    Architecture:
        bottleneck (B, bottleneck_dim)
        → [Linear + BN + LeakyReLU] × N_hidden
        → Linear → (B, n_params)
    """

    def __init__(self, bottleneck_dim: int, n_params: int, hidden_sizes: tuple = (128, 64)):
        super().__init__()
        layers = []
        in_dim = bottleneck_dim
        for h in hidden_sizes:
            layers.extend([
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.LeakyReLU(0.2, inplace=True),
            ])
            in_dim = h
        layers.append(nn.Linear(in_dim, n_params))
        self.net = nn.Sequential(*layers)

    def forward(self, bottleneck: torch.Tensor) -> torch.Tensor:
        """:param bottleneck: (B, bottleneck_dim)
        :return: (B, n_params)"""
        return self.net(bottleneck)


# ---------------------------------------------------------------------------
# Per-Marginal Encoder Trainer (LightningModule)
# ---------------------------------------------------------------------------

class MarginalEncoderTrainer(GPUNoiseMixin, LightningModule):
    """Trains one ConvEncoder + RegressionHead per marginal.

    Each encoder compresses noisy FD data to a bottleneck, and each
    regression head predicts its associated physical parameter(s) from
    the bottleneck with MSE loss on normalised parameter values.

    Training
    --------
    * **Input**  : ``batch["wave_fd"] + batch["noise_fd"]``  (noisy complex signal)
    * **Target** : normalised ground-truth parameter values for each marginal

    After training, wrap this module in :class:`MarginalEncoderWrapper` and
    pass it to :class:`PerMarginalInferenceNetwork` as ``data_summarizer``.
    """

    VALID_REPRESENTATIONS = ("amp_phase", "real_imag")

    def __init__(
        self,
        n_channels: int = 2,
        n_freqs: int = 4096,
        marginals: list = None,           # e.g. [[0], [7, 8], [10]]
        # --- Conv encoder architecture ---
        bottleneck_dim: int = 128,
        hidden_channels: tuple = (32, 64, 128, 256, 256),
        kernel_size: int = 4,
        stride: int = 2,
        dropout: float = 0.0,
        residual: bool = False,
        # --- Regression head ---
        regressor_hidden_sizes: tuple = (128, 64),
        # --- Parameter normalisation (required for MSE to be scale-invariant) ---
        param_mean: list = None,          # shape (n_params_total,) – pass as list
        param_std: list = None,
        # --- Training hyper-parameters ---
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        scheduler_patience: int = 10,
        scheduler_factor: float = 0.5,
        representation: str = "real_imag",
        # --- Optional global amplitude normalisation on top of whitening ---
        amplitude_normalise: bool = False,
        # --- Provenance ---
        prior_bounds: dict = None,
    ):
        super().__init__()
        if representation not in self.VALID_REPRESENTATIONS:
            raise ValueError(
                f"representation must be one of {self.VALID_REPRESENTATIONS}, "
                f"got '{representation}'"
            )
        if marginals is None:
            raise ValueError("marginals must be provided (list of lists of param indices)")
        if amplitude_normalise and representation != "real_imag":
            raise NotImplementedError(
                "amplitude_normalise=True is only supported with "
                "representation='real_imag'."
            )

        self.save_hyperparameters()

        self.n_channels = n_channels
        self.n_freqs = n_freqs
        self.marginals = marginals
        self.bottleneck_dim = bottleneck_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler_patience = scheduler_patience
        self.scheduler_factor = scheduler_factor
        self.representation = representation
        self.amplitude_normalise = amplitude_normalise
        self.prior_bounds = prior_bounds

        if isinstance(hidden_channels, list):
            hidden_channels = tuple(hidden_channels)
        if isinstance(regressor_hidden_sizes, list):
            regressor_hidden_sizes = tuple(regressor_hidden_sizes)

        n_real_channels = n_channels * 2

        # One independent ConvEncoder + RegressionHead per *unique parameter*
        # (not per marginal).  For a 2D marginal like [7, 8], two separate
        # encoders are trained — one for param 7, one for param 8.
        self.param_indices = sorted(set(idx for marginal in marginals for idx in marginal))
        self.encoders = nn.ModuleList()
        self.regressors = nn.ModuleList()
        for _ in self.param_indices:
            self.encoders.append(ConvEncoder(
                n_in_channels=n_real_channels,
                n_freqs=n_freqs,
                bottleneck_dim=bottleneck_dim,
                hidden_channels=hidden_channels,
                kernel_size=kernel_size,
                stride=stride,
                dropout=dropout,
                residual=residual,
            ))
            self.regressors.append(RegressionHead(
                bottleneck_dim=bottleneck_dim,
                n_params=1,
                hidden_sizes=regressor_hidden_sizes,
            ))

        # Whitening scale: noise_scale = ASD * sqrt(T_obs/4), shape (C, F).
        # See DenoisingAutoencoder.set_whitening for the contract.
        dtype = get_torch_dtype()
        self.register_buffer("whitening", torch.ones(n_channels, n_freqs, dtype=dtype))

        # Optional global amplitude scale (scalar). See
        # DenoisingAutoencoder.fit_amplitude_normalisation.
        self.register_buffer("amplitude_scale", torch.tensor(1.0, dtype=dtype))

        # ---- parameter normalisation buffers ---
        n_params_total = 11  # _ORDERED_PRIOR_KEYS has 11 entries
        if param_mean is not None:
            self.register_buffer("param_mean", torch.tensor(param_mean, dtype=dtype))
            self.register_buffer("param_std",  torch.tensor(param_std,  dtype=dtype))
        else:
            self.register_buffer("param_mean", torch.zeros(n_params_total, dtype=dtype))
            self.register_buffer("param_std",  torch.ones(n_params_total, dtype=dtype))

    # ------------------------------------------------------------------
    # Preprocessing helpers (mirrors DenoisingAutoencoder)
    # ------------------------------------------------------------------

    def _complex_to_real(self, z: torch.Tensor) -> torch.Tensor:
        if self.representation == "amp_phase":
            log_amp = torch.log(torch.abs(z) + 1e-33)
            phase   = torch.angle(z)
            return torch.cat([log_amp, phase], dim=1)
        else:  # real_imag
            return torch.cat([z.real, z.imag], dim=1)

    def preprocess(self, z: torch.Tensor) -> torch.Tensor:
        """Whiten complex FD data, convert to real channels, optionally
        divide by the global amplitude scale.

        :return: real tensor (B, 2C, F).
        """
        z_whitened = z / self.whitening
        real = self._complex_to_real(z_whitened)
        if self.amplitude_normalise:
            real = real / self.amplitude_scale
        return real

    def set_whitening(self, noise_scale: torch.Tensor) -> None:
        """Store ``noise_scale = ASD * sqrt(T_obs/4)`` as the whitening scale.

        Zero bins are mapped to ``inf`` so masked frequencies whiten to 0.
        """
        whitening_safe = noise_scale.to(self.whitening.dtype).clone()
        whitening_safe[noise_scale == 0] = float("inf")
        self.whitening.copy_(whitening_safe)
        nonzero = noise_scale[noise_scale > 0]
        print(
            f"[MarginalEncoder] whitening set (shape={tuple(self.whitening.shape)}, "
            f"range=[{nonzero.min().item():.4e}, {nonzero.max().item():.4e}])"
        )

    def fit_amplitude_normalisation(self, dataloader: DataLoader) -> None:
        """Compute the global amplitude scale on whitened clean signals.
        See ``DenoisingAutoencoder.fit_amplitude_normalisation`` for details.
        """
        if not self.amplitude_normalise:
            raise RuntimeError(
                "fit_amplitude_normalisation called but amplitude_normalise=False."
            )
        if self.representation != "real_imag":
            raise NotImplementedError(
                "amplitude_normalise is only supported with representation='real_imag'."
            )
        device = next(self.parameters()).device
        max_val = 0.0
        n_samples = 0
        with torch.no_grad():
            for batch in dataloader:
                batch = materialize_gpu_noise(batch)
                wave_fd = batch["wave_fd"].to(device)
                wave_w = wave_fd / self.whitening
                real = self._complex_to_real(wave_w)
                max_val = max(max_val, real.abs().max().item())
                n_samples += real.shape[0]
        self.amplitude_scale.fill_(max_val)
        print(
            f"[MarginalEncoder] amplitude scale fitted on {n_samples} whitened "
            f"clean samples: amplitude_scale={max_val:.4e}"
        )

    def _denormalize_amplitude(self, x_real: torch.Tensor) -> torch.Tensor:
        """Inverse of the amplitude scaling. For visualisation only."""
        if self.amplitude_normalise:
            return x_real * self.amplitude_scale
        return x_real

    # ------------------------------------------------------------------
    # Lightning steps
    # ------------------------------------------------------------------

    def _step(self, batch, prefix: str):
        noisy     = batch["wave_fd"] + batch["noise_fd"]
        noisy_norm = self.preprocess(noisy)
        params    = batch["source_parameters"]

        total_loss = torch.tensor(0.0, device=self.device, dtype=noisy_norm.dtype)
        for i, (encoder, regressor, param_idx) in enumerate(
            zip(self.encoders, self.regressors, self.param_indices)
        ):
            bottleneck = encoder(noisy_norm)
            predicted  = regressor(bottleneck)                              # (B, 1)
            target_raw = params[:, param_idx:param_idx+1].to(noisy_norm.dtype)
            # Normalise targets so MSE is scale-invariant across parameters
            target_norm = (target_raw - self.param_mean[param_idx]) / (self.param_std[param_idx] + 1e-30)
            loss_i = F.mse_loss(predicted, target_norm)
            total_loss = total_loss + loss_i
            self.log(
                f"{prefix}_mse_param_{param_idx}", loss_i,
                on_step=True, on_epoch=True, prog_bar=False, logger=True,
            )

        self.log(f"{prefix}_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return total_loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=self.scheduler_factor,
            patience=self.scheduler_patience, min_lr=1e-7,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }


# ---------------------------------------------------------------------------
# Wrapper for PerMarginalInferenceNetwork.data_summary
# ---------------------------------------------------------------------------

class MarginalEncoderWrapper(nn.Module):
    """Wraps a trained :class:`MarginalEncoderTrainer` so it can be used as
    ``PerMarginalInferenceNetwork.data_summary``.

    Calling ``forward(d_f, d_t)`` returns a **dict** mapping each unique
    parameter index to its per-parameter bottleneck tensor
    ``(B, bottleneck_dim)``.

    The encoder weights are frozen by default (``freeze=True``).
    """

    def __init__(self, trainer: MarginalEncoderTrainer, freeze: bool = True, device: str = "cuda"):
        super().__init__()
        self.trainer = trainer
        self._device = device
        self._n_features = trainer.bottleneck_dim

        if freeze:
            for p in self.trainer.parameters():
                p.requires_grad = False

    def get_n_features(self) -> int:
        """Bottleneck dimensionality (same for every encoder)."""
        return self._n_features

    def get_n_marginals(self) -> int:
        return len(self.trainer.marginals)

    def get_n_encoders(self) -> int:
        return len(self.trainer.encoders)

    def get_param_indices(self) -> list[int]:
        return self.trainer.param_indices

    def unfreeze_parameters(self):
        for p in self.trainer.parameters():
            p.requires_grad = True

    def forward(self, d_f: torch.Tensor, d_t: torch.Tensor):
        """Encode noisy FD data with each per-parameter encoder.

        :param d_f: complex tensor (B, C, F) — raw frequency-domain data
        :param d_t: ignored (kept for API compatibility)
        :return: (dict mapping param_idx → (B, bottleneck_dim) tensor, d_t)
        """
        x_norm = self.trainer.preprocess(d_f)
        bottleneck_dict = {
            param_idx: encoder(x_norm)
            for param_idx, encoder in zip(self.trainer.param_indices, self.trainer.encoders)
        }
        return bottleneck_dict, d_t


class AutoencoderWrapper(nn.Module):
    """Wraps a trained :class:`DenoisingAutoencoder` so it can be used as
    ``InferenceNetwork.data_summary``.

    Calling ``forward(d_f, d_t)`` returns the **flattened bottleneck**
    representation (not the reconstructed signal).

    The encoder weights are frozen by default (``freeze=True``).
    """

    def __init__(self, autoencoder: DenoisingAutoencoder, freeze: bool = True, device: str = "cuda"):
        super().__init__()
        self.autoencoder = autoencoder
        self._device = device

        if freeze:
            self._freeze_parameters()

        # Pre-compute the number of bottleneck features so
        # InferenceNetwork can query it via get_n_features().
        self._n_features = self._compute_n_features()

    def _freeze_parameters(self):
        """Freeze the autoencoder parameters."""
        for p in self.autoencoder.parameters():
            p.requires_grad = False
    def unfreeze_parameters(self):
        """Unfreeze the autoencoder parameters."""
        for p in self.autoencoder.parameters():
            p.requires_grad = True
    def _compute_n_features(self) -> int:
        """Compute the bottleneck dimensionality."""
        if self.autoencoder.architecture == "conv":
            # For conv architecture, bottleneck_dim is known directly
            return self.autoencoder.bottleneck_dim
        else:
            # For unet, run a dummy forward pass
            n_real_ch = self.autoencoder.n_channels * 2
            dummy = torch.zeros(1, n_real_ch, self.autoencoder.n_freqs)
            with torch.no_grad():
                dummy = dummy.to(self._device)
                bottleneck, _ = self.autoencoder.encoder(dummy)
            return bottleneck.numel()  # sizes[-1] * L_bottleneck

    def get_n_features(self) -> int:
        """Return the dimensionality of the bottleneck (flattened)."""
        return self._n_features

    def forward(self, d_f: torch.Tensor, d_t: torch.Tensor):
        """Encode noisy FD data and return the flattened bottleneck.

        :param d_f: complex tensor (B, C, F)  — raw frequency-domain data
                    (typically ``wave_fd + noise_fd``)
        :param d_t: ignored (kept for API compatibility with other data
                    summarizers like ROMWrapper)
        :return: (bottleneck_flat, d_t) where ``bottleneck_flat`` has shape
                 ``(B, n_features)``
        """
        x_norm = self.autoencoder.preprocess(d_f)
        bottleneck = self.autoencoder.encode(x_norm)
        
        if self.autoencoder.architecture == "conv":
            # bottleneck is already (B, bottleneck_dim)
            bottleneck_flat = bottleneck
        else:
            # unet returns (bottleneck, skips), flatten the bottleneck
            bottleneck, _skips = bottleneck
            bottleneck_flat = bottleneck.reshape(bottleneck.shape[0], -1)

        return bottleneck_flat, d_t


# ---------------------------------------------------------------------------
# ChannelizedMLPCompressor — no-AE learned front-end for joint NRE training
# ---------------------------------------------------------------------------

class ChannelizedMLPCompressor(nn.Module):
    """Learned channelized-MLP compressor that replaces a ``DenoisingAutoencoder``
    in :class:`pembhb.model.JointAEInferenceNetwork`.

    Per real channel (= real or imaginary part of each TDI channel) a small MLP
    ``Linear(n_freqs → hidden) → ReLU → Linear(hidden → out_per_channel)`` is
    applied independently; the per-channel outputs are then concatenated into a
    single ``bottleneck_dim = n_real_channels * out_per_channel`` summary that
    every NRE classifier head consumes.

    There is no decoder.  The module is trained end-to-end with the NRE BCE
    loss; the joint module must therefore be constructed with
    ``encoder_trains_via_nre=True`` so the bottleneck is not detached and the
    AE reconstruction loss term is zeroed out.

    The whitening / mean-subtraction / amplitude-normalisation pipeline mirrors
    :class:`DenoisingAutoencoder` so the comparison against an AE-based run is
    apples-to-apples on identical normalised inputs.
    """

    def __init__(
        self,
        n_channels: int,
        n_freqs: int,
        hidden_dim_per_channel: int = 256,
        out_dim_per_channel: int = 64,
        representation: str = "real_imag",
        whiten: bool = True,
        amplitude_normalise: bool = True,
        subtract_mean_whitened: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        if representation != "real_imag":
            raise NotImplementedError(
                "ChannelizedMLPCompressor only supports representation='real_imag'."
            )
        self.n_channels = n_channels
        self.n_freqs = n_freqs
        self.representation = representation
        self.whiten = whiten
        self.amplitude_normalise = amplitude_normalise
        self.subtract_mean_whitened = subtract_mean_whitened
        self.dropout = float(dropout)

        self.n_real_channels = n_channels * 2  # re + im per TDI channel
        self.hidden_dim_per_channel = hidden_dim_per_channel
        self.out_dim_per_channel = out_dim_per_channel
        # Mark as "conv" so JointAEInferenceNetwork's "conv" code path (which
        # treats encode() output as the bottleneck tensor) is used.
        self.architecture = "conv"
        self.bottleneck_dim = self.n_real_channels * out_dim_per_channel
        # Disable AE features the joint pipeline may query.
        self.compressor = None
        self._mask_active = False
        self.idx_lowerbound = 0
        self.idx_upperbound = n_freqs

        def _make_block():
            layers = [
                nn.Linear(n_freqs, hidden_dim_per_channel),
                nn.ReLU(),
            ]
            if self.dropout > 0.0:
                layers.append(nn.Dropout(self.dropout))
            layers.append(nn.Linear(hidden_dim_per_channel, out_dim_per_channel))
            return nn.Sequential(*layers)

        self.channel_blocks = nn.ModuleList(
            _make_block() for _ in range(self.n_real_channels)
        )

        # Normalisation buffers (same shapes/semantics as DenoisingAutoencoder)
        self.register_buffer(
            "whitening", torch.ones(n_channels, n_freqs, dtype=get_torch_dtype())
        )
        self.register_buffer(
            "amplitude_scale", torch.tensor(1.0, dtype=get_torch_dtype())
        )
        self.register_buffer(
            "amplitude_scale_std", torch.tensor(1.0, dtype=get_torch_dtype())
        )
        self.register_buffer(
            "mean_whitened",
            torch.zeros(self.n_real_channels, n_freqs, dtype=get_torch_dtype()),
        )

    # ------------------------------------------------------------------
    # Complex → real conversion (real_imag only)
    # ------------------------------------------------------------------
    def _complex_to_real(self, z: torch.Tensor) -> torch.Tensor:
        return torch.cat([z.real, z.imag], dim=1)

    def _prewhiten_real(self, z: torch.Tensor) -> torch.Tensor:
        """Complex FD → real channels, whitened iff ``self.whiten``. Shared by
        ``preprocess`` and the amplitude/mean fit (see DenoisingAutoencoder)."""
        return self._complex_to_real(z / self.whitening if self.whiten else z)

    # ------------------------------------------------------------------
    # Whitening
    # ------------------------------------------------------------------
    def set_whitening(self, noise_scale: torch.Tensor) -> None:
        whitening_safe = noise_scale.to(self.whitening.dtype).clone()
        whitening_safe[noise_scale == 0] = float("inf")
        self.whitening.copy_(whitening_safe)
        nonzero = noise_scale[noise_scale > 0]
        print(
            f"[ChannelizedMLP] whitening set (shape={tuple(self.whitening.shape)}, "
            f"range=[{nonzero.min().item():.4e}, {nonzero.max().item():.4e}])"
        )

    # ------------------------------------------------------------------
    # Mean / amplitude normalisation (independent of whitening)
    # ------------------------------------------------------------------
    def fit_white_normalisation(self, dataloader: DataLoader) -> None:
        """Fit ``mean_whitened`` (if subtract_mean_whitened) and
        ``amplitude_scale`` (if amplitude_normalise) on the same real
        representation ``preprocess`` sees (whitened or raw)."""
        if not (self.amplitude_normalise or self.subtract_mean_whitened):
            raise RuntimeError(
                "fit_white_normalisation called but neither amplitude_normalise "
                "nor subtract_mean_whitened is set."
            )
        device = next(self.parameters()).device
        running_sum = torch.zeros(
            self.n_real_channels, self.n_freqs, device=device,
            dtype=self.mean_whitened.dtype,
        )
        n_samples = 0
        with torch.no_grad():
            for batch in dataloader:
                batch = materialize_gpu_noise(batch)
                wave_fd = batch["wave_fd"].to(device)
                real = self._prewhiten_real(wave_fd)
                n_samples += real.shape[0]
                if self.subtract_mean_whitened:
                    running_sum += real.sum(dim=0)
        if self.subtract_mean_whitened:
            self.mean_whitened.copy_(running_sum / n_samples)
            print(
                f"[ChannelizedMLP] mean_whitened fitted on {n_samples} samples"
            )

        if not self.amplitude_normalise:
            return

        max_mean = 0.0
        with torch.no_grad():
            for batch in dataloader:
                batch = materialize_gpu_noise(batch)
                wave_fd = batch["wave_fd"].to(device)
                real = self._prewhiten_real(wave_fd)
                if self.subtract_mean_whitened:
                    real = real - self.mean_whitened
                max_mean = max(max_mean, real.abs().max().item())
        self.amplitude_scale.fill_(max_mean)
        print(f"[ChannelizedMLP] amplitude_scale={max_mean:.4e}")

    # ------------------------------------------------------------------
    # Preprocess (mirrors DenoisingAutoencoder.preprocess, no compressor)
    # ------------------------------------------------------------------
    def preprocess(self, z: torch.Tensor) -> torch.Tensor:
        real = self._prewhiten_real(z)
        if self.subtract_mean_whitened:
            real = real - self.mean_whitened
        if self.amplitude_normalise:
            real = real / self.amplitude_scale
        return real

    # ------------------------------------------------------------------
    # Channelized MLP (encoder)
    # ------------------------------------------------------------------
    def encode(self, x_norm: torch.Tensor) -> torch.Tensor:
        """Apply per-channel MLP, concat across real channels.

        :param x_norm: ``(B, n_real_channels, n_freqs)`` normalised input.
        :return:       ``(B, bottleneck_dim)`` summary.
        """
        outputs = [block(x_norm[:, c, :]) for c, block in enumerate(self.channel_blocks)]
        return torch.cat(outputs, dim=-1)

    def forward(self, x_norm: torch.Tensor) -> torch.Tensor:
        """Same as :meth:`encode`; kept for API parity with DenoisingAutoencoder."""
        return self.encode(x_norm)