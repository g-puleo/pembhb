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

import torch
from torch import nn
from torch.nn import functional as F
from lightning import LightningModule
from torch.utils.data import DataLoader

from pembhb.model import DoubleConv, Down, Up, OutConv


# ---------------------------------------------------------------------------
# Pure Convolutional Encoder / Decoder (no skip connections)
# ---------------------------------------------------------------------------

class ConvEncoder(nn.Module):
    """Pure convolutional encoder without skip connections.
    
    Compresses input to a fixed-size bottleneck vector.
    
    Architecture:
        Input (B, C_in, L) 
        → [Conv1d + BN + LeakyReLU + Dropout] × N_layers (with stride=2 downsampling)
        → Flatten 
        → Linear → bottleneck (B, bottleneck_dim)
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
    ):
        super().__init__()
        self.bottleneck_dim = bottleneck_dim
        self.n_freqs = n_freqs
        
        # Build conv layers with strided downsampling
        layers = []
        in_ch = n_in_channels
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
    ):
        super().__init__()
        self.n_freqs = n_freqs
        self.pre_fc_channels = pre_fc_channels
        self.pre_fc_length = pre_fc_length
        
        # Linear from bottleneck to pre-conv shape
        self.fc = nn.Linear(bottleneck_dim, pre_fc_channels * pre_fc_length)
        self.fc_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
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
        x = self.conv(x)
        
        # Crop or pad to exact output size if needed
        if x.shape[2] != self.n_freqs:
            if x.shape[2] > self.n_freqs:
                x = x[:, :, :self.n_freqs]
            else:
                pad_size = self.n_freqs - x.shape[2]
                x = F.pad(x, (0, pad_size))
        return x


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

class DenoisingAutoencoder(LightningModule):
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
        n_freqs: int = 4096,
        # --- Conv architecture params ---
        architecture: str = "conv",
        bottleneck_dim: int = 128,
        hidden_channels: tuple = (32, 64, 128, 256, 256),
        kernel_size: int = 4,
        stride: int = 2,
        dropout: float = 0.0,
        # --- Unet architecture params (legacy) ---
        sizes: tuple = (16, 32, 64, 128, 256),
        down_sampling: tuple = (2, 2, 2, 2),
        # --- Training params ---
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        scheduler_patience: int = 10,
        scheduler_factor: float = 0.5,
        representation: str = "amp_phase",
        # --- High-frequency only mode ---
        high_freq_only: bool = False,
        freq_split_idx: int = 2048,
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
        self.high_freq_only = high_freq_only
        self.freq_split_idx = freq_split_idx
        self.prior_bounds = prior_bounds  # stored in hparams for checkpoint

        # Determine the number of frequency bins to reconstruct
        if high_freq_only:
            n_freqs_target = n_freqs - freq_split_idx
            print(f"[AutoEncoder] High-freq only mode: reconstructing bins [{freq_split_idx}:{n_freqs}] ({n_freqs_target} bins)")
        else:
            n_freqs_target = n_freqs

        # Complex → real representation doubles the channels
        n_real_channels = n_channels * 2

        if architecture == "conv":
            # Pure convolutional autoencoder (no skip connections)
            self.encoder = ConvEncoder(
                n_in_channels=n_real_channels,
                n_freqs=n_freqs,  # encoder always sees full input
                bottleneck_dim=bottleneck_dim,
                hidden_channels=hidden_channels,
                kernel_size=kernel_size,
                stride=stride,
                dropout=dropout,
            )
            self.decoder = ConvDecoder(
                n_out_channels=n_real_channels,
                n_freqs=n_freqs_target,  # decoder outputs target size
                bottleneck_dim=bottleneck_dim,
                hidden_channels=hidden_channels,
                kernel_size=kernel_size,
                stride=stride,
                pre_fc_channels=self.encoder.pre_fc_channels,
                pre_fc_length=self.encoder.pre_fc_length,
                dropout=dropout,
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

        # ---- normalisation buffers (computed from training data) -----------
        # These are registered as buffers so they are saved/loaded with the
        # checkpoint and moved to the correct device automatically.
        self.register_buffer("mean_vec", torch.zeros(n_real_channels, n_freqs))
        self.register_buffer("global_scale_factor", torch.tensor(1.0))
        self._normalisation_fitted = False

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

    # ------------------------------------------------------------------
    # Normalisation helpers  (mirror ReducedOrderModel._normalize / _denormalize)
    # ------------------------------------------------------------------

    def fit_normalisation(self, dataloader: DataLoader):
        """Compute mean and global scale from the **clean** training signals.

        Call this **once** before training.  The statistics are stored as
        buffers and will be saved together with the model checkpoint.

        :param dataloader: training DataLoader whose batches contain ``wave_fd``.
        """
        device = next(self.parameters()).device
        running_sum = torch.zeros(self.n_channels * 2, self.n_freqs, device=device)
        n_samples = 0
        max_val = 0.0

        with torch.no_grad():
            for batch in dataloader:
                wave_fd = batch["wave_fd"].to(device)
                real = self._complex_to_real(wave_fd)  # (B, 2C, F)
                running_sum += real.sum(dim=0)
                n_samples += real.shape[0]
                max_val = max(max_val, real.abs().max().item())

        self.mean_vec.copy_(running_sum / n_samples)
        self.global_scale_factor.fill_(max_val)
        self._normalisation_fitted = True
        print(
            f"[AutoEncoder] normalisation fitted on {n_samples} samples  "
            f"(representation={self.representation}, "
            f"global_scale={self.global_scale_factor.item():.4e})"
        )

    def _get_mean_vec_for(self, n_freq: int) -> torch.Tensor:
        """Return the appropriate slice of ``mean_vec`` for the given frequency dimension.

        Handles both the full-spectrum case and the high-freq-only case
        where the decoder output has fewer bins than ``mean_vec``.
        """
        if n_freq == self.n_freqs:
            return self.mean_vec
        if self.high_freq_only and n_freq == self.n_freqs - self.freq_split_idx:
            return self.mean_vec[:, self.freq_split_idx:]
        raise ValueError(
            f"Frequency dimension {n_freq} does not match full ({self.n_freqs}) "
            f"or high-freq-only ({self.n_freqs - self.freq_split_idx}) expected size."
        )

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Center and scale: ``(x - mean) / scale``.

        Handles both full-frequency and high-freq-only tensors.
        """
        mean = self._get_mean_vec_for(x.shape[-1])
        return (x - mean) / self.global_scale_factor

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Undo normalisation.

        Handles both full-frequency and high-freq-only tensors.
        """
        mean = self._get_mean_vec_for(x.shape[-1])
        return x * self.global_scale_factor + mean

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
        """Convert complex FD data to normalised real representation.

        :param z: complex tensor (B, C, F)
        :return: normalised real tensor (B, 2C, F)
        """
        return self._normalize(self._complex_to_real(z))

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
        """Get the reconstruction target, possibly sliced for high_freq_only mode.
        
        :param clean_norm: (B, 2C, F) full normalised clean signal
        :return: (B, 2C, F) or (B, 2C, F_target) depending on high_freq_only
        """
        if self.high_freq_only:
            return clean_norm[:, :, self.freq_split_idx:]
        return clean_norm

    def training_step(self, batch, batch_idx):
        noisy = batch["wave_fd"] + batch["noise_fd"]
        clean = batch["wave_fd"]

        # normalise
        noisy_norm = self.preprocess(noisy)
        clean_norm = self.preprocess(clean)

        # forward (full autoencoder)
        reconstructed = self(noisy_norm)
        
        # get target (possibly sliced for high_freq_only)
        target = self._get_target(clean_norm)

        loss = F.mse_loss(reconstructed, target)
        mae, rel_err, max_rel_err = self._compute_extra_metrics(reconstructed, target)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_mae", mae, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_rel_err", rel_err, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_max_rel_err", max_rel_err, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        noisy = batch["wave_fd"] + batch["noise_fd"]
        clean = batch["wave_fd"]

        noisy_norm = self.preprocess(noisy)
        clean_norm = self.preprocess(clean)

        reconstructed = self(noisy_norm)
        
        # get target (possibly sliced for high_freq_only)
        target = self._get_target(clean_norm)

        loss = F.mse_loss(reconstructed, target)
        mae, rel_err, max_rel_err = self._compute_extra_metrics(reconstructed, target)

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_mae", mae, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_rel_err", rel_err, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_max_rel_err", max_rel_err, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        return loss

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
# Wrapper for InferenceNetwork.data_summary
# ---------------------------------------------------------------------------

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
            for p in self.autoencoder.parameters():
                p.requires_grad = False

        # Pre-compute the number of bottleneck features so
        # InferenceNetwork can query it via get_n_features().
        self._n_features = self._compute_n_features()

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
