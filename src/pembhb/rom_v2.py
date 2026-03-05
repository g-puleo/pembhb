import torch
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from line_profiler import profile
import torch.nn as nn
from torch import nn
from pembhb import ROOT_DIR

class ReducedOrderModel:
    """Reduced-order model for frequency-domain (complex) waveform data.

    Builds an orthonormal basis via a greedy algorithm (iterative
    Gram-Schmidt) and uses it to compress / reconstruct waveforms.

    Parameters
    ----------
    batch_size : int
        Batch size used during basis construction (default 1000).
    tolerance : float
        Convergence threshold for the greedy algorithm.
    device : str
        Torch device string.
    filename : str or None
        If given, load a previously saved ROM from this ``.pt`` file.
        All other constructor arguments are ignored in that case.
    debugging : bool
        Enable extra diagnostic plots.
    patience : int
        Number of epochs without improvement before early stopping.
    freq_cutoff_idx : int or None
        If set, discard frequency bins below this index (low-frequency
        cutoff).  The basis, scale factor, and inner-product weights are
        all computed on the truncated spectrum.  At inference time,
        :meth:`compress` slices the input automatically.
    df : float, array-like, or None
        Frequency-bin width(s).  Scalar for uniform grids, 1-D array
        for non-uniform / log-spaced grids.  Used to build the GW
        inner-product weight ``4 Δf``.
    normalize_by_max : bool
        If ``True`` (default) data is divided by the global maximum
        absolute value before building the basis.  Set to ``False`` to
        disable this scaling (the basis is then built on raw whitened
        data).
    """

    def __init__(self, batch_size=1000, tolerance=1e-3, device="cpu",
                 filename=None, debugging=False, patience=1,
                 freq_cutoff_idx=None, df=None, normalize_by_max=True):
        if filename is not None:
            print(f"[ROM] initializing from file {filename}.\n Other arguments will be ignored.")
            self.device = device
            self.from_file(filename)
        else:
            self.debugging = debugging
            self.batch_size = batch_size
            self.device = device
            self.tolerance = tolerance
            self.patience = patience
            self.max_epochs = 4000
            self.freq_cutoff_idx = freq_cutoff_idx
            self.df = df
            self.normalize_by_max = normalize_by_max

            self.basis = None
            self.n_channels = None
            self.n_freq = None
            self.global_scale_factor = None
            self.valid_bins_mask = None
            self.inner_weights = None
            self.asd = None

            if debugging:
                self.plot_dir_debug = os.path.join(
                    ROOT_DIR, "plots",
                    "debug_plots_{time}".format(time=time.strftime("%Y%m%d-%H%M%S")))
                os.makedirs(self.plot_dir_debug, exist_ok=True)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _inner(self, A, B):
        """Weighted inner product ⟨A|B⟩ for complex FD vectors.

        Implements the GW-standard real inner product::

            ⟨a|b⟩ = Re[∑_k a_k* b_k w_k]

        where ``w_k = 4 Δf_k``.  When the data is pre-whitened the
        noise PSD is already absorbed.

        Parameters
        ----------
        A : torch.Tensor  — shape ``(M, D)``
        B : torch.Tensor  — shape ``(N, D)``

        Returns
        -------
        torch.Tensor  — real, shape ``(M, N)``
        """
        w = self.inner_weights
        Aw = A.conj() * w if w is not None else A.conj()
        return (Aw @ B.mT).real

    def _apply_freq_cutoff(self, data):
        """Slice off low-frequency bins when *freq_cutoff_idx* is set.

        Parameters
        ----------
        data : torch.Tensor
            2-D ``(C, F)`` or 3-D ``(B, C, F)`` tensor.

        Returns
        -------
        torch.Tensor
        """
        idx = getattr(self, 'freq_cutoff_idx', None)
        if not idx:
            return data
        if data.ndim == 3:
            return data[:, :, idx:]
        elif data.ndim == 2:
            return data[:, idx:]
        return data

    def _whiten(self, data):
        """Whiten FD data by dividing by the stored ASD.

        No-op if ``self.asd`` is ``None``.

        Parameters
        ----------
        data : torch.Tensor
            2-D ``(C, F)`` or 3-D ``(B, C, F)`` tensor.

        Returns
        -------
        torch.Tensor
        """
        if self.asd is None:
            return data
        asd = self.asd.to(data.device)
        safe_asd = asd.clone()
        safe_asd[safe_asd == 0] = float('inf')
        if data.ndim == 3:
            return data / safe_asd.unsqueeze(0)
        return data / safe_asd

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------

    def from_file(self, filename):
        s = torch.load(filename, map_location=self.device)
        print("Keys in the saved file:", list(s.keys()))
        print(f"name of the saved file: {filename}")

        self.freq_cutoff_idx = s.get('freq_cutoff_idx', None)
        self.df = s.get('df', None)
        self.normalize_by_max = s.get('normalize_by_max', True)

        # ASD
        self.asd = s.get('asd', None)
        if self.asd is not None:
            self.asd = self.asd.to(self.device)
            print(f"[ROM] ASD loaded, shape={tuple(self.asd.shape)}")

        # Try new-style keys first, then _fd-suffixed, then bare legacy
        if 'basis_fd' in s:
            # _fd-suffixed format (from multi-domain era, or recent saves)
            self.basis = s['basis_fd'].to(self.device)
            self.n_channels = s['n_channels_fd']
            self.n_freq = s.get('n_dim_fd', s.get('n_freq'))
            self.global_scale_factor = s['global_scale_factor_fd']
            print(f"[ROM] basis loaded (_fd keys). n_freq={self.n_freq}, "
                  f"n_channels={self.n_channels}, n_basis={len(self.basis)}")
        elif 'basis' in s:
            # Bare legacy format (no _fd suffix)
            self.basis = s['basis'].to(self.device)
            self.n_channels = s['n_channels']
            self.n_freq = s.get('n_freq', s.get('n_dim_fd'))
            self.global_scale_factor = s['global_scale_factor']
            print(f"[ROM] basis loaded (legacy). n_freq={self.n_freq}, "
                  f"n_channels={self.n_channels}, n_basis={len(self.basis)}")
        else:
            raise ValueError(f"Cannot find basis in saved file {filename}. "
                             f"Available keys: {list(s.keys())}")

        # Valid-bins mask
        vbm = s.get('valid_bins_mask_fd', s.get('valid_bins_mask', None))
        if vbm is not None:
            self.valid_bins_mask = vbm.to(self.device)
            print(f"[ROM] valid_bins_mask loaded: {vbm.sum().item()}/{vbm.numel()} bins active")
        else:
            self.valid_bins_mask = None

        # Inner-product weights
        iw = s.get('inner_weights_fd', s.get('inner_weights', None))
        if iw is not None:
            self.inner_weights = iw.to(self.device)
            print(f"[ROM] inner_weights loaded (min={iw.min().item():.4e}, max={iw.max().item():.4e})")
        else:
            self.inner_weights = None

        # Legacy aliases so old code using .basis_fd / .n_dim_fd still works
        self.basis_fd = self.basis
        self.n_channels_fd = self.n_channels
        self.n_dim_fd = self.n_freq
        self.global_scale_factor_fd = self.global_scale_factor
        return

    def load_diagnostics(self, filename):
        self.training_diagnostics = torch.load(filename, map_location=self.device)
        return

    def save_diagnostics(self, filename):
        torch.save(self.training_diagnostics, filename)
        print(f"[ROM] diagnostics saved to {filename}")
        return

    def to_file(self, filename):
        if not filename.endswith(".pt"):
            raise ValueError("[ROM] filename must end with .pt")

        state = {
            "tolerance": self.tolerance,
            "device": self.device,
            "freq_cutoff_idx": self.freq_cutoff_idx,
            "normalize_by_max": self.normalize_by_max,
            "df": self.df if not isinstance(self.df, torch.Tensor) else self.df.cpu(),
            "basis": self.basis.cpu(),
            "n_channels": self.n_channels,
            "n_freq": self.n_freq,
            "global_scale_factor": self.global_scale_factor,
        }
        if self.asd is not None:
            state['asd'] = self.asd.cpu() if isinstance(self.asd, torch.Tensor) else self.asd
        if self.valid_bins_mask is not None:
            vbm = self.valid_bins_mask
            state['valid_bins_mask'] = vbm.cpu() if isinstance(vbm, torch.Tensor) else vbm
        if self.inner_weights is not None:
            iw = self.inner_weights
            state['inner_weights'] = iw.cpu() if isinstance(iw, torch.Tensor) else iw

        # Write _fd-suffixed aliases so that older loading code still works
        state['basis_fd'] = state['basis']
        state['n_channels_fd'] = state['n_channels']
        state['n_dim_fd'] = state['n_freq']
        state['global_scale_factor_fd'] = state['global_scale_factor']
        if 'valid_bins_mask' in state:
            state['valid_bins_mask_fd'] = state['valid_bins_mask']
        if 'inner_weights' in state:
            state['inner_weights_fd'] = state['inner_weights']

        torch.save(state, filename)
        filename_diagnostics = filename.replace(".pt", "_diagnostics.pt")
        self.save_diagnostics(filename_diagnostics)
        print(f"[ROM] basis saved to {filename}")
        return

    # ------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------

    @profile
    def train(self, train_dataloader: torch.utils.data.DataLoader,
              use_pinned_memory: bool = True, prefetch_batches: int = 1,
              convergence_on: str = 'sigma_data'):
        """Train the reduced order model.

        Args:
            train_dataloader: DataLoader whose ``.dataset`` is a ``Subset``.
            use_pinned_memory: Use pinned-memory transfers (low GPU mem).
            prefetch_batches: Batches to prefetch and concatenate.
            convergence_on: ``'sigma'`` or ``'sigma_data'``.
        """
        print(f"[ROM] training with tolerance={self.tolerance:.1e} on device={self.device}")
        print(f"[ROM] convergence criterion: {convergence_on}")
        print(f"[ROM] normalize_by_max: {self.normalize_by_max}")
        if self.freq_cutoff_idx:
            print(f"[ROM] frequency cutoff enabled: discarding bins below index {self.freq_cutoff_idx}")
        train_subset = train_dataloader.dataset
        if not isinstance(train_dataloader.dataset, torch.utils.data.Subset):
            raise ValueError("[ROM] train_dataloader.dataset must be a torch.utils.data.Subset")

        self._use_pinned_memory = use_pinned_memory
        self._prefetch_batches = max(1, int(prefetch_batches))
        if use_pinned_memory:
            print("[ROM] Using pinned memory mode (low GPU memory usage)")
            if hasattr(train_subset.dataset, 'cache_in_memory') and train_subset.dataset.cache_in_memory:
                self._pin_dataset_tensors(train_subset.dataset)
        else:
            print("[ROM] Moving entire dataset to GPU (high memory usage)")
            train_subset.dataset.to(self.device)

        self._train_basis(train_dataloader, convergence_on)

        # Legacy aliases
        self.basis_fd = self.basis
        self.n_channels_fd = self.n_channels
        self.n_dim_fd = self.n_freq
        self.global_scale_factor_fd = self.global_scale_factor

    def _is_convergence_stagnant(self, convergence_value, last_values):
        """Check if convergence_value has not improved for *patience* epochs."""
        if len(last_values) < self.patience:
            return False
        return convergence_value >= min(last_values)

    def _train_basis(self, train_dataloader, convergence_on):
        """Build the reduced basis via greedy Gram-Schmidt."""
        train_subset = train_dataloader.dataset
        print(f"\n[ROM] --- starting basis construction (complex) ---")

        sample = train_subset[0]['wave_fd']
        sample = self._apply_freq_cutoff(sample)
        n_channels = sample.shape[0]
        n_freq = sample.shape[1]
        self.n_channels = n_channels
        self.n_freq = n_freq

        if self.freq_cutoff_idx:
            print(f"[ROM] frequency cutoff at index {self.freq_cutoff_idx}: "
                  f"keeping {n_freq} of {n_freq + self.freq_cutoff_idx} bins")

        # Build valid-bins mask from ASD
        self._build_valid_bins_mask(train_subset, n_channels, n_freq)

        # Build inner-product weights (4*df)
        self._build_inner_weights()

        # Compute global scale factor
        scale = self._compute_global_scale(train_dataloader)
        self.global_scale_factor = scale

        n = len(train_subset)
        seed = torch.randint(0, n, (1,)).item()

        first_data = self._to_device_async(train_subset[seed]['wave_fd'].unsqueeze(0))
        first_data = self._whiten(first_data)
        first_data = self._apply_freq_cutoff(first_data)
        first = self._normalize(first_data).squeeze(0)
        first = first / self._wnorm(first)
        basis = first.unsqueeze(0)

        sigma = float("inf")
        self.epoch = 0
        t0 = time.time()
        pbar = tqdm(total=0, bar_format='{desc}{postfix}', position=0, leave=True)

        log10mc_values = []
        q_values = []
        sigmas = []
        sigmas_unnorm = []
        sigmas_data = []
        picked = [seed]
        picked_set = {seed}
        self.gs_diagnose = {
            "norm_v_original": [],
            "norm_v_trunc": [],
            "norm_projection": [],
            "max_cosine_angle_unnormalised": [],
            "max_cosine_angle_normalised": []
        }
        diagnostics = {
            "log10mc_values": log10mc_values,
            "q_values": q_values,
            "sigmas": sigmas,
            "sigmas_unnorm": sigmas_unnorm,
            "sigmas_data": sigmas_data,
            "max_pointwise_relerrors_ch1": [],
            "max_pointwise_relerrors_ch2": [],
            "picked_indices": picked,
            "gs_diagnose": self.gs_diagnose
        }
        if not hasattr(self, 'training_diagnostics') or not isinstance(self.training_diagnostics, dict):
            self.training_diagnostics = {}
        self.training_diagnostics["diagnostics"] = diagnostics

        self.basis = basis

        try:
            while sigma > self.tolerance and self.epoch < self.max_epochs:
                self.epoch += 1
                sigma, idx, sigma_unnorm, sigma_data, idx_data = self._max_residual_index(
                    train_dataloader, picked_set, diagnostics,
                    prefetch_batches=self._prefetch_batches
                )

                if convergence_on == 'sigma_data':
                    convergence_value = sigma_data
                    last_N_values = diagnostics['sigmas_data'][-self.patience:] if len(diagnostics['sigmas_data']) >= self.patience else diagnostics['sigmas_data']
                elif convergence_on == 'sigma':
                    convergence_value = sigma
                    last_N_values = diagnostics['sigmas'][-self.patience:] if len(diagnostics['sigmas']) >= self.patience else diagnostics['sigmas']

                if self._is_convergence_stagnant(convergence_value, last_N_values):
                    print(f"\n[ROM] convergence stagnation detected (no improvement in {self.patience} epochs). Stopping.")
                    break

                picked_set.add(idx)
                picked.append(idx)
                log10mc_values.append(train_subset[idx]["params"][0])
                q_values.append(train_subset[idx]["params"][1])
                sigmas.append(sigma)
                sigmas_unnorm.append(sigma_unnorm)
                sigmas_data.append(sigma_data)

                v_wave = self._to_device_async(train_subset[idx]['wave_fd'].unsqueeze(0))
                v_wave = self._whiten(v_wave)
                v_wave = self._apply_freq_cutoff(v_wave)
                v = self._normalize(v_wave).squeeze(0)
                v = self._gram_schmidt(v, n_passes=2)
                self.basis = torch.cat([self.basis, v.unsqueeze(0)], dim=0)

                elapsed = time.time() - t0
                pbar.set_postfix({
                    "N_basis": self.basis.shape[0],
                    "iter": self.epoch,
                    "sigma": f"{sigma:.3e}",
                    "sigma_data": f"{sigma_data:.3e}",
                    "conv": f"{convergence_value:.3e}",
                    "elapsed": f"{elapsed:.1f}s",
                    "rate": f"{self.epoch / elapsed:.1f} it/s"
                })
                pbar.update(0)

                if convergence_value <= self.tolerance:
                    print(f"\n[ROM] convergence reached ({convergence_on}={convergence_value:.3e} <= {self.tolerance:.1e})")
                    break

        except KeyboardInterrupt:
            print(f"\n[ROM] training interrupted by user.")

        total = time.time() - t0
        print(f"[ROM] done. basis={len(self.basis)}, time={total:.1f}s")

    # ------------------------------------------------------------------
    # batch projection & residual search
    # ------------------------------------------------------------------

    def _project_batch(self, batch):
        """Project normalised batch onto the basis (compress → reconstruct).

        :param batch: ``(B, D)`` normalised waveforms.
        :return: ``(B, D)`` projected waveforms.
        """
        coeff = self._compress_normalized(batch)
        return self._reconstruct_normalized(coeff)

    def _max_residual_index(self, train_dataloader, picked_set, diagnostics,
                            prefetch_batches: int = 1):
        """Find the training sample with the largest residual."""
        max_seen = -1.0
        max_seen_unnorm = -1.0
        max_seen_data = -1.0
        max_index = None
        max_index_data = None
        seen_points = 0
        max_relerr_ch1 = 0
        max_relerr_ch2 = 0
        dataloader_iter = iter(train_dataloader)
        batch_group_idx = 0

        while True:
            batches = []
            for _ in range(prefetch_batches):
                try:
                    batches.append(next(dataloader_iter))
                except StopIteration:
                    break
            if not batches:
                break

            if len(batches) == 1:
                batch = batches[0]
            else:
                keys = batches[0].keys()
                batch = {k: torch.cat([b[k] for b in batches], dim=0) for k in keys}

            bsize = batch['wave_fd'].shape[0]

            wave_gpu = self._to_device_async(batch['wave_fd'])
            wave_gpu = self._whiten(wave_gpu)
            wave_gpu = self._apply_freq_cutoff(wave_gpu)
            wave_batch = self._normalize(wave_gpu)
            proj = self._project_batch(wave_batch)

            # Weighted residual norms
            r_unnorm = wave_batch - proj
            wnorm_wave = self._wnorm(wave_batch)
            wnorm_resid = self._wnorm(r_unnorm)
            norms = (wnorm_resid / wnorm_wave) ** 2
            norms_unnorm = wnorm_resid ** 2

            # Per-channel max pointwise relative error (real / imag of complex)
            scale_ch1 = wave_batch.real.abs().amax(dim=1, keepdim=True)
            scale_ch2 = wave_batch.imag.abs().amax(dim=1, keepdim=True)
            err_ch1 = torch.amax((wave_batch.real - proj.real).abs() / scale_ch1)
            err_ch2 = torch.amax((wave_batch.imag - proj.imag).abs() / scale_ch2)

            max_relerr_ch1 = max(max_relerr_ch1, err_ch1.item())
            max_relerr_ch2 = max(max_relerr_ch2, err_ch2.item())

            # Noisy-data residual (sigma_data)
            noise_gpu = self._to_device_async(batch['noise_fd'])
            noise_gpu = self._whiten(noise_gpu)
            noise_gpu = self._apply_freq_cutoff(noise_gpu)
            original_data_batch = (wave_gpu + noise_gpu).reshape(bsize, -1)
            normalised_data_batch = self._normalize(original_data_batch)
            data_proj = self._project_batch(normalised_data_batch)
            rel_err_data = self._wnorm(wave_batch - data_proj) / wnorm_wave

            # Mask already-picked indices
            batch_ids = torch.arange(bsize) + seen_points
            mask = torch.tensor(
                [(idx.item() not in picked_set) for idx in batch_ids],
                device=norms.device, dtype=torch.bool,
            )

            if not mask.any():
                seen_points += bsize
                batch_group_idx += 1
                continue

            masked_norms = norms.clone()
            masked_norms[~mask] = -1.0

            current_max, current_idx = masked_norms.max(0)
            current_max_relerr_data, current_idx_data = rel_err_data.max(0)
            if current_max.item() > max_seen:
                max_seen = current_max.item()
                max_index = (current_idx + seen_points).item()
                max_seen_unnorm = norms_unnorm[current_idx].item()

            if current_max_relerr_data.item() > max_seen_data:
                max_seen_data = current_max_relerr_data.item()
                max_index_data = (current_idx_data + seen_points).item()
            seen_points += bsize
            batch_group_idx += 1

        diagnostics["max_pointwise_relerrors_ch1"].append(max_relerr_ch1)
        diagnostics["max_pointwise_relerrors_ch2"].append(max_relerr_ch2)
        return max_seen, max_index, max_seen_unnorm, max_seen_data, max_index_data

    # ------------------------------------------------------------------
    # data transfer helpers
    # ------------------------------------------------------------------

    def _pin_dataset_tensors(self, dataset):
        """Pin dataset tensors in CPU memory for faster GPU transfers."""
        for k in ["wave_fd", "source_parameters", "noise_fd"]:
            if hasattr(dataset, k):
                tensor = getattr(dataset, k)
                if tensor is not None and not tensor.is_pinned():
                    setattr(dataset, k, tensor.pin_memory())
        print("[ROM] Dataset tensors pinned to memory")

    def _to_device_async(self, tensor):
        """Move tensor to device, non-blocking when using pinned memory."""
        if self._use_pinned_memory:
            return tensor.to(self.device, non_blocking=True)
        return tensor.to(self.device)

    # ------------------------------------------------------------------
    # scale factor & inner-product weights
    # ------------------------------------------------------------------

    def _compute_global_scale(self, dataloader):
        """Compute global scale factor (max abs value over training set).

        Returns 1.0 when ``normalize_by_max`` is ``False``.
        """
        if not self.normalize_by_max:
            print(f"[ROM] normalize_by_max=False -> global_scale_factor=1.0")
            return 1.0
        max_val = 0.0
        for batch in dataloader:
            v = self._to_device_async(batch['wave_fd'])
            v = self._whiten(v)
            v = self._apply_freq_cutoff(v)
            max_val = max(max_val, v.abs().max().item())
        print(f"[ROM] global scale factor: {max_val:.3e}")
        return max_val

    def _build_valid_bins_mask(self, train_subset, n_channels, n_freq):
        """Build boolean mask for non-zero ASD bins and store the ASD."""
        import h5py
        dataset = train_subset.dataset if hasattr(train_subset, 'dataset') else train_subset
        filename = dataset.filename
        with h5py.File(filename, 'r') as f:
            asd_np = f['asd'][()]
        asd_full = torch.tensor(asd_np, dtype=torch.float32)
        self.asd = asd_full.to(self.device)

        asd_trimmed = self._apply_freq_cutoff(asd_full)
        asd_flat = asd_trimmed.reshape(-1)
        valid = asd_flat > 0
        n_valid = valid.sum().item()
        n_total = valid.numel()
        print(f"[ROM] valid_bins_mask: {n_valid}/{n_total} bins have non-zero ASD")
        if n_valid < n_total:
            print(f"[ROM] WARNING: {n_total - n_valid} zero-ASD bins detected.")
        self.valid_bins_mask = valid.to(self.device)

    def _build_inner_weights(self):
        """Build the inner-product weight vector ``4 * df`` per frequency bin.

        If ``df`` is ``None`` the weight defaults to 1 (unweighted).
        """
        n_channels = self.n_channels
        n_freq = self.n_freq
        prefactor = 4.0

        if self.df is None:
            w_per_dim = torch.ones(n_freq, device=self.device)
        elif isinstance(self.df, (int, float)):
            w_per_dim = prefactor * float(self.df) * torch.ones(n_freq, device=self.device)
        else:
            w_per_dim = prefactor * torch.as_tensor(self.df, device=self.device, dtype=torch.float32)
            if self.freq_cutoff_idx:
                w_per_dim = w_per_dim[self.freq_cutoff_idx:]
            assert w_per_dim.shape[0] == n_freq, (
                f"bin-width array has {w_per_dim.shape[0]} elements "
                f"after cutoff, but n_freq={n_freq}"
            )

        w = w_per_dim.repeat(n_channels)
        self.inner_weights = w
        print(f"[ROM] inner-product weights built "
              f"(prefactor={prefactor}, n_freq={n_freq}, n_channels={n_channels}, "
              f"w_min={w.min().item():.4e}, w_max={w.max().item():.4e})")

    def _wnorm(self, x):
        """Weighted norm consistent with :meth:`_inner`.

        Parameters
        ----------
        x : torch.Tensor  — 1-D ``(D,)`` or 2-D ``(B, D)``

        Returns
        -------
        torch.Tensor  — scalar or ``(B,)``
        """
        w = self.inner_weights
        if x.ndim == 1:
            if w is None:
                return x.norm()
            return ((x.conj() * x).real * w).sum().sqrt()
        else:
            if w is None:
                return x.norm(dim=-1)
            return ((x.conj() * x).real * w).sum(dim=-1).sqrt()

    # ------------------------------------------------------------------
    # Gram-Schmidt
    # ------------------------------------------------------------------

    def _project_onto_basis_single(self, v):
        """Project a single vector onto the basis (compress -> reconstruct)."""
        coeff = self._compress_normalized(v.unsqueeze(0))
        projection = self._reconstruct_normalized(coeff)
        return projection.squeeze(0)

    @profile
    def _gram_schmidt(self, v, n_passes=2):
        """Gram-Schmidt orthogonalisation with optional reorthogonalisation.

        Parameters
        ----------
        v : torch.Tensor  — 1-D vector to orthogonalise.
        n_passes : int  — 1 = classical GS, 2 = GS + reorthogonalisation.

        Returns
        -------
        normalised : torch.Tensor  — unit vector orthogonal to the basis.
        """
        norm_v_original = self._wnorm(v)
        first_projection_norm = None

        for i in range(n_passes):
            projection = self._project_onto_basis_single(v)
            v = v - projection

            if i == 0:
                first_projection_norm = self._wnorm(projection)
                _vo = norm_v_original.item()
                _vt = self._wnorm(v).item()
                _pr = first_projection_norm.item()
                assert _vo <= _vt + _pr, (
                    f"Triangle inequality violated: "
                    f"v_norm={_vo:.6e} > v_trunc + proj = {_vt:.6e} + {_pr:.6e}")
                assert _vt <= _vo + _pr, (
                    f"Triangle inequality violated: "
                    f"v_trunc={_vt:.6e} > v_norm + proj = {_vo:.6e} + {_pr:.6e}")
                assert _pr <= _vo + _vt, (
                    f"Triangle inequality violated: "
                    f"proj={_pr:.6e} > v_norm + v_trunc = {_vo:.6e} + {_vt:.6e}")

        v_trunc = v
        normalised = v_trunc / self._wnorm(v_trunc)

        # Orthogonality diagnostics
        B = self.basis
        B_wnorms = self._wnorm(B)
        v_trunc_wnorm = self._wnorm(v_trunc)
        normalised_wnorm = self._wnorm(normalised)
        scalar_products = self._inner(B, v_trunc.unsqueeze(0)).squeeze(1)
        cosines_vtrunc = scalar_products / (B_wnorms * v_trunc_wnorm)
        scalar_products_v_normalised = self._inner(B, normalised.unsqueeze(0)).squeeze(1)
        cosines_vnormalised = scalar_products_v_normalised / (B_wnorms * normalised_wnorm)
        self.gs_diagnose["max_cosine_angle_unnormalised"].append(cosines_vtrunc.abs().max().item())
        self.gs_diagnose["max_cosine_angle_normalised"].append(cosines_vnormalised.abs().max().item())
        self.gs_diagnose["norm_v_original"].append(norm_v_original.item())
        self.gs_diagnose["norm_projection"].append(first_projection_norm.item())
        self.gs_diagnose["norm_v_trunc"].append(v_trunc_wnorm.item())

        return normalised

    # ------------------------------------------------------------------
    # normalize / denormalize
    # ------------------------------------------------------------------

    def _normalize(self, data):
        """Scale data into normalized space (divide by global scale factor)."""
        if data.ndim == 3:
            d = data.reshape(data.shape[0], -1)
        else:
            d = data
        return d / self.global_scale_factor

    def _denormalize(self, data):
        """Un-scale data from normalized space."""
        return data * self.global_scale_factor

    # ------------------------------------------------------------------
    # compress / reconstruct (normalized space)
    # ------------------------------------------------------------------

    def _compress_normalized(self, normalized_data):
        """Project normalised data onto basis -> real coefficients.

        Uses the GW inner product ``Re[sum a* b w]``.
        """
        B = self.basis
        coeff = self._inner(B, normalized_data).T  # (B_size, N_basis)
        return coeff

    def _reconstruct_normalized(self, coeff):
        """Reconstruct normalised data from basis coefficients."""
        B = self.basis
        if B.is_complex() and not coeff.is_complex():
            coeff = coeff.to(B.dtype)
        return coeff @ B  # (B_size, N_dim)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def compress(self, data):
        """Compress FD data into reduced-basis coefficients.

        Handles whitening and frequency cutoff automatically.

        :param data: ``(B, C, F)`` complex tensor — raw frequency-domain data.
        :return: ``(B, N_basis)`` real coefficients.
        """
        data = self._whiten(data)
        data = self._apply_freq_cutoff(data)
        normalized_data = self._normalize(data)
        return self._compress_normalized(normalized_data)

    def reconstruct(self, coeff):
        """Reconstruct raw FD data from basis coefficients.

        The result is de-whitened (multiplied by ASD) when appropriate.
        If a frequency cutoff was used, only the retained bins are
        returned.

        :param coeff: ``(B, N_basis)`` real coefficients.
        :return: ``(B, C, n_freq)`` complex tensor.
        """
        normalized_data = self._reconstruct_normalized(coeff)
        denormalized_data = self._denormalize(normalized_data)
        whitened = denormalized_data.reshape(
            denormalized_data.shape[0], self.n_channels, self.n_freq)
        if self.asd is not None:
            asd = self.asd.to(whitened.device)
            asd_trimmed = self._apply_freq_cutoff(asd)
            whitened = whitened * asd_trimmed.unsqueeze(0)
        return whitened


class ROMWrapper(nn.Module):
    """Wrapper ``nn.Module`` around :class:`ReducedOrderModel` for use as
    ``InferenceNetwork.data_summary``.

    Parameters
    ----------
    filename : str
        Path to a saved ROM ``.pt`` file.
    device : str
        Torch device string (default ``"cuda"``).
    max_basis_elems : int or None
        If set, truncate the basis to at most this many elements.
    """

    def __init__(self, filename: str, device: str = "cuda", max_basis_elems=None):
        super().__init__()
        self.rom = ReducedOrderModel(filename=filename, device=device)
        if max_basis_elems is not None and self.rom.basis.shape[0] > max_basis_elems:
            self.rom.basis = self.rom.basis[:max_basis_elems]
            self.rom.basis_fd = self.rom.basis  # keep legacy alias in sync

    def get_n_features(self):
        """Number of scalar features produced by ``forward()``.

        Coefficients are real (from the ``Re[]`` inner product), so
        each basis vector contributes 1 feature.
        """
        return self.rom.basis.shape[0]

    def forward(self, d_f, d_t):
        """Compress FD data and pass TD data through unchanged.

        :param d_f: ``(B, C, F)`` complex tensor — raw frequency-domain data.
        :param d_t: passed through unchanged (kept for API compatibility
                    with other data summarisers).
        :return: ``(compressed, d_t)`` where *compressed* is a real tensor
                 of shape ``(B, n_features)``.
        """
        compressed = self.rom.compress(d_f)
        return compressed, d_t
