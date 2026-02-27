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
    """Reduced-order model that can be trained on frequency-domain (complex),
    time-domain (real) or both representations simultaneously.

    Parameters
    ----------
    domain : str
        Which representation(s) to build a basis for.
        One of ``'fd'``, ``'td'``, or ``'both'``.
    """

    _VALID_DOMAINS = ('fd', 'td', 'both')

    def __init__(self, batch_size=1000, tolerance=1e-3, device="cpu",
                 filename=None, debugging=False, domain='fd', patience=10,
                 freq_cutoff_idx=None, df=None, dt=None):
        if filename is not None: 
            print(f"[ROM] initializing from file {filename}.\n Other arguments will be ignored.")
            self.device = device
            self.from_file(filename)
            
        else:
            if domain not in self._VALID_DOMAINS:
                raise ValueError(f"domain must be one of {self._VALID_DOMAINS}, got '{domain}'")
            self.domain = domain
            self.debugging = debugging
            self.batch_size = batch_size
            self.device = device
            self.tolerance = tolerance
            self.patience = patience
            self.max_epochs = 4000  # safety cap to prevent infinite loops in case of non-convergence
            self.freq_cutoff_idx = freq_cutoff_idx  # discard fd bins below this index
            self.df = df  # frequency bin width(s): scalar (uniform) or 1-D array (non-uniform / logspaced)
            self.dt = dt  # time bin width(s): scalar (uniform) or 1-D array (non-uniform)

            # Per-domain attributes initialised to None; filled during train()
            for dom in self._active_domains:
                setattr(self, f'basis_{dom}', None)
                setattr(self, f'n_channels_{dom}', None)
                setattr(self, f'n_dim_{dom}', None)      # n_freq or n_time
                setattr(self, f'global_scale_factor_{dom}', None)
                setattr(self, f'mean_vec_{dom}', None)
                setattr(self, f'valid_bins_mask_{dom}', None)  # boolean mask for non-zero ASD bins
                setattr(self, f'inner_weights_{dom}', None)    # inner-product weights (4*df or dt)

            self.asd = None  # ASD tensor, shape (n_channels, n_freq_full); stored for inference-time whitening


            # Keep legacy aliases when only fd is used
            if domain == 'fd':
                self.basis = None
                self.n_channels = None
                self.n_freq = None

            if debugging:
                self.plot_dir_debug = os.path.join(ROOT_DIR, "plots", "debug_plots_{time}".format(time=time.strftime("%Y%m%d-%H%M%S"))) 
                os.makedirs(self.plot_dir_debug, exist_ok=True)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @property
    def _active_domains(self):
        """Return list of domain tags that are active."""
        if self.domain == 'both':
            return ['fd', 'td']
        return [self.domain]

    @staticmethod
    def _is_complex(dom):
        return dom == 'fd'

    def _inner(self, A, B, dom):
        """Compute the weighted inner product ⟨A|B⟩ between two sets of vectors.

        For FD (complex) data this implements the GW-standard real inner
        product including the frequency-bin width factor:

            ⟨a|b⟩ = Re[∑_k a_k* b_k w_k]

        where ``w_k = 4 Δf_k``.  When the data has been pre-whitened
        (divided by ASD), the noise weighting 1/S_n(f) is already
        absorbed into the vectors.

        For TD (real) data the weight is ``Δt_k``.

        If no bin-width was supplied at construction time the weight
        defaults to 1 (unweighted dot product, backward-compatible).

        Parameters
        ----------
        A : torch.Tensor
            Shape ``(M, D)`` — *M* row-vectors of dimension *D*.
        B : torch.Tensor
            Shape ``(N, D)`` — *N* row-vectors of dimension *D*.
        dom : str
            Domain tag (``'fd'`` or ``'td'``).

        Returns
        -------
        torch.Tensor
            Real tensor of shape ``(M, N)`` with entries ``⟨A_i | B_j⟩``.
        """
        w = getattr(self, f'inner_weights_{dom}', None)
        if self._is_complex(dom):
            Aw = A.conj() * w if w is not None else A.conj()
            return (Aw @ B.mT).real
        Aw = A * w if w is not None else A
        return Aw @ B.mT

    def _apply_freq_cutoff(self, data, dom):
        """Slice off low-frequency bins when *freq_cutoff_idx* is set (fd only).

        Parameters
        ----------
        data : torch.Tensor
            2-D ``(C, F)`` or 3-D ``(B, C, F)`` tensor.
        dom : str
            Domain tag.

        Returns
        -------
        torch.Tensor
            Sliced tensor if cutoff applies, otherwise the input unchanged.
        """
        idx = getattr(self, 'freq_cutoff_idx', None)
        if dom != 'fd' or not idx:
            return data
        if data.ndim == 3:
            return data[:, :, idx:]
        elif data.ndim == 2:
            return data[:, idx:]
        return data

    def _whiten_fd(self, data):
        """Whiten FD data by dividing by the stored ASD.

        Applies the standard pre-whitening ``data / ASD``, with zero-ASD
        bins guarded (0 → inf so that whitened value → 0).
        No-op if ``self.asd`` is ``None``.

        Parameters
        ----------
        data : torch.Tensor
            2-D ``(C, F)`` or 3-D ``(B, C, F)`` tensor (complex or real).

        Returns
        -------
        torch.Tensor
            Whitened tensor, same shape and dtype as input.
        """
        if self.asd is None:
            return data
        asd = self.asd.to(data.device)
        safe_asd = asd.clone()
        safe_asd[safe_asd == 0] = float('inf')
        if data.ndim == 3:
            return data / safe_asd.unsqueeze(0)
        return data / safe_asd

    def from_file(self, filename):
        s = torch.load(filename, map_location=self.device)
        print("Keys in the saved file:", list(s.keys()))
        print(f"name of the saved file: {filename}")

        # Detect whether this is a legacy (fd-only) file or multi-domain
        if 'domain' in s:
            self.domain = s['domain']
        else:
            self.domain = 'fd'  # legacy files are fd-only

        # Frequency cutoff (None for legacy / uncropped models)
        self.freq_cutoff_idx = s.get('freq_cutoff_idx', None)

        # Bin widths for inner-product weighting
        self.df = s.get('df', None)
        self.dt = s.get('dt', None)

        # ASD (for inference-time whitening)
        self.asd = s.get('asd', None)
        if self.asd is not None:
            self.asd = self.asd.to(self.device)
            print(f"[ROM] ASD loaded, shape={tuple(self.asd.shape)}")

        for dom in self._active_domains:
            if f'basis_{dom}' in s:
                setattr(self, f'basis_{dom}', s[f'basis_{dom}'].to(self.device))
                setattr(self, f'n_channels_{dom}', s[f'n_channels_{dom}'])
                setattr(self, f'n_dim_{dom}', s[f'n_dim_{dom}'])
                setattr(self, f'global_scale_factor_{dom}', s[f'global_scale_factor_{dom}'])
                setattr(self, f'mean_vec_{dom}', s[f'mean_vec_{dom}'].to(self.device))
                # Valid-bins mask (Option B: boolean mask for non-zero ASD bins)
                vbm = s.get(f'valid_bins_mask_{dom}', None)
                if vbm is not None:
                    setattr(self, f'valid_bins_mask_{dom}', vbm.to(self.device))
                    n_valid = vbm.sum().item()
                    n_total = vbm.numel()
                    print(f"[ROM] {dom} valid_bins_mask loaded: {n_valid}/{n_total} bins active")
                else:
                    setattr(self, f'valid_bins_mask_{dom}', None)
                # Inner-product weights (4*df or dt)
                iw = s.get(f'inner_weights_{dom}', None)
                if iw is not None:
                    setattr(self, f'inner_weights_{dom}', iw.to(self.device))
                    print(f"[ROM] {dom} inner_weights loaded (min={iw.min().item():.4e}, max={iw.max().item():.4e})")
                else:
                    setattr(self, f'inner_weights_{dom}', None)
                basis = getattr(self, f'basis_{dom}')
                print(f"[ROM] {dom} basis loaded. n_dim={getattr(self, f'n_dim_{dom}')}, "
                      f"n_channels={getattr(self, f'n_channels_{dom}')}, n_basis={len(basis)}")
            elif dom == 'fd' and 'basis' in s:
                # Legacy fd-only format
                self.basis_fd = s['basis'].to(self.device)
                self.n_channels_fd = s['n_channels']
                self.n_dim_fd = s['n_freq']
                self.global_scale_factor_fd = s['global_scale_factor']
                self.mean_vec_fd = s['mean_vec'].to(self.device)
                print(f"[ROM] fd basis loaded (legacy). n_freq={self.n_dim_fd}, "
                      f"n_channels={self.n_channels_fd}, n_basis={len(self.basis_fd)}")

        # Keep legacy aliases for fd-only models
        if self.domain == 'fd':
            self.basis = self.basis_fd
            self.n_channels = self.n_channels_fd
            self.n_freq = self.n_dim_fd
            self.global_scale_factor = self.global_scale_factor_fd
            self.mean_vec = self.mean_vec_fd
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
            "domain": self.domain,
            "tolerance": self.tolerance,
            "device": self.device,
            "freq_cutoff_idx": self.freq_cutoff_idx,
            "df": self.df if not isinstance(self.df, torch.Tensor) else self.df.cpu(),
            "dt": self.dt if not isinstance(self.dt, torch.Tensor) else self.dt.cpu(),
        }
        # Store ASD for inference-time whitening
        if self.asd is not None:
            state['asd'] = self.asd.cpu() if isinstance(self.asd, torch.Tensor) else self.asd

        for dom in self._active_domains:
            state[f'basis_{dom}'] = getattr(self, f'basis_{dom}').cpu()
            state[f'n_channels_{dom}'] = getattr(self, f'n_channels_{dom}')
            state[f'n_dim_{dom}'] = getattr(self, f'n_dim_{dom}')
            state[f'global_scale_factor_{dom}'] = getattr(self, f'global_scale_factor_{dom}')
            state[f'mean_vec_{dom}'] = getattr(self, f'mean_vec_{dom}').cpu()
            vbm = getattr(self, f'valid_bins_mask_{dom}', None)
            if vbm is not None:
                state[f'valid_bins_mask_{dom}'] = vbm.cpu() if isinstance(vbm, torch.Tensor) else vbm
            iw = getattr(self, f'inner_weights_{dom}', None)
            if iw is not None:
                state[f'inner_weights_{dom}'] = iw.cpu() if isinstance(iw, torch.Tensor) else iw


        # Legacy keys for backward compatibility when fd-only
        if self.domain == 'fd':
            state['basis'] = state['basis_fd']
            state['n_channels'] = state['n_channels_fd']
            state['n_freq'] = state['n_dim_fd']
            state['global_scale_factor'] = state['global_scale_factor_fd']
            state['mean_vec'] = state['mean_vec_fd']

        torch.save(state, filename)
        filename_diagnostics = filename.replace(".pt", "_diagnostics.pt")
        self.save_diagnostics(filename_diagnostics)
        print(f"[ROM] basis saved to {filename}")
        return

    @profile
    def train(self, train_dataloader: torch.utils.data.DataLoader, use_pinned_memory: bool = True, prefetch_batches: int = 1,
              convergence_on: str = 'sigma_data'):
        """Train the reduced order model on the chosen domain(s).

        Args:
            train_dataloader (torch.utils.data.DataLoader): the training dataloader, that has the training Subset as attribute
            use_pinned_memory (bool): If True, uses pinned memory + non-blocking transfers (low GPU memory, fast).
                                      If False, moves entire dataset to GPU (high GPU memory, fastest).
            prefetch_batches (int): Number of DataLoader batches to concatenate before processing.
                                    Larger values reduce Python overhead but use more memory.
            convergence_on (str): Which metric to use as stopping criterion.
                                  ``'sigma'`` = normalised waveform residual (original behaviour),
                                  ``'sigma_data'`` = reconstruction loss on noisy data (default).
        """
        print(f"[ROM] training domain='{self.domain}' with tolerance={self.tolerance:.1e} on device={self.device}")
        print(f"[ROM] convergence criterion: {convergence_on}")
        if self.freq_cutoff_idx:
            print(f"[ROM] frequency cutoff enabled: discarding fd bins below index {self.freq_cutoff_idx}")
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

        # Train each active domain independently
        for dom in self._active_domains:
            self._train_single_domain(dom, train_dataloader, convergence_on)

        # Legacy aliases for fd-only models
        if self.domain == 'fd':
            self.basis = self.basis_fd
            self.n_channels = self.n_channels_fd
            self.n_freq = self.n_dim_fd
            self.global_scale_factor = self.global_scale_factor_fd
            self.mean_vec = self.mean_vec_fd


    def _is_convergence_stagnant(self, convergence_value, last_values):
        """Check if the convergence_value has not improved for a certain number of epochs."""
        if len(last_values) < self.patience:
            return False
        recent_values = last_values
        best_recent = min(recent_values)
        is_stagnant = convergence_value >= best_recent

        return is_stagnant
    def _train_single_domain(self, dom, train_dataloader, convergence_on):
        """Train a reduced basis for a single domain ('fd' or 'td').
        
        """
        wave_key = f'wave_{dom}'
        noise_key = f'noise_{dom}'
        train_subset = train_dataloader.dataset
        is_complex = self._is_complex(dom)
        print(f"\n[ROM][{dom}] --- starting basis construction ({'complex' if is_complex else 'real'}) ---")

        sample = train_subset[0][wave_key]
        sample = self._apply_freq_cutoff(sample, dom)  # slice low-freq bins if cutoff is set
        n_channels = sample.shape[0]
        n_dim = sample.shape[1]  # n_freq (possibly truncated) for fd, n_time for td
        setattr(self, f'n_channels_{dom}', n_channels)
        setattr(self, f'n_dim_{dom}', n_dim)

        if dom == 'fd' and self.freq_cutoff_idx:
            print(f"[ROM][{dom}] frequency cutoff at index {self.freq_cutoff_idx}: "
                  f"keeping {n_dim} of {n_dim + self.freq_cutoff_idx} bins")

        # Build valid-bins mask from ASD (Option B)
        if dom == 'fd':
            self._build_valid_bins_mask(train_subset, dom, n_channels, n_dim)

        # Build inner-product weights (4*df for fd, dt for td)
        self._build_inner_weights(dom)

        # Compute global scale & mean for this domain
        mean, scale = self._compute_global_scale(train_dataloader, wave_key, n_channels * n_dim, dom=dom)
        mean = mean.to(self.device)
        setattr(self, f'global_scale_factor_{dom}', scale)
        setattr(self, f'mean_vec_{dom}', mean)

        n = len(train_subset)
        seed = torch.randint(0, n, (1,)).item()

        first_data = self._to_device_async(train_subset[seed][wave_key].unsqueeze(0))
        if dom == 'fd':
            first_data = self._whiten_fd(first_data)
        first_data = self._apply_freq_cutoff(first_data, dom)
        first = self._normalize_dom(first_data, dom).squeeze(0)
        first = first / self._wnorm(first, dom)
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
        diag_key = f"diagnostics_{dom}" if self.domain == 'both' else "diagnostics"
        diagnostics = {
            "domain": dom,
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
        self.training_diagnostics[diag_key] = diagnostics

        # Temporarily set self.basis so that _gram_schmidt helpers work
        setattr(self, f'basis_{dom}', basis)

        try:
            while sigma > self.tolerance and self.epoch < self.max_epochs:
                self.epoch += 1
                sigma, idx, sigma_unnorm, sigma_data, idx_data = self._max_residual_index_dom(
                    dom, train_dataloader, picked_set, diagnostics,
                    prefetch_batches=self._prefetch_batches
                )
                # Choose convergence metric

                if convergence_on == 'sigma_data':
                    convergence_value = sigma_data
                    last_10_values = diagnostics['sigmas_data'][-self.patience:] if len(diagnostics['sigmas_data']) >= self.patience else diagnostics['sigmas_data']
                
                elif convergence_on == 'sigma':
                    convergence_value = sigma
                    last_10_values = diagnostics['sigmas'][-self.patience:] if len(diagnostics['sigmas']) >= self.patience else diagnostics['sigmas']
                # early stop training if the convergence_value does not decrease after a certain number of epochs (e.g. 10)
                # if self._is_convergence_stagnant(convergence_value, last_10_values):
                #     print(f"\n[ROM][{dom}] convergence stagnation detected (no improvement in {self.patience} epochs). Stopping training.")
                #     break
                picked_set.add(idx)
                picked.append(idx)
                log10mc_values.append(train_subset[idx]["params"][0])
                q_values.append(train_subset[idx]["params"][1])
                sigmas.append(sigma)
                sigmas_unnorm.append(sigma_unnorm)
                sigmas_data.append(sigma_data)

                v_wave = self._to_device_async(train_subset[idx][wave_key].unsqueeze(0))
                if dom == 'fd':
                    v_wave = self._whiten_fd(v_wave)
                v_wave = self._apply_freq_cutoff(v_wave, dom)
                v = self._normalize_dom(v_wave, dom).squeeze(0)
                v = self._gram_schmidt_reorthogonalize_dom(v, dom)
                basis = getattr(self, f'basis_{dom}')
                setattr(self, f'basis_{dom}', torch.cat([basis, v.unsqueeze(0)], dim=0))

                basis = getattr(self, f'basis_{dom}')
                elapsed = time.time() - t0
                pbar.set_postfix({
                    "dom": dom,
                    "N_basis": basis.shape[0],
                    "iter": self.epoch,
                    "sigma": f"{sigma:.3e}",
                    "sigma_data": f"{sigma_data:.3e}",
                    "conv": f"{convergence_value:.3e}",
                    "elapsed": f"{elapsed:.1f}s",
                    "rate": f"{self.epoch / elapsed:.1f} it/s"
                })
                pbar.update(0)

                if convergence_value <= self.tolerance:
                    print(f"\n[ROM][{dom}] convergence reached ({convergence_on}={convergence_value:.3e} <= {self.tolerance:.1e})")
                    break

        except KeyboardInterrupt:
            print(f"\n[ROM][{dom}] training interrupted by user.")
        # except Exception as e:
        #     print(f"\n[ROM][{dom}] training stopped due to error: {e}")

        basis = getattr(self, f'basis_{dom}')
        total = time.time() - t0
        print(f"[ROM][{dom}] done. basis={len(basis)}, time={total:.1f}s")

        
    @profile
    def _project_batch(self, batch):
        """Project the input batch onto the ROM basis (legacy fd-only wrapper)."""
        return self._project_batch_dom(batch, 'fd')

    def _project_batch_dom(self, batch, dom):
        """Project the input batch onto the ROM basis for a given domain.

        :param batch: Input batch of normalized waveforms, with shape (B_size, N_dim). 
        :param dom: Domain tag ('fd' or 'td').
        :return: Projected batch in normalized space.
        """
        coeff = self._compress_normalized_dom(batch, dom)
        proj = self._reconstruct_normalized_dom(coeff, dom)
        return proj

    @profile
    def _max_residual_index(self, train_dataloader, picked_set, prefetch_batches: int = 1):
        """Legacy fd-only wrapper."""
        diag_key = 'diagnostics_fd' if 'diagnostics_fd' in self.training_diagnostics else 'diagnostics'
        return self._max_residual_index_dom('fd', train_dataloader, picked_set,
                                            self.training_diagnostics.get(diag_key, self.training_diagnostics),
                                            prefetch_batches=prefetch_batches)

    def _max_residual_index_dom(self, dom, train_dataloader, picked_set, diagnostics, prefetch_batches: int = 1):
        """Find the training sample with the largest residual for a given domain.

        Works identically for complex (fd) and real (td) data.
        """
        wave_key = f'wave_{dom}'
        noise_key = f'noise_{dom}'
        n_dim = getattr(self, f'n_dim_{dom}')
        is_complex = self._is_complex(dom)

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

            bsize = batch[wave_key].shape[0]

            wave_gpu = self._to_device_async(batch[wave_key])
            if dom == 'fd':
                wave_gpu = self._whiten_fd(wave_gpu)
            wave_gpu = self._apply_freq_cutoff(wave_gpu, dom)
            flattened_wave_batch = wave_gpu.reshape(bsize, -1)# flattened on freq, channel dim
            # here wave_batch is the whitened waveform, minus the mean divided by abs(max)
            wave_batch = self._normalize_dom(wave_gpu, dom)
            # the normalised wave_batch is projected onto the basis (made by normalised whitened waves)
            proj = self._project_batch_dom(wave_batch, dom)


            # and is denormalised to get a comparison with the original waveform
            reconstruction_denormalised = self._denormalize_dom(proj, dom)
            # the difference between the flattened wave (clean) and the denormalised reconstruction is assessed
            #residual_original_space = flattened_wave_batch - reconstruction_denormalised

            # Weighted residual norms
            r_unnorm = wave_batch - proj
            wnorm_wave = self._wnorm(wave_batch, dom)       # (bsize,)
            wnorm_resid = self._wnorm(r_unnorm, dom)        # (bsize,)
            norms = (wnorm_resid / wnorm_wave) ** 2         # squared relative residual
            norms_unnorm = wnorm_resid ** 2                 # squared absolute residual

            # Per-channel max pointwise relative error (element-wise, no inner-product weight)
            if is_complex:
                scale_ch1 = wave_batch.real.abs().amax(dim=1, keepdim=True)
                scale_ch2 = wave_batch.imag.abs().amax(dim=1, keepdim=True)
                err_ch1 = torch.amax((wave_batch.real - proj.real).abs() / scale_ch1)
                err_ch2 = torch.amax((wave_batch.imag - proj.imag).abs() / scale_ch2)
            else:
                # For real data split by channel
                ch1 = wave_batch[:, :n_dim]
                ch1_proj = proj[:, :n_dim]
                ch2 = wave_batch[:, n_dim:]
                ch2_proj = proj[:, n_dim:]
                scale_ch1 = ch1.abs().amax(dim=1, keepdim=True).clamp(min=1e-30)
                scale_ch2 = ch2.abs().amax(dim=1, keepdim=True).clamp(min=1e-30)
                err_ch1 = torch.amax((ch1 - ch1_proj).abs() / scale_ch1)
                err_ch2 = torch.amax((ch2 - ch2_proj).abs() / scale_ch2)

            max_relerr_ch1 = max(max_relerr_ch1, err_ch1.item())
            max_relerr_ch2 = max(max_relerr_ch2, err_ch2.item())

            # Noisy-data residual  (sigma_data)
            noise_gpu = self._to_device_async(batch[noise_key])
            if dom == 'fd':
                noise_gpu = self._whiten_fd(noise_gpu)
            noise_gpu = self._apply_freq_cutoff(noise_gpu, dom)
            original_data_batch = (wave_gpu + noise_gpu).reshape(bsize, -1)
            normalised_data_batch = self._normalize_dom(original_data_batch, dom)
            data_proj = self._project_batch_dom(normalised_data_batch, dom)
            rel_err_data = self._wnorm(wave_batch - data_proj, dom) / wnorm_wave
            # Mask already-picked indices
            batch_ids = torch.arange(bsize) + seen_points
            mask = torch.tensor(
                [(idx.item() not in picked_set) for idx in batch_ids],
                device=norms.device,
                dtype=torch.bool,
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
    
    def _pin_dataset_tensors(self, dataset):
        """Pin dataset tensors in CPU memory for faster GPU transfers."""
        for k in ["wave_fd", "wave_td", "source_parameters", "noise_fd", "noise_td"]:
            if hasattr(dataset, k):
                tensor = getattr(dataset, k)
                if tensor is not None and not tensor.is_pinned():
                    setattr(dataset, k, tensor.pin_memory())
        print("[ROM] Dataset tensors pinned to memory")

    def _to_device_async(self, tensor):
        """Move tensor to device with non-blocking transfer if using pinned memory."""
        if self._use_pinned_memory:
            return tensor.to(self.device, non_blocking=True)
        else:
            return tensor.to(self.device)

    def _compute_global_scale(self, dataloader, wave_key='wave_fd', flat_dim=None, dom='fd'):
        """Compute global scale factor and mean vector for a given data key."""
        if flat_dim is None:
            # Legacy: infer from n_channels * n_freq
            flat_dim = self.n_channels * self.n_freq
        max_val = 0.0
        # Peek at first batch to determine dtype
        peek = self._apply_freq_cutoff(next(iter(dataloader))[wave_key], dom)
        vec_dtype = torch.complex64 if peek.is_complex() else torch.float32
        mean_vec = torch.zeros(flat_dim, device=self.device, dtype=vec_dtype)
        for batch in dataloader:
            v = self._to_device_async(batch[wave_key])
            if dom == 'fd':
                v = self._whiten_fd(v)
            v = self._apply_freq_cutoff(v, dom)
            max_val = max(max_val, v.abs().max().item())
            mean_vec = mean_vec + v.reshape(v.shape[0], -1).sum(dim=0)
        mean_vec = mean_vec / len(dataloader.dataset)
        print(f"[ROM][{dom}] global scale factor: {max_val:.3e}")
        return mean_vec, max_val

    def _build_valid_bins_mask(self, train_subset, dom, n_channels, n_dim):
        """Build a boolean mask identifying frequency bins with non-zero ASD.

        The mask has shape ``(n_channels * n_dim,)`` (flattened) so it can
        be applied directly to the flattened normalised vectors used in
        inner products.  Bins where ASD == 0 are ``False``.

        The ASD is read once from the underlying HDF5 dataset and stored
        on ``self.asd`` for later serialisation.
        """
        import h5py
        dataset = train_subset.dataset if hasattr(train_subset, 'dataset') else train_subset
        filename = dataset.filename
        with h5py.File(filename, 'r') as f:
            asd_np = f['asd'][()]                    # (n_channels, n_freq_full)
        asd_full = torch.tensor(asd_np, dtype=torch.float32)
        self.asd = asd_full.to(self.device)          # store for inference-time whitening

        # Apply freq_cutoff if set
        asd_trimmed = self._apply_freq_cutoff(asd_full, dom)  # (n_channels, n_dim)
        # Flatten to (n_channels * n_dim,) — same layout as normalised vectors
        asd_flat = asd_trimmed.reshape(-1)
        valid = asd_flat > 0
        n_valid = valid.sum().item()
        n_total = valid.numel()
        print(f"[ROM][{dom}] valid_bins_mask: {n_valid}/{n_total} bins have non-zero ASD")
        if n_valid < n_total:
            print(f"[ROM][{dom}] WARNING: {n_total - n_valid} zero-ASD bins detected. "
                  f"These will be excluded from inner products.")
        setattr(self, f'valid_bins_mask_{dom}', valid.to(self.device))

    # ------------------------------------------------------------------
    # Inner-product weights & weighted norm
    # ------------------------------------------------------------------
    def _build_inner_weights(self, dom):
        """Build the inner-product weight vector for a given domain.

        For FD the weight per frequency bin is ``4 * Δf``  (the standard
        GW inner-product prefactor).  For TD it is ``Δt``.

        The per-bin weight is tiled across channels to match the flattened
        vector layout ``(n_channels * n_dim,)``.

        If no ``df`` / ``dt`` was supplied at construction time the weight
        defaults to all ones (backward-compatible, unweighted inner product).
        """
        n_channels = getattr(self, f'n_channels_{dom}')
        n_dim = getattr(self, f'n_dim_{dom}')

        if dom == 'fd':
            raw = self.df
            prefactor = 4.0
        else:  # td
            raw = self.dt
            prefactor = 1.0

        if raw is None:
            w_per_dim = torch.ones(n_dim, device=self.device)
        elif isinstance(raw, (int, float)):
            w_per_dim = prefactor * float(raw) * torch.ones(n_dim, device=self.device)
        else:
            # raw is a tensor / array — non-uniform bin widths
            w_per_dim = prefactor * torch.as_tensor(raw, device=self.device, dtype=torch.float32)
            # Apply freq cutoff if set (only the retained bins matter)
            if dom == 'fd' and self.freq_cutoff_idx:
                w_per_dim = w_per_dim[self.freq_cutoff_idx:]
            assert w_per_dim.shape[0] == n_dim, (
                f"bin-width array for '{dom}' has {w_per_dim.shape[0]} elements "
                f"after cutoff, but n_dim={n_dim}"
            )

        # Tile across channels: [w_0, …, w_{D-1}, w_0, …, w_{D-1}, …]
        w = w_per_dim.repeat(n_channels)
        setattr(self, f'inner_weights_{dom}', w)
        print(f"[ROM][{dom}] inner-product weights built "
              f"(prefactor={prefactor}, n_dim={n_dim}, n_channels={n_channels}, "
              f"w_min={w.min().item():.4e}, w_max={w.max().item():.4e})")

    def _wnorm(self, x, dom):
        """Weighted norm consistent with :meth:`_inner`.

        Parameters
        ----------
        x : torch.Tensor
            1-D ``(D,)`` or 2-D ``(B, D)`` tensor.
        dom : str
            Domain tag.

        Returns
        -------
        torch.Tensor
            Scalar (1-D input) or shape ``(B,)`` (2-D input).
        """
        w = getattr(self, f'inner_weights_{dom}', None)
        if x.ndim == 1:
            if w is None:
                return x.norm()
            return ((x.conj() * x).real * w).sum().sqrt()
        else:
            if w is None:
                return x.norm(dim=-1)
            return ((x.conj() * x).real * w).sum(dim=-1).sqrt()

    # ------------------------------------------------------------------
    # Gram-Schmidt orthogonalization (unified)
    # ------------------------------------------------------------------
    def _project_onto_basis_single(self, v, dom):
        """Project a single vector onto the current basis (compress → reconstruct).

        Parameters
        ----------
        v : torch.Tensor
            1-D vector of shape ``(N_dim,)``.
        dom : str
            Domain tag.

        Returns
        -------
        projection : torch.Tensor
            Component of *v* lying in the span of the basis, shape ``(N_dim,)``.
        """
        coeff = self._compress_normalized_dom(v.unsqueeze(0), dom)   # (1, N_basis)
        projection = self._reconstruct_normalized_dom(coeff, dom)    # (1, N_dim)
        return projection.squeeze(0)

    @profile
    def _gram_schmidt_dom(self, v, dom, n_passes=2):
        """Gram-Schmidt orthogonalization, domain-aware.

        Each pass projects *v* onto the current basis via
        compress → reconstruct and subtracts the result.
        Setting ``n_passes=2`` (default) gives the classical
        *Gram-Schmidt with one reorthogonalization step*.

        Parameters
        ----------
        v : torch.Tensor
            1-D vector to orthogonalize against the current basis.
        dom : str
            Domain tag (``'fd'`` or ``'td'``).
        n_passes : int
            Number of orthogonalization passes.
            Use 1 for classical GS, 2 for GS + reorthogonalization.

        Returns
        -------
        normalised : torch.Tensor
            Unit vector orthogonal to the basis.
        """
        norm_v_original = self._wnorm(v, dom)
        first_projection_norm = None

        for i in range(n_passes):
            projection = self._project_onto_basis_single(v, dom)
            v = v - projection

            if i == 0:
                first_projection_norm = self._wnorm(projection, dom)
                # Triangle inequality checks (first pass only)
                _vo = norm_v_original.item()
                _vt = self._wnorm(v, dom).item()
                _pr = first_projection_norm.item()
                assert _vo <= _vt + _pr, (
                    f"Triangular inequality violated: "
                    f"v_norm={_vo:.6e} > v_trunc_norm + projection_norm = "
                    f"{_vt:.6e} + {_pr:.6e} = {_vt + _pr:.6e} "
                    f"(excess={_vo - (_vt + _pr):.6e})"
                )
                assert _vt <= _vo + _pr, (
                    f"Triangular inequality violated: "
                    f"v_trunc_norm={_vt:.6e} > v_norm + projection_norm = "
                    f"{_vo:.6e} + {_pr:.6e} = {_vo + _pr:.6e} "
                    f"(excess={_vt - (_vo + _pr):.6e})"
                )
                assert _pr <= _vo + _vt, (
                    f"Triangular inequality violated: "
                    f"projection_norm={_pr:.6e} > v_norm + v_trunc_norm = "
                    f"{_vo:.6e} + {_vt:.6e} = {_vo + _vt:.6e} "
                    f"(excess={_pr - (_vo + _vt):.6e})"
                )

        v_trunc = v
        normalised = v_trunc / self._wnorm(v_trunc, dom)

        # Orthogonality diagnostics (using the same weighted inner product)
        B = getattr(self, f'basis_{dom}')
        B_wnorms = self._wnorm(B, dom)           # (N_basis,)
        v_trunc_wnorm = self._wnorm(v_trunc, dom)
        normalised_wnorm = self._wnorm(normalised, dom)
        # _inner(B, v.unsqueeze(0), dom) → (N_basis, 1); squeeze to (N_basis,)
        scalar_products = self._inner(B, v_trunc.unsqueeze(0), dom).squeeze(1)
        cosines_vtrunc = scalar_products / (B_wnorms * v_trunc_wnorm)
        scalar_products_v_normalised = self._inner(B, normalised.unsqueeze(0), dom).squeeze(1)
        cosines_vnormalised = scalar_products_v_normalised / (B_wnorms * normalised_wnorm)
        self.gs_diagnose["max_cosine_angle_unnormalised"].append(cosines_vtrunc.abs().max().item())
        self.gs_diagnose["max_cosine_angle_normalised"].append(cosines_vnormalised.abs().max().item())

        self.gs_diagnose["norm_v_original"].append(norm_v_original.item())
        self.gs_diagnose["norm_projection"].append(first_projection_norm.item())
        self.gs_diagnose["norm_v_trunc"].append(v_trunc_wnorm.item())

        return normalised

    def _gram_schmidt_reorthogonalize(self, v):
        """Legacy fd-only wrapper (2-pass Gram-Schmidt)."""
        return self._gram_schmidt_dom(v, 'fd', n_passes=2)

    def _gram_schmidt_reorthogonalize_dom(self, v, dom):
        """Gram-Schmidt with reorthogonalization (2-pass), domain-aware."""
        return self._gram_schmidt_dom(v, dom, n_passes=2)

    def _gram_schmidt(self, v):
        """Legacy fd-only wrapper (single-pass Gram-Schmidt)."""
        return self._gram_schmidt_dom(v, 'fd', n_passes=1)

    # ------------------------------------------------------------------
    # normalize / denormalize  (domain-aware + legacy wrappers)
    # ------------------------------------------------------------------
    def _normalize(self, data):
        """Legacy fd-only wrapper."""
        return self._normalize_dom(data, 'fd')

    def _normalize_dom(self, data, dom):
        """Center and scale data to normalized space for a given domain."""
        if data.ndim == 3:
            d = data.reshape(data.shape[0], -1)
        else:
            d = data
        mean = getattr(self, f'mean_vec_{dom}')
        scale = getattr(self, f'global_scale_factor_{dom}')
        return (d - mean) / scale
    
    def _denormalize(self, data):
        """Legacy fd-only wrapper."""
        return self._denormalize_dom(data, 'fd')

    def _denormalize_dom(self, data, dom):
        """Unscale and uncenter data from normalized space for a given domain."""
        mean = getattr(self, f'mean_vec_{dom}')
        scale = getattr(self, f'global_scale_factor_{dom}')
        return data * scale + mean

    # ------------------------------------------------------------------
    # compress / reconstruct  (domain-aware + legacy wrappers)
    # ------------------------------------------------------------------
    def _compress_normalized(self, normalized_data):
        """Legacy fd-only wrapper."""
        return self._compress_normalized_dom(normalized_data, 'fd')

    def _compress_normalized_dom(self, normalized_data, dom):
        """Compress normalized data into reduced basis coefficients for a given domain.

        Uses the unified ``_inner`` helper so that FD coefficients are
        real-valued (consistent with the GW inner product
        ⟨a|b⟩ = Re[∑ a* b]) and only valid (non-zero-ASD) bins
        contribute to the sum.
        """
        B = getattr(self, f'basis_{dom}')
        # _inner returns (N_basis, B_size); transpose to (B_size, N_basis)
        coeff = self._inner(B, normalized_data, dom).T
        return coeff

    def _reconstruct_normalized(self, coeff):
        """Legacy fd-only wrapper."""
        return self._reconstruct_normalized_dom(coeff, 'fd')

    def _reconstruct_normalized_dom(self, coeff, dom):
        """Reconstruct normalized data from reduced basis coefficients for a given domain."""
        B = getattr(self, f'basis_{dom}')
        # coeff is real (from the Re[] inner product); promote to complex
        # when the basis is complex so that @ doesn't raise a dtype error.
        if B.is_complex() and not coeff.is_complex():
            coeff = coeff.to(B.dtype)
        return coeff @ B  # shape (B_size, N_dim)

    # ------------------------------------------------------------------
    # public API with flatten → project → unflatten
    # ------------------------------------------------------------------
    def compress(self, data, dom='fd'):
        """Compress data into reduced basis coefficients.

        If the ROM was trained with a frequency cutoff, the low-frequency
        bins are sliced off automatically — callers may pass the full
        spectrum without manual truncation.

        :param data: Input data with shape (Batch_size, Channels, Dim).
        :param dom: Domain tag ('fd' or 'td').
        :return: Coefficients of shape (Batch_size, N_basis).
        """
        data_white = self._whiten_fd(data) if dom == 'fd' else data
        data_white_cutoff = self._apply_freq_cutoff(data_white, dom)
        normalized_data = self._normalize_dom(data_white_cutoff, dom)
        return self._compress_normalized_dom(normalized_data, dom)

    def reconstruct(self, coeff, dom='fd'):
        """Reconstruct raw data from reduced basis coefficients.

        Inverse of :meth:`compress`: returns data in the same space as the
        original input (i.e. raw, un-whitened).  For FD the whitened
        reconstruction is multiplied back by the ASD.

        If the ROM was trained with a frequency cutoff, the returned
        tensor contains only the retained (high-frequency) bins, i.e.
        shape ``(B, C, n_dim_fd)`` where ``n_dim_fd = n_freq - freq_cutoff_idx``.

        :param coeff: Coefficients with shape (Batch_size, N_basis).
        :param dom: Domain tag ('fd' or 'td').
        :return: Reconstructed raw data with shape (Batch_size, Channels, Dim).
        """
        normalized_data = self._reconstruct_normalized_dom(coeff, dom)
        denormalized_data = self._denormalize_dom(normalized_data, dom)
        n_channels = getattr(self, f'n_channels_{dom}')
        n_dim = getattr(self, f'n_dim_{dom}')
        whitened = denormalized_data.reshape(denormalized_data.shape[0], n_channels, n_dim)
        if dom == 'fd' and self.asd is not None:
            # de-whiten: multiply back by ASD to return to raw-data space
            asd = self.asd.to(whitened.device)   # (C, F_full) — may be wider than n_dim
            asd_trimmed = self._apply_freq_cutoff(asd, dom)  # (C, n_dim)
            whitened = whitened * asd_trimmed.unsqueeze(0)
        return whitened
    

class ROMWrapper(nn.Module): 
    """A wrapper nn.Module around the ReducedOrderModel to be used as a data
    summarizer in the InferenceNetwork.  Supports fd-only, td-only, or both
    domains depending on the loaded ROM.

    Parameters
    ----------
    filename : str
        Path to a saved ROM ``.pt`` file.
    device : str
        Torch device string (default ``"cuda"``).
    compress : str or list of str or None
        Which domain(s) to compress via the ROM basis.

        * ``'fd'``   – compress only the frequency-domain input.
        * ``'td'``   – compress only the time-domain input.
        * ``'both'`` – compress both domains (requires the ROM to have
          trained bases for both).
        * ``None`` (default) – automatically compress every domain for
          which the ROM has a trained basis (original behaviour).

        Any domain that is **not** compressed is returned as passthrough.
    """

    _VALID_COMPRESS = ('fd', 'td', 'both')

    def __init__(self, filename: str, device: str = "cuda", compress=None, max_basis_elems=None):
        super().__init__()
        self.rom = ReducedOrderModel(filename=filename, device=device)
        self.rom.basis = self.rom.basis[:max_basis_elems] if max_basis_elems is not None else self.rom.basis
        # Resolve which domains to compress
        if compress is None:
            # Auto: compress every domain that has a basis
            self._compress_domains = [
                dom for dom in ('fd', 'td')
                if getattr(self.rom, f'basis_{dom}', None) is not None
            ]
        elif compress == 'both':
            self._compress_domains = ['fd', 'td']
        elif isinstance(compress, str):
            if compress not in ('fd', 'td'):
                raise ValueError(f"compress must be one of {self._VALID_COMPRESS} or None, got '{compress}'")
            self._compress_domains = [compress]
        elif isinstance(compress, (list, tuple)):
            for c in compress:
                if c not in ('fd', 'td'):
                    raise ValueError(f"Invalid domain '{c}' in compress list")
            self._compress_domains = list(compress)
        else:
            raise TypeError(f"compress must be str, list, or None, got {type(compress)}")

        # Validate that the ROM actually has a basis for every requested domain
        for dom in self._compress_domains:
            basis_dom = getattr(self.rom, f'basis_{dom}', None)
            if basis_dom is None:
                raise ValueError(
                    f"compress='{compress}' requests domain '{dom}', but the ROM "
                    f"loaded from '{filename}' has no trained basis for it."
                )
            # and retain at most max_basis_elems if specified
            n_basis_elems = basis_dom.shape[0]
            
            if max_basis_elems is not None and n_basis_elems > max_basis_elems:
                setattr(self.rom, f'basis_{dom}', basis_dom[:max_basis_elems])

    def get_n_features(self):
        """Return the total number of scalar features produced by forward().

        With the noise-weighted real inner product, FD coefficients are
        real-valued, so each basis vector contributes 1 feature (not 2).
        """
        n = 0
        for dom in self._compress_domains:
            basis = getattr(self.rom, f'basis_{dom}')
            n += basis.shape[0]   # real coefficients for both fd and td
        return n

    def forward(self, d_f, d_t): 
        """Forward pass of the ROMWrapper.

        :param d_f: raw data in the frequency domain, complex tensor of shape (B, C, F)
        :param d_t: raw data in the time domain, real tensor of shape (B, C, T)
        :return: tuple of (compressed_features, passthrough) where
                 *compressed_features* is real-valued with shape ``(B, n_features)``
                 (or ``None`` if no domain is compressed), and *passthrough* is a
                 dict of ``{domain: tensor}`` for every domain that was **not**
                 compressed (empty dict when everything is compressed).
        """
        parts = []
        passthrough = {}

        # --- frequency domain ---
        if 'fd' in self._compress_domains:
            # compress() handles whitening internally.
            compressed_fd = self.rom.compress(d_f, dom='fd')
            # Coefficients are real-valued under the Re[] inner product
            parts.append(compressed_fd)
        else:
            passthrough['fd'] = d_f  # pass through uncompressed

        # --- time domain ---
        if 'td' in self._compress_domains:
            compressed_td = self.rom.compress(d_t, dom='td')
            parts.append(compressed_td)  # already real
        else:
            passthrough['td'] = d_t  # pass through uncompressed

        compressed = torch.cat(parts, dim=1) if parts else None

        return compressed, passthrough
