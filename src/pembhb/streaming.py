"""Concurrent data generation for streaming TMNRE rounds.

A background producer thread fills a pool of ``n_buffers`` GPU-resident buffers
while the trainer (consumer) iterates over them, one buffer per epoch. The
``RingBuffer`` below is the synchronisation core: it guarantees a buffer is
never overwritten before it has been trained on at least once, and applies
back-pressure (the producer blocks) when the consumer falls behind.
"""

import threading

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from pembhb import get_torch_complex_dtype, get_torch_dtype
from pembhb.utils import mbhb_collate_fn


class RingBuffer:
    """Pool of ``n_buffers`` fixed-size buffers shared by one producer thread
    and one consumer (the trainer).

    Per-buffer state, all guarded by a single ``threading.Condition``:

    - ``in_use``   : the consumer is iterating this buffer this epoch.
    - ``consumed`` : the consumer has finished >=1 epoch since the last fill.
    - ``filling``  : the producer is currently writing this buffer.

    A buffer is *free to fill* iff ``consumed and not in_use and not filling``.
    """

    def __init__(self, n_buffers, buffer_size, sample_shapes, dtypes, device="cpu",
                 host_fields=()):
        """
        :param sample_shapes: per-field per-sample shape, e.g.
            ``{"wave_fd": (n_channels, n_freq), "params": (n_params,)}``.
        :param dtypes: per-field torch dtype, same keys as ``sample_shapes``.
        :param host_fields: field names kept on CPU regardless of ``device``
            (e.g. ``params`` — tiny, and downstream callbacks read them with
            ``np.asarray`` exactly as in the HDF5 path).
        """
        self.n = n_buffers
        self.M = buffer_size
        self.device = device
        self.host_fields = set(host_fields)
        self.fields = {
            name: [
                torch.empty((buffer_size, *shape), dtype=dtypes[name],
                            device="cpu" if name in self.host_fields else device)
                for _ in range(n_buffers)
            ]
            for name, shape in sample_shapes.items()
        }

        self._in_use = [False] * n_buffers
        self._consumed = [False] * n_buffers  # nothing trained on yet
        self._filling = [False] * n_buffers
        self._fill_seq = [0] * n_buffers  # when each buffer was last filled (LRU)
        self._fill_clock = 0  # monotonic counter, bumped on every fill
        self._rr = 0  # consumer round-robin pointer
        self._stop = False
        self._cond = threading.Condition()

    # --- producer side --------------------------------------------------

    def seed_fill(self, j, chunk):
        """Write the initial contents of buffer ``j`` (bypasses the
        free-to-fill precondition; used once before the producer starts)."""
        self._copy_in(j, chunk)
        with self._cond:
            self._filling[j] = False
            self._consumed[j] = False
            self._fill_clock += 1
            self._fill_seq[j] = self._fill_clock
            self._cond.notify_all()

    def acquire_writable(self):
        """Block until a buffer is free to fill, mark it ``filling`` and return
        its index. Returns ``None`` if the buffer was stopped."""
        with self._cond:
            while not self._stop:
                eligible = [
                    j for j in range(self.n)
                    if self._consumed[j] and not self._in_use[j] and not self._filling[j]
                ]
                if eligible:
                    j = min(eligible, key=lambda k: self._fill_seq[k])  # least-recently filled
                    self._filling[j] = True
                    return j
                self._cond.wait()
        return None

    def write(self, j, chunk):
        """Fill buffer ``j`` (previously acquired) and mark it ready."""
        self._copy_in(j, chunk)
        with self._cond:
            self._filling[j] = False
            self._consumed[j] = False
            self._fill_clock += 1
            self._fill_seq[j] = self._fill_clock
            self._cond.notify_all()

    # --- consumer side --------------------------------------------------

    def next_readable(self):
        """Pick the next non-``filling`` buffer round-robin, mark it ``in_use``
        and return its index. Blocks only in the (rare) case every other buffer
        is currently being filled."""
        with self._cond:
            while not self._stop:
                for _ in range(self.n):
                    j = self._rr
                    self._rr = (self._rr + 1) % self.n
                    if not self._filling[j]:
                        self._in_use[j] = True
                        return j
                self._cond.wait()
        return None

    def release_epoch(self, j):
        """Mark buffer ``j`` consumed (>=1 epoch done) and no longer in use."""
        with self._cond:
            self._in_use[j] = False
            self._consumed[j] = True
            self._cond.notify_all()

    # --- lifecycle ------------------------------------------------------

    def stop(self):
        """Wake any blocked producer/consumer so they can exit cleanly."""
        with self._cond:
            self._stop = True
            self._cond.notify_all()

    # --- internals ------------------------------------------------------

    def _copy_in(self, j, chunk):
        """Copy a dict of ``(M, *shape)`` field tensors into buffer ``j``."""
        for name, dst_list in self.fields.items():
            dst_list[j].copy_(chunk[name])


class Producer:
    """Background thread that continuously refills a ``RingBuffer`` from a
    simulator.

    Each iteration claims a free-to-fill buffer, generates ``ring.M`` samples on
    the GPU (``sim.sample(M, keep_on_gpu=True)``) and writes them in. It blocks
    in ``acquire_writable`` when no buffer is free (back-pressure) and exits
    cleanly when ``ring.stop()`` is called. Any exception is stored on
    ``self.error`` and the ring is stopped so the consumer never hangs.
    """

    def __init__(self, ring, sim):
        self.ring = ring
        self.sim = sim
        self.error = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _chunk(self, sample):
        """Map one ``sim.sample`` output onto the ring's field tensors."""
        # sim.sample returns parameters as (n_params, N); the dataset/collate
        # convention is sample-first (N, n_params), matching the HDF5 path
        # (simulator stores ``parameters.T``). Transpose to match.
        params = torch.as_tensor(sample["parameters"]).t().contiguous()
        chunk = {
            "wave_fd": sample["wave_fd"],   # torch CUDA tensor, (N, C, F)
            "params": params,               # (N, n_params), host -> copied on write
        }
        if "wave_td" in self.ring.fields:
            chunk["wave_td"] = sample["wave_td"]
        return chunk

    def seed_fill_all(self):
        """Blocking initial fill of every buffer; call once before ``start()``."""
        for j in range(self.ring.n):
            sample = self.sim.sample(self.ring.M, keep_on_gpu=True)
            self.ring.seed_fill(j, self._chunk(sample))

    def start(self):
        self._thread.start()

    def join(self, timeout=None):
        self._thread.join(timeout=timeout)

    def _run(self):
        try:
            while True:
                j = self.ring.acquire_writable()
                if j is None:
                    return
                sample = self.sim.sample(self.ring.M, keep_on_gpu=True)
                self.ring.write(j, self._chunk(sample))
        except Exception as e:  # noqa: BLE001 - surface to the main thread
            self.error = e
            self.ring.stop()


class LiveBufferDataset(Dataset):
    """Map-style view over a single ring buffer index ``j``.

    ``__getitem__`` returns the same per-sample dict the HDF5 ``MBHBDataset``
    yields (``wave_fd`` + ``params``), but sliced straight from the GPU-resident
    buffer (zero host round-trip). Built fresh each epoch by
    ``StreamingDataModule.train_dataloader`` so consecutive epochs read different
    buffers.
    """

    def __init__(self, ring, j):
        self.wave_fd = ring.fields["wave_fd"][j]   # (M, C, F) on device
        self.params = ring.fields["params"][j]     # (M, P) on device
        self.has_td = "wave_td" in ring.fields
        if self.has_td:
            self.wave_td = ring.fields["wave_td"][j]

    def __len__(self):
        return self.wave_fd.shape[0]

    def __getitem__(self, idx):
        out = {"wave_fd": self.wave_fd[idx], "params": self.params[idx]}
        if self.has_td:
            out["wave_td"] = self.wave_td[idx]
        return out


class _FrozenPoolDataset(Dataset):
    """Map-style dataset over a fixed (val/test) pool of GPU tensors."""

    def __init__(self, pool):
        self.w, self.p = pool["wave_fd"], pool["params"]

    def __len__(self):
        return self.w.shape[0]

    def __getitem__(self, i):
        return {"wave_fd": self.w[i], "params": self.p[i]}


class StreamingDataModule(L.LightningDataModule):
    """Drop-in replacement for ``MBHBDataModule`` backed by a live ring buffer.

    Training reads one buffer per epoch (round-robin via ``ring.next_readable``);
    the previous epoch's buffer is released back to the producer at the start of
    the next ``train_dataloader`` call, so the trainer must run with
    ``reload_dataloaders_every_n_epochs=1``. Validation/test read a *frozen* pool
    generated once at round start. Noise is produced on-device via the existing
    ``gpu_noise`` deferred path (``mbhb_collate_fn(..., gpu_noise=True)`` +
    ``GPUNoiseMixin.on_after_batch_transfer``).

    The public method surface mirrors ``MBHBDataModule`` so ``_train_joint`` is
    unchanged: ``setup``, ``get_noise_scale``, ``get_asd``, ``get_freqs``,
    ``get_times``, ``get_max_td``, ``get_params_mean_std``,
    ``get_sincos_mean_std``, ``median_snr``, ``train/val/test_dataloader``.
    """

    def __init__(self, ring, sim, val_pool, batch_size, noise_factor=1.0,
                 n_train_noise_realisations=1, device="cuda"):
        super().__init__()
        self.ring = ring
        self.sim = sim
        self.val_pool = val_pool          # dict: {"wave_fd": (Mv,C,F), "params": (Mv,P)} on device
        self.batch_size = batch_size
        self.noise_factor = noise_factor
        self.n_train_noise_realisations = n_train_noise_realisations
        self.device = device

        # noise_scale = filtered_asd / sqrt(4*df); on-device so materialize_gpu_noise
        # draws coloured noise on the same device as the (GPU-resident) waveforms.
        noise_scale_np = sim.filtered_asd / np.sqrt(4.0 * sim.df)
        # Train waveforms live on GPU -> GPU noise_scale. The frozen val/test
        # pool lives on CPU (matching the HDF5 raw-batch semantics that manual
        # callback loops read via np.asarray) -> CPU noise_scale.
        self.noise_scale = torch.as_tensor(noise_scale_np, dtype=get_torch_dtype(), device=device)
        self._val_noise_scale = self.noise_scale.cpu()
        self.td_params = None             # FD-only streaming for now

        # Normalisation stats + SNR safety check from the seed-filled buffers
        # (computed once, before the producer starts overwriting anything).
        # Kept on CPU to match MBHBDataModule.get_params_mean_std (downstream
        # code does np.array(mean) on the result).
        self._params_pool = torch.cat(
            [ring.fields["params"][j] for j in range(ring.n)], dim=0).cpu()
        self.median_snr = self._compute_median_snr()

        self._active_j = None             # buffer currently held for the epoch
        self.test = _FrozenPoolDataset(val_pool)  # exposed for PP-KS eval

    # --- normalisation / metadata (MBHBDataModule contract) --------------

    def _compute_median_snr(self):
        wave = self.val_pool["wave_fd"].detach().cpu().numpy()
        snr = self.sim.get_SNR_FD(wave)
        return float(np.median(np.asarray(snr)))

    def setup(self, stage=None):
        pass  # buffers are already allocated and seed-filled by the caller

    def get_noise_scale(self):
        return self.noise_scale

    def get_asd(self):
        return torch.as_tensor(self.sim.asd, dtype=get_torch_dtype(), device=self.device)

    def get_freqs(self):
        return self.sim.freqs

    def get_times(self):
        return None

    def get_max_td(self):
        return None

    def get_params_mean_std(self):
        p = self._params_pool
        return p.mean(dim=0), p.std(dim=0)

    def get_sincos_mean_std(self, periodic_bc_params):
        p = self._params_pool
        sincos_mean, sincos_std = [], []
        for idx in periodic_bc_params:
            col = p[:, idx]
            s, c = torch.sin(col), torch.cos(col)
            sincos_mean.extend([s.mean().item(), c.mean().item()])
            sincos_std.extend([s.std().item(), c.std().item()])
        return sincos_mean, sincos_std

    # --- dataloaders ------------------------------------------------------

    def _collate(self, shuffle, n_noise, noise_scale):
        nf, td = self.noise_factor, self.td_params
        return lambda b: mbhb_collate_fn(b, noise_scale, nf, noise_shuffling=shuffle, td_params=td,
                                         n_noise_realisations=n_noise, gpu_noise=True)

    def train_dataloader(self, shuffle=True, num_workers=None, pin_memory=False):
        # Release the buffer trained on last epoch, then claim the next one.
        # Relies on Trainer(reload_dataloaders_every_n_epochs=1).
        if self._active_j is not None:
            self.ring.release_epoch(self._active_j)
        self._active_j = self.ring.next_readable()
        ds = LiveBufferDataset(self.ring, self._active_j)
        # num_workers MUST be 0: workers are separate processes and cannot share
        # the parent process's GPU tensors.
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle, num_workers=0,
                          collate_fn=self._collate(shuffle, self.n_train_noise_realisations,
                                                   self.noise_scale))

    def _val_loader(self):
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False,
                          num_workers=0, collate_fn=self._collate(False, 1, self._val_noise_scale))

    def val_dataloader(self):
        return self._val_loader()

    def test_dataloader(self):
        return self._val_loader()

    def release_active(self):
        """Release the final epoch's buffer (call once after trainer.fit)."""
        if self._active_j is not None:
            self.ring.release_epoch(self._active_j)
            self._active_j = None
