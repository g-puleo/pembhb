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
from torch.utils.data import DataLoader, Dataset, IterableDataset

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

    def write_slice(self, j, off, chunk):
        """Copy a sub-batch of ``n`` samples into buffer ``j`` at offset ``off``.

        Used to fill a buffer incrementally (sub-batched generation) without
        ever materialising a full ``M``-sample chunk or flipping any flag.
        """
        n = next(iter(chunk.values())).shape[0]
        for name, dst_list in self.fields.items():
            dst_list[j][off:off + n].copy_(chunk[name])

    def commit_fill(self, j):
        """Mark buffer ``j`` ready after an incremental fill (see ``write_slice``)."""
        with self._cond:
            self._filling[j] = False
            self._consumed[j] = False
            self._fill_clock += 1
            self._fill_seq[j] = self._fill_clock
            self._cond.notify_all()


class Producer:
    """Background thread that continuously refills a ``RingBuffer`` from a
    simulator.

    Each iteration claims a free-to-fill buffer, generates ``ring.M`` samples on
    the GPU (``sim.sample(M, keep_on_gpu=True)``) and writes them in. It blocks
    in ``acquire_writable`` when no buffer is free (back-pressure) and exits
    cleanly when ``ring.stop()`` is called. Any exception is stored on
    ``self.error`` and the ring is stopped so the consumer never hangs.
    """

    def __init__(self, ring, sim, gen_batch_size=250):
        self.ring = ring
        self.sim = sim
        self.gen_batch_size = int(gen_batch_size)  # sub-batch per bbhx call (VRAM)
        self.error = None
        self.n_seed = 0    # buffers filled by the blocking seed_fill_all()
        self.n_chunks = 0  # buffers refreshed by the running producer loop
        self._thread = threading.Thread(target=self._run, daemon=True)

    @property
    def samples_generated(self):
        """Total waveforms generated this round = (seed + running) * M."""
        return (self.n_seed + self.n_chunks) * self.ring.M

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

    def _fill(self, j):
        """Fill buffer ``j`` with ``M`` samples, generated in sub-batches of
        ``gen_batch_size`` so bbhx never builds the whole buffer at once
        (bounds the transient VRAM) and written straight into the buffer slice
        (no separate full-M chunk tensor)."""
        M, gb = self.ring.M, self.gen_batch_size
        for off in range(0, M, gb):
            n = min(gb, M - off)
            sample = self.sim.sample(n, keep_on_gpu=True)
            self.ring.write_slice(j, off, self._chunk(sample))
        self.ring.commit_fill(j)

    def seed_fill_all(self):
        """Blocking initial fill of every buffer; call once before ``start()``."""
        for j in range(self.ring.n):
            self._fill(j)
            self.n_seed += 1

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
                self._fill(j)
                self.n_chunks += 1
        except Exception as e:  # noqa: BLE001 - surface to the main thread
            self.error = e
            self.ring.stop()


class StreamingChunkIterable(IterableDataset):
    """Iterable view that streams whole chunks round-robin until ``length``
    samples have been yielded, decoupling the epoch (a sample count) from the
    buffer size.

    Each ``__iter__`` locks **one** buffer at a time via ``acquire`` (=
    ``ring.next_readable``), yields its ``M`` samples in a fresh within-chunk
    permutation, then ``release``s it (= ``ring.release_epoch``) before moving
    to the next buffer. The final chunk of an epoch is read only partially when
    ``length`` is not a multiple of ``M`` — so the epoch ends at exactly
    ``length`` samples, with no ``idx % M`` replay. Consecutive epochs continue
    the round-robin from the ring's internal pointer, so every chunk is read
    once before any is revisited.

    ``acquire``/``release`` are callbacks onto ``StreamingDataModule`` so it can
    track the held buffer (``_active_j``) for the post-``fit`` safety net; the
    ring's one-buffer-at-a-time locking invariant is unchanged.
    """

    def __init__(self, ring, acquire, release, length, seed=None):
        self.ring = ring
        self.acquire = acquire
        self.release = release
        self.length = int(length)
        self.has_td = "wave_td" in ring.fields
        self._seed = seed

    def __len__(self):
        return self.length

    def __iter__(self):
        g = torch.Generator()
        if self._seed is not None:
            g.manual_seed(int(self._seed))
        yielded = 0
        while yielded < self.length:
            j = self.acquire()          # locks one buffer (blocks if none free)
            if j is None:               # ring stopped (producer died / round end)
                return
            try:
                wave_fd = self.ring.fields["wave_fd"][j]
                params = self.ring.fields["params"][j]
                wave_td = self.ring.fields["wave_td"][j] if self.has_td else None
                M = wave_fd.shape[0]
                for i in torch.randperm(M, generator=g).tolist():
                    out = {"wave_fd": wave_fd[i], "params": params[i]}
                    if self.has_td:
                        out["wave_td"] = wave_td[i]
                    yield out
                    yielded += 1
                    if yielded >= self.length:
                        break
            finally:
                self.release(j)         # free the chunk as soon as its pass ends


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

    Training streams whole chunks round-robin (via ``StreamingChunkIterable``):
    one buffer is locked at a time, read once, then released, until the epoch's
    ``samples_per_epoch`` sample budget is met. The epoch is thus a pure sample
    count, decoupled from the buffer size. Validation/test read a *frozen* pool
    generated once at round start. Noise is produced on-device via the existing
    ``gpu_noise`` deferred path (``mbhb_collate_fn(..., gpu_noise=True)`` +
    ``GPUNoiseMixin.on_after_batch_transfer``).

    The public method surface mirrors ``MBHBDataModule`` so ``_train_joint`` is
    unchanged: ``setup``, ``get_noise_scale``, ``get_asd``, ``get_freqs``,
    ``get_times``, ``get_max_td``, ``get_params_mean_std``,
    ``get_sincos_mean_std``, ``median_snr``, ``train/val/test_dataloader``.
    """

    def __init__(self, ring, sim, val_pool, batch_size, noise_factor=1.0,
                 n_train_noise_realisations=1, device="cuda", samples_per_epoch=None):
        super().__init__()
        self.ring = ring
        self.sim = sim
        self.val_pool = val_pool          # dict: {"wave_fd": (Mv,C,F), "params": (Mv,P)} on device
        self.batch_size = batch_size
        self.noise_factor = noise_factor
        self.n_train_noise_realisations = n_train_noise_realisations
        self.device = device
        # Epoch = samples_per_epoch training examples (a sample count, not a
        # buffer count): the trainer streams chunks round-robin until this many
        # are read. Defaults to one buffer.
        self.samples_per_epoch = int(samples_per_epoch) if samples_per_epoch else ring.M

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

        self._active_j = None             # buffer currently locked by the iterator
        self._epoch = 0                   # per-epoch seed for within-chunk shuffle
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

    def _acquire(self):
        """Lock the next buffer round-robin and remember it (safety-net for
        ``release_active``). Returns ``None`` if the ring was stopped."""
        j = self.ring.next_readable()
        self._active_j = j
        return j

    def _release(self, j):
        if j is not None:
            self.ring.release_epoch(j)
        if self._active_j == j:
            self._active_j = None

    def train_dataloader(self, shuffle=True, num_workers=None, pin_memory=False,
                         single_chunk=False):
        # Stream chunks round-robin until samples_per_epoch samples are read; the
        # iterable locks one buffer at a time (invariant unchanged). shuffle is
        # honoured *within* each chunk. single_chunk caps the epoch at one buffer
        # (M samples) for cheap normalisation-stat passes.
        length = self.ring.M if single_chunk else self.samples_per_epoch
        seed = self._epoch if shuffle else None
        self._epoch += 1
        ds = StreamingChunkIterable(self.ring, self._acquire, self._release,
                                    length=length, seed=seed)
        # num_workers MUST be 0: workers are separate processes and cannot share
        # the parent process's GPU tensors. shuffle=False at the DataLoader level
        # (an IterableDataset can't use it); shuffling is done inside the iterable.
        return DataLoader(ds, batch_size=self.batch_size, shuffle=False, num_workers=0,
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
