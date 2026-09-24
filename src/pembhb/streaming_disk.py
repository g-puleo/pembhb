"""Disk-backed variant of the streaming data generator (:mod:`pembhb.streaming`).

The GPU ring keeps ``n_buffers`` chunks of ``M`` waveforms resident in VRAM
(``5 x 7000`` complex waveforms is >30 GB). Here each "buffer" is instead an
HDF5 file on disk, overwritten in place by the producer. VRAM drops to a single
``gen_batch_size`` sub-batch, at the cost of disk read/write traffic — which is
hidden by reading with ``num_workers > 0`` (each worker opens its own h5 handle,
exactly like :class:`pembhb.data.MBHBDataset`). Multi-worker reads are the whole
point: the GPU path is pinned at ``num_workers=0`` because GPU tensors cannot
cross a process boundary; disk buffers have no such restriction.

The synchronisation core is *unchanged*: :class:`DiskRingBuffer` subclasses
:class:`RingBuffer` and inherits ``acquire_writable`` / ``next_readable`` /
``release_epoch`` / ``seed_fill`` / ``write`` / ``commit_fill`` / ``stop`` /
``mark_seen`` verbatim. Only the storage of a buffer (a GPU tensor -> an h5
file) is overridden. Consumption is ``k = samples_per_epoch / buffer_size``
buffers per epoch, picked round-robin in one atomic step
(``RingBuffer.acquire_readable_many``) that skips buffers being written or
already locked — the disk analogue of ``LiveBufferDataset`` (the
``dataset_style="mapstyle"`` path), generalised so the file size and the epoch
length are independent knobs.
"""

import os

import h5py
import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from pembhb import get_numpy_complex_dtype, get_numpy_dtype, get_torch_dtype
from pembhb.streaming import Producer, RingBuffer, _FrozenPoolDataset
from pembhb.utils import mbhb_collate_fn


class DiskRingBuffer(RingBuffer):
    """A :class:`RingBuffer` whose ``n_buffers`` buffers are HDF5 files on disk
    rather than GPU tensors.

    The scheduling state machine (``_in_use`` / ``_consumed`` / ``_filling`` /
    ``reuse_threshold`` / round-robin / LRU) is inherited unchanged. Buffer
    ``j`` lives at ``buffer_{j}.h5``; the producer overwrites it in place (fixed
    ``M``-row datasets, no resize) so total disk stays bounded at
    ``n_buffers * M`` waveforms. The ring guarantees a file being written is
    never read (skip-filling) and never overwritten before ``reuse_threshold``
    epochs, so there is never concurrent read+write of the *same* file — no
    HDF5 SWMR needed.
    """

    def __init__(self, n_buffers, buffer_size, buffer_dir, n_channels, n_freq,
                 n_params, has_td=False, n_time=None, reuse_threshold=1):
        self.buffer_dir = buffer_dir
        os.makedirs(buffer_dir, exist_ok=True)
        self.paths = [os.path.join(buffer_dir, f"buffer_{j}.h5")
                      for j in range(n_buffers)]
        self.has_td = has_td
        # h5 dataset name -> (per-sample shape, numpy dtype)
        self._specs = {
            "wave_fd": ((n_channels, n_freq), get_numpy_complex_dtype()),
            "params": ((n_params,), get_numpy_dtype()),
        }
        if has_td:
            assert n_time is not None, "has_td requires n_time"
            self._specs["wave_td"] = ((n_channels, n_time), get_numpy_dtype())
        # No in-memory field tensors: consumers read from disk. ``fields`` is
        # kept as an empty dict so any inherited ``"x" in ring.fields`` check
        # (GPU-path helpers) is simply False on the disk ring.
        self.fields = {}
        self._init_state(n_buffers, buffer_size, reuse_threshold)
        self._alloc_files()

    def _alloc_files(self):
        """Pre-create every buffer file with fixed-size datasets so a path exists
        before the first fill. Fills don't touch these in place — they write a
        fresh ``.tmp`` and atomically ``os.replace`` it in (see ``create_file`` /
        ``commit_rename``), so the producer never opens a buffer that a reader
        (worker *or* this process) may hold open."""
        for j, p in enumerate(self.paths):
            t = self.tmp_path(j)
            if os.path.exists(t):        # partial tmp from an aborted prior-round fill
                os.remove(t)
            with h5py.File(p, "w") as f:
                for name, (shape, dt) in self._specs.items():
                    f.create_dataset(name, shape=(self.M, *shape), dtype=dt)

    def path(self, j):
        return self.paths[j]

    # --- storage: write-to-tmp + atomic rename ---------------------------
    # HDF5 refuses to open a file ``r+`` while it is open read-only *in the same
    # process* (errors even with HDF5_USE_FILE_LOCKING=FALSE). The producer runs
    # in the main process alongside transient main-process reads, so an in-place
    # ``r+`` overwrite races with them. Writing a brand-new ``.tmp`` file (which
    # nothing has open) and ``os.replace``-ing it in sidesteps that entirely and
    # is crash-safe: readers see either the whole old file or the whole new one.

    def create_file(self, path):
        """Create a fresh writable buffer file with the schema's datasets."""
        f = h5py.File(path, "w")
        for name, (shape, dt) in self._specs.items():
            f.create_dataset(name, shape=(self.M, *shape), dtype=dt)
        return f

    def tmp_path(self, j):
        return self.paths[j] + ".tmp"

    def commit_rename(self, j):
        """Atomically swap buffer ``j``'s freshly-written ``.tmp`` into place."""
        os.replace(self.tmp_path(j), self.paths[j])

    def read_field(self, j, name):
        """Read a full field off buffer ``j`` (used for normalisation stats)."""
        with h5py.File(self.paths[j], "r") as f:
            return np.asarray(f[name][()])


class DiskProducer(Producer):
    """Background producer that writes generated waveforms to the disk ring.

    Identical control flow to :class:`Producer` (``acquire_writable`` -> fill ->
    ``commit_fill``, ``n_seed`` / ``n_chunks`` / ``samples_generated`` counters);
    only ``_fill`` differs — it writes a fresh ``.tmp`` file in
    ``gen_batch_size`` sub-batches (host numpy, so ``keep_on_gpu=False``, bounding
    transient VRAM to one sub-batch) then atomically renames it into place.
    """

    def _chunk(self, sample):
        # keep_on_gpu=False -> host numpy already. Match the HDF5 convention:
        # parameters stored sample-first (N, n_params) (simulator writes .T).
        params = np.asarray(sample["parameters"]).T
        chunk = {"wave_fd": np.asarray(sample["wave_fd"]), "params": params}
        if self.ring.has_td:
            chunk["wave_td"] = np.asarray(sample["wave_td"])
        return chunk

    def _fill(self, j):
        # Write a brand-new tmp file (no reader can have it open) then atomically
        # swap it in. Avoids the same-process r+/read-only open conflict.
        # Returns True if committed, False if aborted because the ring was
        # stopped mid-fill (see the abort check below).
        M, gb = self.ring.M, self.gen_batch_size
        tmp = self.ring.tmp_path(j)
        f = self.ring.create_file(tmp)
        try:
            dsets = {name: f[name] for name in self.ring._specs}
            for off in range(0, M, gb):
                # Bail out promptly on stop() so the round-end join() succeeds and
                # this (now-defunct) producer can never write into the buffer_dir
                # that the NEXT round reuses. The partial .tmp is harmless — it is
                # never renamed in, and the next round's _alloc_files removes it.
                if self.ring._stop:
                    return False
                n = min(gb, M - off)
                chunk = self._chunk(self.sim.sample(n, keep_on_gpu=False))
                for name, ds in dsets.items():
                    ds[off:off + n] = chunk[name]
        finally:
            f.close()
        self.ring.commit_rename(j)   # atomic swap, then mark ready
        self.ring.commit_fill(j)
        return True

    def _run(self):
        # Same loop as Producer._run, but surface the traceback (the base class
        # swallows it into self.error, which otherwise shows up only as a
        # confusing None from the consumer's next_readable()). Only completed
        # fills bump n_chunks.
        import traceback
        try:
            while True:
                j = self.ring.acquire_writable()
                if j is None:
                    return
                if self._fill(j):
                    self.n_chunks += 1
        except Exception as e:  # noqa: BLE001 - surface to the main thread
            self.error = e
            print("[streaming][disk] PRODUCER ERROR:\n" + traceback.format_exc(),
                  flush=True)
            self.ring.stop()


class DiskBufferDataset(Dataset):
    """Map-style view over ``k`` buffer files — the disk analogue of
    :class:`pembhb.streaming.LiveBufferDataset`, generalised from one file to the
    ``k = samples_per_epoch / buffer_size`` files that make up one epoch.

    Row ``i`` maps to ``(file i // M, row i % M)``, so an epoch of
    ``k * M`` rows reads every row of every locked file exactly once — the file
    size (``buffer_size``) and the epoch length (``samples_per_epoch``) are
    independent again. Being map-style, the DataLoader shards ``__getitem__``
    across ``num_workers`` worker processes, each of which lazily opens its own
    h5 handle per file (fork-safe: handles are only created inside the worker).
    Ring locking stays in the main process — see
    :meth:`DiskStreamingDataModule.train_dataloader`.
    """

    def __init__(self, paths, has_td=False, length=None):
        self.paths = [paths] if isinstance(paths, str) else list(paths)
        self.has_td = has_td
        with h5py.File(self.paths[0], "r") as f:
            self.M = f["wave_fd"].shape[0]      # rows per file (all equal)
        self.total = self.M * len(self.paths)   # distinct rows available
        self.length = int(length) if length else self.total
        self._files = {}  # per-process handles {file index: h5py.File}, lazy

    def _file(self, b):
        f = self._files.get(b)
        if f is None:
            f = self._files[b] = h5py.File(self.paths[b], "r")
        return f

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # idx % total is a defensive no-op in the normal regime (length == total);
        # it only wraps if a caller deliberately asks for length > total.
        b, i = divmod(idx % self.total, self.M)
        f = self._file(b)
        out = {
            "wave_fd": torch.as_tensor(f["wave_fd"][i]),
            "params": torch.as_tensor(f["params"][i]),
        }
        if self.has_td:
            out["wave_td"] = torch.as_tensor(f["wave_td"][i])
        return out

    def close(self):
        """Close this process's handles (if any). Worker handles live in their
        own processes and are freed when those workers exit."""
        for f in self._files.values():
            f.close()
        self._files.clear()


class DiskStreamingDataModule(L.LightningDataModule):
    """Disk-backed drop-in for :class:`pembhb.streaming.StreamingDataModule`.

    Same public surface as ``MBHBDataModule`` / ``StreamingDataModule`` so
    ``_train_joint`` is unchanged. Training locks the ``k = samples_per_epoch /
    buffer_size`` buffer files an epoch needs (one atomic round-robin pick via
    ``ring.acquire_readable_many``, skipping any buffer being written), reads
    each of them exactly once with ``num_workers`` workers, and releases all
    ``k`` at the next epoch's ``train_dataloader`` call (relies on
    ``reload_dataloaders_every_n_epochs=1``, as the ``mapstyle`` GPU path does).
    So an epoch yields exactly ``samples_per_epoch`` **distinct** examples with
    no ``idx % M`` reuse, while ``buffer_size`` independently sets the file size
    (``samples_per_epoch`` must be a multiple of it, and ``n_buffers >= k + 1``
    so the producer always keeps a buffer it may refresh).
    Validation/test read a frozen CPU pool generated once at round start.

    Noise follows the proven HDF5 recipe: a CPU ``noise_scale`` is shipped with
    the batch and coloured on-device by ``materialize_gpu_noise`` /
    ``on_after_batch_transfer`` (``gpu_noise=True``) — the disk waveforms arrive
    on CPU from the workers, exactly like ``MBHBDataset``.
    """

    def __init__(self, ring, sim, val_pool, batch_size, noise_factor=1.0,
                 n_train_noise_realisations=1, device="cuda", samples_per_epoch=None,
                 num_workers=8, prefetch_factor=4):
        super().__init__()
        self.ring = ring
        self.sim = sim
        self.val_pool = val_pool
        self.batch_size = batch_size
        self.noise_factor = noise_factor
        self.n_train_noise_realisations = n_train_noise_realisations
        self.device = device
        self.num_workers = int(num_workers)
        self.prefetch_factor = int(prefetch_factor)
        self.samples_per_epoch = int(samples_per_epoch) if samples_per_epoch else ring.M
        # Buffers consumed per epoch. buffer_size (ring.M) sizes the *file*;
        # samples_per_epoch sizes the *epoch*; k ties them together.
        if self.samples_per_epoch % ring.M:
            raise ValueError(
                f"samples_per_epoch ({self.samples_per_epoch}) must be a multiple of "
                f"buffer_size ({ring.M}): an epoch is read as whole buffer files"
            )
        self.k = self.samples_per_epoch // ring.M
        if self.k > ring.n - 1:
            raise ValueError(
                f"epoch needs k={self.k} buffers but n_buffers={ring.n}: keep "
                f"n_buffers >= k+1 so the producer always has one free to refresh"
            )

        # noise_scale = filtered_asd / sqrt(4*df), kept on CPU: it rides along in
        # the batch and is moved to the GPU with it, where materialize_gpu_noise
        # draws the coloured noise (matches the MBHBDataModule / gpu_noise path).
        noise_scale_np = sim.filtered_asd / np.sqrt(4.0 * sim.df)
        self.noise_scale = torch.as_tensor(noise_scale_np, dtype=get_torch_dtype())
        self.td_params = None  # FD-only streaming for now

        # Normalisation stats + SNR check from the seed-filled buffers, read once
        # before the producer overwrites anything (safe: nothing consumed yet, so
        # the producer is blocked in acquire_writable).
        self._params_pool = torch.as_tensor(
            np.concatenate([ring.read_field(j, "params") for j in range(ring.n)], axis=0),
            dtype=get_torch_dtype(),
        )
        self.median_snr = self._compute_median_snr()

        self._active_js = []    # buffers currently locked for the epoch
        self._active_ds = None  # its dataset (to close the main-process handles)
        # Reuse accounting for StreamReuseLogger. The GPU ring counts distinct
        # sims via ring.mark_seen() inside its (worker-free) iterator; with disk
        # workers that is impossible, so we count in the main process instead:
        # acquiring a buffer whose _fill_seq is new since we last read it means M
        # fresh distinct sims. n_train_noise_realisations tiling is divided out by
        # the callback, so _examples_consumed counts pre-noise training examples.
        self._examples_consumed = 0
        self._counted_fill_seq = {}   # buffer j -> last _fill_seq counted distinct
        self._producer = None         # set by _setup_streaming_disk; for error surfacing
        self.test = _FrozenPoolDataset(val_pool)

    # --- normalisation / metadata (MBHBDataModule contract) --------------

    def _compute_median_snr(self):
        wave = self.val_pool["wave_fd"].detach().cpu().numpy()
        snr = self.sim.get_SNR_FD(wave)
        return float(np.median(np.asarray(snr)))

    def setup(self, stage=None):
        pass  # buffers are allocated + seed-filled by the caller

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

    def _collate(self, shuffle, n_noise):
        nf, td = self.noise_factor, self.td_params
        return lambda b: mbhb_collate_fn(b, self.noise_scale, nf, noise_shuffling=shuffle,
                                         td_params=td, n_noise_realisations=n_noise,
                                         gpu_noise=True)

    def _release_active(self):
        """Close this process's handles and release every buffer locked for the
        epoch that just ended, so the producer may refresh them. The DataLoader
        workers have already exited (no persistent_workers), so the files are no
        longer open for reading."""
        if self._active_ds is not None:
            self._active_ds.close()
            self._active_ds = None
        for j in self._active_js:
            self.ring.release_epoch(j)
        self._active_js = []

    def train_dataloader(self, shuffle=True, num_workers=None, pin_memory=False,
                         single_chunk=False):
        # single_chunk caps the epoch at one buffer (cheap normalisation passes).
        k = 1 if single_chunk else self.k
        length = self.ring.M * k
        nw = self.num_workers if num_workers is None else int(num_workers)

        # Release last epoch's buffers, then lock this epoch's k in ONE atomic
        # round-robin pick (distinct, not being written). All ring bookkeeping
        # happens here, in the main process: DataLoader workers are separate
        # processes whose ring copy the producer would never see.
        self._release_active()
        js = self.ring.acquire_readable_many(k)
        if js is None:
            # The ring was stopped before this epoch could start — almost always
            # a dead producer. Surface its real error instead of an opaque
            # None-index crash downstream.
            perr = getattr(self._producer, "error", None)
            raise RuntimeError(
                "disk streaming ring stopped before epoch start"
                + (f"; producer failed: {perr!r}" if perr is not None else "")
            )
        self._active_js = list(js)
        # Distinct-sims accounting (main-process; count_seen is armed by
        # StreamReuseLogger only during fit, so pre-fit norm passes don't count).
        # A buffer with a _fill_seq we haven't counted contributes M fresh sims;
        # re-reading the same fill contributes none -> reuse_factor reflects
        # reuse_threshold.
        if self.ring.count_seen:
            for j in js:
                fseq = self.ring._fill_seq[j]
                if self._counted_fill_seq.get(j) != fseq:
                    self.ring.distinct_sims_seen += self.ring.M
                    self._counted_fill_seq[j] = fseq
        ds = DiskBufferDataset([self.ring.path(j) for j in js],
                               has_td=self.ring.has_td, length=length)
        self._active_ds = ds

        extra = {}
        if nw > 0:
            # persistent_workers deliberately OFF: workers must die at epoch end
            # so their read handles on these buffers are closed before the
            # producer is allowed to overwrite the files.
            extra = {"prefetch_factor": self.prefetch_factor}
        # pin_memory is honoured (not forced): the single pinning thread is a
        # throughput ceiling at large n_freq, where it measured strictly slower.
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle, num_workers=nw,
                          pin_memory=pin_memory,
                          collate_fn=self._collate(shuffle, self.n_train_noise_realisations),
                          **extra)

    def _val_loader(self):
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False,
                          num_workers=0, collate_fn=self._collate(False, 1))

    def val_dataloader(self):
        return self._val_loader()

    def test_dataloader(self):
        return self._val_loader()

    def release_active(self):
        """Release the final epoch's buffers (call once after trainer.fit)."""
        self._release_active()

    @property
    def _active_j(self):
        """First buffer locked this epoch (``None`` if idle) — convenience for
        logs/tests written when an epoch was always exactly one buffer."""
        return self._active_js[0] if self._active_js else None
