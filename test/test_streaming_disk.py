"""Unit + integration tests for the disk-backed streaming ring
(:mod:`pembhb.streaming_disk`). No bbhx/GPU required — a tiny ``FakeSim`` stands
in for :class:`pembhb.simulator.MBHBSimulatorFD`.

Covers: file allocation, params transpose, normalisation stats, strict
round-robin one-buffer-per-epoch consumption with ``num_workers>0``, producer
in-place overwrite after ``reuse_threshold``, bounded disk usage, and that the
GPU ``RingBuffer`` refactor (shared ``_init_state``) is behaviour-preserving.
"""
import hashlib
import os
import time

import h5py
import numpy as np
import pytest
import torch

import pembhb
pembhb.set_precision("float64")
from pembhb.streaming import RingBuffer  # noqa: E402
from pembhb.streaming_disk import (  # noqa: E402
    DiskProducer, DiskRingBuffer, DiskStreamingDataModule)

C, F, P = 3, 8, 5


class FakeSim:
    """Minimal MBHBSimulatorFD stand-in producing distinguishable fills."""

    def __init__(self):
        self.filtered_asd = np.ones((C, F))
        self.asd = np.ones((C, F))
        self.df = 1.0
        self.freqs = np.arange(F)
        self._k = 0     # bumped per call -> distinguishable wave fills (overwrite test)
        self._row = 0   # running global row id -> every row unique (distinct test)

    def sample(self, n, keep_on_gpu=False):
        self._k += 1
        # parameters shape (n_params, n); params[p, i] = globally-unique row id.
        ids = np.arange(self._row, self._row + n, dtype=float)
        self._row += n
        params = np.tile(ids, (P, 1))
        wave = np.full((n, C, F), self._k, dtype=np.complex128)
        return {"parameters": params, "wave_fd": wave}

    def get_SNR_FD(self, wave):
        return np.full(wave.shape[0], 20.0)


def _buf_hash(path):
    with h5py.File(path, "r") as f:
        return hashlib.md5(np.ascontiguousarray(f["wave_fd"][()]).tobytes()).hexdigest()


def test_disk_ring_alloc_and_transpose(tmp_path):
    n, M = 3, 40
    ring = DiskRingBuffer(n, M, str(tmp_path), n_channels=C, n_freq=F, n_params=P)
    DiskProducer(ring, FakeSim(), gen_batch_size=16).seed_fill_all()
    for j in range(n):
        assert os.path.exists(ring.path(j))
        with h5py.File(ring.path(j), "r") as f:
            assert f["wave_fd"].shape == (M, C, F)
            assert f["params"].shape == (M, P)
            # parameters stored sample-first (N, P): each row is constant across
            # the P columns (transpose correct), and rows are distinct ids.
            p = f["params"][()]
            assert np.allclose(p, p[:, :1])          # row i == [id_i]*P
            assert len(np.unique(p[:, 0])) == M       # M distinct row ids
    # buffer 0 is seed-filled first -> ids 0..M-1, so row i has value i
    with h5py.File(ring.path(0), "r") as f:
        assert np.allclose(f["params"][3], 3.0)


def test_norm_stats_from_seed_buffers(tmp_path):
    n, M = 3, 40
    sim = FakeSim()
    ring = DiskRingBuffer(n, M, str(tmp_path), n_channels=C, n_freq=F, n_params=P)
    DiskProducer(ring, sim, gen_batch_size=16).seed_fill_all()
    dm = DiskStreamingDataModule(
        ring, sim,
        val_pool={"wave_fd": torch.zeros(10, C, F, dtype=torch.complex128),
                  "params": torch.zeros(10, P)},
        batch_size=8, samples_per_epoch=M, num_workers=0)
    assert dm._params_pool.shape == (n * M, P)
    ref = np.concatenate([ring.read_field(j, "params") for j in range(n)], 0)
    mean, _ = dm.get_params_mean_std()
    assert np.allclose(mean.numpy(), ref.mean(0))
    assert dm.median_snr == 20.0


@pytest.mark.parametrize("num_workers", [0, 2])
def test_distinct_examples_per_epoch(tmp_path, num_workers):
    """With buffer_size == samples_per_epoch (the size chosen in
    _setup_streaming_disk), one epoch reads the buffer exactly once: M rows,
    every id present exactly once, no idx%M reuse. Producer left stopped so the
    acquired buffer is stable for the read."""
    n, M = 3, 40
    sim = FakeSim()
    ring = DiskRingBuffer(n, M, str(tmp_path), n_channels=C, n_freq=F, n_params=P)
    DiskProducer(ring, sim, gen_batch_size=16).seed_fill_all()
    dm = DiskStreamingDataModule(
        ring, sim,
        val_pool={"wave_fd": torch.zeros(10, C, F, dtype=torch.complex128),
                  "params": torch.zeros(10, P)},
        batch_size=8, samples_per_epoch=M, num_workers=num_workers, prefetch_factor=2)

    loader = dm.train_dataloader(shuffle=True, num_workers=num_workers)
    expected = np.sort(ring.read_field(dm._active_j, "params")[:, 0])
    seen = np.sort(np.concatenate(
        [b["source_parameters"][:, 0].numpy() for b in loader]))
    dm.release_active()
    assert seen.shape[0] == M                       # exactly one pass (no padding)
    assert len(np.unique(seen)) == M                # all distinct (no reuse)
    assert np.array_equal(seen, expected)           # exactly the buffer's rows


def test_stream_reuse_logger_contract(tmp_path):
    """DiskStreamingDataModule must satisfy StreamReuseLogger's contract:
    expose ``_examples_consumed`` and update ``ring.distinct_sims_seen`` — M per
    newly-filled buffer read while ``count_seen`` is armed, 0 for reuse. Producer
    left stopped so buffers are never refreshed (every re-read is pure reuse)."""
    n, M = 3, 40
    sim = FakeSim()
    ring = DiskRingBuffer(n, M, str(tmp_path), n_channels=C, n_freq=F, n_params=P)
    DiskProducer(ring, sim, gen_batch_size=16).seed_fill_all()
    dm = DiskStreamingDataModule(
        ring, sim,
        val_pool={"wave_fd": torch.zeros(10, C, F, dtype=torch.complex128),
                  "params": torch.zeros(10, P)},
        batch_size=8, samples_per_epoch=M, num_workers=0)

    assert hasattr(dm, "_examples_consumed")        # the attribute that crashed

    # count_seen off (pre-fit norm pass): no distinct counting
    dm.train_dataloader(num_workers=0)
    assert ring.distinct_sims_seen == 0
    dm._release_active()

    # armed (fit): 3 distinct buffers -> +M each; the 4th read wraps to buffer 0
    # (same fill, producer stopped) -> no increment.
    ring.count_seen = True
    for expected_distinct in (M, 2 * M, 3 * M, 3 * M):
        dm.train_dataloader(num_workers=0)
        assert ring.distinct_sims_seen == expected_distinct
        dm._release_active()
    # examples/reuse as the callback would compute them after 4 epochs of M
    dm._examples_consumed = 4 * M
    assert dm._examples_consumed / max(ring.distinct_sims_seen, 1) > 1.0


def test_fill_while_reader_open(tmp_path):
    """The bug that killed the first real run: HDF5 refuses to open a file ``r+``
    while it is open read-only in the same process, so the producer's in-place
    overwrite crashed. The atomic write-tmp+rename fill must succeed even with a
    stale read handle held on the old file."""
    n, M = 2, 20
    sim = FakeSim()
    ring = DiskRingBuffer(n, M, str(tmp_path), n_channels=C, n_freq=F, n_params=P)
    prod = DiskProducer(ring, sim, gen_batch_size=8)
    prod.seed_fill_all()
    fr = h5py.File(ring.path(0), "r")          # hold a read handle (same process)
    old = complex(fr["wave_fd"][0, 0, 0])
    prod._fill(0)                              # must NOT raise (rename sidesteps r+)
    fr.close()
    with h5py.File(ring.path(0), "r") as f:    # new content swapped in
        assert complex(f["wave_fd"][0, 0, 0]) != old
    # no leftover tmp
    assert not os.path.exists(ring.tmp_path(0))


def test_fill_aborts_on_stop(tmp_path):
    """A fill must abort promptly once the ring is stopped, without renaming its
    partial tmp in — so the round-end join() succeeds and the defunct producer
    can never write into the buffer_dir the next round reuses."""
    n, M = 2, 40
    sim = FakeSim()
    ring = DiskRingBuffer(n, M, str(tmp_path), n_channels=C, n_freq=F, n_params=P)
    prod = DiskProducer(ring, sim, gen_batch_size=8)
    prod.seed_fill_all()
    before = _buf_hash(ring.path(0))
    ring.stop()                                # simulate round-end stop
    assert prod._fill(0) is False              # aborts, no commit
    assert _buf_hash(ring.path(0)) == before   # buffer file unchanged (no rename)


@pytest.mark.parametrize("num_workers", [0, 2])
def test_round_robin_and_overwrite(tmp_path, num_workers):
    n, M = 3, 40
    sim = FakeSim()
    ring = DiskRingBuffer(n, M, str(tmp_path), n_channels=C, n_freq=F,
                          n_params=P, reuse_threshold=1)
    prod = DiskProducer(ring, sim, gen_batch_size=16)
    prod.seed_fill_all()
    dm = DiskStreamingDataModule(
        ring, sim,
        val_pool={"wave_fd": torch.zeros(10, C, F, dtype=torch.complex128),
                  "params": torch.zeros(10, P)},
        batch_size=8, samples_per_epoch=M, num_workers=num_workers, prefetch_factor=2)

    seed_hashes = [_buf_hash(ring.path(j)) for j in range(n)]
    prod.start()
    order = []
    for _ in range(6):
        loader = dm.train_dataloader(shuffle=True, num_workers=num_workers)
        order.append(dm._active_j)
        total = sum(b["wave_fd"].shape[0] for b in loader)
        assert total == M
        time.sleep(0.05)
    dm.release_active()
    time.sleep(0.2)
    ring.stop()
    prod.join(timeout=10)

    assert prod.error is None
    # strict round-robin: each epoch advances the buffer index by 1 (mod n)
    for a, b in zip(order, order[1:]):
        assert (b - a) % n == 1
    # producer refreshed at least one buffer, overwriting the file in place
    assert prod.n_chunks > 0
    new_hashes = [_buf_hash(ring.path(j)) for j in range(n)]
    assert sum(a != b for a, b in zip(seed_hashes, new_hashes)) > 0
    # disk stays bounded at exactly n files
    assert len([x for x in os.listdir(tmp_path) if x.endswith(".h5")]) == n


def test_gpu_ring_refactor_preserved():
    """The shared ``_init_state`` refactor must not change GPU ring behaviour."""
    r = RingBuffer(n_buffers=3, buffer_size=4,
                   sample_shapes={"wave_fd": (2, 5), "params": (6,)},
                   dtypes={"wave_fd": torch.complex128, "params": torch.float64},
                   device="cpu", host_fields=("params",), reuse_threshold=2)
    assert r.n == 3 and r.M == 4 and r.reuse_threshold == 2
    assert len(r._in_use) == len(r._consumed) == len(r._filling) == 3
    assert r._seen_rows[0].shape[0] == 4
    assert set(r.fields) == {"wave_fd", "params"}
    r.seed_fill(0, {"wave_fd": torch.zeros(4, 2, 5, dtype=torch.complex128),
                    "params": torch.zeros(4, 6)})
    j = r.next_readable()
    assert j is not None
    r.release_epoch(j)


# --------------------------------------------------------------------------
# k buffers per epoch: buffer_size (file) decoupled from samples_per_epoch
# --------------------------------------------------------------------------

def test_acquire_readable_many_distinct_and_atomic():
    """k distinct buffers in one pick, round-robin continuing across calls, and
    never a buffer that is in use or being filled."""
    r = RingBuffer(n_buffers=6, buffer_size=4,
                   sample_shapes={"params": (2,)}, dtypes={"params": torch.float64},
                   device="cpu")
    js = r.acquire_readable_many(5)
    assert len(set(js)) == 5                       # distinct
    assert all(r._in_use[j] for j in js)
    for j in js:
        r.release_epoch(j)
    js2 = r.acquire_readable_many(5)               # rotation continues
    assert len(set(js2)) == 5 and js2[0] == (js[-1] + 1) % 6

    # a buffer being filled is skipped; one already in use is skipped too
    for j in js2:
        r.release_epoch(j)
    r._filling[2] = True
    r._in_use[3] = True
    js3 = r.acquire_readable_many(4)
    assert 2 not in js3 and 3 not in js3 and len(set(js3)) == 4


def test_acquire_readable_many_rejects_k_without_spare():
    """k must leave at least one buffer free for the producer."""
    r = RingBuffer(n_buffers=3, buffer_size=4,
                   sample_shapes={"params": (2,)}, dtypes={"params": torch.float64},
                   device="cpu")
    with pytest.raises(ValueError):
        r.acquire_readable_many(3)
    with pytest.raises(ValueError):
        r.acquire_readable_many(0)


@pytest.mark.parametrize("num_workers", [0, 2])
def test_epoch_spans_k_buffers(tmp_path, num_workers):
    """The fix: an epoch of samples_per_epoch = k * buffer_size rows reads k whole
    files, every row exactly once — buffer_size no longer has to equal the epoch."""
    n, M, k = 4, 20, 3
    sim = FakeSim()
    ring = DiskRingBuffer(n, M, str(tmp_path), n_channels=C, n_freq=F, n_params=P)
    DiskProducer(ring, sim, gen_batch_size=8).seed_fill_all()
    dm = DiskStreamingDataModule(
        ring, sim,
        val_pool={"wave_fd": torch.zeros(10, C, F, dtype=torch.complex128),
                  "params": torch.zeros(10, P)},
        batch_size=5, samples_per_epoch=k * M, num_workers=num_workers, prefetch_factor=2)
    assert dm.k == k

    loader = dm.train_dataloader(shuffle=True, num_workers=num_workers)
    assert len(dm._active_js) == k and len(set(dm._active_js)) == k
    expected = np.sort(np.concatenate(
        [ring.read_field(j, "params")[:, 0] for j in dm._active_js]))
    seen = np.sort(np.concatenate(
        [b["source_parameters"][:, 0].numpy() for b in loader]))
    assert seen.shape[0] == k * M            # exactly one pass over k files
    assert len(np.unique(seen)) == k * M     # all distinct (no reuse)
    assert np.array_equal(seen, expected)    # exactly the locked files' rows

    dm.release_active()                      # all k released for the producer
    assert dm._active_js == []
    assert not any(ring._in_use)
    assert sum(ring._consumed) == k          # each locked file counted once


def test_epoch_config_validation(tmp_path):
    """samples_per_epoch must be a multiple of buffer_size, and n_buffers >= k+1."""
    n, M = 3, 20
    sim = FakeSim()
    ring = DiskRingBuffer(n, M, str(tmp_path), n_channels=C, n_freq=F, n_params=P)
    DiskProducer(ring, sim, gen_batch_size=8).seed_fill_all()
    pool = {"wave_fd": torch.zeros(10, C, F, dtype=torch.complex128),
            "params": torch.zeros(10, P)}
    with pytest.raises(ValueError, match="multiple of"):
        DiskStreamingDataModule(ring, sim, pool, batch_size=5,
                                samples_per_epoch=int(1.5 * M), num_workers=0)
    with pytest.raises(ValueError, match="n_buffers"):
        DiskStreamingDataModule(ring, sim, pool, batch_size=5,
                                samples_per_epoch=3 * M, num_workers=0)


def test_multi_buffer_round_robin_and_refresh(tmp_path):
    """Consecutive k-buffer epochs rotate (no file read twice in a row) and the
    spare buffer is what the producer refreshes."""
    n, M, k = 4, 20, 2
    sim = FakeSim()
    ring = DiskRingBuffer(n, M, str(tmp_path), n_channels=C, n_freq=F,
                          n_params=P, reuse_threshold=1)
    prod = DiskProducer(ring, sim, gen_batch_size=10)
    prod.seed_fill_all()
    dm = DiskStreamingDataModule(
        ring, sim,
        val_pool={"wave_fd": torch.zeros(10, C, F, dtype=torch.complex128),
                  "params": torch.zeros(10, P)},
        batch_size=5, samples_per_epoch=k * M, num_workers=0)
    seed_hashes = [_buf_hash(ring.path(j)) for j in range(n)]
    prod.start()
    picks = []
    for _ in range(4):
        loader = dm.train_dataloader(shuffle=True, num_workers=0)
        picks.append(tuple(dm._active_js))
        assert sum(b["wave_fd"].shape[0] for b in loader) == k * M
        time.sleep(0.05)
    dm.release_active()
    time.sleep(0.2)
    ring.stop()
    prod.join(timeout=10)

    assert prod.error is None
    for a, b in zip(picks, picks[1:]):
        assert not set(a) & set(b)           # consecutive epochs use other files
    assert prod.n_chunks > 0                 # the spare got refreshed
    new_hashes = [_buf_hash(ring.path(j)) for j in range(n)]
    assert sum(a != b for a, b in zip(seed_hashes, new_hashes)) > 0
    assert len([x for x in os.listdir(tmp_path) if x.endswith(".h5")]) == n
