"""Unit tests for pembhb.streaming.RingBuffer.

These run on CPU tensors with a fake producer/consumer to prove the two
synchronisation guarantees in isolation from CUDA/bbhx:

1. no buffer is overwritten before it has been consumed at least once;
2. the producer blocks (back-pressure) when no buffer is free to fill.
"""

import threading
import time

import torch

from pembhb.streaming import RingBuffer


def _make_ring(n_buffers=3, M=4):
    return RingBuffer(
        n_buffers=n_buffers,
        buffer_size=M,
        sample_shapes={"x": (1,)},
        dtypes={"x": torch.float32},
        device="cpu",
    )


def _chunk(M, tag):
    return {"x": torch.full((M, 1), float(tag))}


def test_no_unseen_overwrite():
    """Drive a real producer + consumer thread and check that every value the
    producer destroys was read by the consumer at least once."""
    n, M = 3, 4
    rb = _make_ring(n, M)

    for j in range(n):
        rb.seed_fill(j, _chunk(M, 0))  # all buffers start tagged 0

    current_tag = [0] * n          # tag currently stored in each buffer
    overwritten = []               # tags the producer destroys
    consumed = []                  # tags the consumer reads
    next_tag = [1]                 # next unique tag to write (list = mutable closure)

    def producer():
        while True:
            j = rb.acquire_writable()
            if j is None:
                return
            overwritten.append(current_tag[j])   # safe: j is `filling`, consumer skips it
            tag = next_tag[0]
            next_tag[0] += 1
            current_tag[j] = tag
            rb.write(j, _chunk(M, tag))           # producer fast: no sleep

    def consumer(n_epochs):
        for _ in range(n_epochs):
            j = rb.next_readable()
            if j is None:
                return
            consumed.append(int(rb.fields["x"][j][0, 0].item()))
            time.sleep(0.005)                     # consumer slower -> stresses back-pressure
            rb.release_epoch(j)
        rb.stop()

    pt = threading.Thread(target=producer)
    ct = threading.Thread(target=consumer, args=(60,))
    pt.start(); ct.start()
    ct.join(timeout=10); pt.join(timeout=10)
    assert not pt.is_alive() and not ct.is_alive(), "threads did not finish"

    # The core guarantee: nothing was overwritten without being seen first.
    assert set(overwritten).issubset(set(consumed)), (
        f"overwritten-but-never-consumed tags: {set(overwritten) - set(consumed)}"
    )
    # Sanity: the producer actually did work (otherwise the test is vacuous).
    assert len(overwritten) > n


def test_backpressure_blocks_then_releases():
    """With nothing consumed, acquire_writable must block; releasing an epoch
    must unblock it and hand back exactly that buffer."""
    n, M = 2, 4
    rb = _make_ring(n, M)
    for j in range(n):
        rb.seed_fill(j, _chunk(M, 0))  # all READY but consumed=False

    result = {}

    def one_acquire():
        result["j"] = rb.acquire_writable()

    t = threading.Thread(target=one_acquire)
    t.start()
    t.join(timeout=0.2)
    assert t.is_alive(), "producer should be blocked: no buffer consumed yet"

    rb.release_epoch(1)          # consumer signs off on buffer 1
    t.join(timeout=2)
    assert not t.is_alive(), "producer should have unblocked after release"
    assert result["j"] == 1, "producer must claim the buffer that was released"


def test_stop_unblocks_with_none():
    """stop() must wake a blocked producer and make acquire_writable return None."""
    rb = _make_ring(2, 4)
    for j in range(2):
        rb.seed_fill(j, _chunk(4, 0))

    result = {}

    def one_acquire():
        result["j"] = rb.acquire_writable()

    t = threading.Thread(target=one_acquire)
    t.start()
    t.join(timeout=0.2)
    assert t.is_alive(), "producer should be blocked before stop()"

    rb.stop()
    t.join(timeout=2)
    assert not t.is_alive()
    assert result["j"] is None
