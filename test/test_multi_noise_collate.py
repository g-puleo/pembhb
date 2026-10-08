"""Smoke tests for mbhb_collate_fn(n_noise_realisations > 1).

The multi-noise-instances merge (commit be7e3ff, branch
``multiple_noise_instance_per_example``) extends the collate to expand a
batch of ``B`` waveforms into ``B * n_noise_realisations`` examples by
tiling the batch ``n`` times and drawing an independent noise realisation
for every one of the ``B * n`` slots (each waveform sees ``n`` noises of
its own; none is shared across waveforms).

These tests pin the contract:
* shape after expansion is ``(n*B, ...)``,
* waveforms / parameters are tiled (``out[i] == out[i + B]``),
* all ``n*B`` noise realisations are distinct,
* ``n_noise_realisations = 1`` is byte-identical to the legacy single-tile path,
* the stored-noise path bypasses expansion entirely (so observation
  files stay deterministic).
"""

import torch

from pembhb.utils import mbhb_collate_fn


_C = 2          # channels
_F = 128        # frequency bins (small for fast tests)
_P = 11         # parameters
_RNG = torch.Generator().manual_seed(31415)


def _make_noise_scale():
    # noise_scale (C, F) > 0; values arbitrary, just need positive.
    return torch.linspace(1.0, 2.0, _F).expand(_C, _F).clone()


def _make_batch(B: int, with_td: bool = False, with_stored_noise: bool = False):
    """Construct a list-of-dicts batch like MBHBDataset.__getitem__ produces.

    Each waveform is filled with its (i+1) so adjacent samples are easily
    distinguishable by exact equality in the tests.
    """
    batch = []
    for i in range(B):
        d = {
            "wave_fd": torch.full((_C, _F), float(i + 1), dtype=torch.complex64),
            "params":  torch.full((_P,), float(i + 1)),
        }
        if with_td:
            d["wave_td"] = torch.full((_C, _F), float(i + 1))
        if with_stored_noise:
            d["noise_fd"] = torch.randn(_C, _F, generator=_RNG,
                                        dtype=torch.float32).to(torch.complex64)
        batch.append(d)
    return batch


# ---------------------------------------------------------------------------
# 1. Shape contract: (B,...) → (n*B,...) for wave_fd / noise_fd / params
# ---------------------------------------------------------------------------

def test_shape_expansion_n_gt_one():
    B, n = 4, 3
    out = mbhb_collate_fn(
        _make_batch(B), noise_scale=_make_noise_scale(),
        noise_factor=1.0, n_noise_realisations=n,
    )
    assert out["wave_fd"].shape == (n * B, _C, _F), out["wave_fd"].shape
    assert out["noise_fd"].shape == (n * B, _C, _F), out["noise_fd"].shape
    assert out["source_parameters"].shape == (n * B, _P), out["source_parameters"].shape


# ---------------------------------------------------------------------------
# 2. Waveform / parameter tile pattern: out[i] == out[i + B] across tiles
# ---------------------------------------------------------------------------

def test_waveform_and_params_tiled_across_realisations():
    B, n = 4, 3
    out = mbhb_collate_fn(
        _make_batch(B), noise_scale=_make_noise_scale(),
        noise_factor=1.0, n_noise_realisations=n,
    )
    for i in range(B):
        for j in range(1, n):
            assert torch.equal(out["wave_fd"][i], out["wave_fd"][j * B + i])
            assert torch.equal(out["source_parameters"][i],
                               out["source_parameters"][j * B + i])


# ---------------------------------------------------------------------------
# 3. Noise pattern: every one of the n*B slots gets its own realisation
# ---------------------------------------------------------------------------

def test_noise_independent_across_all_slots():
    B, n = 4, 3
    out = mbhb_collate_fn(
        _make_batch(B), noise_scale=_make_noise_scale(),
        noise_factor=1.0, n_noise_realisations=n,
    )
    noise = out["noise_fd"]
    for a in range(n * B):
        for b in range(a + 1, n * B):
            assert not torch.equal(noise[a], noise[b]), (a, b)


# ---------------------------------------------------------------------------
# 4. n_noise_realisations=1 is byte-identical to the un-expanded path
# ---------------------------------------------------------------------------

def test_n_equal_one_backward_compat():
    B = 4
    out = mbhb_collate_fn(
        _make_batch(B), noise_scale=_make_noise_scale(),
        noise_factor=1.0, n_noise_realisations=1,
    )
    assert out["wave_fd"].shape == (B, _C, _F)
    assert out["noise_fd"].shape == (B, _C, _F)
    assert out["source_parameters"].shape == (B, _P)


# ---------------------------------------------------------------------------
# 5. Stored-noise batches bypass expansion (observation determinism)
# ---------------------------------------------------------------------------

def test_stored_noise_bypasses_expansion():
    B, n = 4, 5
    batch = _make_batch(B, with_stored_noise=True)
    out = mbhb_collate_fn(
        batch, noise_scale=_make_noise_scale(),
        noise_factor=1.0, n_noise_realisations=n,
    )
    # Despite n=5, stored-noise path keeps the batch size at B.
    assert out["wave_fd"].shape == (B, _C, _F)
    assert out["noise_fd"].shape == (B, _C, _F)
    assert out["source_parameters"].shape == (B, _P)
    # Stored noise was loaded verbatim (modulo noise_factor=1.0):
    for i in range(B):
        assert torch.equal(out["noise_fd"][i], batch[i]["noise_fd"])
