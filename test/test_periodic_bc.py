"""Tests for generalised periodic-BC handling (psi has period pi, not 2pi)."""
import numpy as np
import torch

from pembhb.utils import parse_periodic_bc_spec
from pembhb.model import reparametrise_periodic_bc


def test_parse_list_form_is_legacy_2pi():
    # A plain list of indices => every index has angular frequency k = 1 (period 2pi).
    indices, k = parse_periodic_bc_spec([5, 7])
    assert indices == [5, 7]
    assert k == {5: 1.0, 7: 1.0}


def test_parse_none_is_empty():
    assert parse_periodic_bc_spec(None) == ([], {})


def test_parse_dict_numeric_and_string_periods():
    indices, k = parse_periodic_bc_spec({5: "2*pi", 7: 2 * np.pi, 9: "pi"})
    assert indices == [5, 7, 9]
    assert k[5] == k[7] == 1.0
    assert abs(k[9] - 2.0) < 1e-12  # period pi => k = 2


def test_embedding_uses_k_frequency():
    p = torch.tensor([[0.3, 1.1]])
    out = reparametrise_periodic_bc(p, [0], {0: 2.0})
    assert out.shape == (1, 3)
    assert torch.allclose(out[0, 0], torch.sin(torch.tensor(0.6)))
    assert torch.allclose(out[0, 1], torch.cos(torch.tensor(0.6)))
    assert torch.allclose(out[0, 2], torch.tensor(1.1))  # non-periodic col untouched


def test_embedding_legacy_call_defaults_to_k1():
    p = torch.tensor([[0.3]])
    out = reparametrise_periodic_bc(p, [0])  # no k_by_index
    assert torch.allclose(out[0, 0], torch.sin(torch.tensor(0.3)))
    assert torch.allclose(out[0, 1], torch.cos(torch.tensor(0.3)))


def test_psi_period_pi_invariance():
    # The whole point: with k=2 the embedding is invariant under theta -> theta + pi.
    theta = torch.tensor([[0.4]])
    shifted = theta + torch.pi
    a = reparametrise_periodic_bc(theta, [0], {0: 2.0})
    b = reparametrise_periodic_bc(shifted, [0], {0: 2.0})
    assert torch.allclose(a, b, atol=1e-6)
    # ...and a k=1 embedding is NOT invariant under +pi (only under +2pi).
    c = reparametrise_periodic_bc(theta, [0], {0: 1.0})
    d = reparametrise_periodic_bc(shifted, [0], {0: 1.0})
    assert not torch.allclose(c, d, atol=1e-3)
