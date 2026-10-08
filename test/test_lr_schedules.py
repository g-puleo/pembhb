"""`joint_training.lr_schedule`: whole-optimiser LR shapes.

The per-group `SingleGroupReduceLROnPlateau` only touches one param group;
OneCycle and the cosine family drive every group at once. These pin the shapes
and the "absent config changes nothing" contract.
"""
import types
import numpy as np
import pytest
import torch

from pembhb.model import JointAEInferenceNetwork


class _Stub:
    """Plain object carrying the REAL method; LightningModule.hparams is a
    read-only property, so the method is bound here instead of onto an
    instance of the model."""
    _build_global_lr_schedule = JointAEInferenceNetwork._build_global_lr_schedule


def _fake(lr_schedule, total_steps=1000, lr_ae=1e-4, lr_nre=1e-3):
    """A stand-in carrying only what _build_global_lr_schedule reads."""
    obj = _Stub()
    obj.hparams = {"train_conf": {"joint_training": {"lr_schedule": lr_schedule}}}
    obj.trainer = types.SimpleNamespace(estimated_stepping_batches=total_steps)
    p1 = torch.nn.Parameter(torch.zeros(2))
    p2 = torch.nn.Parameter(torch.zeros(2))
    opt = torch.optim.AdamW([{"params": [p1], "lr": lr_ae},
                             {"params": [p2], "lr": lr_nre}])
    return obj, opt


def _trace(obj, opt, n):
    cfg = obj._build_global_lr_schedule(opt)
    sched = cfg["scheduler"]
    assert cfg["interval"] == "step"
    out = []
    for _ in range(n):
        out.append([g["lr"] for g in opt.param_groups])
        opt.step(); sched.step()
    return np.asarray(out)


def test_absent_or_plateau_changes_nothing():
    for cfg in ({}, {"type": "plateau"}, {"type": "none"}, None):
        obj, opt = _fake(cfg)
        assert obj._build_global_lr_schedule(opt) is None


def test_onecycle_rises_then_falls_below_base():
    obj, opt = _fake({"type": "onecycle", "max_lr_mult": 10.0, "warmup_frac": 0.3})
    tr = _trace(obj, opt, 1000)
    peak = tr[:, 1].argmax()
    assert 250 < peak < 350                       # peak near pct_start
    assert tr[peak, 1] > 5 * tr[0, 1]             # genuinely rises
    assert tr[-1, 1] < tr[0, 1]                   # anneals below the start
    # both groups scheduled, each from its own base lr
    assert tr[peak, 1] / tr[peak, 0] == pytest.approx(10.0, rel=1e-6)


def test_cosine_warms_up_then_decays_monotonically():
    obj, opt = _fake({"type": "cosine", "warmup_frac": 0.1, "min_lr": 1e-7})
    tr = _trace(obj, opt, 1000)
    warm = 100
    assert tr[0, 1] < tr[warm, 1]                 # warmup climbs
    assert tr[warm, 1] == pytest.approx(1e-3, rel=1e-3)   # reaches base lr
    tail = tr[warm + 5:, 1]
    assert np.all(np.diff(tail) <= 1e-12)         # then only decays
    assert tail[-1] < 1e-5


def test_warm_restarts_restarts():
    obj, opt = _fake({"type": "warm_restarts", "t0_frac": 0.25, "t_mult": 2})
    tr = _trace(obj, opt, 1000)
    d = np.diff(tr[:, 1])
    assert (d > 0).sum() >= 1                     # at least one jump back up
    assert tr[:, 1].max() == pytest.approx(1e-3, rel=1e-6)


def test_unknown_type_is_rejected_loudly():
    obj, opt = _fake({"type": "magic"})
    with pytest.raises(ValueError, match="not recognised"):
        obj._build_global_lr_schedule(opt)


def test_missing_step_count_is_rejected():
    obj, opt = _fake({"type": "onecycle"}, total_steps=0)
    with pytest.raises(RuntimeError, match="estimated_stepping_batches"):
        obj._build_global_lr_schedule(opt)
