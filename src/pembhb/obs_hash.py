"""Content hash of an observation file, used to match runs with MCMC samples."""
import hashlib

import h5py
import numpy as np

HASHED_DATASETS = ("frequencies", "wave_fd", "noise_fd")


def obs_hash(path: str) -> str:
    """12-hex SHA-256 over the arrays inference uses (not the file bytes)."""
    h = hashlib.sha256()
    with h5py.File(path, "r") as f:
        for name in HASHED_DATASETS:
            a = np.ascontiguousarray(f[name][()])
            h.update(f"{name}|{a.dtype.str}|{a.shape}|".encode())
            h.update(a.tobytes())
    return h.hexdigest()[:12]
