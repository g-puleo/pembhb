import os
import torch
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
print(f"ROOT_DIR: {ROOT_DIR}")
DATA_ROOT_DIR = "/data/gpuleo/mbhb"

# ---------------------------------------------------------------------------
# Global precision configuration
# ---------------------------------------------------------------------------
# These module-level variables hold the current precision setting.  They are
# read by simulator, data, model, ROM and autoencoder modules whenever a
# tensor or array is created.  float32 is the default so that existing
# workflows continue to work without any changes.
#
# Call ``set_precision("float64")`` (or pass ``precision: "float64"`` in
# train_config.yaml) **before** instantiating any model / data object to
# switch the entire pipeline to double precision.
#
# NOTE (future work – per-round precision):
#   To allow switching precision between TMNRE rounds, every component that
#   caches tensors (ROM basis, autoencoder buffers, dataset cache, model
#   buffers) would need a ``.to(new_dtype)`` pass after ``set_precision``
#   is called.  The registry itself already supports being called multiple
#   times; only the downstream "cast existing objects" logic is missing.
# ---------------------------------------------------------------------------

_PRECISION: str = "float32"  # "float32" or "float64"


def set_precision(precision: str = "float32") -> None:
    """Set the global numerical precision for the pipeline.

    Parameters
    ----------
    precision : str
        ``"float32"`` (default) or ``"float64"``.

    This function also calls :func:`torch.set_default_dtype` so that newly
    created ``nn.Parameter`` / ``nn.Linear`` weights automatically use the
    chosen precision.  When *precision* is ``"float32"`` we additionally
    set ``torch.set_float32_matmul_precision("medium")`` for performance;
    that call is skipped for ``"float64"`` since it is irrelevant.
    """
    global _PRECISION
    precision = precision.lower().strip()
    if precision not in ("float32", "float64"):
        raise ValueError(f"Unsupported precision '{precision}'. Use 'float32' or 'float64'.")
    _PRECISION = precision

    torch.set_default_dtype(get_torch_dtype())
    if precision == "float32":
        torch.set_float32_matmul_precision("medium")

    print(f"[pembhb] Precision set to {_PRECISION} "
          f"(torch default dtype: {torch.get_default_dtype()})")


def get_precision() -> str:
    """Return the current precision string (``'float32'`` or ``'float64'``)."""
    return _PRECISION


def get_torch_dtype() -> torch.dtype:
    """Return the PyTorch real dtype matching the current precision."""
    return torch.float64 if _PRECISION == "float64" else torch.float32


def get_torch_complex_dtype() -> torch.dtype:
    """Return the PyTorch complex dtype matching the current precision."""
    return torch.complex128 if _PRECISION == "float64" else torch.complex64


def get_numpy_dtype() -> np.dtype:
    """Return the NumPy real dtype matching the current precision."""
    return np.float64 if _PRECISION == "float64" else np.float32


def get_numpy_complex_dtype() -> np.dtype:
    """Return the NumPy complex dtype matching the current precision."""
    return np.complex128 if _PRECISION == "float64" else np.complex64