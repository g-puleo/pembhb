"""GPU memory reservation and monitoring utilities.

Provides functions to measure peak GPU usage via a dry-run forward+backward
pass and then pre-claim exactly the needed memory (plus a safety margin) in
PyTorch's CUDA caching allocator so that other processes on the same GPU
cannot steal it.
"""

import torch


def reserve_gpu_memory(model, dataloader, safety_factor=1.25, device=None):
    """Measure peak GPU memory with a dry-run training step, then reserve it.

    1. Runs one forward + backward pass on a real batch.
    2. Reads ``torch.cuda.max_memory_allocated()`` for the true peak.
    3. Allocates a dummy tensor of ``peak * safety_factor`` bytes, then
       deletes it.  The memory stays in PyTorch's CUDA caching allocator,
       effectively reserved for this process.

    **Important**: do NOT call ``torch.cuda.empty_cache()`` after this
    function — that would release the cached memory back to CUDA.

    Parameters
    ----------
    model : torch.nn.Module
        The model that will be trained (e.g. ``JointAEInferenceNetwork``).
    dataloader : torch.utils.data.DataLoader
        A training dataloader — one batch will be consumed for the dry run.
    safety_factor : float
        Multiplicative safety margin applied to the measured peak.
        Default 1.25 (reserve 25 % more than measured).
    device : torch.device or None
        CUDA device. Defaults to ``cuda:0``.

    Returns
    -------
    float
        Amount of memory reserved, in GiB.
    """
    if device is None:
        device = torch.device("cuda")

    if not torch.cuda.is_available():
        print("[gpu_utils] CUDA not available — skipping reservation.")
        return 0.0

    model = model.to(device)
    model.train()

    # Reset peak stats so we measure only the dry run
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    # --- dry-run forward + backward with one batch --------------------------
    batch = next(iter(dataloader))
    # Move batch tensors to device
    batch_gpu = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch_gpu[k] = v.to(device)
        else:
            batch_gpu[k] = v

    loss = model.training_step(batch_gpu, batch_idx=0)
    if isinstance(loss, dict):
        loss = loss["loss"]
    loss.backward()

    torch.cuda.synchronize(device)
    peak_bytes = torch.cuda.max_memory_allocated(device)

    # Clean up the dry-run gradients and optimizer state
    model.zero_grad(set_to_none=True)
    del loss, batch, batch_gpu
    # Free the memory used by the dry run itself before reserving
    torch.cuda.empty_cache()

    # --- reserve peak * safety_factor bytes ---------------------------------
    reserve_bytes = int(peak_bytes * safety_factor)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)

    # Don't try to reserve more than what's actually free
    reserve_bytes = min(reserve_bytes, int(free_bytes * 0.95))

    if reserve_bytes > 0:
        _dummy = torch.empty(reserve_bytes, dtype=torch.uint8, device=device)
        del _dummy
        # Do NOT call torch.cuda.empty_cache() here — the memory must stay
        # in the caching allocator so other processes cannot claim it.

    reserved_gib = reserve_bytes / (1024 ** 3)
    peak_gib = peak_bytes / (1024 ** 3)
    total_gib = total_bytes / (1024 ** 3)
    print(
        f"[gpu_utils] Measured peak: {peak_gib:.2f} GiB | "
        f"Reserved: {reserved_gib:.2f} GiB (x{safety_factor}) | "
        f"GPU total: {total_gib:.1f} GiB"
    )
    return reserved_gib


def log_gpu_memory(label=""):
    """Print a one-line GPU memory summary."""
    if not torch.cuda.is_available():
        return
    free, total = torch.cuda.mem_get_info()
    alloc = torch.cuda.memory_allocated()
    cached = torch.cuda.memory_reserved()
    prefix = f"[gpu_mem {label}] " if label else "[gpu_mem] "
    print(
        f"{prefix}"
        f"free={free / 1e9:.2f}G  "
        f"alloc={alloc / 1e9:.2f}G  "
        f"cached={cached / 1e9:.2f}G  "
        f"total={total / 1e9:.1f}G"
    )
