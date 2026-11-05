"""Runtime selection shared by automatic CVAE entry points."""

import torch


def training_device(requested):
    """Resolve CPU/CUDA selection and fail before creating run output."""

    if requested == "cpu":
        return "cpu"
    available = torch.cuda.is_available()
    if requested == "auto" and not available:
        return "cpu"
    if not available:
        raise RuntimeError(
            "CUDA is unavailable. Check that nvidia-smi works, or explicitly "
            "choose --device cpu. No run directory was created."
        )
    try:
        torch.empty(1, device="cuda")
    except RuntimeError as error:
        raise RuntimeError(
            "CUDA could not initialize. Check the NVIDIA driver with "
            "nvidia-smi, or explicitly choose --device cpu. No run directory "
            "was created."
        ) from error
    return "cuda"
