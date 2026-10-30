"""Device and seed helpers. Every script and test uses these so nothing hard-codes a device."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def get_device() -> torch.device:
    """MPS on Apple Silicon when available, else CUDA, else CPU. FORCE_CPU=1 overrides (used by tests)."""
    if os.environ.get("FORCE_CPU") == "1":
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def describe_device(device: torch.device) -> str:
    if device.type == "mps":
        return "Apple Metal (MPS)"
    if device.type == "cuda":
        return torch.cuda.get_device_name(0)
    return "CPU"
