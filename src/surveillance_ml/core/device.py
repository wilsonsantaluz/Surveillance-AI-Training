"""Compute-device selection and deterministic random seeding."""

import random

import numpy as np
import torch


def select_device() -> torch.device:
    """Prefer CUDA when available and otherwise run safely on CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
