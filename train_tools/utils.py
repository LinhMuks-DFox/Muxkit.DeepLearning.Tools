"""
Training utilities: seeding and reproducibility helpers.

Note: Enabling deterministic behavior may reduce CUDA performance.
Behavior unchanged; docstrings improved.
"""

import random
import numpy as np
import torch


def set_manual_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs (CPU/GPU) for reproducibility.

    Args:
        seed (int): Seed value for RNGs.
    """
    # Python random seed
    random.seed(seed)

    # NumPy random seed
    np.random.seed(seed)

    # PyTorch random seed for CPU
    torch.manual_seed(seed)

    # PyTorch deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # PyTorch GPU (CUDA)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # If using multi-GPU setups

    # PyTorch MPS (Apple Silicon)
    if torch.backends.mps.is_available():
        print("Warning: MPS may not support full deterministic behavior.")

    # PyTorch ROCm (AMD GPUs)
    if torch.backends.mps.is_built():
        pass  # No separate seed API
    if torch.backends.cuda.is_built():
        torch.cuda.manual_seed(seed)  # Applies to ROCm backends too.
