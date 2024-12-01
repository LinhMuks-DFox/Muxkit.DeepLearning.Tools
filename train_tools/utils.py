import random
import numpy as np
import torch


def set_manual_seed(seed: int) -> None:
    """
    Set the random seed for Python's random, NumPy, and PyTorch across all supported devices.

    Args:
        seed (int): The seed to set for all random number generators.
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
        print("Warning: MPS does not support manual seed setting for deterministic behavior.")

    # PyTorch ROCm (AMD GPUs)
    if torch.has_mps:
        # Currently, no separate manual seed API for MPS
        pass
    if torch.has_cuda and torch.backends.cuda.is_built():
        torch.cuda.manual_seed(seed)  # Apply to ROCm if CUDA is using AMD backend.
