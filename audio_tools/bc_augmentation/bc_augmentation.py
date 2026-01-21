"""
bc_augmentation (preferred import)

This module provides a correctly spelled import path for the Between-Class
augmentation utilities. It re-exports the public API from
``.bc_augmentaion`` for backward compatibility without changing behavior.

Example
    from audio_tools.bc_augmentation import mix_sounds, compute_gain
"""

from .bc_augmentaion import *  # noqa: F401,F403 - re-export compatibility

__all__ = [
    name for name in globals().keys()
    if not name.startswith("_")
]
