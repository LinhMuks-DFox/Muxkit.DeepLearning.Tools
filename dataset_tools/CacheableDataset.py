"""
Compatibility alias for the correctly spelled module name.

Prefer importing from ``dataset_tools.CacheableDataset``. This file re-exports
``CacheableDataset`` from ``CachableDataset`` without changing behavior.
"""

from .CachableDataset import CacheableDataset  # noqa: F401

__all__ = ["CacheableDataset"]

