"""
CacheableDataset: lightweight in-memory caching wrapper for any Dataset.

Purpose
- Speed up data loading by caching up to ``max_cache_size`` samples in RAM.
- Optional multiprocessing-safe cache via ``multiprocessing.Manager``.

Behavior is unchanged; documentation added for clarity.
"""

from multiprocessing import Manager
import torch
import torch.utils.data as data


class CacheableDataset(data.Dataset):
    """Wrap any dataset and cache items by index.

    Args:
        dataset (Dataset): The underlying dataset ``__getitem__`` returns (x, y).
        max_cache_size (int): Max number of samples to keep in memory.
        multiprocessing (bool): Use process-shared dict for cache.
        device (str): Device to return tensors on ("cpu"/"cuda").
    """

    def __init__(self, dataset: data.Dataset, max_cache_size: int = 1000, multiprocessing: bool = False, device: str = "cpu") -> None:
        self.dataset = dataset
        self.max_cache_size = max_cache_size
        self.multiprocessing = multiprocessing

        # Choose which type of caching method to use
        if self.multiprocessing:
            self.cache = Manager().dict()  # Multiprocessing shared cache
        else:
            self.cache = {}  # Single process cache

        self.device = device

    def __len__(self) -> int:
        return len(self.dataset)

    @torch.no_grad()
    def __getitem__(self, idx) -> tuple:
        """Return (x, y), reading from cache if available and cloning to avoid aliasing."""
        if idx in self.cache:
            x, y = self.cache[idx]
            return x.to(self.device), y.to(self.device)
        x, y = self.dataset[idx]
        if len(self.cache) < self.max_cache_size:
            self.cache[idx] = (x.cpu().clone(), y.cpu().clone())
        return x.to(self.device), y.to(self.device)

    def __str__(self) -> str:
        """Return a compact summary of cache strategy and size."""
        strategy = "Multiprocessing Cache" if self.multiprocessing else "Single Process Cache"
        cache_size = len(self.cache)
        info = (
            f"CacheableDataset Info:\n"
            f"- Cache Strategy: {strategy}\n"
            f"- Max Cache Size: {self.max_cache_size}\n"
            f"- Current Cache Size: {cache_size}\n"
            f"- Device: {self.device}\n"
        )
        return info
