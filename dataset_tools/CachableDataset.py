from multiprocessing import Manager

import torch.utils.data as data


class CacheableDataset(data.Dataset):

    def __init__(self, dataset: data.Dataset, max_cache_size: int = 1000, multiprocessing: bool = False, device:str="cpu") -> None:
        """
        :param dataset: Original dataset
        :param max_cache_size: Maximum cache size
        :param multiprocessing: Whether to use multiprocessing shared cache
        """
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

    def __getitem__(self, idx) -> tuple:
        if idx in self.cache:
            x, y = self.cache[idx]
            return x.to(self.device), y.to(self.device)
        x, y = self.dataset[idx]
        if len(self.cache) < self.max_cache_size:
            self.cache[idx] = (x.detach().cpu().clone(), y.clone().detach().cpu().clone())
        return x.to(self.device), y.to(self.device)

    def __str__(self) -> str:
        """
        Return debugging information about the dataset, including the cache strategy and cache size.
        """
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
