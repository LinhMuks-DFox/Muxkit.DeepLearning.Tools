import unittest

import torch
from torch.utils.data import Dataset

from dataset_tools.CachableDataset import CacheableDataset


class DummyDataset(Dataset):
    """
    A dummy dataset for testing purposes.
    """

    def __init__(self, length=100):
        self.data = [(torch.randn(3, 224, 224), torch.tensor(i))
                     for i in range(length)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class TestCacheableDataset(unittest.TestCase):

    def setUp(self):
        self.dataset = DummyDataset(length=10)

    def test_single_process_cache(self):
        """
        Test the single process cache functionality.
        """
        cacheable_dataset = CacheableDataset(
            self.dataset, max_cache_size=5, multiprocessing=False)
        # Access the first element twice
        x1, y1 = cacheable_dataset[0]
        x2, y2 = cacheable_dataset[0]

    def test_multiprocessing_cache(self):
        """
        Test the multiprocessing cache functionality.
        """
        cacheable_dataset = CacheableDataset(
            self.dataset, max_cache_size=5, multiprocessing=True)
        # Access the first element twice
        x1, y1 = cacheable_dataset[0]
        x2, y2 = cacheable_dataset[0]

    def test_cache_limit(self):
        """
        Test that the cache size does not exceed max_cache_size.
        """
        cacheable_dataset = CacheableDataset(
            self.dataset, max_cache_size=3, multiprocessing=False)

        # Access more elements than the max_cache_size
        for i in range(5):
            _ = cacheable_dataset[i]

        # Check that cache does not exceed the max size
        self.assertEqual(len(cacheable_dataset.cache), 3)

    def test_device_handling(self):
        """
        Test that tensors are correctly moved to the specified device in single-process mode.
        """
        cacheable_dataset = CacheableDataset(
            self.dataset, max_cache_size=5, multiprocessing=False)
        x, y = cacheable_dataset[0]

        # Move data to a specific device (e.g., CPU)
        device = torch.device("cpu")
        x_device, y_device = x.to(device), y.to(device)

        # Access the cached data and ensure it's on the correct device
        cached_x, cached_y = cacheable_dataset[0]
        self.assertTrue(cached_x.device == device)
        self.assertTrue(cached_y.device == device)

    def test_str_output(self):
        """
        Test the __str__ method to ensure it provides correct debug information.
        """
        cacheable_dataset = CacheableDataset(
            self.dataset, max_cache_size=5, multiprocessing=False)
        info_str = str(cacheable_dataset)
        self.assertIn("Cache Strategy: Single Process Cache", info_str)
        self.assertIn("Max Cache Size: 5", info_str)
        self.assertIn("Current Cache Size: 0", info_str)

        # Access some data and check if cache size updates
        _ = cacheable_dataset[0]
        info_str_updated = str(cacheable_dataset)
        self.assertIn("Current Cache Size: 1", info_str_updated)


if __name__ == '__main__':
    unittest.main()
