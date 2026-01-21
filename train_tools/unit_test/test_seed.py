import unittest
import random
import numpy as np
import torch

from train_tools.utils import set_manual_seed


class TestSeed(unittest.TestCase):

    def test_reproducibility(self):
        set_manual_seed(123)
        a = random.random(); b = np.random.rand(); c = torch.randn(3)
        set_manual_seed(123)
        a2 = random.random(); b2 = np.random.rand(); c2 = torch.randn(3)
        self.assertEqual(a, a2)
        self.assertEqual(b, b2)
        self.assertTrue(torch.allclose(c, c2))

