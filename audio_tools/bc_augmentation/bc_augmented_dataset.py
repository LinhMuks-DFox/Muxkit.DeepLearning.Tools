"""
BC Augmented Dataset

Reference
- Tokozume et al., 2017. "Between-Class Learning for Image Classification" (arXiv:1711.10282).

This module defines ``BCLearningDataset``, which creates mixed audio samples and
soft labels via Between-Class (BC) augmentation. Behavior unchanged; docs clarified.
"""

import torch
from torch.utils.data import Dataset

from .bc_augmentation import mix_sounds


class BCLearningDataset(Dataset):
    """Dataset wrapper that applies BC mixing on-the-fly.

    Args:
        dataset (Dataset|Sequence): Source dataset returning (audio, label).
        sample_rate (int): Sample rate for gain computation.
        num_classes (int): Number of classes for one-hot conversion.
        device (str): Torch device to host tensors ("cpu" or "cuda").
    """
    def __init__(self, dataset, sample_rate, num_classes, device="cpu"):
        super().__init__()
        self.dataset = dataset
        self.sample_rate = sample_rate
        self.num_classes = num_classes
        self.device = device

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """Return a BC-mixed example (mixed_audio, soft_label)."""
        while True:
            idx1 = torch.randint(len(self.dataset), (1,)).item()
            idx2 = torch.randint(len(self.dataset), (1,)).item()
            sound1, label1 = self.dataset[idx1]
            sound2, label2 = self.dataset[idx2]
            label1, label2 = [i if isinstance(i, torch.Tensor) else torch.tensor(i) for i in [label1, label2]]
            if not torch.equal(label1, label2):
                break

        sound1 = sound1.to(self.device)
        sound2 = sound2.to(self.device)

        if isinstance(label1, int) or label1.ndim == 0:
            label1 = torch.eye(self.num_classes, device=self.device)[label1]
            label2 = torch.eye(self.num_classes, device=self.device)[label2]

        r = torch.rand(1, device=label1.device)  # Random mix ratio in [0,1]
        mixed_sound = mix_sounds(sound1, sound2, r.to(sound1.device), self.sample_rate, device=sound1.device)
        label = label1 * r.squeeze() + label2 * (1 - r.squeeze())

        return mixed_sound, label
