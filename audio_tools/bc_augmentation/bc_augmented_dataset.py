"""
bc_augmented_dataset.py

This module provides a PyTorch Dataset class, `BCLearningDataset`,
which uses Between-Class (BC) augmentation for generating mixed audio samples and labels.

Key formulas:
1. **Mixed Label**:
   Given two one-hot encoded labels \( y_1 \) and \( y_2 \) and a mixing ratio \( r \):
   \[
   y_{mix} = r \cdot y_1 + (1 - r) \cdot y_2
   \]
"""

import torch
from torch.utils.data import Dataset

from .bc_augmention import mix_sounds


class BCLearningDataset(Dataset):
    def __init__(self, dataset, sample_rate, num_classes, device='cpu'):
        """
        Args:
            dataset (torch.utils.data.Dataset): 原始数据集，标签可以是整数或one-hot编码。
            sample_rate (int): 音频的采样率。
            num_classes (int): 类别数量，用于one-hot编码。
            device (str): 'cpu' 或 'cuda'。
        """
        self.dataset = dataset
        self.sample_rate = sample_rate
        self.num_classes = num_classes
        self.device = device

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
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

        r = torch.rand(1, device=self.device)  # Adjust r to match the number of channels
        # print(sound1.shape, sound2.shape)
        mixed_sound = mix_sounds(sound1, sound2, r, self.sample_rate, device=self.device)
        label = label1 * r.squeeze() + label2 * (1 - r.squeeze())  # Adjust label calculation for channels

        return mixed_sound, label
