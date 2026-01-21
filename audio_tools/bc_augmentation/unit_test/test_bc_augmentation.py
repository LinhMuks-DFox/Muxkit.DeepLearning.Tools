import unittest
import torch
from ..bc_augmented_dataset import BCLearningDataset
from ..bc_augmentaion import mix_sounds, compute_gain
from torch.utils.data import DataLoader


class TestBCLearningDataset(unittest.TestCase):
    def setUp(self):
        self.num_classes = 5
        self.sample_rate = 16000
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Mock dataset with 10 samples, each sample is a tuple (audio tensor, label)
        self.mock_data = [
            (torch.rand(16000), torch.randint(0, self.num_classes, (1,)).item())
            for _ in range(10)
        ]
        self.dataset = BCLearningDataset(
            self.mock_data, self.sample_rate, self.num_classes, device=self.device)

    def test_mixed_sample_shape(self):
        mixed_sample, label = self.dataset[0]
        self.assertEqual(
            mixed_sample.shape[0], 16000, "Mixed sample length should match sample_rate.")
        self.assertEqual(label.shape[0], self.num_classes,
                         "Label should be one-hot encoded with num_classes elements.")

    def test_data_loader(self):
        data_loader = DataLoader(self.dataset, batch_size=2)
        for batch in data_loader:
            sounds, labels = batch
            self.assertEqual(
                sounds.shape[1], 16000, "Each sound in the batch should have length 16000.")
            self.assertEqual(
                labels.shape[1], self.num_classes, "Each label should have num_classes elements.")

    def test_gain_adjustment(self):
        sound1, _ = self.dataset[0]
        sound2, _ = self.dataset[1]
        r = 0.5
        mixed_sound = mix_sounds(
            sound1, sound2, r, self.sample_rate, device=self.device)

        # 计算增益调整系数 t，确保使用 torch.tensor() 包装
        gain1 = compute_gain(sound1, self.sample_rate,
                             device=self.device).max()
        gain2 = compute_gain(sound2, self.sample_rate,
                             device=self.device).max()
        t = 1.0 / (1 + torch.pow(torch.tensor(10.0, device=self.device),
                   (gain1 - gain2) / 20.0) * (1 - r) / r)

        # 使用与 mix_sounds 中相同的公式计算 expected_norm
        expected_norm = (sound1 * t + sound2 * (1 - t)).norm().item() / \
            torch.sqrt(t ** 2 + (1 - t) ** 2).item()

        self.assertAlmostEqual(mixed_sound.norm().item(),
                               expected_norm, delta=0.01)


if __name__ == "__main__":
    unittest.main()
