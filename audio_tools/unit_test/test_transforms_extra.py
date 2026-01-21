import unittest
import torch

from audio_tools.transforms import (
    TimeSequenceLengthFixer,
    SoundTrackSelector,
    create_mask,
    tensor_masking,
)


class TestTransforms(unittest.TestCase):

    def test_time_sequence_length_fixer(self):
        fixer = TimeSequenceLengthFixer(1, 16000, mode="start")
        x = torch.randn(2, 8000)
        y = fixer(x)
        self.assertEqual(y.shape, (2, 16000))

    def test_sound_track_selector(self):
        sel = SoundTrackSelector("mix")
        x = torch.randn(2, 100)
        y = sel(x)
        self.assertEqual(y.shape[0], 1)

    def test_mask_and_tensor_masking(self):
        mask = create_mask([2, 10], 0.5)
        self.assertEqual(mask.shape, torch.Size([2, 10]))
        x = torch.arange(20).reshape(2, 10).float()
        m1, m2, m = tensor_masking(x, 0.5)
        self.assertTrue((m1 + m2).equal(x))

