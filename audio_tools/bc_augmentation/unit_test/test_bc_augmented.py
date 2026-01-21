import pathlib

from ..bc_augmentaion import mix_sounds as my_mix_sounds
import torchaudio
import unittest
import torch
import numpy as np
import random
# Tested Method, implemented by the ACDNet authors


def padding(pad):
    def f(sound):
        return np.pad(sound, pad, 'constant')
    return f


def random_gain(db):
    def f(sound):
        return sound * np.power(10, random.uniform(-db, db) / 20.0)

    return f


def a_weight(fs, n_fft, min_db=-80.0):
    freq = np.linspace(0, fs // 2, n_fft // 2 + 1)
    freq_sq = np.power(freq, 2)
    freq_sq[0] = 1.0
    weight = 2.0 + 20.0 * (2 * np.log10(12194) + 2 * np.log10(freq_sq)
                           - np.log10(freq_sq + 12194 ** 2)
                           - np.log10(freq_sq + 20.6 ** 2)
                           - 0.5 * np.log10(freq_sq + 107.7 ** 2)
                           - 0.5 * np.log10(freq_sq + 737.9 ** 2))
    weight = np.maximum(weight, min_db)

    return weight


def compute_gain(sound, fs, min_db=-80.0, mode='A_weighting'):
    if fs == 16000 or fs == 20000:
        n_fft = 2048
    elif fs == 44100:
        n_fft = 4096
    else:
        raise Exception('Invalid fs {}'.format(fs))
    stride = n_fft // 2

    gain = []
    # no xrange anymore supported
    for i in range(0, len(sound) - n_fft + 1, stride):
        if mode == 'RMSE':
            g = np.mean(sound[i: i + n_fft] ** 2)
        elif mode == 'A_weighting':
            spec = np.fft.rfft(np.hanning(n_fft + 1)[:-1] * sound[i: i + n_fft])
            power_spec = np.abs(spec) ** 2
            a_weighted_spec = power_spec * np.power(10, a_weight(fs, n_fft) / 10)
            g = np.sum(a_weighted_spec)
        else:
            raise Exception('Invalid mode {}'.format(mode))
        gain.append(g)

    gain = np.array(gain)
    gain = np.maximum(gain, np.power(10, min_db / 10))
    gain_db = 10 * np.log10(gain)

    return gain_db


def mix(sound1, sound2, r, fs):
    gain1 = np.max(compute_gain(sound1, fs))  # Decibel
    gain2 = np.max(compute_gain(sound2, fs))
    t = 1.0 / (1 + np.power(10, (gain1 - gain2) / 20.) * (1 - r) / r)
    sound = ((sound1 * t + sound2 * (1 - t)) / np.sqrt(t ** 2 + (1 - t) ** 2))

    return sound

# Convert time representation


def to_hms(time):
    h = int(time // 3600)
    m = int((time - h * 3600) // 60)
    s = int(time - h * 3600 - m * 60)
    if h > 0:
        line = '{}h{:02d}m'.format(h, m)
    else:
        line = '{}m{:02d}s'.format(m, s)

    return line


# Tested Method


class TestBCAugmentation(unittest.TestCase):

    def setUp(self):
        # 加载音频文件并初始化标签
        audio1_path = "bc_augmentation/unit_test" / pathlib.Path("./1-137-A-32.wav")
        audio2_path = "bc_augmentation/unit_test" / pathlib.Path("./1-1791-A-26.wav")
        assert audio1_path.exists(), audio1_path.resolve()
        assert audio2_path.exists(), audio2_path.resolve()

        self.audio1, _ = torchaudio.load(audio1_path)
        self.audio2, _ = torchaudio.load(audio2_path)
        self.label1 = torch.tensor([0, 1], dtype=torch.float)
        self.label2 = torch.tensor([1, 0], dtype=torch.float)
        self.sample_rate = 20000
        self.mix_ratio = 0.5  # 混合比例

    def test_mix_audio(self):
        # 使用已知的ACDNet实现的 mix 方法生成混合音频
        audio1_np = self.audio1.numpy().squeeze()  # 转换为 numpy 格式
        audio2_np = self.audio2.numpy().squeeze()
        acdnet_mixed_audio = mix(audio1_np, audio2_np, self.mix_ratio, self.sample_rate)

        # 使用您自己的实现生成混合音频
        audio1_torch = self.audio1.squeeze()  # 确保音频数据为 1D
        audio2_torch = self.audio2.squeeze()

        # mix_sounds(sound1, sound2, r, fs, device='cpu'):

        my_mixed_audio = my_mix_sounds(audio1_torch, audio2_torch, self.mix_ratio, self.sample_rate)

        # 比较两个混合音频的差异
        loss = torch.nn.MSELoss()
        with torch.no_grad():
            difference = loss(torch.from_numpy(acdnet_mixed_audio), my_mixed_audio)
        print(difference)
        self.assertAlmostEqual(difference, 0, delta=0.01, msg="The mixed audio waveforms differ.")

    def test_label(self):
        # 使用指定的混合比例生成标签
        mixed_label = self.label1 * self.mix_ratio + self.label2 * (1 - self.mix_ratio)

        # 检查混合标签的正确性
        expected_label = torch.tensor([0.5, 0.5], dtype=torch.float)
        self.assertTrue(torch.allclose(mixed_label, expected_label, atol=0.01), "The mixed label is incorrect.")


if __name__ == "__main__":
    unittest.main()
