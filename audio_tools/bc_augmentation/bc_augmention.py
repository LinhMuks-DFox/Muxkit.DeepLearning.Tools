'''
This module provides utility functions for Between-Class (BC) augmentation,
as described in the paper "Learning from Between-Class Examples for Deep Sound Recognition".

Key formulas:
1. **Mixed Audio Formula**:
   Given two audio samples \( x_1 \) and \( x_2 \) and a random mixing ratio \( r \), the mixed audio \( x_{mix} \) is calculated as:
   \[
   x_{mix} = \frac{r \cdot x_1 + (1 - r) \cdot x_2}{\sqrt{r^2 + (1 - r)^2}}
   \]

2. **Gain Adjustment**:
   To adjust for perceptual loudness, the gain-adjusted mixing coefficient \( p \) is computed using the gains \( G_1 \) and \( G_2 \) of \( x_1 \) and \( x_2 \):
   \[
   p = \frac{1}{1 + 10^{(G_1 - G_2)/20} \cdot \frac{1 - r}{r}}
   \]
   The adjusted mixed audio formula then becomes:
   \[
   x_{mix} = \frac{p \cdot x_1 + (1 - p) \cdot x_2}{\sqrt{p^2 + (1 - p)^2}}
   \]
"""

'''

import torch


def a_weight(fs, n_fft, min_db=-80.0, device='cpu'):
    """Compute the A-weighting filter for perceptual audio processing."""
    freq = torch.linspace(0, fs // 2, n_fft // 2 + 1, device=device)
    freq_sq = freq ** 2
    freq_sq[0] = 1.0  # Prevent division by zero
    weight = 2.0 + 20.0 * (2 * torch.log10(torch.tensor(12194.0)) + 2 * torch.log10(freq_sq)
                           - torch.log10(freq_sq + 12194 ** 2)
                           - torch.log10(freq_sq + 20.6 ** 2)
                           - 0.5 * torch.log10(freq_sq + 107.7 ** 2)
                           - 0.5 * torch.log10(freq_sq + 737.9 ** 2))
    weight = torch.clamp(weight, min=min_db)
    return weight


def compute_gain(sound: torch.Tensor, fs, min_db=-80.0, mode='A_weighting', device='cpu'):
    """Compute the gain (sound pressure level) of an audio signal with optimized batch processing."""
    n_fft = 2048 if fs in [16000, 20000] else 4096
    stride = n_fft // 2

    # 检查输入音频格式
    if sound.ndim >= 2:
        assert sound.shape[-2] == 1, "Currently, only support mono-track audio"
    length = sound.shape[-1]

    # 将音频分块
    num_segments = (length - n_fft) // stride + 1  # 计算总分段数
    frames = torch.as_strided(
        sound,
        size=(num_segments, n_fft),
        stride=(stride * sound.stride(-1), sound.stride(-1))
    ).to(device)

    if mode == 'A_weighting':
        windowed_frames = frames * torch.hann_window(n_fft, device=device)  # 应用窗函数
        spec = torch.fft.rfft(windowed_frames)  # 对所有片段同时执行FFT
        power_spec = torch.abs(spec) ** 2
        a_weighted_spec = power_spec * torch.pow(10, a_weight(fs, n_fft, device=device) / 10)
        gain = a_weighted_spec.sum(dim=-1)  # 对频率轴进行求和
    else:
        gain = (frames ** 2).mean(dim=-1)

    # 应用增益下限
    min_gain_value = torch.pow(torch.tensor(10.0, device=device), torch.tensor(min_db / 10, device=device))
    gain = torch.clamp(gain, min=min_gain_value)

    gain_db = 10 * torch.log10(gain)
    return gain_db


def mix_sounds(sound1, sound2, r, fs, device='cpu'):
    """Mix two audio signals with gain adjustment for perceptual consistency."""
    gain1 = compute_gain(sound1, fs, device=device).max()
    gain2 = compute_gain(sound2, fs, device=device).max()
    t = 1.0 / (1 + torch.pow(10, (gain1 - gain2) / 20.0) * (1 - r) / r)
    mixed_sound = (sound1 * t + sound2 * (1 - t)) / torch.sqrt(t ** 2 + (1 - t) ** 2)
    return mixed_sound


class BCAugmentor(torch.nn.Module):
    def __init__(self, sample_rate, device='cpu'):
        self.sample_rate = sample_rate
        self.device = device

    @torch.no_grad
    def mix_sounds(self, sound1, sound2, r):
        gain1 = compute_gain(sound1, self.sample_rate, device=self.device).max()
        gain2 = compute_gain(sound2, self.sample_rate, device=self.device).max()
        t = 1.0 / (1 + torch.pow(10, (gain1 - gain2) / 20.0) * (1 - r) / r)
        mixed_sound = (sound1 * t + sound2 * (1 - t)) / torch.sqrt(t ** 2 + (1 - t) ** 2)
        return mixed_sound

    def forward(self, sound1, sound2, r):
        return self.mix_sounds(sound1, sound2, r)
