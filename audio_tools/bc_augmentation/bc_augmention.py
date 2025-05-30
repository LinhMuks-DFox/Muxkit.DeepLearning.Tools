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

    length = sound.shape[-1]

    # 将音频分块
    num_segments = (length - n_fft) // stride + 1  # 计算总分段数
    frames = torch.as_strided(
        sound,
        size=(sound.shape[0], num_segments, n_fft),
        stride=(sound.stride(0), stride * sound.stride(-1), sound.stride(-1))
    ).to(device)

    if mode == 'A_weighting':
        windowed_frames = frames * torch.hann_window(n_fft, device=device).view(1, 1, n_fft)  # 应用窗函数
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
    """Mix two multi-channel audio signals channel-wise with gain adjustment for perceptual consistency."""
    assert sound1.shape == sound2.shape, "Input sounds must have the same shape [C, T]"
    if not (sound1.device == sound2.device == r.device == device):
        sound1 = sound1.to(device)
        sound2 = sound2.to(device)
        r = r.to(device)

    gain1 = compute_gain(sound1, fs, device=device)  # shape: [C, NumFrames]
    gain2 = compute_gain(sound2, fs, device=device)  # shape: [C, NumFrames]
    
    gain1_max = gain1.max(dim=-1, keepdim=True)[0]  # shape: [C, 1]
    gain2_max = gain2.max(dim=-1, keepdim=True)[0]  # shape: [C, 1]
    t = 1.0 / (1 + torch.pow(10, (gain1_max - gain2_max) / 20.0) * (1 - r) / r)  # shape: [C, 1]
    mixed_sound = (sound1 * t + sound2 * (1 - t)) / torch.sqrt(t ** 2 + (1 - t) ** 2)
    
    return mixed_sound


class BCAugmentor(torch.nn.Module):
    def __init__(self, sample_rate, device='cpu'):
        self.sample_rate = sample_rate
        self.device = device

    @torch.no_grad
    def mix_sounds(self, sound1, sound2, r):
        gain1 = compute_gain(sound1, self.sample_rate, device=self.device)  # shape: [C, NumFrames]
        gain2 = compute_gain(sound2, self.sample_rate, device=self.device)  # shape: [C, NumFrames]
        
        gain1_max = gain1.max(dim=-1, keepdim=True)[0]  # shape: [C, 1]
        gain2_max = gain2.max(dim=-1, keepdim=True)[0]  # shape: [C, 1]

        t = 1.0 / (1 + torch.pow(10, (gain1_max - gain2_max) / 20.0) * (1 - r) / r).view(-1, 1, 1)  # shape: [C, 1, 1]
        mixed_sound = (sound1 * t + sound2 * (1 - t)) / torch.sqrt(t ** 2 + (1 - t) ** 2)
        
        return mixed_sound

    def forward(self, sound1, sound2, r):
        return self.mix_sounds(sound1, sound2, r)
