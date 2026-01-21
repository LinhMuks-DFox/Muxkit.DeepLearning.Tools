"""
Common audio transforms: crop/fix length, channel selection, and masking helpers.

These utilities are used in training pipelines. Behavior remains unchanged;
docstrings now clarify inputs/outputs and usage examples.
"""

import random
import torchvision.transforms as vision_transforms
import torchvision.transforms.functional as vision_transform_fnc
from typing import Union, List, Optional, Tuple
import numpy as np
import torch


class TimeSequenceLengthFixer(torch.nn.Module):
    """Fix the waveform length to a target duration.

    Example:
        >>> import torchaudio
        >>> wav_form, sample_rate = torchaudio.load("test.wav")
        >>> fixer = TimeSequenceLengthFixer(5, sample_rate)
        >>> fixed_wav_form = fixer(wav_form)
        >>> fixed_wav_form.shape
    """
    _FIX_MODE = {
        "random", "r", "random-timezone",
        "s", "start",
        "e", "end"
    }

    def __init__(self, fixed_length: int, sample_rate: int, mode="r"):
        super().__init__()
        if mode not in self._FIX_MODE:
            raise ValueError(f"Invalid mode:{mode}, mode shall be {self._FIX_MODE}")
        self.mode_ = mode
        self.fixed_length = int(fixed_length * sample_rate)

    def forward(self, audio_data: torch.Tensor) -> torch.Tensor:
        """Return a slice or padded waveform of exact target length.

        Args:
            audio_data (Tensor): Shape [C, T].
        Returns:
            Tensor: Shape [C, fixed_length*sample_rate].
        """
        if self.mode_ in {"r", "random", "random-timezone"}:
            if audio_data.shape[1] < self.fixed_length:
                return self.select_time_zone(audio_data, 0)[0]
            return self.select_time_zone(audio_data,
                                         random.randint(0, audio_data.shape[1] - self.fixed_length)
                                         )[0]
        if self.mode_ in {"s", "start"}:
            return self.select_time_zone(audio_data, 0)[0]
        if self.mode_ in {"e", "end"}:
            return self.select_time_zone(audio_data, audio_data.shape[1] - self.fixed_length)[0]
        else:
            raise ValueError(f"Invalid mode:{self.mode_}")

    def select_time_zone(self, audio_data: torch.Tensor, start_time: int):
        """Helper to slice at ``start_time`` with right padding when needed."""
        if audio_data.shape[1] < self.fixed_length:
            audio_data = torch.nn.functional.pad(audio_data,
                                                 (0, self.fixed_length - audio_data.shape[1]))
        return audio_data[..., start_time: start_time + self.fixed_length], start_time


class SoundTrackSelector(torch.nn.Module):
    """Select or mix channels from stereo/multi-channel audio.

    Example:
        >>> import torchaudio
        >>> wav_form, sample_rate = torchaudio.load("test.wav")
        >>> selector = SoundTrackSelector("left")
        >>> selected_wav_form = selector(wav_form)
    """
    _VALID_TRACKS = {"all", "left", "right", "mix", "random-single", "random"}
    _MOD_SELECT_KERNEL = {
        "all": lambda x: x,
        "left": lambda x: x[0].unsqueeze(0),
        "right": lambda x: x[1].unsqueeze(0),
        "mix": lambda x: x.mean(dim=0).unsqueeze(0),
    }

    def __init__(self, mode: str):
        super().__init__()
        if mode not in self._VALID_TRACKS:
            raise ValueError(f"mode must be one of {', '.join(self._VALID_TRACKS)}, but got {mode}")
        if mode == "random":
            mode = random.choice(["all", "left", "right", "mix"])
        if mode == "random-single":
            mode = random.choice(["left", "right", "mix"])
        self.select_mode = mode
        self.select_kernel = self._MOD_SELECT_KERNEL[mode]

    def forward(self, audio_data: torch.Tensor) -> torch.Tensor:
        return self.select_kernel(audio_data)


def create_mask(size: Union[torch.Tensor, torch.Size, List[int]],
                mask_rate: float = 0.5) -> torch.Tensor:
    """Return a random boolean mask with given mask rate.

    Args:
        size (Tensor|Size|List[int]): Output shape.
        mask_rate (float): Probability of True.
    """
    mask = torch.randn(*size) > mask_rate
    return mask


def create_mask_chunk_2d(size: torch.Size,
                         mask_rate: float = 0.5, b: int = 8) -> torch.Tensor:
    """
    1. Create a mini mask with the given (size // b)
        when b = 2, size=(4*4), mini_mask be-like:
        [[1 0]
         [0 1]]
    2. resize mini mask, up sample it to the size, with mode nearest
        [[1 0]       [[1 1 0 0]
         [0 1]]  -->  [1 1 0 0]
                      [0 0 1 1]
                      [0 0 1 1]]
    3. return the mask
    Args:
        size (torch.Size): Target 2D size.
        mask_rate (float): Probability of True in mini mask.
        b (int): Downsample factor for mini mask.
    Returns:
        Tensor: Boolean mask of shape ``size``.
    """
    mini_mask_size: torch.Tensor = torch.div(torch.tensor(size, dtype=torch.int), b, rounding_mode="floor")
    mini_mask: torch.Tensor = create_mask(mini_mask_size, mask_rate)
    size: list = [*size]
    mask: torch.Tensor = vision_transform_fnc.resize(img=mini_mask.unsqueeze(0), size=size,
                                                     interpolation=vision_transforms.InterpolationMode.NEAREST)
    return mask.squeeze(0)


def tensor_masking(tensor_to_mask: torch.Tensor,
                   mask_rate: float = .5, mask: Optional = None) -> \
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply a boolean mask to split a tensor into masked and unmasked parts.

    Example:
        Mask              [T, F, T, F]
        Original          [1, 2, 3, 4]
        Masked (keep T)   [1, 0, 3, 0]
        Unmasked (keep F) [0, 2, 0, 4]

    Args:
        tensor_to_mask (Tensor): Input tensor (NumPy path used for convenience).
        mask_rate (float): Probability of True if ``mask`` not provided.
        mask (Tensor|None): Boolean mask; if None, a random mask is created.
    Returns:
        Tuple[Tensor, Tensor, Tensor]: (masked, unmasked, mask)
    """
    if mask is None:
        mask = create_mask(tensor_to_mask.shape, mask_rate)
    return torch.tensor(np.where(mask, tensor_to_mask.numpy(), 0)), \
        torch.tensor(np.where(~mask, tensor_to_mask.numpy(), 0)), mask


class TimeSequenceMaskingTransformer(torch.nn.Module):
    """
    Mask the time sequence of audio data.
    >>> import torchaudio
    >>> wav_form, sample_rate = torchaudio.load("test.wav")
    >>> masker = TimeSequenceMaskingTransformer(0.1)
    >>> masked_wav_form, unmasked_wav_form = masker(wav_form)
    >>> masked_wav_form.shape
    >>> unmasked_wav_form.shape
    """

    def __init__(self, mask_rate: float):
        self.mask_rate_ = mask_rate
        super().__init__()

    def forward(self, audio_data: torch.Tensor):
        masked_audio_data, unmasked_audio_data, mask = tensor_masking(audio_data, self.mask_rate_)
        return masked_audio_data, unmasked_audio_data


class SpectrogramMaskingTransformer(torch.nn.Module):
    """
    Create a masked spectrogram.
    >>> import torchaudio
    >>> wav_form, sample_rate = torchaudio.load("test.wav")
    >>> masker = SpectrogramMaskingTransformer(0.1)
    >>> masked_wav_form, unmasked_wav_form = masker(wav_form)
    >>> masked_wav_form.shape
    >>> unmasked_wav_form.shape
    >>> masker.get_mask().shape
    """

    def __init__(self, mask_rate: float):
        self.mask_rate_ = mask_rate
        super().__init__()
        self.prev_mask_ = None

    def forward(self, audio_data: torch.Tensor):
        mask = create_mask_chunk_2d(audio_data.shape[1:], self.mask_rate_)
        masked_audio_data, unmasked_audio_data, mask = tensor_masking(audio_data, self.mask_rate_, mask=mask)
        self.prev_mask_ = mask
        return masked_audio_data, unmasked_audio_data

    def get_mask(self):
        return self.prev_mask_


if __name__ == "__main__":
    time_fixer = TimeSequenceLengthFixer(5, 16000)
    print(time_fixer(torch.randn([1, 160000])).shape)
