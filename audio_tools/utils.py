"""
Audio utility helpers used across training and data pipelines.

Currently includes length adjustment for tensors representing audio
waveforms. Behaviors unchanged; documentation clarified.
"""

import torch

from ..utl import api_tags as tags


@tags.stable_api
def fix_length(audio_data: torch.Tensor, sample_length: int) -> torch.Tensor:
    """Pad or truncate an audio tensor on the time axis to ``sample_length``.

    Args:
        audio_data (Tensor): Shape [..., T] or [C, T].
        sample_length (int): Target length along the last dimension.

    Returns:
        Tensor: Tensor with last dimension exactly ``sample_length``.
    """
    current_length = audio_data.shape[-1]
    if current_length > sample_length:
        audio_data = audio_data[..., :sample_length]
    elif current_length < sample_length:
        padding = sample_length - current_length
        audio_data = torch.nn.functional.pad(audio_data, (0, padding))
    return audio_data
