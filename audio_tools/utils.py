import torch

from ..utl import api_tags as tags


@tags.stable_api
def fix_length(audio_data: torch.Tensor, sample_length: int) -> torch.Tensor:
    """
    Adjust the length of the input audio data to match the given sample length.
    
    Args:
        audio_data (torch.Tensor): Input audio tensor.
        sample_length (int): The target sample length.
    
    Returns:
        torch.Tensor: Adjusted audio data, either padded or truncated to the specified length.
    """
    current_length = audio_data.shape[-1]
    if current_length > sample_length:
        audio_data = audio_data[..., :sample_length]
    elif current_length < sample_length:
        padding = sample_length - current_length
        audio_data = torch.nn.functional.pad(audio_data, (0, padding))
    return audio_data

