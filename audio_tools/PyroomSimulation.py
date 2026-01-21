import typing

import numpy as np
import pyroomacoustics
import torch
import torch.nn as nn
import torchaudio.transforms as transforms

from ..utl import api_tags as tags

PyRoomAcousticRoomMaker = typing.Callable[[], pyroomacoustics.Room]


@tags.stable_api
class RoomSimulation(nn.Module):
    """Apply PyRoomAcoustics RIRs via torchaudio convolution.

    Args:
        make_room (Callable): Returns a configured ``pyroomacoustics.Room``.
        n_mic (int): Number of microphones.

    Forward:
        audio (Tensor): Shape [C, T]
        -> Tensor: Shape [n_mic, T]

    Notes:
        - RIR kernels are stored as non-trainable parameters.
        - Works on CPU and GPU.
    """

    def __init__(self, make_room, n_mic):
        super().__init__()
        self.rir = nn.Parameter(self.__init_rir(
            make_room), requires_grad=False)
        self.n_mic = n_mic
        self.convolver = transforms.Convolve(mode='full')

    @staticmethod
    def __init_rir(make_room):
        """Initialize RIR kernels from the provided room configuration.

        Raises:
            ValueError: If computed RIR data is empty or incomplete.
        """
        room = make_room()
        room.compute_rir()
        rir_arrays = room.rir
        if not rir_arrays or not all(rir_arrays):
            raise ValueError(
                "RIR data is empty or incomplete. Check room configuration and RIR computation.")
        # Pad all RIRs to the same length
        max_length = max(len(rir)
                         for mic_rirs in rir_arrays for rir in mic_rirs)
        padded_rirs = np.array(
            [np.pad(rir[0], (0, max_length - len(rir[0])), 'constant') for rir in rir_arrays])
        return torch.from_numpy(padded_rirs)

    def forward(self, audio):
        """Convolve waveform with stored RIR kernels using torchaudio."""
        return self.convolver(audio, self.rir)


def random_microphone_array_position(space_size: typing.List[float], n_mic: int) -> np.ndarray:
    """
    Generate random positions for microphones in a given space.

    Args:
        space_size (List[float]): The size of the space (1D, 2D, or 3D).
        n_mic (int): The number of microphones.

    Returns:
        np.ndarray: A matrix of shape (n_mic, space_dimensions) representing the microphone positions.
    """
    assert len(space_size) in [1, 2, 3], "Space dimensions must be 1, 2, or 3"
    min_coords = [0.0 for _ in range(len(space_size))]
    max_coords = space_size
    coords = torch.FloatTensor(n_mic, len(space_size))
    for i in range(len(space_size)):
        coords[:, i] = torch.FloatTensor(
            n_mic).uniform_(min_coords[i], max_coords[i])
    return coords.double().numpy().T
