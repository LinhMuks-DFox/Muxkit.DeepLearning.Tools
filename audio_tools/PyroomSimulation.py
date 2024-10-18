import typing
import numpy as np
import pyroomacoustics
import torch
import torch.nn as nn
import torchaudio.transforms as transforms
import api_tags as tags

PyRoomAcousticRoomMaker = typing.Callable[[], pyroomacoustics.Room]

@tags.stable_api
class TorchRoomAcoustic(nn.Module):
    """
    A PyTorch module that integrates PyRoomAcoustics for room simulation 
    and torchaudio for applying the resulting room simulation on audio data.

    Args:
        new_room_creator (Callable): A callable that creates a PyRoomAcoustics Room object.
        source_location (List[float]): The location of the sound source in the room.

    Forward Input:
        sample (torch.Tensor): A tensor containing the audio sample to simulate. Expected to be 1D.

    Forward Output:
        torch.Tensor: Simulated microphone signals after room simulation.
    """
    
    def __init__(self,
                 new_room_creator: PyRoomAcousticRoomMaker,
                 source_location: typing.List[float],
                 ):
        super().__init__()
        self.room_creator = new_room_creator
        self.source_location = source_location

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to simulate the room acoustics for the given audio sample.

        Args:
            sample (torch.Tensor): Audio tensor to simulate, expected as [n_sample].

        Returns:
            torch.Tensor: The resulting microphone signals after room simulation.
        """
        # Convert the sample to numpy for PyRoomAcoustics compatibility
        sample = sample.detach().numpy().flatten()
        room = self.room_creator()
        room.add_source(self.source_location, signal=sample)
        room.simulate()
        # Convert the room's mic array signals back to torch.Tensor
        output = torch.tensor(room.mic_array.signals)
        return output

@tags.untested
class RoomSimulation(nn.Module):
    """
    A PyTorch module to apply Room Impulse Response (RIR) convolution to an audio sample 
    using precomputed RIR kernels from PyRoomAcoustics and torchaudio for GPU/CPU accelerated convolution.

    Args:
        make_room (Callable): A callable that returns a computable PyRoomAcoustics Room object.
        n_mic (int): The number of microphones in the room setup.

    Forward Input:
        audio (torch.Tensor): A tensor representing the audio input to apply the RIR. Shape: [n_channel, n_sample].

    Forward Output:
        torch.Tensor: A tensor after applying RIR convolution, with shape [n_mic, n_sample].

    Notes:
        - The RIR kernels are stored as non-updatable torch tensors.
        - Convolution can be performed on both GPU and CPU.
    """
    
    def __init__(self, make_room, n_mic):
        super().__init__()
        self.rir = nn.Parameter(self.__init_rir(make_room), requires_grad=False)
        self.n_mic = n_mic
        self.convolver = transforms.Convolve(mode='full')

    @staticmethod
    def __init_rir(make_room):
        """
        Initializes the RIR kernels from the given room configuration.

        Args:
            make_room (Callable): A function that returns a PyRoomAcoustics Room object.

        Returns:
            torch.Tensor: A tensor containing the RIR kernels for each microphone.

        Raises:
            ValueError: If the RIR computation returns incomplete or empty data.
        """
        room = make_room()
        room.compute_rir()
        rir_arrays = room.rir
        if not rir_arrays or not all(rir_arrays):
            raise ValueError("RIR data is empty or incomplete. Check room configuration and RIR computation.")
        # Pad all RIRs to the same length
        max_length = max(len(rir) for mic_rirs in rir_arrays for rir in mic_rirs)
        padded_rirs = np.array([np.pad(rir[0], (0, max_length - len(rir[0])), 'constant') for rir in rir_arrays])
        return torch.from_numpy(padded_rirs)

    def forward(self, audio):
        """
        Applies the RIR convolution to the input audio using torchaudio's GPU-accelerated convolution.

        Args:
            audio (torch.Tensor): A tensor representing the audio input, with shape [n_channel, n_sample].

        Returns:
            torch.Tensor: The resulting audio tensor after applying the RIR convolution. Shape: [n_mic, n_sample].
        """
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
        coords[:, i] = torch.FloatTensor(n_mic).uniform_(min_coords[i], max_coords[i])
    return coords.double().numpy().T