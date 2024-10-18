import torch
import pyroomacoustics
import unittest
from audio_tools import TorchRoomAcoustic, RoomSimulation

class TestPyRoomSimulation(unittest.TestCase):

    def setUp(self):
        # Create a simple mock room generator for testing
        self.mock_room = pyroomacoustics.Room.shoeBox([5, 4, 6])
        self.mock_room.compute_rir = lambda: None
        self.mock_room.rir = [[[np.random.randn(100)] for _ in range(2)]]
        self.mock_room.mic_array = pyroomacoustics.MicrophoneArray(np.random.randn(2, 3), self.mock_room.fs)
        self.mock_room.simulate = lambda: None

    def test_torch_room_acoustic_forward(self):
        # Mock room generator
        room_creator = lambda: self.mock_room
        module = TorchRoomAcoustic(room_creator, [1, 1, 1])
        sample = torch.randn(16000)  # Example 1 second sample at 16kHz

        # Run forward
        output = module(sample)
        self.assertEqual(output.shape[0], 2)  # Expected 2 mic signals

    def test_room_simulation_forward(self):
        room_creator = lambda: self.mock_room
        module = RoomSimulation(room_creator, 2)
        audio = torch.randn(1, 16000)  # Example mono audio input

        # Run forward
        output = module(audio)
        self.assertEqual(output.shape[0], 2)  # Expected 2 mic signals

if __name__ == "__main__":
    unittest.main()
