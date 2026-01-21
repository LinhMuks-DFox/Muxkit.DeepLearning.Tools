import unittest

import pyroomacoustics
import torch

from audio_tools.PyroomSimulation import RoomSimulation, random_microphone_array_position


class TestPyRoomSimulation(unittest.TestCase):

    def setUp(self):
        # Create a simple mock room generator for testing
        self.material = pyroomacoustics.make_materials(**{
            "ceiling": "hard_surface",
            "floor": "brickwork",
            "east": "brickwork",
            "west": "brickwork",
            "north": "brickwork",
            "south": "brickwork",
        })

        self.n_mic = 3

    def make_room(self):
        mock_room = pyroomacoustics.ShoeBox([5, 4, 6], materials=self.material)
        mock_room.add_microphone_array(random_microphone_array_position(
            [5, 4, 6], n_mic=self.n_mic
        ))
        mock_room.add_source([1.5, 1.5, 1.5])
        return mock_room

    def test_room_simulation_forward(self):
        module = RoomSimulation(self.make_room, self.n_mic).float()
        # Example mono audio input
        audio = torch.randn(1, 16000, dtype=torch.float)

        # Run forward
        output = module(audio)
        self.assertEqual(output.shape[0], self.n_mic)  # Expected 2 mic signals


if __name__ == "__main__":
    unittest.main()
