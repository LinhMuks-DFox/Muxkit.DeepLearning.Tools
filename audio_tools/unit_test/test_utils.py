import unittest
import torch
from audio_tools.utils import fix_length

class TestAudioToolsUtils(unittest.TestCase):

    def test_fix_length_truncate(self):
        """
        Test if fix_length correctly truncates the input when it's longer than the target length.
        """
        # Create a tensor longer than the target length
        audio_data = torch.randn(1, 2000)
        target_length = 1000
        
        # Apply fix_length
        output = fix_length(audio_data, target_length)
        
        # Assert the length is truncated to the target length
        self.assertEqual(output.shape[-1], target_length)
    
    def test_fix_length_pad(self):
        """
        Test if fix_length correctly pads the input when it's shorter than the target length.
        """
        # Create a tensor shorter than the target length
        audio_data = torch.randn(1, 800)
        target_length = 1000
        
        # Apply fix_length
        output = fix_length(audio_data, target_length)
        
        # Assert the length is padded to the target length
        self.assertEqual(output.shape[-1], target_length)
    
    def test_fix_length_no_change(self):
        """
        Test if fix_length leaves the input unchanged when it's equal to the target length.
        """
        # Create a tensor with the same length as the target length
        audio_data = torch.randn(1, 1000)
        target_length = 1000
        
        # Apply fix_length
        output = fix_length(audio_data, target_length)
        
        # Assert the length remains the same
        self.assertEqual(output.shape[-1], target_length)
    
    def test_fix_length_shape(self):
        """
        Test if fix_length maintains the correct shape (n_channel, n_sample).
        """
        # Create a tensor with multiple channels
        audio_data = torch.randn(2, 1200)  # 2 channels
        target_length = 1000
        
        # Apply fix_length
        output = fix_length(audio_data, target_length)
        
        # Assert the shape has 2 channels and the correct number of samples
        self.assertEqual(output.shape, (2, target_length))

if __name__ == '__main__':
    unittest.main()
