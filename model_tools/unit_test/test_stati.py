import unittest

import torch

from model_tools.stati import freeze_module, unfreeze_module, get_size, stati_model


class TestModelToolsStati(unittest.TestCase):

    def test_freeze_module(self):
        """
        Test if freeze_module correctly sets requires_grad to False.
        """
        # Create a simple model
        model = torch.nn.Linear(10, 10)
        freeze_module(model)

        # Check that all parameters are frozen
        for param in model.parameters():
            self.assertFalse(param.requires_grad)

    def test_unfreeze_module(self):
        """
        Test if unfreeze_module correctly sets requires_grad to True.
        """
        # Create a simple model and freeze it first
        model = torch.nn.Linear(10, 10)
        freeze_module(model)

        # Unfreeze the model
        unfreeze_module(model)

        # Check that all parameters are unfrozen
        for param in model.parameters():
            self.assertTrue(param.requires_grad)

    def test_get_size(self):
        """
        Test if get_size correctly calculates the size of a tensor in bytes.
        """
        # Create a simple tensor
        tensor = torch.randn(10, 10)
        expected_size = tensor.nelement() * tensor.element_size()

        # Check if get_size returns the correct value
        self.assertEqual(get_size(tensor), expected_size)

    def test_stati_model(self):
        """
        Test if stati_model correctly returns model statistics, including parameter counts and model size.
        """
        # Create a simple model
        model = torch.nn.Sequential(
            torch.nn.Linear(100, 200),
            torch.nn.ReLU(),
            torch.nn.Linear(200, 400)
        )

        # Get the model stats in megabytes (mb)
        stats = stati_model(model, unit="mb")

        # Calculate expected parameter counts
        param_count_with_grad = sum(p.numel() for p in model.parameters() if p.requires_grad)
        param_count_without_grad = sum(p.numel() for p in model.parameters() if not p.requires_grad)

        # Assert the parameter counts
        self.assertEqual(stats["param_count_with_grad"], param_count_with_grad)
        self.assertEqual(stats["param_count_without_grad"], param_count_without_grad)

        # Check if model size is in the expected range (since it's in mb, it won't be exact)
        self.assertTrue(stats["model_size"] > 0)


if __name__ == '__main__':
    unittest.main()
