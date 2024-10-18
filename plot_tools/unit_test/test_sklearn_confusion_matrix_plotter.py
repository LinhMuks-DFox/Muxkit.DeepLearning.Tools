import unittest
import numpy as np
from unittest.mock import MagicMock, patch
from plot_tools.SklearnConfusionMatrixPlotter import \
    ConfusionMatrixPlotter  # Adjust the import path to match your project structure
import matplotlib.pyplot as plt  # Import matplotlib directly


class TestConfusionMatrixPlotter(unittest.TestCase):

    def setUp(self):
        # Setup the mock compose path function
        self.mock_compose_path = lambda x: f"{x}"
        # Create a class to label mapping
        self.class2label = {str(i): {"display_name": f"Class_{i}"} for i in range(5)}
        # Initialize the plotter with the mock compose path function
        self.plotter = ConfusionMatrixPlotter(class2label=self.class2label, compose_path_func=self.mock_compose_path)

    def test_plot_multi_label_confusion_matrix(self):
        confusion_matrix = np.array([[[100, 10], [5, 85]] for _ in range(5)])
        self.plotter.plot_sklearn_multi_label_confusion_matrix(confusion_matrix, prefix="None")


if __name__ == '__main__':
    unittest.main()
