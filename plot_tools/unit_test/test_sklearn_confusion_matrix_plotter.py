import unittest
import numpy as np
import os
import tempfile
from unittest.mock import MagicMock
from plot_tools.SklearnConfusionMatrixPlotter import ConfusionMatrixPlotter

class TestConfusionMatrixPlotter(unittest.TestCase):

    def setUp(self):
        # Create a mock function to simulate path composition
        self.mock_compose_path = MagicMock(return_value=tempfile.mkdtemp())
        self.class2label = {str(i): {"display_name": f"Class_{i}"} for i in range(5)}
        self.plotter = ConfusionMatrixPlotter(class2label=self.class2label, compose_path_func=self.mock_compose_path)

    def test_plot_sklearn_multi_label_confusion_matrix(self):
        """
        Test plotting multi-label confusion matrix with shape [n_class, 2, 2].
        """
        # Create a mock confusion matrix of shape [5, 2, 2] (5 classes)
        confusion_matrix = np.array([[[100, 10], [5, 85]] for _ in range(5)])
        
        # Call the plotting method
        self.plotter.plot_sklearn_multi_label_confusion_matrix(confusion_matrix, prefix="test_multilabel", n_rows=2, n_cols=2)
        
        # Check if the path composition function was called
        self.mock_compose_path.assert_called()
        
    def test_plot_sklearn_multi_class_confusion_matrix(self):
        """
        Test plotting multi-class confusion matrix with shape [n_class, n_class].
        """
        # Create a mock confusion matrix of shape [5, 5]
        confusion_matrix = np.random.randint(0, 100, (5, 5))
        
        # Call the plotting method
        self.plotter.plot_sklearn_multi_label_confusion_matrix(confusion_matrix, prefix="test_multiclass")
        
        # Check if the path composition function was called
        self.mock_compose_path.assert_called()

    def test_invalid_confusion_matrix_shape(self):
        """
        Test passing an invalid confusion matrix shape.
        """
        # Create a confusion matrix with an invalid shape
        invalid_confusion_matrix = np.random.randint(0, 100, (3, 3, 3))  # Invalid shape
        
        # Check if ValueError is raised for invalid shape
        with self.assertRaises(ValueError):
            self.plotter.plot_sklearn_multi_label_confusion_matrix(invalid_confusion_matrix, prefix="test_invalid")

if __name__ == '__main__':
    unittest.main()
