import unittest
import torch
from torch.utils.data import DataLoader, TensorDataset

from score_tools.ClassifierTester import MonoLabelClassificationTester


class SimpleModel(torch.nn.Module):
    def __init__(self, in_features=4, num_classes=3):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, num_classes)
    def forward(self, x):
        return self.linear(x)


class TestMonoLabelClassificationTester(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        X = torch.randn(20, 4)
        y = torch.randint(0, 3, (20,))
        self.loader = DataLoader(TensorDataset(X, y), batch_size=5)
        self.model = SimpleModel()
        self.device = torch.device('cpu')
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

    def test_predict_and_metrics(self):
        tester = MonoLabelClassificationTester(self.model, self.device, self.loss_fn)
        tester.set_dataloader(self.loader, n_class=3)
        tester.predict_all()
        tester.calculate_all_metrics()
        status = tester.status_map()
        self.assertIn('accuracy', status)
        self.assertIn('f1_score', status)
        report = tester.classification_report()
        self.assertIsInstance(report, str)

