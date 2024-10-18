import unittest
import torch
import numpy as np
from sklearn.metrics import accuracy_score
from score_tools.ClassifierTester import ClassifierTester 
# 模拟的模型
class DummyModel(torch.nn.Module):
    def __init__(self, output_size):
        super(DummyModel, self).__init__()
        self.linear = torch.nn.Linear(10, output_size)

    def forward(self, x):
        return self.linear(x)

class TestClassifierTester(unittest.TestCase):
    def setUp(self):
        # 设置单标签模型
        self.single_label_model = DummyModel(output_size=3)  # 3类分类问题
        self.device = torch.device("cpu")
        self.single_label_tester = ClassifierTester(self.single_label_model, self.device, n_classes=3)

        # 设置多标签模型
        self.multi_label_model = DummyModel(output_size=3)  # 假设有3个标签
        self.multi_label_tester = ClassifierTester(self.multi_label_model, self.device, n_classes=3, sigmoid_before_thresholding=True)

        # 模拟的数据
        self.batch_size = 5
        self.num_features = 10
        self.num_classes = 3

        self.single_label_data = torch.randn(self.batch_size, self.num_features)
        self.single_label_target = torch.randint(0, self.num_classes, (self.batch_size,))

        self.multi_label_data = torch.randn(self.batch_size, self.num_features)
        self.multi_label_target = torch.randint(0, 2, (self.batch_size, self.num_classes))  # 假设二进制标签

        # 模拟的dataloader
        self.single_label_dataloader = [(self.single_label_data, self.single_label_target)]
        self.multi_label_dataloader = [(self.multi_label_data, self.multi_label_target)]

    def test_single_label_prediction(self):
        """测试单标签分类的预测"""
        self.single_label_tester.set_dataloader(self.single_label_dataloader)
        self.single_label_tester.predict_all(multi_label=False)

        self.assertEqual(self.single_label_tester.y_true_.shape[0], self.batch_size)
        self.assertEqual(self.single_label_tester.y_predict_.shape[0], self.batch_size)

    def test_single_label_metrics(self):
        """测试单标签分类的指标计算"""
        self.single_label_tester.set_dataloader(self.single_label_dataloader)
        self.single_label_tester.predict_all(multi_label=False)
        self.single_label_tester.calculate_all_metrics(multi_label=False)

        self.assertIsNotNone(self.single_label_tester.accuracy_)
        self.assertIsNotNone(self.single_label_tester.precision_)
        self.assertIsNotNone(self.single_label_tester.recall_)
        self.assertIsNotNone(self.single_label_tester.f1_score_)

    def test_multi_label_prediction(self):
        """测试多标签分类的预测"""
        self.multi_label_tester.set_dataloader(self.multi_label_dataloader)
        self.multi_label_tester.predict_all(multi_label=True)

        self.assertEqual(self.multi_label_tester.y_true_.shape[0], self.batch_size)
        self.assertEqual(self.multi_label_tester.y_predict_binary_.shape[0], self.batch_size)

    def test_multi_label_metrics(self):
        """测试多标签分类的指标计算"""
        self.multi_label_tester.set_dataloader(self.multi_label_dataloader)
        self.multi_label_tester.predict_all(multi_label=True)
        self.multi_label_tester.calculate_all_metrics(multi_label=True)

        self.assertIsNotNone(self.multi_label_tester.accuracy_)
        self.assertIsNotNone(self.multi_label_tester.precision_)
        self.assertIsNotNone(self.multi_label_tester.recall_)
        self.assertIsNotNone(self.multi_label_tester.f1_score_)

    def test_hamming_loss(self):
        """测试Hamming损失"""
        self.multi_label_tester.set_dataloader(self.multi_label_dataloader)
        self.multi_label_tester.predict_all(multi_label=True)
        self.multi_label_tester.calculate_hamming_loss()

        self.assertIsNotNone(self.multi_label_tester.hamming_loss_)

    def test_classification_report(self):
        """测试分类报告"""
        self.single_label_tester.set_dataloader(self.single_label_dataloader)
        self.single_label_tester.predict_all(multi_label=False)
        report = self.single_label_tester.classification_report(multi_label=False)
        self.assertIsInstance(report, str)

        self.multi_label_tester.set_dataloader(self.multi_label_dataloader)
        self.multi_label_tester.predict_all(multi_label=True)
        report_multi = self.multi_label_tester.classification_report(multi_label=True)
        self.assertIsInstance(report_multi, str)

if __name__ == '__main__':
    unittest.main()
