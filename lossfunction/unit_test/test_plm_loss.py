import unittest
import torch

from lossfunction.PLMLoss import PLMLoss


class TestPLMLoss(unittest.TestCase):

    def test_forward_returns_scalar(self):
        C = 3
        N = 5
        label_dist = torch.full((C,), 0.2)
        loss_fn = PLMLoss(label_distribution=label_dist,
                          lambda_rate=0.1, hist_bins=5, loss_kernel="bce")
        y_pred = torch.randn(N, C)
        y_true = torch.randint(0, 2, (N, C)).float()
        out = loss_fn(y_pred, y_true)
        self.assertIsInstance(out.item(), float)

    def test_update_ratios_changes_ideal(self):
        C = 2
        label_dist = torch.tensor([0.5, 0.5])
        loss_fn = PLMLoss(label_distribution=label_dist,
                          lambda_rate=0.5, hist_bins=3, loss_kernel="bce")
        # Manually set histograms to create a positive difference
        loss_fn.hist_pos_true = torch.tensor([[1., 0.], [0., 1.], [0., 0.]])
        loss_fn.hist_pos_pred = torch.tensor([[0., 1.], [1., 0.], [0., 0.]])
        loss_fn.hist_neg_true = torch.zeros_like(loss_fn.hist_pos_true)
        loss_fn.hist_neg_pred = torch.zeros_like(loss_fn.hist_pos_true)
        before = loss_fn.positive_ratio_ideal.clone()
        loss_fn.update_ratios()
        after = loss_fn.positive_ratio_ideal
        self.assertFalse(torch.allclose(before, after))
