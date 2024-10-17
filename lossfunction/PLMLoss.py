import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import typing



class PLMLoss(nn.Module):
    """
        Partial Label Masking Loss (PLMLoss) function:

        This class implements the Partial Label Masking (PLM) loss as proposed in the paper 
        "PLM: Partial Label Masking for Imbalanced Multi-label Classification" by Kevin Duarte et al.
        The PLM loss addresses the problem of imbalanced multi-label classification by adaptively 
        masking labels based on the class distributions and their imbalance.

        Parameters:
            label_distribution (torch.Tensor): A tensor representing the distribution of positive labels 
                                            for each class, which is used to adjust the mask.
            lambda_rate (float): The learning rate to update the positive ratio during the training 
                                process. Default is 0.1.
            hist_bins (int): The number of bins used to create histograms for estimating label distribution.
                            Default is 10.
            loss_kernel (str): The loss function to use, either "bce" (binary cross-entropy) or "cross_entropy". 
                            Default is "bce".

        Raises:
            TypeError: If the loss kernel is not "bce" or "cross_entropy".

        Returns:
            A scalar loss value after applying the Partial Label Masking (PLM) method.

        Functions:
            forward(y_pred, y_true):
                Computes the masked loss by applying the PLM mechanism on the predicted and true labels.
            
            generate_mask(y_true):
                Generates a mask based on the ratio of positive and negative samples in relation to 
                their ideal distributions. The mask is used to weight the loss.
            
            update_ratios():
                Updates the positive ratios based on the difference between the predicted and true 
                label distributions, adjusting the imbalance.
            
            _clear_probability_histogram():
                Resets the histograms that track the true and predicted label distributions.
            
            _compute_probabilities_difference():
                Computes the difference between true and predicted probabilities using KL divergence.
            
            _compute_kl_divergence(hist_true, hist_pred):
                Calculates the Kullback-Leibler divergence between two histograms, normalized to form 
                probability distributions.

            _compute_histogram(y_true, y_pred):
                Computes histograms for positive and negative samples, for both true and predicted labels, 
                across all classes.

            __str__():
                Provides a string representation of the PLMLoss object, including the lambda rate, number of bins, 
                and the loss function used.
        """

    _LOSS_KERNEL_ = {
        "bce" : F.binary_cross_entropy_with_logits,
        "cross_entropy": F.cross_entropy
    }
    def __init__(self,
                 label_distribution: torch.Tensor,
                 lambda_rate=0.1,
                 hist_bins=10,
                 loss_kernel: str = "bce"):
        super().__init__()
        self.loss_kernel = loss_kernel
        if self.loss_kernel not in self._LOSS_KERNEL_.keys():
            raise TypeError("Only bce or cross entropy is valid")
        self.loss_function = self._LOSS_KERNEL_.get(self.loss_kernel)
        self.positive_ratio: torch.Tensor = nn.Parameter(
            label_distribution.clone().detach(), requires_grad=False)
        self.positive_ratio_ideal: torch.Tensor = nn.Parameter(
            label_distribution.clone().detach(), requires_grad=False)
        self.change_rate = lambda_rate
        self.n_bins = hist_bins
        self.n_classes = self.positive_ratio.shape[0]
        self._clear_probability_histogram()
        self.store_hist = True

    def _clear_probability_histogram(self):
        self.hist_pos_true = torch.zeros(self.n_bins, self.n_classes)
        self.hist_pos_pred = torch.zeros(self.n_bins, self.n_classes)
        self.hist_neg_true = torch.zeros(self.n_bins, self.n_classes)
        self.hist_neg_pred = torch.zeros(self.n_bins, self.n_classes)

    def forward(self, y_pred, y_true):
        if self.store_hist:
            (
                hist_pos_true,
                hist_neg_true,
                hist_pos_pred,
                hist_neg_pred,
            ) = self._compute_histogram(y_true, y_pred)
            self.hist_pos_true += hist_pos_true
            self.hist_pos_pred += hist_pos_pred
            self.hist_neg_true += hist_neg_true
            self.hist_neg_pred += hist_neg_pred
        loss = self.loss_function(
            y_pred, y_true, reduction='none')
        mask = self.generate_mask(y_true)
        loss *= mask
        return loss.mean()

    def generate_mask(self, y_true):
        # p = torch.ones_like(y_true, device=y_true.device, dtype=y_true.dtype)
        # _y_true = y_true.detach().clone().to(torch.int32)
        # p[(_y_true == 1) & (self.positive_ratio > self.positive_ratio_ideal)] = self.positive_ratio_ideal / self.positive_ratio
        # p[(_y_true == 0) & (self.positive_ratio < self.positive_ratio_ideal)] = self.positive_ratio / self.positive_ratio_ideal
        # mask = torch.bernoulli(p).to(y_true.device)

        p = torch.ones_like(
            y_true, device=y_true.device, dtype=y_true.dtype)
        _y_true = y_true.detach().clone().to(torch.int32)
        mask_pos = (_y_true == 1) & (self.positive_ratio.unsqueeze(
            0) > self.positive_ratio_ideal.unsqueeze(0))
        mask_neg = (_y_true == 0) & (self.positive_ratio.unsqueeze(
            0) < self.positive_ratio_ideal.unsqueeze(0))
        pos_prob = (self.positive_ratio_ideal /
                    self.positive_ratio).expand_as(p)
        neg_prob = (self.positive_ratio /
                    self.positive_ratio_ideal).expand_as(p)
        # 根据条件计算每个类的概率
        p[mask_pos] = pos_prob[mask_pos].to(p.dtype)
        p[mask_neg] = neg_prob[mask_neg].to(p.dtype)
        # 生成伯努利分布
        mask = torch.bernoulli(p).to(y_true.device)
        return mask

    def update_ratios(self):
        prob_diff = self._compute_probabilities_difference()
        self.positive_ratio_ideal *= torch.exp(self.change_rate * prob_diff)
        self._clear_probability_histogram()

    def _compute_probabilities_difference(self):
        hist_diff_pos = self._compute_kl_divergence(
            self.hist_pos_true, self.hist_pos_pred)
        hist_diff_neg = self._compute_kl_divergence(
            self.hist_neg_true, self.hist_neg_pred)
        return hist_diff_pos - hist_diff_neg

    @staticmethod
    def _compute_kl_divergence(hist_true, hist_pred):
        # Normalize histograms to form probability distributions
        hist_true_norm = hist_true / hist_true.sum(dim=0, keepdim=True)
        hist_pred_norm = hist_pred / hist_pred.sum(dim=0, keepdim=True)

        # Avoid log of zero by clamping
        epsilon = 1e-10
        hist_pred_norm = hist_pred_norm.clamp(min=epsilon)
        hist_true_norm = hist_true_norm.clamp(min=epsilon)

        # Compute KL divergence
        kl_div = F.kl_div(hist_pred_norm.log(),
                          hist_true_norm, reduction="batchmean")
        return kl_div

    def _compute_histogram(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy()
        n_classes = y_true.shape[1]
        n_bins = self.n_bins
        value_range = (0.0, 1.0)
        hist_pos_true, hist_neg_true, hist_pos_pred, hist_neg_pred = [np.zeros((n_bins, n_classes), np.float32)
                                                                      for _ in range(4)]
        for class_i in range(n_classes):
            y_true_class = y_true[:, class_i]
            y_pred_class = y_pred[:, class_i]
            pos_indices = y_true_class == 1
            neg_indices = ~pos_indices
            hist_pos_true[:, class_i], _ = np.histogram(
                y_true_class[pos_indices], bins=n_bins, range=value_range)
            hist_neg_true[:, class_i], _ = np.histogram(
                y_true_class[neg_indices], bins=n_bins, range=value_range)
            hist_pos_pred[:, class_i], _ = np.histogram(
                y_pred_class[pos_indices], bins=n_bins, range=value_range)
            hist_neg_pred[:, class_i], _ = np.histogram(
                y_pred_class[neg_indices], bins=n_bins, range=value_range)
        return [torch.from_numpy(hist) for hist in [hist_pos_true, hist_neg_true, hist_pos_pred, hist_neg_pred]]

    def __str__(self) -> str:
        return f"PLMLoss(lambda_rate={self.change_rate}, hist_bins={self.n_bins}, using loss {str(self.loss_function)})"

    __repr__ = __str__
