import abc
import warnings
import typing
import numpy as np
import sklearn.metrics as metrics
import torch
import tqdm


class ClassifierTester(abc.ABC):
    def __init__(self, model, device, n_classes: int, binary_decision_threshold=0.5, sigmoid_before_thresholding=False):
        self.model_ = model
        self.device_ = device
        self.n_classes_ = n_classes
        self.binary_decision_threshold = binary_decision_threshold
        self.sigmoid_before_thresholding = sigmoid_before_thresholding
        self.dataloader_ = None
        self.loss_fn_ = None

        # Metrics
        self.confusion_matrix_ = None
        self.accuracy_ = None
        self.precision_ = None
        self.recall_ = None
        self.f1_score_ = None
        self.hamming_loss_ = None

        # Predictions and ground truth
        self.y_predict_ = None
        self.y_true_ = None
        self.y_predict_binary_ = None  # For multi-label classification
        self.loss_ = torch.zeros(0, dtype=torch.float).to(self.device_)

    def set_loss_function(self, loss: typing.Callable):
        self.loss_fn_ = loss
        return self

    def set_dataloader(self, dataloader):
        self.dataloader_ = dataloader
        self.y_true_ = torch.zeros((0, self.n_classes_), dtype=torch.int).to(self.device_)
        self.y_predict_ = torch.zeros((0, self.n_classes_), dtype=torch.float).to(self.device_)
        self.y_predict_binary_ = torch.zeros((0, self.n_classes_), dtype=torch.int).to(self.device_)  # Multi-label support
        return self

    @torch.no_grad()
    def predict_all(self, multi_label=False):
        self.model_.eval()
        self.model_.to(self.device_)

        for data, label in tqdm.tqdm(self.dataloader_):
            data, label = data.to(self.device_), label.to(self.device_)
            output = self.model_(data)

            if self.loss_fn_:
                loss = self.loss_fn_(output, label)
                self.loss_ = torch.cat([self.loss_, loss])

            if multi_label:
                if self.sigmoid_before_thresholding:
                    output = torch.sigmoid(output)
                binary_prediction = self.make_binary_prediction(output)
                self.y_predict_binary_ = torch.cat([self.y_predict_binary_, binary_prediction])

            else:
                predicted_y = torch.argmax(output, dim=1)
                self.y_predict_ = torch.cat([self.y_predict_, predicted_y])

            # Handle ground truth for both label types
            if len(label.shape) > 1:
                label = torch.argmax(label, dim=1)  # One-hot to int
            self.y_true_ = torch.cat([self.y_true_, label])

        self.y_true_ = self.y_true_.detach().cpu().numpy()
        self.y_predict_ = self.y_predict_.detach().cpu().numpy()
        if multi_label:
            self.y_predict_binary_ = self.y_predict_binary_.detach().cpu().numpy()

        return self

    def make_binary_prediction(self, y_probability: torch.Tensor):
        ret = torch.zeros_like(y_probability, dtype=torch.int)
        ret[y_probability >= self.binary_decision_threshold] = 1
        return ret

    def calculate_confusion_matrix(self, multi_label=False):
        if multi_label:
            self.confusion_matrix_ = metrics.multilabel_confusion_matrix(self.y_true_, self.y_predict_binary_)
        else:
            self.confusion_matrix_ = metrics.confusion_matrix(self.y_true_, self.y_predict_)
        return self

    def calculate_accuracy(self, multi_label=False):
        if multi_label:
            self.accuracy_ = self.multilabel_accuracy(self.y_true_, self.y_predict_binary_)
        else:
            self.accuracy_ = metrics.accuracy_score(self.y_true_, self.y_predict_)
        return self

    def calculate_precision(self, multi_label=False):
        if multi_label:
            self.precision_ = metrics.precision_score(self.y_true_, self.y_predict_binary_, average="macro", zero_division=0)
        else:
            self.precision_ = metrics.precision_score(self.y_true_, self.y_predict_, average="macro", zero_division=0)
        return self

    def calculate_recall(self, multi_label=False):
        if multi_label:
            self.recall_ = metrics.recall_score(self.y_true_, self.y_predict_binary_, average="macro", zero_division=0)
        else:
            self.recall_ = metrics.recall_score(self.y_true_, self.y_predict_, average="macro", zero_division=0)
        return self

    def calculate_f1_score(self, multi_label=False):
        if multi_label:
            self.f1_score_ = metrics.f1_score(self.y_true_, self.y_predict_binary_, average="macro", zero_division=0)
        else:
            self.f1_score_ = metrics.f1_score(self.y_true_, self.y_predict_, average="macro", zero_division=0)
        return self

    def calculate_hamming_loss(self):
        self.hamming_loss_ = metrics.hamming_loss(self.y_true_, self.y_predict_binary_)
        return self

    def evaluate_model(self, multi_label=False):
        self.predict_all(multi_label)
        self.calculate_all_metrics(multi_label)
        return self.status_map()

    def calculate_all_metrics(self, multi_label=False):
        self.calculate_accuracy(multi_label)
        self.calculate_precision(multi_label)
        self.calculate_recall(multi_label)
        self.calculate_f1_score(multi_label)
        self.calculate_confusion_matrix(multi_label)
        if multi_label:
            self.calculate_hamming_loss()
        return self

    def classification_report(self, multi_label=False):
        if multi_label:
            return metrics.classification_report(self.y_true_, self.y_predict_binary_, zero_division=0)
        return metrics.classification_report(self.y_true_, self.y_predict_, zero_division=0)

    def status_map(self):
        return {
            "f1_score": self.f1_score_,
            "accuracy": self.accuracy_,
            "precision": self.precision_,
            "recall": self.recall_,
            "confusion_matrix": self.confusion_matrix_,
        }

    @staticmethod
    def multilabel_accuracy(y_true: np.ndarray, y_predict: np.ndarray):
        n_class = y_true.shape[1]
        acc = np.zeros(n_class)
        for i in range(n_class):
            acc[i] = metrics.accuracy_score(y_true[:, i], y_predict[:, i])
        return np.mean(acc)
