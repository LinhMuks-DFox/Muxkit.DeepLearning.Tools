import numpy as np
import sklearn.metrics as metrics
import torch
import tqdm

import abc
import typing
from dataclasses import dataclass, asdict
from typing import Dict, Any


@dataclass
class MetricsStatusMap:
    f1_score: float
    accuracy: float
    precision: float
    recall: float
    confusion_matrix: np.ndarray

    @staticmethod
    def create_metrics(f1_score: float, accuracy: float, precision: float, recall: float,
                       confusion_matrix: np.ndarray) -> Dict[str, Any]:
        _me = MetricsStatusMap(
            f1_score=f1_score,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            confusion_matrix=confusion_matrix
        )
        return asdict(_me)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(
            f1_score=data["f1_score"],
            accuracy=data["accuracy"],
            precision=data["precision"],
            recall=data["recall"],
            confusion_matrix=np.array(data["confusion_matrix"])
        )

    def copy_from(self, other: "MetricsStatusMap"):
        self.f1_score = other.f1_score
        self.accuracy = other.accuracy
        self.precision = other.precision
        self.recall = other.recall
        self.confusion_matrix = np.copy(other.confusion_matrix)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def __str__(self):
        return (f"MetricsStatusMap(f1_score={self.f1_score}, accuracy={self.accuracy}, "
                f"precision={self.precision}, recall={self.recall}, "
                f"confusion_matrix=\n{self.confusion_matrix})")


class ClassifierTester(abc.ABC):

    def __init__(self, model, device, loss_fn):
        self.model_ = model
        self.dataloader_ = None
        self.device_ = device
        self.loss_fn_ = loss_fn
        self.n_classes_ = None

        self.confusion_matrix_ = None
        self.sklearn_confusion_matrix_ = None
        self.accuracy_ = None
        self.precision_ = None
        self.recall_ = None
        self.f1_score_ = None

        self.y_predict_ = None  # np.ndarray
        self.y_true_ = None  # np.ndarray
        self.loss_ = None  # torch.Tensor

    def set_loss_function(self, loss: typing.Callable):
        self.loss_fn_ = loss
        return self

    @abc.abstractmethod
    def set_dataloader(self, dataloader, n_class: int):
        pass

    @abc.abstractmethod
    def predict_all(self):
        pass

    @abc.abstractmethod
    def calculate_confusion_matrix(self):
        pass

    @abc.abstractmethod
    def calculate_accuracy(self, ):
        pass

    @abc.abstractmethod
    def calculate_precision(self, ):
        pass

    @abc.abstractmethod
    def calculate_recall(self, ):
        pass

    @abc.abstractmethod
    def calculate_f1_score(self, ):
        pass

    def status_map(self) -> typing.Dict:
        if not all([
            self.f1_score_ is not None,
            self.accuracy_ is not None,
            self.precision_ is not None,
            self.recall_ is not None,
            self.confusion_matrix_ is not None,
        ]):
            raise ValueError("None metrics exist, use calculate_all_metrics() before calling status_map")
        return MetricsStatusMap.create_metrics(
            f1_score=self.f1_score_,
            accuracy=self.accuracy_,
            precision=self.precision_,
            recall=self.recall_,
            confusion_matrix=self.confusion_matrix_
        )

    @abc.abstractmethod
    def evaluate_model(self):
        pass

    @abc.abstractmethod
    def calculate_all_metrics(self):
        pass

    @abc.abstractmethod
    def classification_report(self, ):
        pass


class MonoLabelClassificationTester(ClassifierTester):

    def __init__(self,
                 model: torch.nn.Module,
                 device: torch.device,
                 loss_fn: torch.nn.Module
                 ):
        super().__init__(model, device, loss_fn)

    def set_dataloader(self, dataloader, n_class: int) -> "MonoLabelClassificationTester":
        self.dataloader_ = dataloader
        self.n_classes_ = n_class
        self.y_predict_ = torch.empty(0, dtype=torch.int32).to(self.device_)
        self.y_true_ = torch.empty(0, dtype=torch.int32).to(self.device_)
        self.loss_ = torch.empty(0, dtype=torch.float).to(self.device_)
        return self

    @torch.no_grad()
    def predict_all(self) -> "MonoLabelClassificationTester":
        if self.dataloader_ is None or self.y_predict_ is None or self.y_true_ is None:
            raise ValueError("dataloader, y_predict, y_true is None, use set_dataloader() before calling predict_all")
        self.model_.eval()
        self.model_.to(self.device_)
        data: torch.Tensor
        label: torch.Tensor
        for data, label in tqdm.tqdm(self.dataloader_):
            data = data.to(self.device_)
            label = label.to(self.device_)
            model_out = self.model_(data)
            predicted_y = torch.argmax(model_out, dim=1)
            if self.loss_fn_ is not None:
                self.loss_ = torch.hstack([self.loss_, self.loss_fn_(model_out, label)])
            # if label is one-hot, convert it to int
            if len(label.shape) > 1:
                label = torch.argmax(label, dim=1)
            self.y_true_ = torch.cat([self.y_true_, label])
            self.y_predict_ = torch.cat([self.y_predict_, predicted_y])

        self.y_true_: np.ndarray = self.y_true_.detach().cpu().numpy()
        self.y_predict_: np.ndarray = self.y_predict_.detach().cpu().numpy()
        return self

    def calculate_confusion_matrix(self) -> "MonoLabelClassificationTester":
        self.confusion_matrix_ = metrics.confusion_matrix(self.y_true_, self.y_predict_)
        self.sklearn_confusion_matrix_ = self.confusion_matrix_
        return self

    def calculate_accuracy(self, ) -> "MonoLabelClassificationTester":
        self.accuracy_ = metrics.accuracy_score(self.y_true_, self.y_predict_)
        return self

    def calculate_precision(self, ) -> "MonoLabelClassificationTester":
        self.precision_ = metrics.precision_score(self.y_true_, self.y_predict_, average="macro", zero_division=0)
        return self

    def calculate_recall(self, ) -> "MonoLabelClassificationTester":
        self.recall_ = metrics.recall_score(self.y_true_, self.y_predict_, average="macro", zero_division=0)
        return self

    def calculate_f1_score(self, ) -> "MonoLabelClassificationTester":
        self.f1_score_ = metrics.f1_score(self.y_true_, self.y_predict_, average="macro", zero_division=0)
        return self

    def evaluate_model(self):
        self.predict_all()
        self.calculate_all_metrics()
        return self.status_map()

    def calculate_all_metrics(self) -> "MonoLabelClassificationTester":
        if self.y_true_ is None or self.y_predict_ is None:
            raise ValueError("y_true, y_predict is None, use predict_all() before calling calculate_all_metrics")
        if isinstance(self.y_true_, torch.Tensor) or isinstance(self.y_predict_, torch.Tensor):
            raise ValueError("y_true, y_predict is None, use predict_all() before calling calculate_all_metrics")
        self.calculate_recall()
        self.calculate_f1_score()
        self.calculate_precision()
        self.calculate_accuracy()
        self.calculate_confusion_matrix()
        return self

    def classification_report(self, ):
        return metrics.classification_report(self.y_true_, self.y_predict_, zero_division=np.nan)
