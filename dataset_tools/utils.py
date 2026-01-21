import typing

import numpy as np
import torch


def label_digit2tensor(label_digits: typing.List[int], class_num=527) -> torch.Tensor:
    """
    Convert a list of label indices into a one-hot encoded tensor.

    Args:
        label_digits (List[int]): A list of integer indices representing the labels.
        class_num (int): The total number of classes.

    Returns:
        torch.Tensor: A one-hot encoded tensor representing the labels.
    """
    label_digits: np.ndarray = np.array(label_digits)
    label: np.ndarray = np.zeros(class_num)
    label[label_digits] = 1
    return torch.tensor(label)
