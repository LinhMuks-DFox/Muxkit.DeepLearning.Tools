"""
Confusion matrix plotting utilities (matplotlib + seaborn).

Provides a simple function for a single matrix and a class to render
per-class 2x2 matrices or a full square matrix with both proportions and counts.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from ..utl import api_tags as tags


def plot_confusion_matrix(matrix: np.ndarray):
    """Plot a confusion matrix heatmap with normalized values and raw counts.

    Args:
        matrix (ndarray): Confusion matrix (square), e.g., from scikit-learn.
    Returns:
        matplotlib.figure.Figure: The created figure.
    """
    # 计算每个类别的占比（按行归一化）
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized_matrix = np.divide(matrix, row_sums, where=row_sums != 0)

    # 创建图形和轴
    fig, ax = plt.subplots(figsize=(10, 8))

    # 绘制带热图的混淆矩阵
    sns.heatmap(
        normalized_matrix, annot=True, fmt=".2f", cmap="Blues", cbar=True,
        xticklabels=True, yticklabels=True, ax=ax, annot_kws={"size": 10}
    )

    # 添加真实数值的注释
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j + 0.5, i + 0.5, f"({int(matrix[i, j])})",
                    ha="center", va="center", color="black", fontsize=8)

    # 设置标题和轴标签
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    ax.set_title("Confusion Matrix with Proportions and Counts")

    return fig


@tags.stable_api
class ConfusionMatrixPlotter:
    def __init__(self, class2label):
        """Initialize with a mapping from class index to label/metadata.

        Args:
            class2label (dict): Maps class index -> { 'display_name': str, ... }.
        """
        self.class2label = class2label

    def _plot_individual_confusion_matrix(self, cm, ax, idx):
        """Render a single 2x2 confusion matrix on the provided axes.

        Args:
            cm (ndarray): 2x2 matrix for a single label.
            ax (Axes): Matplotlib axes to draw on.
            idx (int): Class index for title lookup.
        """
        # 归一化混淆矩阵
        norm_cm = cm / (cm.sum(axis=1, keepdims=True) + 1e-10)
        norm_cm = np.nan_to_num(norm_cm)  # 处理除以零的情况
        sns.heatmap(norm_cm, annot=True, fmt=".2f", cmap="Blues", cbar=False, ax=ax,
                    annot_kws={"fontsize": 8}, linewidths=.5, linecolor='black', square=True)

        # 根据背景颜色调整字体颜色，并添加原始值
        for k, text in enumerate(ax.texts):
            row, col = divmod(k, 2)
            original_value = cm[row, col]
            new_text = f"{text.get_text()}\n({original_value})"
            text.set_text(new_text)
            text.set_color('white' if float(
                text.get_text().split('\n')[0]) > 0.5 else 'black')

        ax.axis('off')
        ax.set_title(
            f"{idx} ({self.class2label[idx]['display_name']})", fontsize=10)

    def plot(self, confusion_matrix, n_rows=1, n_cols=1):
        """Plot multi-class or multi-label confusion matrices.

        Args:
            confusion_matrix (ndarray): Square matrix (multi-class) or stack of 2x2 (multi-label).
            n_rows (int): Grid rows for multi-label visualization.
            n_cols (int): Grid cols for multi-label visualization.
        Returns:
            matplotlib.figure.Figure: The created figure.
        """
        if confusion_matrix.ndim == 3 and confusion_matrix.shape[1:] == (2, 2):
            # 多标签分类情况，每个类别一个2x2的混淆矩阵
            num_images_per_plot = n_rows * n_cols
            total_images = len(confusion_matrix)
            num_plots = (total_images + num_images_per_plot -
                         1) // num_images_per_plot

            adjusted_fig_size = (n_cols * 4, n_rows * 4)
            fig, axs = plt.subplots(n_rows, n_cols, figsize=adjusted_fig_size)
            axs = axs.flatten() if num_images_per_plot > 1 else [axs]

            for idx, ax in enumerate(axs):
                matrix_idx = idx
                if matrix_idx < total_images:
                    self._plot_individual_confusion_matrix(
                        confusion_matrix[matrix_idx], ax, matrix_idx)
                else:
                    ax.axis('off')  # Hide any extra subplots

            plt.tight_layout()
            return fig

        elif confusion_matrix.ndim == 2 and confusion_matrix.shape[0] == confusion_matrix.shape[1]:
            # 多类分类情况，方阵的混淆矩阵
            norm_confusion_matrix = confusion_matrix / \
                (confusion_matrix.sum(axis=1, keepdims=True) + 1e-10)
            norm_confusion_matrix = np.nan_to_num(
                norm_confusion_matrix)  # 处理除以零的情况
            fig, ax = plt.subplots(
                figsize=(confusion_matrix.shape[0] * 1.2, confusion_matrix.shape[0] * 1.2))
            sns.heatmap(norm_confusion_matrix, annot=True, fmt=".2f", cmap="Blues", cbar=True, ax=ax,
                        annot_kws={"fontsize": 8}, linewidths=.5, linecolor='black', square=True)

            # 根据背景颜色调整字体颜色，并添加原始值
            for k, text in enumerate(ax.texts):
                row = k // confusion_matrix.shape[1]
                col = k % confusion_matrix.shape[1]
                original_value = confusion_matrix[row, col]
                new_text = f"{text.get_text()}\n({original_value})"
                text.set_text(new_text)
                text.set_color('white' if float(
                    text.get_text().split('\n')[0]) > 0.5 else 'black')

            plt.tight_layout()
            return fig

        else:
            raise ValueError("Invalid confusion matrix shape")
