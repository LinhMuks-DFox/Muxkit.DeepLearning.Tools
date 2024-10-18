import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

class ConfusionMatrixPlotter:

    def __init__(self, class2label, compose_path_func):
        """
        Initializes the ConfusionMatrixPlotter class.
        
        :param class2label: A dictionary mapping class indices to labels and display names.
        :param compose_path_func: A function to compose the path for saving plots.
        """
        self.class2label = class2label
        self.compose_path = compose_path_func

    def plot_sklearn_multi_label_confusion_matrix(self,
                                                  confusion_matrix: np.ndarray,
                                                  prefix: str,
                                                  n_rows: int = 5,
                                                  n_cols: int = 5):
        """
        Plots the confusion matrix for multi-label classification or multi-class classification.
        
        :param confusion_matrix: Confusion matrix, either multi-class or multi-label.
        :param prefix: Prefix for the file path where the plots will be saved.
        :param n_rows: Number of rows for the multi-label confusion matrix plot.
        :param n_cols: Number of columns for the multi-label confusion matrix plot.
        """
        def plot_individual_confusion_matrix(cm, ax, idx):
            # Normalize the confusion matrix
            norm_cm = cm / (cm.sum(axis=1, keepdims=True) + 1e-10)
            norm_cm = np.nan_to_num(norm_cm)  # Handle division by zero
            sns.heatmap(norm_cm, annot=True, fmt=".2f", cmap="Blues", cbar=False, ax=ax,
                        annot_kws={"fontsize": 8}, linewidths=.5, linecolor='black', square=True)

            # Change font color based on the background color and add original values
            for k, text in enumerate(ax.texts):
                row, col = divmod(k, 2)
                original_value = cm[row, col]
                new_text = f"{text.get_text()}\n({original_value})"
                text.set_text(new_text)
                text.set_color('white' if float(text.get_text().split('\n')[0]) > 0.5 else 'black')

            ax.axis('off')
            ax.set_title(f"{idx} ({self.class2label[str(idx)]['display_name']})", fontsize=10)

        if confusion_matrix.ndim == 3 and confusion_matrix.shape[1:] == (2, 2):
            # Multi-label case with [n_class, 2, 2] shape
            num_images_per_plot = n_rows * n_cols
            total_images = len(confusion_matrix)
            num_plots = total_images // num_images_per_plot
            if total_images % num_images_per_plot != 0:
                num_plots += 1
            
            adjusted_fig_size = (n_cols * 4, n_rows * 4)
            if not os.path.exists(self.compose_path(f"{prefix}_confusion_matrix")):
                os.makedirs(self.compose_path(f"{prefix}"))

            for plot_index in range(num_plots):
                fig, ax = plt.subplots(n_rows, n_cols, figsize=adjusted_fig_size)
                start_index = plot_index * num_images_per_plot
                end_index = min((plot_index + 1) * num_images_per_plot, total_images)

                for i in range(n_rows):
                    for j in range(n_cols):
                        idx = start_index + i * n_cols + j
                        if idx >= end_index:
                            break
                        plot_individual_confusion_matrix(confusion_matrix[idx], ax[i, j], idx)

                plt.tight_layout()
                plt_path = self.compose_path(f"{prefix}/cfx_{plot_index}.png")
                os.makedirs(os.path.dirname(plt_path), exist_ok=True)  # Ensure directory exists
                plt.savefig(plt_path, dpi=400)
                plt.close()

        elif confusion_matrix.ndim == 2 and confusion_matrix.shape[0] == confusion_matrix.shape[1]:
            # Multi-class mono-label case with [n_class, n_class] shape
            norm_confusion_matrix = confusion_matrix / (confusion_matrix.sum(axis=1, keepdims=True) + 1e-10)
            norm_confusion_matrix = np.nan_to_num(norm_confusion_matrix)  # Handle division by zero
            fig, ax = plt.subplots(figsize=(confusion_matrix.shape[0], confusion_matrix.shape[0]))
            sns.heatmap(norm_confusion_matrix, annot=True, fmt=".2f", cmap="Blues", cbar=True, ax=ax,
                        annot_kws={"fontsize": 8}, linewidths=.5, linecolor='black', square=True)

            # Change font color based on the background color and add original values
            for k, text in enumerate(ax.texts):
                row = k // confusion_matrix.shape[1]
                col = k % confusion_matrix.shape[1]
                original_value = confusion_matrix[row, col]
                new_text = f"{text.get_text()}\n({original_value})"
                text.set_text(new_text)
                text.set_color('white' if float(text.get_text().split('\n')[0]) > 0.5 else 'black')

            plt.tight_layout()
            plt_path = self.compose_path(f"{prefix}/cfx_multiclass.pdf")
            os.makedirs(os.path.dirname(plt_path), exist_ok=True)  # Ensure directory exists
            plt.savefig(plt_path, dpi=300)
            plt.close()

        else:
            raise ValueError("Invalid shape for confusion_matrix")
