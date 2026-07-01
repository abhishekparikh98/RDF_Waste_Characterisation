"""
Evaluation metrics and validation utilities for model performance analysis
"""

from typing import Dict, Tuple, List
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns


class MetricsCalculator:
    """Calculate comprehensive evaluation metrics from predictions and ground truth"""
    
    def __init__(self, class_names: List[str], average_type: str = 'weighted'):
        """
        Initialize metrics calculator.
        
        Args:
            class_names: List of class names for classification report
            average_type: Type of averaging for multi-class metrics ('weighted', 'macro', 'micro')
        """
        self.class_names = class_names
        self.average_type = average_type
    
    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate all metrics given predictions and ground truth.
        
        Args:
            y_true: Ground truth labels (class indices)
            y_pred: Predicted labels (class indices)
        
        Returns:
            Dictionary of metrics {metric_name: value}
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average=self.average_type, zero_division=0),
            'recall': recall_score(y_true, y_pred, average=self.average_type, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average=self.average_type, zero_division=0)
        }
        return metrics
    
    def get_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> np.ndarray:
        """
        Generate confusion matrix.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
        
        Returns:
            Confusion matrix (num_classes x num_classes)
        """
        labels = list(range(len(self.class_names)))
        return confusion_matrix(y_true, y_pred, labels=labels)
    
    def get_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        output_dict: bool = False
    ) -> str or Dict:
        """
        Generate detailed classification report.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            output_dict: Return as dictionary if True, else string
        
        Returns:
            Classification report (string or dict)
        """
        return classification_report(
            y_true, y_pred,
            labels=list(range(len(self.class_names))),
            target_names=self.class_names,
            digits=4,
            output_dict=output_dict,
            zero_division=0
        )


class ConfusionMatrixVisualizer:
    """Visualize confusion matrix with heatmap"""
    
    @staticmethod
    def plot(
        cm: np.ndarray,
        class_names: List[str],
        figsize: Tuple[int, int] = (10, 8),
        save_path: str = None,
        dpi: int = 300
    ) -> plt.Figure:
        """
        Plot confusion matrix as heatmap.
        
        Args:
            cm: Confusion matrix (num_classes x num_classes)
            class_names: List of class names
            figsize: Figure size
            save_path: Path to save figure
            dpi: DPI for saved figure
        
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Normalize confusion matrix by row (actual counts)
        row_sums = cm.sum(axis=1, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            cm_percent = np.divide(
                cm.astype('float') * 100,
                row_sums,
                out=np.zeros_like(cm, dtype=float),
                where=row_sums != 0
            )
        
        # Create heatmap
        sns.heatmap(
            cm,
            annot=cm_percent,
            fmt='.1f',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'label': 'Percentage (%)'},
            ax=ax,
            annot_kws={'size': 9}
        )
        
        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        ax.set_title('Confusion Matrix - Test Set', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        
        return fig


class TrainingHistoryVisualizer:
    """Visualize training history (accuracy and loss curves)"""
    
    @staticmethod
    def plot_training_history(
        history: Dict,
        figsize: Tuple[int, int] = (14, 5),
        save_dir: str = None,
        filename_prefix: str = "",
        dpi: int = 300
    ) -> Tuple[plt.Figure, plt.Figure]:
        """
        Plot accuracy and loss curves from training history.
        
        Args:
            history: Training history dict with keys like 'accuracy', 'loss', 'val_accuracy', 'val_loss'
            figsize: Figure size
            save_dir: Directory to save figures
            dpi: DPI for saved figures
        
        Returns:
            Tuple of (accuracy_fig, loss_fig)
        """
        # Plot Accuracy
        fig_acc, ax_acc = plt.subplots(figsize=figsize)
        ax_acc.plot(history['accuracy'], label='Training Accuracy', linewidth=2, marker='o', markersize=4)
        ax_acc.plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2, marker='s', markersize=4)
        ax_acc.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax_acc.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax_acc.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        ax_acc.legend(fontsize=11)
        ax_acc.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_dir:
            acc_name = f"{filename_prefix}training_accuracy.png" if filename_prefix else "training_accuracy.png"
            fig_acc.savefig(f"{save_dir}/{acc_name}", dpi=dpi, bbox_inches='tight')
        
        # Plot Loss
        fig_loss, ax_loss = plt.subplots(figsize=figsize)
        ax_loss.plot(history['loss'], label='Training Loss', linewidth=2, marker='o', markersize=4)
        ax_loss.plot(history['val_loss'], label='Validation Loss', linewidth=2, marker='s', markersize=4)
        ax_loss.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax_loss.set_ylabel('Loss', fontsize=12, fontweight='bold')
        ax_loss.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax_loss.legend(fontsize=11)
        ax_loss.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_dir:
            loss_name = f"{filename_prefix}training_loss.png" if filename_prefix else "training_loss.png"
            fig_loss.savefig(f"{save_dir}/{loss_name}", dpi=dpi, bbox_inches='tight')
        
        return fig_acc, fig_loss


class TabularModelVisualizer:
    """Visualize tabular model explainability outputs."""

    @staticmethod
    def plot_feature_importance(
        importances: np.ndarray,
        feature_names: List[str],
        title: str = "Feature Importance",
        save_path: str = None,
        figsize: Tuple[int, int] = (10, 6),
        dpi: int = 300
    ) -> plt.Figure:
        """Plot sorted feature importance values."""
        indices = np.argsort(importances)[::-1]
        sorted_importances = importances[indices]
        sorted_names = [feature_names[i] for i in indices]

        fig, ax = plt.subplots(figsize=figsize)
        ax.bar(range(len(sorted_importances)), sorted_importances, color="steelblue")
        ax.set_xticks(range(len(sorted_names)))
        ax.set_xticklabels(sorted_names, rotation=45, ha="right")
        ax.set_ylabel("Importance")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

        return fig
