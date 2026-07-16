import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve
)

logger = logging.getLogger(__name__)

def evaluate_predictions(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    plot_dir: Path
) -> Dict[str, float]:
    
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob))
    }
    
    logger.info(f"--- {model_name} Evaluation Metrics ---")
    for k, v in metrics.items():
        logger.info(f"{k.capitalize()}: {v:.4f}")
        
    cm = confusion_matrix(y_true, y_pred)
    logger.info(f"Confusion Matrix:\n{cm}")
    
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    axes[0].plot(fpr, tpr, label=f"ROC (AUC = {metrics['roc_auc']:.4f})")
    axes[0].plot([0, 1], [0, 1], "k--")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title(f"ROC Curve - {model_name}")
    axes[0].legend()
    
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    axes[1].plot(rec, prec, label="Precision-Recall Curve")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title(f"Precision-Recall Curve - {model_name}")
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(plot_dir / f"{model_name.lower().replace(' ', '_')}_curves.png", dpi=150)
    plt.close()
    
    return metrics
