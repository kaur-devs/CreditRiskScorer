import logging
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.calibration import calibration_curve
from sklearn.model_selection import cross_val_score
from src.config import (
    PROCESSED_TRAIN_PATH,
    PROCESSED_VAL_PATH,
    MODEL_PATH,
    PREPROCESSOR_PATH,
    TARGET_COLUMN,
    RANDOM_STATE,
    setup_logging
)

logger = logging.getLogger(__name__)

def run_final_evaluation() -> None:
    logger.info("Loading splits and serialized artifacts...")
    train_df = pd.read_csv(PROCESSED_TRAIN_PATH, index_col=0)
    val_df = pd.read_csv(PROCESSED_VAL_PATH, index_col=0)
    
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    
    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN]
    X_val = val_df.drop(columns=[TARGET_COLUMN])
    y_val = val_df[TARGET_COLUMN]
    
    X_train_trans = preprocessor.transform(X_train)
    X_val_trans = preprocessor.transform(X_val)
    
    logger.info("Computing 5-fold cross-validation scores on training set...")
    cv_scores = cross_val_score(model, X_train_trans, y_train, cv=5, scoring="roc_auc", n_jobs=-1)
    logger.info(f"Cross-Validation ROC-AUC Scores: {cv_scores}")
    logger.info(f"Mean CV ROC-AUC: {cv_scores.mean():.4f} (Std: {cv_scores.std():.4f})")
    
    y_pred = model.predict(X_val_trans)
    y_prob = model.predict_proba(X_val_trans)[:, 1]
    
    print("\n" + "="*50)
    print("               CLASSIFICATION REPORT")
    print("="*50)
    print(classification_report(y_val, y_pred, digits=4))
    print("="*50)
    
    cm = confusion_matrix(y_val, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    plot_dir = Path("notebooks/plots")
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Calibration Curve Plot
    prob_true, prob_pred = calibration_curve(y_val, y_prob, n_bins=10)
    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker="o", label="XGBoost")
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Probability Calibration Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "final_calibration_curve.png", dpi=150)
    plt.close()
    logger.info("Calibration curve saved.")
    
    # 2. Feature Importance Plot
    importances = model.feature_importances_
    features = list(X_train_trans.columns)
    
    feat_imp = pd.Series(importances, index=features).sort_values(ascending=True)
    
    plt.figure(figsize=(10, 6))
    feat_imp.plot(kind="barh", color="steelblue")
    plt.xlabel("F-Score / Relative Importance")
    plt.title("XGBoost Tuned - Feature Importances")
    plt.tight_layout()
    plt.savefig(plot_dir / "final_feature_importance.png", dpi=150)
    plt.close()
    logger.info("Feature importance plot saved.")


if __name__ == "__main__":
    setup_logging()
    run_final_evaluation()
