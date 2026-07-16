import logging
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.metrics import f1_score, precision_recall_curve
from src.config import (
    PROCESSED_DATA_DIR,
    PROCESSED_TRAIN_PATH,
    PROCESSED_VAL_PATH,
    PROCESSED_TEST_PATH,
    MODEL_PATH,
    PREPROCESSOR_PATH,
    MODEL_DIR,
    TARGET_COLUMN,
    RANDOM_STATE,
    setup_logging
)
from src.feature_engineering import build_feature_pipeline
from src.evaluate import evaluate_predictions

logger = logging.getLogger(__name__)

def split_and_save_data() -> None:
    clean_data_path = PROCESSED_DATA_DIR / "cs-training-clean.csv"
    logger.info(f"Loading cleaned dataset from {clean_data_path}...")
    df = pd.read_csv(clean_data_path, index_col=0)
    
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=df[TARGET_COLUMN]
    )
    
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=temp_df[TARGET_COLUMN]
    )
    
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(PROCESSED_TRAIN_PATH)
    val_df.to_csv(PROCESSED_VAL_PATH)
    test_df.to_csv(PROCESSED_TEST_PATH)
    logger.info("Successfully saved stratified train, validation, and test splits.")


def get_preprocessed_splits():
    train_df = pd.read_csv(PROCESSED_TRAIN_PATH, index_col=0)
    val_df = pd.read_csv(PROCESSED_VAL_PATH, index_col=0)
    
    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN]
    X_val = val_df.drop(columns=[TARGET_COLUMN])
    y_val = val_df[TARGET_COLUMN]
    
    pipeline = build_feature_pipeline()
    X_train_trans = pipeline.fit_transform(X_train, y_train)
    X_val_trans = pipeline.transform(X_val)
    
    return X_train_trans, y_train, X_val_trans, y_val, pipeline


def train_baseline_model() -> None:
    X_train_trans, y_train, X_val_trans, y_val, _ = get_preprocessed_splits()
    
    logger.info("Training baseline Logistic Regression model...")
    model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    model.fit(X_train_trans, y_train)
    
    y_pred = model.predict(X_val_trans)
    y_prob = model.predict_proba(X_val_trans)[:, 1]
    
    plot_dir = Path("notebooks/plots")
    evaluate_predictions(y_val, y_pred, y_prob, "Baseline Logistic Regression", plot_dir)


def compare_imbalance_handling() -> None:
    X_train, y_train, X_val, y_val, _ = get_preprocessed_splits()
    plot_dir = Path("notebooks/plots")
    
    results = {}
    
    model_base = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    model_base.fit(X_train, y_train)
    y_pred_base = model_base.predict(X_val)
    y_prob_base = model_base.predict_proba(X_val)[:, 1]
    results["Baseline"] = evaluate_predictions(y_val, y_pred_base, y_prob_base, "Baseline", plot_dir)
    
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    model_smote = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    model_smote.fit(X_train_smote, y_train_smote)
    y_pred_sm = model_smote.predict(X_val)
    y_prob_sm = model_smote.predict_proba(X_val)[:, 1]
    results["SMOTE"] = evaluate_predictions(y_val, y_pred_sm, y_prob_sm, "SMOTE", plot_dir)
    
    ros = RandomOverSampler(random_state=RANDOM_STATE)
    X_train_ros, y_train_ros = ros.fit_resample(X_train, y_train)
    model_ros = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    model_ros.fit(X_train_ros, y_train_ros)
    y_pred_ros = model_ros.predict(X_val)
    y_prob_ros = model_ros.predict_proba(X_val)[:, 1]
    results["Random Oversampling"] = evaluate_predictions(y_val, y_pred_ros, y_prob_ros, "Random Oversampling", plot_dir)
    
    model_cw = LogisticRegression(random_state=RANDOM_STATE, class_weight="balanced", max_iter=1000)
    model_cw.fit(X_train, y_train)
    y_pred_cw = model_cw.predict(X_val)
    y_prob_cw = model_cw.predict_proba(X_val)[:, 1]
    results["Class Weights"] = evaluate_predictions(y_val, y_pred_cw, y_prob_cw, "Class Weights", plot_dir)
    
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_prob_base)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    
    y_pred_tuned = (y_prob_base >= best_threshold).astype(int)
    results["Threshold Tuning"] = evaluate_predictions(y_val, y_pred_tuned, y_prob_base, "Threshold Tuning", plot_dir)
    
    print("\n" + "="*70)
    print("             CLASS IMBALANCE COMPARISON RESULTS")
    print("="*70)
    print(f"{'Method':<25} | {'Accuracy':<8} | {'Precision':<9} | {'Recall':<6} | {'F1-Score':<8} | {'ROC-AUC':<7}")
    print("-"*70)
    for method, metrics in results.items():
        print(f"{method:<25} | {metrics['accuracy']:<8.4f} | {metrics['precision']:<9.4f} | {metrics['recall']:<6.4f} | {metrics['f1']:<8.4f} | {metrics['roc_auc']:<7.4f}")
    print("="*70 + "\n")


def train_xgb_model() -> None:
    logger.info("Starting hyperparameter tuning for XGBoost...")
    X_train, y_train, X_val, y_val, pipeline = get_preprocessed_splits()
    
    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5],
        'scale_pos_weight': [1, 5, 10, 13.9]
    }
    
    xgb = XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss')
    
    search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_dist,
        n_iter=10,
        scoring='roc_auc',
        cv=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1
    )
    
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    
    logger.info(f"Hyperparameter tuning complete. Best parameters found:\n{search.best_params_}")
    logger.info(f"Best CV ROC-AUC Score: {search.best_score_:.4f}")
    
    y_pred = best_model.predict(X_val)
    y_prob = best_model.predict_proba(X_val)[:, 1]
    
    plot_dir = Path("notebooks/plots")
    evaluate_predictions(y_val, y_pred, y_prob, "XGBoost Tuned", plot_dir)
    
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(pipeline, PREPROCESSOR_PATH)
    logger.info(f"Model serialized and saved to: {MODEL_PATH}")
    logger.info(f"Preprocessor serialized and saved to: {PREPROCESSOR_PATH}")


if __name__ == "__main__":
    setup_logging()
    
    if not PROCESSED_TRAIN_PATH.exists():
        split_and_save_data()
        
    train_xgb_model()
