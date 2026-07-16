import logging
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from src.config import (
    PROCESSED_DATA_DIR,
    PROCESSED_TRAIN_PATH,
    PROCESSED_VAL_PATH,
    PROCESSED_TEST_PATH,
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
    
    logger.info(f"Train set: {train_df.shape}")
    logger.info(f"Val set: {val_df.shape}")
    logger.info(f"Test set: {test_df.shape}")
    
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(PROCESSED_TRAIN_PATH)
    val_df.to_csv(PROCESSED_VAL_PATH)
    test_df.to_csv(PROCESSED_TEST_PATH)
    logger.info("Successfully saved stratified train, validation, and test splits.")


def train_baseline_model() -> None:
    logger.info("Loading split datasets for baseline training...")
    train_df = pd.read_csv(PROCESSED_TRAIN_PATH, index_col=0)
    val_df = pd.read_csv(PROCESSED_VAL_PATH, index_col=0)
    
    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN]
    X_val = val_df.drop(columns=[TARGET_COLUMN])
    y_val = val_df[TARGET_COLUMN]
    
    pipeline = build_feature_pipeline()
    X_train_trans = pipeline.fit_transform(X_train, y_train)
    X_val_trans = pipeline.transform(X_val)
    
    logger.info("Training baseline Logistic Regression model...")
    model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    model.fit(X_train_trans, y_train)
    
    y_pred = model.predict(X_val_trans)
    y_prob = model.predict_proba(X_val_trans)[:, 1]
    
    plot_dir = Path("notebooks/plots")
    metrics = evaluate_predictions(
        y_true=y_val,
        y_pred=y_pred,
        y_prob=y_prob,
        model_name="Baseline Logistic Regression",
        plot_dir=plot_dir
    )
    
    # Explain why accuracy is misleading
    logger.info("\n--- ANALYSIS: Why Accuracy is Misleading ---")
    logger.info(f"Baseline Accuracy: {metrics['accuracy']:.4f}")
    
    # Calculate zero-rate classifier (always predicts negative class 0)
    majority_class_ratio = (y_val == 0).mean()
    logger.info(f"Majority Class Ratio (Zero-Rate Accuracy): {majority_class_ratio:.4f}")
    
    logger.info(
        "A dummy model that predicts zero default for every borrower achieves "
        f"{majority_class_ratio*100:.2f}% accuracy. However, this model finds "
        f"0 of the {y_val.sum()} default cases (Recall = 0.0). "
        "Thus, high accuracy is purely a reflection of class imbalance, not predictive value."
    )


if __name__ == "__main__":
    setup_logging()
    
    # Ensure splits exist, if not, generate them
    if not PROCESSED_TRAIN_PATH.exists():
        split_and_save_data()
        
    train_baseline_model()
