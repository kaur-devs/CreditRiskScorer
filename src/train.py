import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import (
    PROCESSED_DATA_DIR,
    PROCESSED_TRAIN_PATH,
    PROCESSED_VAL_PATH,
    PROCESSED_TEST_PATH,
    TARGET_COLUMN,
    RANDOM_STATE,
    setup_logging
)

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


if __name__ == "__main__":
    setup_logging()
    split_and_save_data()
    
    # Verification print
    for name, path in [("Train", PROCESSED_TRAIN_PATH), ("Val", PROCESSED_VAL_PATH), ("Test", PROCESSED_TEST_PATH)]:
        df_split = pd.read_csv(path, index_col=0)
        target_counts = df_split[TARGET_COLUMN].value_counts().to_dict()
        target_pct = (df_split[TARGET_COLUMN].value_counts(normalize=True) * 100).to_dict()
        print(f"\n{name} Split ({df_split.shape[0]} rows):")
        for cls, count in target_counts.items():
            print(f" - Class {cls}: {count} ({target_pct[cls]:.2f}%)")
