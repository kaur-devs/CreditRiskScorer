import logging
import pandas as pd
import numpy as np
from typing import Dict, Any

logger = logging.getLogger(__name__)

class DataCleaner:
    def __init__(self) -> None:
        self.medians: Dict[str, float] = {}
        self.capping_thresholds: Dict[str, float] = {}
        self.is_fitted = False

    def fit(self, df: pd.DataFrame) -> "DataCleaner":
        logger.info("Fitting DataCleaner on training set...")
        
        target_col = "SeriousDlqin2yrs"
        features_df = df.drop(columns=[target_col]) if target_col in df.columns else df
        
        self.medians["MonthlyIncome"] = float(features_df["MonthlyIncome"].median())
        self.medians["NumberOfDependents"] = float(features_df["NumberOfDependents"].median())
        
        delinquency_cols = [
            "NumberOfTime30-59DaysPastDueNotWorse",
            "NumberOfTimes90DaysLate",
            "NumberOfTime60-89DaysPastDueNotWorse"
        ]
        for col in delinquency_cols:
            if col in features_df.columns:
                clean_col = features_df[col].replace([96, 98], np.nan)
                self.medians[col] = float(clean_col.median())
                
        self.capping_thresholds["RevolvingUtilizationOfUnsecuredLines"] = float(
            features_df["RevolvingUtilizationOfUnsecuredLines"].quantile(0.99)
        )
        self.capping_thresholds["DebtRatio"] = float(
            features_df["DebtRatio"].quantile(0.99)
        )
        
        self.is_fitted = True
        logger.info(f"Fitted parameters: Medians={self.medians}, Thresholds={self.capping_thresholds}")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("DataCleaner must be fitted before calling transform.")
            
        logger.info("Transforming dataset...")
        df_clean = df.copy()
        
        if "MonthlyIncome" in df_clean.columns:
            df_clean["MonthlyIncome"] = df_clean["MonthlyIncome"].fillna(self.medians["MonthlyIncome"])
            
        if "NumberOfDependents" in df_clean.columns:
            df_clean["NumberOfDependents"] = df_clean["NumberOfDependents"].fillna(self.medians["NumberOfDependents"])
            
        delinquency_cols = [
            "NumberOfTime30-59DaysPastDueNotWorse",
            "NumberOfTimes90DaysLate",
            "NumberOfTime60-89DaysPastDueNotWorse"
        ]
        for col in delinquency_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].replace([96, 98], self.medians[col])
                
        if "RevolvingUtilizationOfUnsecuredLines" in df_clean.columns:
            limit = self.capping_thresholds["RevolvingUtilizationOfUnsecuredLines"]
            df_clean["RevolvingUtilizationOfUnsecuredLines"] = df_clean["RevolvingUtilizationOfUnsecuredLines"].clip(upper=limit)
            
        if "DebtRatio" in df_clean.columns:
            limit = self.capping_thresholds["DebtRatio"]
            df_clean["DebtRatio"] = df_clean["DebtRatio"].clip(upper=limit)
            
        return df_clean

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


if __name__ == "__main__":
    from src.config import RAW_DATA_PATH, PROCESSED_DATA_DIR, setup_logging
    
    setup_logging()
    
    df_raw = pd.read_csv(RAW_DATA_PATH, index_col=0)
    print("--- BEFORE CLEANING ---")
    print(f"Dataset Shape: {df_raw.shape}")
    print(f"Total Missing:\n{df_raw.isnull().sum()[df_raw.isnull().sum() > 0]}")
    print(f"Max Utilization: {df_raw['RevolvingUtilizationOfUnsecuredLines'].max()}")
    print(f"Max DebtRatio: {df_raw['DebtRatio'].max()}")
    print(f"Max Delinquency 30-59 days: {df_raw['NumberOfTime30-59DaysPastDueNotWorse'].max()}")
    
    # Remove duplicates
    df_no_dups = df_raw.drop_duplicates()
    print(f"\nDuplicates removed: {df_raw.shape[0] - df_no_dups.shape[0]}")
    
    # Fit and transform
    cleaner = DataCleaner()
    df_clean = cleaner.fit_transform(df_no_dups)
    
    print("\n--- AFTER CLEANING ---")
    print(f"Dataset Shape: {df_clean.shape}")
    print(f"Total Missing: {df_clean.isnull().sum().sum()}")
    print(f"Max Utilization: {df_clean['RevolvingUtilizationOfUnsecuredLines'].max()}")
    print(f"Max DebtRatio: {df_clean['DebtRatio'].max()}")
    print(f"Max Delinquency 30-59 days: {df_clean['NumberOfTime30-59DaysPastDueNotWorse'].max()}")
    
    output_path = PROCESSED_DATA_DIR / "cs-training-clean.csv"
    df_clean.to_csv(output_path)
    print(f"\nSaved cleaned dataset to {output_path}")

