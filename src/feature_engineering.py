import logging
from typing import Any
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer

logger = logging.getLogger(__name__)

class CreditFeatureAdder(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y: None = None) -> "CreditFeatureAdder":
        self.feature_names_in_ = np.array(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        logger.info("Adding interaction features...")
        X_out = X.copy()
        
        X_out["IncomePerPerson"] = X_out["MonthlyIncome"] / (X_out["NumberOfDependents"] + 1)
        
        X_out["TotalLatePayments"] = (
            X_out["NumberOfTime30-59DaysPastDueNotWorse"] +
            X_out["NumberOfTime60-89DaysPastDueNotWorse"] +
            X_out["NumberOfTimes90DaysLate"]
        )
        
        return X_out

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        if input_features is None:
            input_features = getattr(self, "feature_names_in_", [])
        return np.array(list(input_features) + ["IncomePerPerson", "TotalLatePayments"], dtype=object)


def build_feature_pipeline() -> Pipeline:
    skewed_cols = [
        "RevolvingUtilizationOfUnsecuredLines",
        "DebtRatio",
        "MonthlyIncome",
        "IncomePerPerson"
    ]
    
    other_cols = [
        "age",
        "NumberOfOpenCreditLinesAndLoans",
        "NumberRealEstateLoansOrLines",
        "NumberOfDependents",
        "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfTimes90DaysLate",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "TotalLatePayments"
    ]
    
    log_scaler = Pipeline([
        ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scaler", StandardScaler())
    ])
    
    col_transformer = ColumnTransformer(
        transformers=[
            ("skewed", log_scaler, skewed_cols),
            ("other", StandardScaler(), other_cols)
        ],
        remainder="drop"
    )
    
    pipeline = Pipeline([
        ("feature_adder", CreditFeatureAdder()),
        ("transformer", col_transformer)
    ])
    pipeline.set_output(transform="pandas")
    
    return pipeline


if __name__ == "__main__":
    from src.config import PROCESSED_DATA_DIR, setup_logging
    
    setup_logging()
    
    clean_data_path = PROCESSED_DATA_DIR / "cs-training-clean.csv"
    df = pd.read_csv(clean_data_path, index_col=0)
    
    X = df.drop(columns=["SeriousDlqin2yrs"])
    y = df["SeriousDlqin2yrs"]
    
    pipeline = build_feature_pipeline()
    X_trans = pipeline.fit_transform(X, y)
    
    print("--- FEATURE ENGINEERING PIPELINE VERIFICATION ---")
    print(f"Original shape: {X.shape}")
    print(f"Transformed shape: {X_trans.shape}")
    print("\nTransformed columns:")
    print(list(X_trans.columns))
    print("\nSummary stats of transformed features (mean, std, min, max):")
    stats = pd.DataFrame({
        "Mean": X_trans.mean(),
        "Std": X_trans.std(),
        "Min": X_trans.min(),
        "Max": X_trans.max()
    })
    print(stats.round(4))

