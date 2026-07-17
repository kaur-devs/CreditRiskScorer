import logging
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Union
from src.config import MODEL_DIR, setup_logging
from src.preprocessing import DataCleaner

logger = logging.getLogger(__name__)

class CreditRiskPredictor:
    def __init__(self, model_dir: Path = MODEL_DIR) -> None:
        self.model_dir = Path(model_dir)
        self.model_path = self.model_dir / "model.pkl"
        self.preprocessor_path = self.model_dir / "preprocessor.pkl"
        self.cleaner_path = self.model_dir / "cleaner.pkl"
        
        self.model = joblib.load(self.model_path)
        self.preprocessor = joblib.load(self.preprocessor_path)
        
        if self.cleaner_path.exists():
            self.cleaner = joblib.load(self.cleaner_path)
        else:
            logger.info("cleaner.pkl not found. Fitting a new DataCleaner on raw training data...")
            raw_train_path = Path("data/raw/cs-training.csv")
            if not raw_train_path.exists():
                raise FileNotFoundError(f"Raw data required to fit cleaner at {raw_train_path}")
            
            df_raw = pd.read_csv(raw_train_path, index_col=0)
            df_no_dups = df_raw.drop_duplicates()
            
            self.cleaner = DataCleaner()
            self.cleaner.fit(df_no_dups)
            joblib.dump(self.cleaner, self.cleaner_path)
            logger.info(f"Fitted cleaner serialized to {self.cleaner_path}")
            
        self.threshold = 0.1587

    def predict(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        df_clean = self.cleaner.transform(df_raw)
        
        if "SeriousDlqin2yrs" in df_clean.columns:
            df_clean = df_clean.drop(columns=["SeriousDlqin2yrs"])
            
        X_trans = self.preprocessor.transform(df_clean)
        
        probabilities = self.model.predict_proba(X_trans)[:, 1]
        
        predictions = (probabilities >= self.threshold).astype(int)
        
        confidences = np.where(
            probabilities >= 0.5, "Very High Risk",
            np.where(probabilities >= self.threshold, "High Risk",
            np.where(probabilities >= 0.05, "Medium Risk", "Low Risk"))
        )
        
        results = pd.DataFrame({
            "probability": probabilities,
            "prediction": predictions,
            "risk_label": np.where(predictions == 1, "High Risk", "Low Risk"),
            "confidence_level": confidences
        }, index=df_raw.index)
        
        return results

    def predict_single(self, record: Dict[str, Any]) -> Dict[str, Any]:
        df_raw = pd.DataFrame([record])
        results = self.predict(df_raw)
        row = results.iloc[0]
        
        return {
            "probability": float(row["probability"]),
            "prediction": int(row["prediction"]),
            "risk_label": str(row["risk_label"]),
            "confidence_level": str(row["confidence_level"])
        }


if __name__ == "__main__":
    setup_logging()
    
    predictor = CreditRiskPredictor()
    
    mock_customers = [
        {
            "RevolvingUtilizationOfUnsecuredLines": 0.05,
            "age": 45,
            "NumberOfTime30-59DaysPastDueNotWorse": 0,
            "DebtRatio": 0.25,
            "MonthlyIncome": 8500.0,
            "NumberOfOpenCreditLinesAndLoans": 8,
            "NumberOfTimes90DaysLate": 0,
            "NumberRealEstateLoansOrLines": 1,
            "NumberOfTime60-89DaysPastDueNotWorse": 0,
            "NumberOfDependents": 1.0
        },
        {
            "RevolvingUtilizationOfUnsecuredLines": 0.95,
            "age": 32,
            "NumberOfTime30-59DaysPastDueNotWorse": 2,
            "DebtRatio": 0.85,
            "MonthlyIncome": 2500.0,
            "NumberOfOpenCreditLinesAndLoans": 4,
            "NumberOfTimes90DaysLate": 1,
            "NumberRealEstateLoansOrLines": 0,
            "NumberOfTime60-89DaysPastDueNotWorse": 1,
            "NumberOfDependents": 2.0
        }
    ]
    
    print("--- MOCK CUSTOMERS PREDICTIONS ---")
    for i, customer in enumerate(mock_customers, 1):
        res = predictor.predict_single(customer)
        print(f"\nCustomer {i} Input: age={customer['age']}, income={customer['MonthlyIncome']}, utilization={customer['RevolvingUtilizationOfUnsecuredLines']}")
        print(f"Prediction Output: {res}")
