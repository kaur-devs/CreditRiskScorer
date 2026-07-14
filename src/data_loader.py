import logging
import pandas as pd
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, filepath: Path) -> None:
        self.filepath = Path(filepath)
        self.data: pd.DataFrame = pd.DataFrame()

    def load_data(self) -> pd.DataFrame:
        logger.info(f"Ingesting raw data from {self.filepath}...")
        
        if not self.filepath.exists():
            error_msg = f"Data file not found at: {self.filepath.absolute()}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
            
        try:
            self.data = pd.read_csv(self.filepath, index_col=0)
            logger.info(f"Successfully loaded {self.data.shape[0]} rows and {self.data.shape[1]} columns.")
            return self.data
        except pd.errors.EmptyDataError as e:
            logger.error("The file is empty.")
            raise ValueError("The provided data file is empty.") from e
        except Exception as e:
            logger.error(f"Error parsing data file: {str(e)}")
            raise ValueError(f"Failed to load dataset: {str(e)}") from e

    def inspect_dataset(self) -> Dict[str, Any]:
        if self.data.empty:
            logger.warning("No data loaded. Call load_data() first.")
            return {}
            
        logger.info("Inspecting dataset structure and stats...")
        
        target_col = "SeriousDlqin2yrs"
        target_distribution = {}
        if target_col in self.data.columns:
            counts = self.data[target_col].value_counts().to_dict()
            percentages = self.data[target_col].value_counts(normalize=True).to_dict()
            target_distribution = {
                "counts": counts,
                "percentages": percentages
            }
            logger.info(f"Target distribution: {counts} (Percentages: {percentages})")
        else:
            logger.warning(f"Target column '{target_col}' not found in dataset columns.")
            
        missing_counts = self.data.isnull().sum().to_dict()
        missing_percentages = (self.data.isnull().mean() * 100).to_dict()
        
        summary = {
            "shape": self.data.shape,
            "columns": list(self.data.columns),
            "missing_counts": missing_counts,
            "missing_percentages": missing_percentages,
            "target_distribution": target_distribution
        }
        
        return summary

    def save_data(self, df: pd.DataFrame, output_path: Path) -> None:
        logger.info(f"Saving data to: {output_path}...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            df.to_csv(output_path, index=True)
            logger.info(f"Successfully saved data file ({df.shape[0]} rows).")
        except Exception as e:
            logger.error(f"Failed to save data: {str(e)}")
            raise IOError(f"Could not write CSV to disk: {str(e)}") from e


if __name__ == "__main__":
    from src.config import RAW_DATA_PATH, PROCESSED_DATA_DIR, setup_logging
    
    setup_logging()
    loader = DataLoader(RAW_DATA_PATH)
    
    try:
        df = loader.load_data()
        summary = loader.inspect_dataset()
        
        print("\n--- DATASET PROFILE SUMMARY ---")
        print(f"Dataset Shape: {summary['shape']}")
        
        print("\nMissing Values by Column:")
        for col in summary['columns']:
            count = summary['missing_counts'][col]
            pct = summary['missing_percentages'][col]
            print(f" - {col}: {count} ({pct:.2f}%)")
            
        print("\nTarget Class Distribution ('SeriousDlqin2yrs'):")
        counts = summary['target_distribution']['counts']
        pcts = summary['target_distribution']['percentages']
        for k, v in counts.items():
            print(f" - Class {k}: {v} ({pcts[k]*100:.2f}%)")
            
        processed_output_path = PROCESSED_DATA_DIR / "cs-training-raw.csv"
        loader.save_data(df, processed_output_path)
        
    except Exception as e:
        logger.exception(f"Execution failed: {str(e)}")
