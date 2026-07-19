import logging
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from pathlib import Path
from src.config import (
    PROCESSED_VAL_PATH,
    MODEL_PATH,
    PREPROCESSOR_PATH,
    TARGET_COLUMN,
    setup_logging
)

logger = logging.getLogger(__name__)

def generate_shap_explanations() -> None:
    logger.info("Loading model, preprocessor, and validation split...")
    val_df = pd.read_csv(PROCESSED_VAL_PATH, index_col=0)
    
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    
    X_val = val_df.drop(columns=[TARGET_COLUMN])
    y_val = val_df[TARGET_COLUMN]
    
    X_val_trans = preprocessor.transform(X_val)
    
    logger.info("Initializing SHAP TreeExplainer...")
    # Use TreeExplainer for optimized, exact SHAP computation on tree models
    explainer = shap.TreeExplainer(model)
    
    # Compute SHAP values on a representative sample of 1,000 records for performance
    sample_size = min(1000, len(X_val_trans))
    X_sample = X_val_trans.iloc[:sample_size]
    
    logger.info(f"Computing SHAP values for {sample_size} records...")
    shap_values = explainer(X_sample)
    
    plot_dir = Path("notebooks/plots")
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Global Summary Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.title("SHAP Global Feature Importance & Impact Plot", fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(plot_dir / "shap_summary.png", dpi=150)
    plt.close()
    logger.info("SHAP Summary Plot saved.")
    
    # Identify a high-risk default customer to explain (first record where y_val == 1 in sample)
    sample_y = y_val.iloc[:sample_size]
    default_indices = sample_y[sample_y == 1].index
    if len(default_indices) == 0:
        logger.warning("No default cases found in validation sample to explain. Defaulting to first row.")
        target_idx = 0
    else:
        # Use get_indexer to find position of the index key in the sample DataFrame
        target_idx = X_sample.index.get_indexer([default_indices[0]])[0]
        
    logger.info(f"Explaining predictions for customer at index {target_idx} (Val Set index: {X_val.index[target_idx]})...")
    
    # 2. Local Waterfall Plot
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_values[target_idx], show=False)
    plt.title("SHAP Waterfall Plot - Single Customer Explanation", fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(plot_dir / "shap_waterfall.png", dpi=150)
    plt.close()
    logger.info("SHAP Waterfall Plot saved.")
    
    # 3. Local Force Plot
    # force_plot needs matplotlib=True for saving to a file
    plt.figure(figsize=(12, 4))
    shap.plots.force(
        explainer.expected_value,
        shap_values.values[target_idx],
        X_sample.iloc[target_idx],
        matplotlib=True,
        show=False
    )
    plt.title("SHAP Force Plot - Single Customer Explanation", fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(plot_dir / "shap_force.png", dpi=150)
    plt.close()
    logger.info("SHAP Force Plot saved.")
    
    # Print Text Explanation of the specific customer
    print("\n" + "="*70)
    print(f"         CUSTOMER DEFAULT RISK MODEL EXPLANATION (INDEX {X_val.index[target_idx]})")
    print("="*70)
    
    # Calculate base expected probability vs actual predicted probability
    expected_prob = 1 / (1 + np.exp(-explainer.expected_value))
    raw_output = explainer.expected_value + np.sum(shap_values.values[target_idx])
    pred_prob = 1 / (1 + np.exp(-raw_output))
    
    print(f"Base (Expected) Default Probability: {expected_prob * 100:.2f}%")
    print(f"Actual Predicted Default Probability: {pred_prob * 100:.2f}%")
    
    # Highlight top features contributing to the risk increment
    X_val_added = preprocessor.named_steps["feature_adder"].transform(X_val)
    
    # Strip prefixes to align the rearranged columns with the original features
    original_cols = [col.replace("skewed__", "").replace("other__", "") for col in X_sample.columns]
    raw_values = X_val_added.loc[X_val.index[target_idx], original_cols].values
    
    features = X_sample.columns
    contributions = shap_values.values[target_idx]
    
    contrib_df = pd.DataFrame({
        "Feature": features,
        "Raw Value": raw_values,
        "Scaled/Transformed Value": X_sample.iloc[target_idx].values,
        "SHAP Contribution (Log-Odds)": contributions
    }).sort_values(by="SHAP Contribution (Log-Odds)", key=abs, ascending=False)
    
    print("\nTop Contributing Features to Risk Shift:")
    print(contrib_df.round(4).to_string(index=False))
    print("="*70 + "\n")


if __name__ == "__main__":
    setup_logging()
    generate_shap_explanations()
