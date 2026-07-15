import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_loader import DataLoader
from src.config import RAW_DATA_PATH

def run_eda_visualizations():
    loader = DataLoader(RAW_DATA_PATH)
    df = loader.load_data()
    
    plot_dir = Path("notebooks/plots")
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    sns.set_theme(style="whitegrid")
    
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(x="SeriousDlqin2yrs", data=df, hue="SeriousDlqin2yrs", palette="muted", legend=False)
    plt.title("Target Class Distribution", fontsize=14, pad=15)
    plt.xlabel("Serious Delinquency in 2 Years", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    
    total = len(df)
    for p in ax.patches:
        percentage = f"{100 * p.get_height() / total:.2f}%"
        ax.annotate(percentage, (p.get_x() + p.get_width() / 2., p.get_height() + 1500),
                    ha="center", va="center", fontsize=10, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(plot_dir / "target_imbalance.png", dpi=150)
    plt.close()
    
    plt.figure(figsize=(10, 8))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", 
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title("Feature Correlation Heatmap", fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(plot_dir / "correlation_heatmap.png", dpi=150)
    plt.close()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    sns.histplot(df["age"], bins=30, kde=True, ax=axes[0, 0], color="skyblue")
    axes[0, 0].set_title(f"Age Distribution (Skew: {df['age'].skew():.2f})")
    
    sns.histplot(df["NumberOfOpenCreditLinesAndLoans"], bins=20, kde=True, ax=axes[0, 1], color="salmon")
    axes[0, 1].set_title(f"Open Credit Lines Distribution (Skew: {df['NumberOfOpenCreditLinesAndLoans'].skew():.2f})")
    
    income_non_null = df["MonthlyIncome"].dropna()
    sns.histplot(income_non_null[income_non_null < 25000], bins=30, kde=True, ax=axes[1, 0], color="lightgreen")
    axes[1, 0].set_title(f"Monthly Income (<25k) (Skew: {df['MonthlyIncome'].skew():.2f})")
    
    debt_ratio_filtered = df[df["DebtRatio"] < 2]["DebtRatio"]
    sns.histplot(debt_ratio_filtered, bins=30, kde=True, ax=axes[1, 1], color="gold")
    axes[1, 1].set_title(f"Debt Ratio (<2) (Skew: {df['DebtRatio'].skew():.2f})")
    
    plt.tight_layout()
    plt.savefig(plot_dir / "feature_distributions.png", dpi=150)
    plt.close()
    
    cols_to_plot = [
        "RevolvingUtilizationOfUnsecuredLines", 
        "DebtRatio", 
        "MonthlyIncome"
    ]
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    for i, col in enumerate(cols_to_plot):
        sns.boxplot(x=df[col].dropna(), ax=axes[i], color="lightcoral")
        axes[i].set_title(f"Boxplot of {col}", fontsize=12)
    plt.tight_layout()
    plt.savefig(plot_dir / "outliers_boxplot.png", dpi=150)
    plt.close()
    
    duplicates_count = df.duplicated().sum()
    print(f"Total Duplicates: {duplicates_count}")
    print("\nFeature Skewness:")
    print(df.skew())

if __name__ == "__main__":
    run_eda_visualizations()
