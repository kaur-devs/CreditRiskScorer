# Credit Risk Scorer

A production-quality Credit Risk Prediction and Scoring System built with Python, scikit-learn, XGBoost, and FastAPI.

This system predicts whether a customer will experience financial distress (specifically, a serious credit delinquency) within the next 2 years using the Kaggle dataset **"Give Me Some Credit"**.

---

## Project Overview

In credit scoring, predicting defaults accurately is crucial for risk management. The dataset contains **150,000 credit records** and exhibits a severe class imbalance where only **6.68%** of customers represent default instances. 

This project demonstrates industrial machine learning engineering practices:
* **Anti-Data Leakage Pipeline**: Features strict separation of fitting (medians, outlier thresholds) on training data only.
* **Modern Preprocessing**: Standardized pipeline using custom scikit-learn transformers (`ColumnTransformer`, `Pipeline`) configured to output pandas DataFrames natively.
* **Class Imbalance Solutions**: Compares SMOTE, Random Oversampling, Class Weights, and optimal post-prediction decision threshold tuning.
* **Advanced XGBoost Tuning**: Hyperparameter tuning via `RandomizedSearchCV` optimizing the **ROC-AUC** metric.
* **SHAP Explainability**: Integrates SHapley Additive exPlanations to unpack predictions globally and locally for individual customers.
* **Production Web API**: High-performance FastAPI server with strict input validation using Pydantic aliases.
* **Robust Docker Containerization**: Lightweight multi-stage equivalent Docker configurations.
* **Automated Unit Testing**: Comprehensive endpoint validation using `pytest` and FastAPI's `TestClient`.

---

## Repository Structure

```text
credit-risk-scorer/
│
├── api/
│   ├── app.py          # FastAPI application server
│   └── schemas.py      # Pydantic request/response validation schemas
│
├── data/
│   ├── raw/            # Ingested raw Kaggle dataset (gitignored)
│   └── processed/      # Stratified train, val, and test splits (gitignored)
│
├── docker/             # Docker deployment configurations
│
├── model/              # Serialized cleaner, scaler, and XGBoost models (gitignored)
│
├── notebooks/          # Exploratory notebooks and reports
│   ├── plots/          # Generated distribution, curve, and SHAP plots
│   ├── 01_EDA.ipynb    # Exploratory Data Analysis notebook
│   └── 02_Modeling.ipynb # XGBoost modeling and tuning experiments
│
├── src/                # Modular codebase
│   ├── config.py       # Base pathing and column parameters configuration
│   ├── data_loader.py  # Data loading and profiling module
│   ├── preprocessing.py # DataCleaner logic (duplicates, missingness, capping)
│   ├── feature_engineering.py # Scaler and interaction feature transformer pipeline
│   ├── train.py        # Splitting, baseline training, and XGBoost tuning
│   ├── evaluate.py     # Evaluation metrics and plotting helpers
│   ├── evaluate_final.py # Final model cross-validation and calibration diagnostics
│   ├── explain.py      # SHAP global and local explainability generator
│   ├── predict.py      # Inference predictor engine
│   └── utils.py        # Generic utilities
│
├── tests/
│   └── test_api.py     # pytest API endpoint and Pydantic schema validation tests
│
├── .dockerignore       # Docker build exclusions file
├── .gitignore          # Git tracking exclusions file
├── Dockerfile          # Production Docker container build script
├── docker-compose.yml  # Multi-container orchestration configurations
├── requirements.txt    # Declared project dependencies
└── README.md           # Comprehensive project documentation
```

---

## Installation & Setup

### Prerequisites
* Python 3.12+
* Git
* Docker (Optional, for containerized run)

### 1. Clone the Repository
```bash
git clone https://github.com/kaur-devs/CreditRiskScorer.git
cd CreditRiskScorer
```

### 2. Set Up the Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Pipeline Execution (Usage)

To build the artifacts and train the models step-by-step:

### 1. Ingest Data
Loads raw dataset, generates profile statistics, and saves a verification copy:
```bash
python3 -m src.data_loader
```

### 2. Clean Data
Removes duplicates, caps utilization/debt outliers at the 99th percentile, replaces missing late-payment placeholders (96, 98), and saves the clean dataset:
```bash
python3 -m src.preprocessing
```

### 3. Feature Engineering Pipeline
Verifies scaling, log-transforms right-skewed variables (`MonthlyIncome`, `DebtRatio`), and appends engineered interaction variables (`IncomePerPerson`, `TotalLatePayments`):
```bash
python3 -m src.feature_engineering
```

### 4. Splitting & Model Tuning
Splits data into stratified train (70%), validation (15%), and test (15%) subsets, runs class imbalance comparisons, performs `RandomizedSearchCV` on XGBoost, and saves the best model/preprocessor:
```bash
python3 -m src.train
```

### 5. Final Model Evaluation
Runs 5-fold cross-validation, prints the validation classification report, and exports calibration curves and feature importances:
```bash
python3 -m src.evaluate_final
```

### 6. SHAP Explanations
Generates global summary plots, waterfalls, and force charts explaining local risk scores:
```bash
python3 -m src.explain
```

---

## Model Evaluation & Results

### 1. Class Imbalance Benchmark (Logistic Regression)

| Method | Validation Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Thresh=0.5)** | **93.59%** | **58.10%** | 15.51% | 24.49% | 85.29% |
| **SMOTE** | 78.79% | 20.71% | 76.56% | 32.61% | 85.76% |
| **Random Oversampling** | 79.09% | 21.00% | 76.70% | 32.97% | **85.85%** |
| **Class Weights (Balanced)** | 79.07% | 20.99% | **76.83%** | 32.98% | **85.85%** |
| **Threshold Tuning (Thresh=0.1587)** | 91.23% | 37.94% | 48.60% | **42.62%** | 85.29% |

### 2. Tuned XGBoost Model (Final Model)
* **Cross-Validation ROC-AUC**: **0.8639 ± 0.0032** (displays high model stability)
* **Validation ROC-AUC**: **0.8674**
* **Validation Accuracy**: **93.76%**
* **Validation Precision**: **64.76%**

---

## FastAPI Web Service

### Launch the Server
```bash
uvicorn api.app:app --reload
```
Access the interactive API documentation at: `http://127.0.0.1:8000/docs`

### Example Request (`POST /predict`)
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
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
     }'
```

### Example Response
```json
{
  "probability": 0.012605533003807068,
  "prediction": 0,
  "risk_label": "Low Risk",
  "confidence_level": "Low Risk"
}
```

---

## Docker Deployment

Build and orchestrate the containerized FastAPI server using Docker Compose:

```bash
# Build and run the server in the background
docker compose up --build -d

# Verify health status
curl http://localhost:8000/health
```

---

## Explainability (SHAP Insights)

SHAP global feature importance maps show that the top default risk drivers are:
1. **Revolving Credit Utilization**: Balanced card limits relative to total outstanding balance.
2. **Total Historical Late Payments**: Sum of delinquency occurrences.
3. **Debt-to-Income Ratio**: Monthly expenses relative to gross income.

For local explanations, a sample waterfall chart showing contribution log-odds shifts is exported to `notebooks/plots/shap_waterfall.png`.

---

## Future Improvements
* **Advanced Ensembles**: Stack XGBoost with LightGBM and CatBoost to form meta-ensembles.
* **SMOTE + XGBoost Tuning**: Hyperparameter optimize scale-pos-weight in combination with training set oversampling.
* **API SHAP Endpoint**: Expose a `/predict/explain` route to return JSON-formatted local SHAP values alongside prediction results.

---

## License

This project is licensed under the MIT License.
