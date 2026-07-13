# Credit Risk Scorer

A production-quality Credit Risk Prediction System built with Python, scikit-learn, XGBoost, and FastAPI.

This system predicts whether a customer will experience financial distress within the next 2 years using the Kaggle dataset **"Give Me Some Credit"**.

## Project Architecture & Structure

```text
credit-risk-scorer/
│
├── data/
│   ├── raw/            # Raw data files (gitignored)
│   └── processed/      # Cleaned and processed data files (gitignored)
│
├── notebooks/          # Jupyter notebooks for EDA and modeling experiments
│
├── src/                # Modular source code
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── explain.py
│   └── utils.py
│
├── model/              # Serialized model artifacts (gitignored)
│
├── api/                # FastAPI application code
│   ├── app.py
│   └── schemas.py
│
├── tests/              # Unit and integration tests
│
├── docker/             # Docker deployment files
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Installation & Setup

### Prerequisites
- Python 3.12+
- Git

### 1. Clone the Repository
```bash
git clone <repository-url>
cd credit-risk-scorer
```

### 2. Set Up Virtual Environment
Create and activate a virtual environment to manage dependencies in isolation:

**On macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**On Windows (Command Prompt):**
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

**On Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\activate.ps1
```

### 3. Install Dependencies
Install all package dependencies defined in `requirements.txt`:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Next Steps
- **Phase 2**: Dataset Loading and ingestion utilities.
- **Phase 3**: Exploratory Data Analysis (EDA).
