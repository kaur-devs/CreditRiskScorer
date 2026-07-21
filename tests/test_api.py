import pytest
from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_predict_valid_low_risk():
    payload = {
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
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "probability" in data
    assert data["prediction"] == 0
    assert data["risk_label"] == "Low Risk"


def test_predict_valid_high_risk():
    payload = {
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
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == 1
    assert data["risk_label"] == "High Risk"


def test_predict_invalid_negative_utilization():
    payload = {
        "RevolvingUtilizationOfUnsecuredLines": -0.5,
        "age": 45,
        "NumberOfTime30-59DaysPastDueNotWorse": 0,
        "DebtRatio": 0.25,
        "MonthlyIncome": 8500.0,
        "NumberOfOpenCreditLinesAndLoans": 8,
        "NumberOfTimes90DaysLate": 0,
        "NumberRealEstateLoansOrLines": 1,
        "NumberOfTime60-89DaysPastDueNotWorse": 0,
        "NumberOfDependents": 1.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_invalid_age_out_of_bounds():
    payload = {
        "RevolvingUtilizationOfUnsecuredLines": 0.5,
        "age": 150,
        "NumberOfTime30-59DaysPastDueNotWorse": 0,
        "DebtRatio": 0.25,
        "MonthlyIncome": 8500.0,
        "NumberOfOpenCreditLinesAndLoans": 8,
        "NumberOfTimes90DaysLate": 0,
        "NumberRealEstateLoansOrLines": 1,
        "NumberOfTime60-89DaysPastDueNotWorse": 0,
        "NumberOfDependents": 1.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_missing_required_field():
    payload = {
        "RevolvingUtilizationOfUnsecuredLines": 0.5,
        "age": 45,
        "MonthlyIncome": 8500.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
