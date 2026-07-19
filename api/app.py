import logging
from fastapi import FastAPI, HTTPException
from api.schemas import CreditInputSchema, CreditPredictionResponse
from src.predict import CreditRiskPredictor

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Credit Risk Scorer API",
    description="Production-grade API to predict credit default risk.",
    version="1.0.0"
)

# Load the predictor once globally at startup to avoid reload overhead
try:
    predictor = CreditRiskPredictor()
except Exception as e:
    logger.exception("Failed to load ML models at API startup.")
    predictor = None


@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Credit Risk Scorer API.",
        "docs_url": "/docs",
        "health_check": "/health"
    }


@app.get("/health")
def health_check():
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model assets are not loaded.")
    return {"status": "healthy"}


@app.post("/predict", response_model=CreditPredictionResponse)
def predict_risk(payload: CreditInputSchema):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model is currently unavailable.")
        
    try:
        # Convert input schema to dict, using the aliases (with hyphens) to match feature names
        input_data = payload.model_dump(by_alias=True)
        prediction_result = predictor.predict_single(input_data)
        return prediction_result
    except Exception as e:
        logger.exception("Error occurred during credit risk prediction.")
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
