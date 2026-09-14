from __future__ import annotations

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

from database import save_prediction_record
from src.model_service import get_prediction_service

app = FastAPI(title="Credit Card Fraud Detection API", version="1.0.0")


class TransactionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    Time: float = Field(..., description="Transaction time value from the dataset")
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float


class PredictionResponse(BaseModel):
    prediction: str
    fraud_probability: float
    risk_level: str
    recommended_action: str


@app.get("/health")
def health_check() -> dict[str, object]:
    try:
        service = get_prediction_service()
        model_loaded = service is not None and hasattr(service, "model")
    except Exception:
        model_loaded = False

    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_transaction(payload: TransactionRequest) -> PredictionResponse:
    try:
        service = get_prediction_service()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model files are missing. Train the model first.",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model failed to load: {exc}",
        ) from exc

    transaction_dict = payload.model_dump()

    try:
        result = service.predict_transaction(transaction_dict)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {exc}",
        ) from exc

    response = PredictionResponse(
        prediction=result["prediction"],
        fraud_probability=result["fraud_probability"],
        risk_level=result["risk_level"],
        recommended_action=result["recommended_action"],
    )

    save_prediction_record(
        transaction_amount=float(payload.Amount),
        fraud_probability=float(response.fraud_probability),
        prediction=response.prediction,
        risk_level=response.risk_level,
        recommended_action=response.recommended_action,
    )

    return response
