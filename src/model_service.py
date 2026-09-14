from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.risk_engine import classify_fraud_risk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "fraud_model.pkl"
FEATURES_PATH = PROJECT_ROOT / "models" / "features.pkl"
SCALER_PATH = PROJECT_ROOT / "models" / "scaler.pkl"


class FraudPredictionService:
    def __init__(self, model_path: str | Path = MODEL_PATH, features_path: str | Path = FEATURES_PATH, scaler_path: str | Path = SCALER_PATH):
        self.model_path = Path(model_path)
        self.features_path = Path(features_path)
        self.scaler_path = Path(scaler_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        if not self.features_path.exists():
            raise FileNotFoundError(f"Feature list not found: {self.features_path}")

        self.model = joblib.load(self.model_path)
        self.feature_names = joblib.load(self.features_path)
        self.scaler = joblib.load(self.scaler_path) if self.scaler_path.exists() else None

    def validate_input(self, transaction: dict[str, Any]) -> dict[str, float]:
        if not isinstance(transaction, dict):
            raise ValueError("Transaction must be a dictionary of feature names to numeric values.")
        if not transaction:
            raise ValueError("Transaction payload is empty.")

        missing = [name for name in self.feature_names if name not in transaction]
        if missing:
            raise ValueError(f"Missing required transaction features: {missing}")

        validated: dict[str, float] = {}
        for name in self.feature_names:
            value = transaction[name]
            try:
                validated[name] = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Feature '{name}' must be numeric. Received: {value!r}") from exc

        return validated

    def predict_transaction(self, transaction: dict[str, Any]) -> dict[str, Any]:
        validated = self.validate_input(transaction)
        feature_frame = pd.DataFrame([validated], columns=self.feature_names)

        if self.scaler is not None:
            transformed = self.scaler.transform(feature_frame)
        else:
            transformed = feature_frame

        prediction = int(self.model.predict(transformed)[0])
        fraud_probability = float(self.model.predict_proba(transformed)[0, 1])
        decision = classify_fraud_risk(fraud_probability)

        return {
            "prediction": "FRAUD" if prediction == 1 else "NORMAL",
            "fraud_probability": round(fraud_probability, 6),
            "risk_level": decision.risk_level,
            "recommended_action": decision.recommended_action,
        }


@lru_cache(maxsize=1)
def get_prediction_service() -> FraudPredictionService:
    return FraudPredictionService()
