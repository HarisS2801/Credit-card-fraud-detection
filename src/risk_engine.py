from __future__ import annotations

from dataclasses import dataclass

RISK_THRESHOLDS = {
    "low_threshold": 0.30,
    "high_threshold": 0.70,
}


@dataclass(frozen=True)
class RiskDecision:
    risk_level: str
    recommended_action: str


def classify_fraud_risk(fraud_probability: float) -> RiskDecision:
    """Classify transaction risk using demonstration thresholds.

    These thresholds are for project demonstration and portfolio use only.
    They are not real banking rules or production fraud-policy settings.
    """
    if fraud_probability < RISK_THRESHOLDS["low_threshold"]:
        return RiskDecision(risk_level="LOW", recommended_action="APPROVE")
    if fraud_probability < RISK_THRESHOLDS["high_threshold"]:
        return RiskDecision(risk_level="MEDIUM", recommended_action="REVIEW")
    return RiskDecision(risk_level="HIGH", recommended_action="BLOCK")
