"""
Shared helper utilities for the MammoFusion inference API.

Contains the medical-wording rules (risk level, predicted class,
interpretation) and small formatting helpers. Kept dependency-free so it
can be imported by both inference.py and main.py without side effects.
"""
from __future__ import annotations

# Default decision threshold (probability >= THRESHOLD => positive class).
DEFAULT_THRESHOLD = 0.5

# Always attached to every prediction response.
DISCLAIMER = "This AI result is for research and decision-support purposes only."

VALID_MODES = ("image", "tabular", "fused")


def risk_level_from_probability(probability: float) -> str:
    """
    Map a malignancy probability to a coarse risk level.

      probability < 0.35            -> "Low"
      0.35 <= probability < 0.65    -> "Moderate"
      probability >= 0.65           -> "High"
    """
    if probability < 0.35:
        return "Low"
    if probability < 0.65:
        return "Moderate"
    return "High"


def predicted_class_from_risk(risk_level: str) -> str:
    """Human-readable class label derived from the risk level."""
    return {
        "Low": "Benign / Low Risk",
        "Moderate": "Moderate Risk",
        "High": "Malignant Risk",
    }.get(risk_level, "Moderate Risk")


def interpretation_for(risk_level: str) -> str:
    """
    Medically-responsible interpretation text. Never phrased as a diagnosis.
    """
    if risk_level == "High":
        return (
            "The model predicts elevated malignancy risk. This result is not a "
            "medical diagnosis and should be reviewed by a qualified clinician."
        )
    if risk_level == "Moderate":
        return (
            "The model predicts intermediate malignancy risk. This is not a "
            "diagnosis; further clinical investigation is recommended."
        )
    return (
        "The model predicts low malignancy risk. This is not a diagnosis; "
        "routine screening guidance still applies."
    )


def round_prob(probability: float, ndigits: int = 4) -> float:
    """Round a probability for clean JSON output."""
    return round(float(probability), ndigits)
