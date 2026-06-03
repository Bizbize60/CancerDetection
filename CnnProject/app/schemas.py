"""
Pydantic response schemas.

These are used mainly for the auto-generated OpenAPI docs and for the
static endpoints (/api/health, /api/model-info). The /api/predict endpoint
returns either a single or a batch payload depending on the request, so it
is documented loosely and returns a plain dict (validated by hand).
"""
from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str = "ok"
    message: str = "MammoFusion API is running"


class ModelInfoResponse(BaseModel):
    project: str
    description: str
    available_modes: List[str]
    input_image_size: int
    image_branch: str
    tabular_branch: str
    fusion_strategy: str
    output: str


class SinglePrediction(BaseModel):
    mode: str
    probability: float
    predicted_class: str
    risk_level: str
    threshold: float
    interpretation: str
    disclaimer: str
    heatmap_url: Optional[str] = None
    overlay_url: Optional[str] = None
    warning: Optional[str] = None


class BatchPredictionRow(BaseModel):
    row_id: int
    probability: float
    predicted_class: str
    risk_level: str


class BatchPrediction(BaseModel):
    mode: str
    count: int
    predictions: List[BatchPredictionRow]
    threshold: float
    disclaimer: str
    heatmap_url: Optional[str] = None
    overlay_url: Optional[str] = None
    warning: Optional[str] = None
