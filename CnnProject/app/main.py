"""
MammoFusion FastAPI application.

Routes
------
GET  /                  -> static/index.html (the UI)
GET  /api/health        -> health check
GET  /api/model-info    -> model/architecture summary
POST /api/predict       -> multipart prediction (image / tabular / fused)

Static mounts
-------------
/static/...   -> static/      (frontend assets)
/outputs/...  -> outputs/      (generated heatmaps, served to the browser)
"""
from __future__ import annotations

import os
import logging

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles

import config
from app.inference import predict, InferenceError
from app.schemas import HealthResponse, ModelInfoResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mammofusion.api")

PROJECT_ROOT = config.PROJECT_ROOT
STATIC_DIR = os.path.join(PROJECT_ROOT, "static")
OUTPUT_DIR = config.OUTPUT_DIR
HEATMAP_DIR = os.path.join(OUTPUT_DIR, "heatmaps")

# Make sure output dirs exist before mounting.
os.makedirs(HEATMAP_DIR, exist_ok=True)

app = FastAPI(
    title="MammoFusion API",
    description="Hybrid breast cancer classification (image + tabular fusion).",
    version="1.0.0",
)

# Serve generated heatmaps and frontend assets as static files.
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ---------------------------------------------------------------------------
# Root -> UI
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="UI not found.")
    return FileResponse(index_path)


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    icon = os.path.join(STATIC_DIR, "assets", "mammofusion-icon.png")
    if os.path.exists(icon):
        return FileResponse(icon)
    return Response(status_code=204)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@app.get("/api/health", response_model=HealthResponse)
def health():
    return {"status": "ok", "message": "MammoFusion API is running"}


# ---------------------------------------------------------------------------
# Model info
# ---------------------------------------------------------------------------
@app.get("/api/model-info", response_model=ModelInfoResponse)
def model_info():
    return {
        "project": "MammoFusion",
        "description": (
            "Hybrid breast cancer classification using mammography images "
            "and CSV-based clinical metadata"
        ),
        "available_modes": ["image", "tabular", "fused"],
        "input_image_size": config.IMAGE_SIZE,
        "image_branch": "ResNet50 + SpatialAttention",
        "tabular_branch": "CSV Metadata + Embedding + MLP",
        "fusion_strategy": "Intermediate Feature Fusion",
        "output": "Binary malignancy probability",
    }


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------
@app.post("/api/predict")
async def api_predict(
    mode: str = Form(...),
    image: UploadFile | None = File(default=None),
    csv_file: UploadFile | None = File(default=None),
):
    mode = (mode or "").strip().lower()
    if mode not in ("image", "tabular", "fused"):
        raise HTTPException(
            status_code=400,
            detail="Invalid mode. Expected one of: image, tabular, fused.",
        )

    # Read uploaded bytes (only what the mode needs).
    image_bytes = None
    csv_bytes = None
    try:
        if image is not None and getattr(image, "filename", None):
            image_bytes = await image.read()
        if csv_file is not None and getattr(csv_file, "filename", None):
            csv_bytes = await csv_file.read()
    except Exception:
        logger.exception("Failed reading uploaded files")
        raise HTTPException(status_code=400, detail="Could not read uploaded files.")

    try:
        result = predict(mode=mode, image_bytes=image_bytes, csv_bytes=csv_bytes)
        return JSONResponse(content=result)
    except InferenceError as exc:
        # Expected, user-facing validation/processing error.
        raise HTTPException(status_code=400, detail=str(exc))
    except HTTPException:
        raise
    except Exception:
        # Unexpected: log full trace, return a clean generic message.
        logger.exception("Unexpected error during prediction")
        raise HTTPException(
            status_code=500,
            detail="Internal error during prediction. Please try again.",
        )
