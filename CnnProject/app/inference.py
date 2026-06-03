"""
MammoFusion inference pipeline.
 
Responsibilities
----------------
* Lazy-load + cache the three model variants (image / tabular / fused).
* Preprocess uploaded images and CSV metadata so they match the
  training/validation pipeline in dataset.py / tabular_encoder.py.
* Run inference (always under model.eval() + torch.no_grad()).
* Generate a spatial-attention heatmap for image / fused modes.
* Format single & batch responses with medically-responsible wording.
 
All user-facing failures raise InferenceError(message); main.py turns these
into clean JSON {"detail": ...} responses. Full stack traces stay in logs.
"""
from __future__ import annotations
 
import io
import os
import logging
import uuid
 
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
 
import config
from model import build_model
from tabular_encoder import (
    TabularPreprocessor, normalise_columns, fill_missing_tabular_cols,
    CAT_FEATURES, NUM_FEATURES,
)
from app.utils import (
    DEFAULT_THRESHOLD, DISCLAIMER, VALID_MODES,
    risk_level_from_probability, predicted_class_from_risk,
    interpretation_for, round_prob,
)
from app.heatmap import generate_spatial_attention_heatmap
 
logger = logging.getLogger("mammofusion.inference")
 
# ImageNet normalization — must match dataset.py.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
 
DEVICE = config.DEVICE
HEATMAP_DIR = os.path.join(config.OUTPUT_DIR, "heatmaps")
PREPROCESSOR_PATH = os.path.join(config.CHECKPOINT_DIR, "tabular_preprocessor.pkl")
 
# Global lazy model cache: mode -> bundle dict.
loaded_models: dict[str, dict] = {}
 
 
class InferenceError(Exception):
    """User-facing error; message is safe to surface as JSON `detail`."""
    pass
 
 
# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def get_checkpoint_path(mode: str) -> str:
    if mode not in VALID_MODES:
        raise InferenceError(
            "Invalid mode. Expected one of: image, tabular, fused."
        )
    return os.path.join(config.CHECKPOINT_DIR, f"best_{mode}.pt")
 
 
def load_model_bundle(mode: str) -> dict:
    """
    Lazily load (and cache) the model bundle for a mode.
 
    Returns: {"model", "preprocessor", "device", "cat_dims"}
    """
    if mode in loaded_models:
        return loaded_models[mode]
 
    if mode not in VALID_MODES:
        raise InferenceError(
            "Invalid mode. Expected one of: image, tabular, fused."
        )
 
    ckpt_path = get_checkpoint_path(mode)
    if not os.path.exists(ckpt_path):
        raise InferenceError(f"Checkpoint not found: checkpoints/best_{mode}.pt")
 
    try:
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    except Exception as exc:  # corrupt / incompatible checkpoint
        logger.exception("Failed to load checkpoint %s", ckpt_path)
        raise InferenceError(f"Could not load checkpoint: best_{mode}.pt") from exc
 
    cat_dims = ckpt.get("cat_dims")
 
    preprocessor = None
    if mode in ("tabular", "fused"):
        if not os.path.exists(PREPROCESSOR_PATH):
            raise InferenceError(
                "Checkpoint not found: checkpoints/tabular_preprocessor.pkl"
            )
        preprocessor = TabularPreprocessor.load(PREPROCESSOR_PATH)
 
    try:
        model = build_model(mode=mode, cat_dims=cat_dims, pretrained=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(DEVICE)
        model.eval()
    except Exception as exc:
        logger.exception("Failed to build/load model for mode=%s", mode)
        raise InferenceError(f"Could not initialise model for mode: {mode}") from exc
 
    bundle = {
        "model": model,
        "preprocessor": preprocessor,
        "device": DEVICE,
        "cat_dims": cat_dims,
    }
    loaded_models[mode] = bundle
    logger.info("Loaded model bundle for mode=%s (val_auc=%s)",
                mode, ckpt.get("val_auc"))
    return bundle
 
 
# ---------------------------------------------------------------------------
# Image preprocessing  (matches dataset.py: grayscale -> 3ch -> resize+pad ->
# ImageNet normalize)
# ---------------------------------------------------------------------------
def preprocess_image(image_bytes: bytes):
    """
    Returns (image_tensor[1,3,H,W] on DEVICE, original_rgb_pil).
 
    The returned PIL image is the un-normalized grayscale mammogram (as RGB),
    used as the base for the heatmap overlay.
    """
    try:
        pil = Image.open(io.BytesIO(image_bytes))
        pil.load()
    except Exception as exc:
        raise InferenceError(
            "Could not read image file. Please upload a PNG or JPEG image."
        ) from exc
 
    # Grayscale -> 3 identical channels (mirrors dataset._load_image).
    gray = pil.convert("L")
    original_rgb_pil = gray.convert("RGB")
 
    size = config.IMAGE_SIZE
    arr = np.asarray(gray, dtype=np.uint8)
    h, w = arr.shape[:2]
    if h == 0 or w == 0:
        raise InferenceError(
            "Could not read image file. Please upload a PNG or JPEG image."
        )
 
    import cv2  # local import; opencv is a backend dependency
 
    # LongestMaxSize: scale so the longest side == size, keep aspect ratio.
    scale = size / float(max(h, w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
 
    # PadIfNeeded (centered, constant 0) to size x size.
    pad_h = size - new_h
    pad_w = size - new_w
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        borderType=cv2.BORDER_CONSTANT, value=0,
    )
 
    # Stack to 3 channels, scale to [0,1], ImageNet-normalize.
    rgb = np.stack([padded] * 3, axis=-1).astype(np.float32) / 255.0
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
 
    # HWC -> CHW -> [1,3,H,W]
    tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).contiguous()
    tensor = tensor.to(DEVICE)
    return tensor, original_rgb_pil
 
 
# ---------------------------------------------------------------------------
# CSV preprocessing  (matches tabular_encoder transform)
# ---------------------------------------------------------------------------
def preprocess_csv(csv_bytes: bytes, preprocessor: TabularPreprocessor, device):
    """
    Returns (cat_inputs: dict[str, LongTensor[N]], num_inputs: FloatTensor[N, F], n_rows).
    """
    if preprocessor is None:
        raise InferenceError("Tabular preprocessor is not available.")
 
    try:
        df = pd.read_csv(io.BytesIO(csv_bytes))
    except Exception as exc:
        raise InferenceError(
            "Could not read CSV file. Please upload a valid CSV."
        ) from exc
 
    if df is None or len(df) == 0:
        raise InferenceError(
            "Could not read CSV file. Please upload a valid CSV."
        )
 
    # Mirror the training column handling (transform also does this, but we
    # normalise explicitly per the pipeline contract).
    df = fill_missing_tabular_cols(normalise_columns(df))
 
    try:
        transformed = preprocessor.transform(df)
    except Exception as exc:
        logger.exception("Tabular transform failed")
        raise InferenceError(
            "Could not process CSV metadata. Please check the columns."
        ) from exc
 
    cat_arrays = transformed["cat"]   # {col: np.int64[N]}
    num_array = transformed["num"]    # np.float32[N, F]
 
    cat_inputs = {
        col: torch.as_tensor(cat_arrays[col], dtype=torch.long).to(device)
        for col in CAT_FEATURES
    }
    num_inputs = torch.as_tensor(num_array, dtype=torch.float32).to(device)
    n_rows = int(num_inputs.shape[0])
    return cat_inputs, num_inputs, n_rows
 
 
# ---------------------------------------------------------------------------
# Response formatting
# ---------------------------------------------------------------------------
def format_single_prediction(probability: float, mode: str,
                             heatmap_urls: dict | None = None,
                             spatial_attention_url: str | None = None,
                             warning: str | None = None) -> dict:
    risk = risk_level_from_probability(probability)
    resp = {
        "mode": mode,
        "probability": round_prob(probability),
        "predicted_class": predicted_class_from_risk(risk),
        "risk_level": risk,
        "threshold": DEFAULT_THRESHOLD,
        "interpretation": interpretation_for(risk),
        "disclaimer": DISCLAIMER,
    }
    if heatmap_urls:
        resp["heatmap_url"] = heatmap_urls.get("heatmap_url")
        resp["overlay_url"] = heatmap_urls.get("overlay_url")   # Grad-CAM
    if spatial_attention_url:
        resp["spatial_attention_url"] = spatial_attention_url   # SpatialAttention
    if warning:
        resp["warning"] = warning
    return resp
 
 
def format_batch_prediction(probabilities, mode: str,
                            heatmap_urls: dict | None = None,
                            spatial_attention_url: str | None = None,
                            warning: str | None = None) -> dict:
    predictions = []
    for i, p in enumerate(probabilities, start=1):
        risk = risk_level_from_probability(float(p))
        predictions.append({
            "row_id": i,
            "probability": round_prob(float(p)),
            "predicted_class": predicted_class_from_risk(risk),
            "risk_level": risk,
        })
    resp = {
        "mode": mode,
        "count": len(predictions),
        "predictions": predictions,
        "threshold": DEFAULT_THRESHOLD,
        "disclaimer": DISCLAIMER,
    }
    if heatmap_urls:
        resp["heatmap_url"] = heatmap_urls.get("heatmap_url")
        resp["overlay_url"] = heatmap_urls.get("overlay_url")   # Grad-CAM
    if spatial_attention_url:
        resp["spatial_attention_url"] = spatial_attention_url   # SpatialAttention
    if warning:
        resp["warning"] = warning
    return resp
 
 
# ---------------------------------------------------------------------------
# Heatmap helper
# ---------------------------------------------------------------------------
def _make_heatmap(original_pil, attention_map, kind: str = "heatmap"):
    """Generate heatmap; return (urls|None, warning|None). Never raises."""
    try:
        prefix = f"pred_{uuid.uuid4().hex[:8]}_{kind}"
        urls = generate_spatial_attention_heatmap(
            original_image_pil=original_pil,
            attention_map=attention_map,
            output_dir=HEATMAP_DIR,
            prefix=prefix,
        )
        return urls, None
    except Exception:
        logger.exception("Heatmap generation failed (kind=%s)", kind)
        return None, f"{kind} heatmap could not be generated."
 
 
# ---------------------------------------------------------------------------
# Grad-CAM  (model.py'ye dokunmaz — dışarıdan PyTorch hook kullanır)
# ---------------------------------------------------------------------------
def compute_grad_cam(model, mode: str, image_tensor,
                     cat_inputs=None, num_inputs=None):
    """
    Grad-CAM: ResNet50'nin son konv bloğu (layer4) üzerinde forward + backward
    hook'lar kullanarak sınıf aktivasyon haritası hesaplar.
 
    Gradient-weighted Class Activation Mapping (Selvaraju et al., 2017).
    Formül: CAM = ReLU( Σ_c  mean_{h,w}(∂score/∂A^c_{h,w}) · A^c )
 
    Parameters
    ----------
    image_tensor : [1, 3, H, W]  — her zaman tek görüntü.
    cat_inputs   : dict[str, LongTensor[1]]  — fused için ilk CSV satırı.
    num_inputs   : FloatTensor[1, F]          — fused için ilk CSV satırı.
 
    Returns
    -------
    cam : [1, 1, h, w] CPU tensor  (normalize edilmemiş, _make_heatmap normalize eder)
    None on failure.
    """
    # ResNet50Backbone.features[-1] = layer4 (son konv bloğu)
    target_layer = model.image_branch.backbone.features[-1]
 
    _activations: dict = {}
    _gradients: dict = {}
 
    def _fwd_hook(m, inp, out):
        _activations["a"] = out  # [1, 2048, 12, 12]
 
    def _bwd_hook(m, gin, gout):
        if gout[0] is not None:
            _gradients["g"] = gout[0]  # [1, 2048, 12, 12]
 
    h1 = target_layer.register_forward_hook(_fwd_hook)
    h2 = target_layer.register_full_backward_hook(_bwd_hook)
 
    try:
        model.zero_grad()
 
        # Forward: model.eval() hâlinde ama no_grad YOK — grafik gerekli.
        if mode == "image":
            logits = model(image_tensor)
        else:  # fused
            logits = model(image_tensor, cat_inputs, num_inputs)
 
        # Malignansi lojitine göre gradyan (pozitif sınıf skoru).
        logits[0, 0].backward()
 
        if "g" not in _gradients:
            logger.warning("Grad-CAM: gradient hook boş kaldı, atlanıyor.")
            return None
 
        acts  = _activations["a"]   # [1, 2048, 12, 12]
        grads = _gradients["g"]     # [1, 2048, 12, 12]
 
        # Kanal ağırlıkları: her kanalın ortalama gradyanı
        weights = grads.mean(dim=[2, 3], keepdim=True)  # [1, 2048, 1, 1]
 
        # Ağırlıklı aktivasyon toplamı + ReLU (negatif katkıları kes)
        cam = F.relu(
            (weights * acts).sum(dim=1, keepdim=True)   # [1, 1, 12, 12]
        )
        return cam.detach().cpu()
 
    except Exception:
        logger.exception("Grad-CAM hesaplaması başarısız")
        return None
 
    finally:
        h1.remove()
        h2.remove()
        model.zero_grad()  # birikmiş gradyanları temizle
 
 
# ---------------------------------------------------------------------------
# Top-level predict
# ---------------------------------------------------------------------------
def predict(mode: str, image_bytes: bytes | None = None,
            csv_bytes: bytes | None = None) -> dict:
    """
    Orchestrate a prediction. Returns a single or batch response dict.
    Raises InferenceError on any user-facing validation/processing failure.
    """
    if mode not in VALID_MODES:
        raise InferenceError(
            "Invalid mode. Expected one of: image, tabular, fused."
        )
 
    # ---- input validation ------------------------------------------------
    if mode in ("image", "fused") and not image_bytes:
        raise InferenceError(f"Image file is required for {mode} mode.")
    if mode in ("tabular", "fused") and not csv_bytes:
        raise InferenceError(f"CSV file is required for {mode} mode.")
 
    bundle = load_model_bundle(mode)
    model = bundle["model"]
    preprocessor = bundle["preprocessor"]
    device = bundle["device"]
 
    model.eval()  # guarantees BatchNorm uses running stats (safe at N==1)
 
    # ---- IMAGE -----------------------------------------------------------
    if mode == "image":
        image_tensor, original_pil = preprocess_image(image_bytes)
 
        # Step 1: olasılık + spatial attention (no_grad, verimli)
        with torch.no_grad():
            logits, attn = model.forward_with_attention(image_tensor)
            probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
 
        # Step 2: Grad-CAM (ayrı pass, gradyan gerekli — model.py'ye dokunulmaz)
        cam = compute_grad_cam(model, "image", image_tensor)
 
        # Step 3: her iki haritayı PNG'ye çevir
        uid = uuid.uuid4().hex[:8]
        gradcam_urls, w1 = _make_heatmap(original_pil, cam, kind=f"{uid}_gradcam") \
            if cam is not None else (None, "Grad-CAM üretilemedi.")
        spatial_urls,  w2 = _make_heatmap(original_pil, attn[0:1], kind=f"{uid}_spatial")
 
        # overlay_url → Grad-CAM (UI'da gösterilen)
        # spatial_attention_url → SpatialAttention (JSON'da extra alan)
        sa_url = spatial_urls.get("overlay_url") if spatial_urls else None
        warning = " | ".join(filter(None, [w1, w2])) or None
        return format_single_prediction(
            float(probs[0]), mode,
            heatmap_urls=gradcam_urls,
            spatial_attention_url=sa_url,
            warning=warning,
        )
 
    # ---- TABULAR ---------------------------------------------------------
    if mode == "tabular":
        cat_inputs, num_inputs, n_rows = preprocess_csv(
            csv_bytes, preprocessor, device
        )
        with torch.no_grad():
            logits = model(None, cat_inputs, num_inputs)
            probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
        if n_rows == 1:
            return format_single_prediction(float(probs[0]), mode)
        return format_batch_prediction(probs, mode)
 
    # ---- FUSED -----------------------------------------------------------
    # mode == "fused"
    image_tensor, original_pil = preprocess_image(image_bytes)
    cat_inputs, num_inputs, n_rows = preprocess_csv(csv_bytes, preprocessor, device)
 
    # Tahmin için: tekrarlanan N görüntü
    if n_rows > 1:
        images = image_tensor.repeat(n_rows, 1, 1, 1)
    else:
        images = image_tensor
 
    # Step 1: olasılıklar + spatial attention (no_grad)
    with torch.no_grad():
        logits, attn = model.forward_with_attention(images, cat_inputs, num_inputs)
        probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
 
    # Step 2: Grad-CAM — tek görüntü + ilk CSV satırı (heatmap tek olacak)
    cat_single = {k: v[0:1] for k, v in cat_inputs.items()}
    num_single = num_inputs[0:1]
    cam = compute_grad_cam(model, "fused", image_tensor, cat_single, num_single)
 
    # Step 3: her iki haritayı üret
    uid = uuid.uuid4().hex[:8]
    gradcam_urls, w1 = _make_heatmap(original_pil, cam, kind=f"{uid}_gradcam") \
        if cam is not None else (None, "Grad-CAM üretilemedi.")
    spatial_urls,  w2 = _make_heatmap(original_pil, attn[0:1], kind=f"{uid}_spatial")
 
    sa_url = spatial_urls.get("overlay_url") if spatial_urls else None
    warning = " | ".join(filter(None, [w1, w2])) or None
 
    if n_rows == 1:
        return format_single_prediction(
            float(probs[0]), mode,
            heatmap_urls=gradcam_urls,
            spatial_attention_url=sa_url,
            warning=warning,
        )
    return format_batch_prediction(
        probs, mode,
        heatmap_urls=gradcam_urls,
        spatial_attention_url=sa_url,
        warning=warning,
    )