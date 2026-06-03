"""
Spatial-attention heatmap generation.

Takes the original (pre-normalization) image plus the [B,1,h,w] sigmoid
attention map produced by SpatialAttention.get_attention_map / the model's
forward_with_attention, and renders:

  * a pure colorized heatmap PNG
  * an overlay PNG (heatmap blended over the original image)

Both are written under outputs/heatmaps/ and exposed as static URLs.
"""
from __future__ import annotations

import os
import numpy as np
import cv2

# Display canvas size (matches config.IMAGE_SIZE). Imported lazily-safe.
try:
    import config
    DISPLAY_SIZE = int(getattr(config, "IMAGE_SIZE", 384))
except Exception:  # pragma: no cover - config should always import
    DISPLAY_SIZE = 384


def _to_numpy_2d(attention_map) -> np.ndarray:
    """Convert a torch tensor / ndarray attention map to a 2D float array."""
    # Lazy torch handling so this module can be imported without torch present.
    if hasattr(attention_map, "detach"):
        attention_map = attention_map.detach().cpu().numpy()
    arr = np.asarray(attention_map, dtype=np.float32)
    arr = np.squeeze(arr)          # [1,1,h,w] / [1,h,w] -> [h,w]
    if arr.ndim == 1:
        side = int(np.sqrt(arr.shape[0]))
        arr = arr.reshape(side, side)
    elif arr.ndim > 2:
        # Fall back to the first channel if still multi-dim.
        arr = arr.reshape(arr.shape[-2], arr.shape[-1])
    return arr


def generate_spatial_attention_heatmap(
    original_image_pil,
    attention_map,
    output_dir: str,
    prefix: str,
    colormap: int = cv2.COLORMAP_JET,
    display_size: int = DISPLAY_SIZE,
):
    """
    Render heatmap + overlay PNGs and return their static URLs.

    Parameters
    ----------
    original_image_pil : PIL.Image
        Original RGB image BEFORE ImageNet normalization.
    attention_map : torch.Tensor | np.ndarray
        Attention map shaped [1,1,h,w], [1,h,w] or [h,w].
    output_dir : str
        Absolute directory to write the PNGs into (created if missing).
    prefix : str
        Filename prefix. Files become {prefix}_heatmap.png / {prefix}_overlay.png.

    Returns
    -------
    dict with keys:
        heatmap_url : "/outputs/heatmaps/{prefix}_heatmap.png"
        overlay_url : "/outputs/heatmaps/{prefix}_overlay.png"
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1-2) attention map -> 2D numpy
    attn = _to_numpy_2d(attention_map)

    # 3) normalize to [0, 1]
    a_min, a_max = float(attn.min()), float(attn.max())
    if a_max - a_min > 1e-8:
        attn = (attn - a_min) / (a_max - a_min)
    else:
        attn = np.zeros_like(attn)

    # 4) resize attention to display size (smooth upsample)
    attn_resized = cv2.resize(
        attn, (display_size, display_size), interpolation=cv2.INTER_CUBIC
    )
    attn_resized = np.clip(attn_resized, 0.0, 1.0)
    attn_uint8 = (attn_resized * 255.0).astype(np.uint8)

    # 5) colorize -> RGB heatmap
    heatmap_bgr = cv2.applyColorMap(attn_uint8, colormap)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    # 6-7) original image -> RGB, resized to display size
    orig_rgb = np.asarray(original_image_pil.convert("RGB"), dtype=np.uint8)
    orig_rgb = cv2.resize(
        orig_rgb, (display_size, display_size), interpolation=cv2.INTER_AREA
    )

    # 8) blend overlay
    overlay_rgb = cv2.addWeighted(orig_rgb, 0.55, heatmap_rgb, 0.45, 0.0)

    # 9) save (cv2 writes BGR)
    heatmap_path = os.path.join(output_dir, f"{prefix}_heatmap.png")
    overlay_path = os.path.join(output_dir, f"{prefix}_overlay.png")
    cv2.imwrite(heatmap_path, cv2.cvtColor(heatmap_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(overlay_path, cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))

    # 10) return static URLs
    return {
        "heatmap_url": f"/outputs/heatmaps/{prefix}_heatmap.png",
        "overlay_url": f"/outputs/heatmaps/{prefix}_overlay.png",
    }
