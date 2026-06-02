"""
Mode-aware evaluation:
  "image"   → TTA x5 (görüntü augmentasyonları üzerinde)
  "tabular" → Tek geçiş (TTA anlamsız — augment edilecek görüntü yok)
  "fused"   → TTA x5 (görüntü tarafı augment edilir, tabular sabit)

Threshold seçimi val seti üzerinde yapılır, test setine uygulanır.
"""
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix,
                             f1_score, precision_score)

import config
from dataset import (MammographyDataset, TTADataset,
                     get_eval_transform, get_tta_transforms,
                     get_collate_fn)
from model import build_model
from tabular_encoder import TabularPreprocessor


# ---------------------------------------------------------------------------
# Batch unpack — train.py ile aynı mantık
# ---------------------------------------------------------------------------
def _unpack_batch(batch, mode, device):
    if mode == "image":
        images, labels = batch
        images = images.to(device, non_blocking=True)
        return images, labels, None, None

    images, labels, cat_inputs, num_inputs = batch
    cat_inputs = {k: v.to(device, non_blocking=True) for k, v in cat_inputs.items()}
    num_inputs = num_inputs.to(device, non_blocking=True)
    if mode == "fused":
        images = images.to(device, non_blocking=True)
    else:
        images = None
    return images, labels, cat_inputs, num_inputs


# ---------------------------------------------------------------------------
# Inference fonksiyonları
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict_tta(model, base_ds, device, mode, n_tta=5):
    """TTA: her örnek için n_tta augmentation → olasılık ortalaması.
    Sadece "image" ve "fused" modlarında çağrılır."""
    model.eval()
    tta_transforms = get_tta_transforms()
    tta_ds = TTADataset(base_ds, tta_transforms)
    collate_fn = get_collate_fn(mode)
    loader = DataLoader(tta_ds, batch_size=config.BATCH_SIZE,
                        shuffle=False, num_workers=config.NUM_WORKERS,
                        pin_memory=True, collate_fn=collate_fn)

    all_probs, all_labels = [], []
    for batch in loader:
        images, labels, cat_inputs, num_inputs = _unpack_batch(batch, mode, device)
        logits = model(images, cat_inputs, num_inputs)
        probs  = torch.sigmoid(logits).cpu().numpy().ravel()
        all_probs.append(probs)
        all_labels.append(labels.numpy().ravel())

    probs_all  = np.concatenate(all_probs)
    labels_all = np.concatenate(all_labels)
    n_samples  = len(base_ds)

    probs_mat  = probs_all.reshape(n_samples, n_tta)
    labels_mat = labels_all.reshape(n_samples, n_tta)
    probs_avg  = probs_mat.mean(axis=1)
    labels_final = labels_mat[:, 0]
    return probs_avg, labels_final


@torch.no_grad()
def predict_single(model, ds, device, mode):
    """TTA olmadan tek geçiş. Tabular modda da bu kullanılır."""
    model.eval()
    collate_fn = get_collate_fn(mode)
    loader = DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=False,
                        num_workers=config.NUM_WORKERS, pin_memory=True,
                        collate_fn=collate_fn)
    all_probs, all_labels = [], []
    for batch in loader:
        images, labels, cat_inputs, num_inputs = _unpack_batch(batch, mode, device)
        logits = model(images, cat_inputs, num_inputs)
        all_probs.append(torch.sigmoid(logits).cpu().numpy().ravel())
        all_labels.append(labels.numpy().ravel())
    return np.concatenate(all_probs), np.concatenate(all_labels)


# ---------------------------------------------------------------------------
# Metrik / threshold yardımcıları
# ---------------------------------------------------------------------------
def metrics_at_threshold(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "threshold":   threshold,
        "sensitivity": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
        "precision":   precision_score(y_true, y_pred, zero_division=0),
        "f1":          f1_score(y_true, y_pred),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def find_threshold_for_sensitivity(y_true, y_prob, target=0.90):
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    valid = tpr >= target
    if not valid.any():
        return 0.5
    idx  = np.where(valid)[0]
    best = idx[np.argmin(fpr[idx])]
    return float(thr[best]) if np.isfinite(thr[best]) else 0.5


def youden_threshold(y_true, y_prob):
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    return float(thr[np.argmax(tpr - fpr)])


def format_block(name, m):
    return (
        f"--- {name} (threshold={m['threshold']:.3f}) ---\n"
        f"  Sensitivity: {m['sensitivity']:.4f}\n"
        f"  Specificity: {m['specificity']:.4f}\n"
        f"  Precision:   {m['precision']:.4f}\n"
        f"  F1:          {m['f1']:.4f}\n"
        f"  Confusion: TN={m['tn']} FP={m['fp']} FN={m['fn']} TP={m['tp']}\n"
        f"  FN (kaçırılan kanser): {m['fn']}\n"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    device = config.DEVICE
    mode   = config.MODE
    print(f"Device: {device} | MODE: {mode}")

    ckpt_path = os.path.join(config.CHECKPOINT_DIR, config.CHECKPOINT_NAME)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Checkpoint'teki mode ile config.MODE uyuşmalı (kullanıcı uyarısı)
    ckpt_mode = ckpt.get("mode", "fused")
    if ckpt_mode != mode:
        print(f"⚠️  WARNING: checkpoint mode={ckpt_mode!r} ama config.MODE={mode!r}. "
              f"Aynı modu kullanmanız önerilir.")

    cat_dims     = ckpt.get("cat_dims")
    preprocessor = None
    if mode in {"tabular", "fused"}:
        prep_path = os.path.join(config.CHECKPOINT_DIR, "tabular_preprocessor.pkl")
        preprocessor = TabularPreprocessor.load(prep_path)

    print(f"Loaded checkpoint: epoch={ckpt['epoch']} "
          f"val_AUC={ckpt['val_auc']:.4f} stage={ckpt.get('stage','?')}")

    # ---- Model ----
    model = build_model(mode=mode, cat_dims=cat_dims, pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    # ---- Datasets ----
    # Image / fused: TTA kendi dönüşümlerini uygular → transform=None
    # Tabular: image hiç okunmuyor → transform önemsiz
    if mode in {"image", "fused"}:
        test_ds = MammographyDataset(
            os.path.join(config.SPLITS_DIR, "test.csv"),
            transform=None,
            preprocessor=preprocessor,
            mode=mode,
        )
    else:  # tabular
        test_ds = MammographyDataset(
            os.path.join(config.SPLITS_DIR, "test.csv"),
            transform=None,
            preprocessor=preprocessor,
            mode=mode,
        )

    val_ds = MammographyDataset(
        os.path.join(config.SPLITS_DIR, "val.csv"),
        transform=(get_eval_transform() if mode in {"image", "fused"} else None),
        preprocessor=preprocessor,
        mode=mode,
    )

    # ---- Predict ----
    if mode == "tabular":
        print("\nRunning single-pass on test set (TTA tabular'da uygulanmaz)...")
        probs, labels = predict_single(model, test_ds, device, mode)
        tta_label = "no TTA"
    else:
        print(f"\nRunning TTA on test set (5 augmentations, MODE={mode})...")
        probs, labels = predict_tta(model, test_ds, device, mode, n_tta=5)
        tta_label = "TTA x5"

    print("Running single-pass on val set (threshold selection)...")
    val_probs, val_labels = predict_single(model, val_ds, device, mode)

    auc = roc_auc_score(labels, probs)

    thr_default = 0.5
    thr_youden  = youden_threshold(val_labels, val_probs)
    thr_sens90  = find_threshold_for_sensitivity(val_labels, val_probs, 0.90)
    thr_sens95  = find_threshold_for_sensitivity(val_labels, val_probs, 0.95)

    mode_label = {"image": "Image only", "tabular": "Tabular only",
                  "fused": "Image + Tabular"}[mode]
    report = (
        f"=== TEST RESULTS ({mode_label} | {tta_label}) ===\n"
        f"Test size: {len(labels)}\n\n"
        f"ROC-AUC: {auc:.4f}\n\n"
        f"Threshold secimi val set uzerinde yapildi.\n\n"
        f"{format_block('Default 0.5',        metrics_at_threshold(labels, probs, thr_default))}"
        f"{format_block('Youden J',            metrics_at_threshold(labels, probs, thr_youden))}"
        f"{format_block('Sensitivity >= 0.90', metrics_at_threshold(labels, probs, thr_sens90))}"
        f"{format_block('Sensitivity >= 0.95', metrics_at_threshold(labels, probs, thr_sens95))}"
    )
    print(report)

    out_path = os.path.join(config.OUTPUT_DIR, config.RESULTS_NAME)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)

    np.savez(os.path.join(config.OUTPUT_DIR, config.PREDICTIONS_NAME),
             probs=probs, labels=labels)
    print(f"Results saved → {out_path}")


if __name__ == "__main__":
    main()