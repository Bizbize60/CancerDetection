"""Test set'te best model'i değerlendir + threshold tuning."""
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix,
                             f1_score, precision_score)

import config
from dataset import MammographyDataset, get_eval_transform
from model import BreastCancerModel


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        probs = torch.sigmoid(logits).cpu().numpy().ravel()
        all_probs.append(probs)
        all_labels.append(labels.numpy().ravel())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def metrics_at_threshold(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "threshold": threshold,
        "sensitivity": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
        "precision":   precision_score(y_true, y_pred, zero_division=0),
        "f1":          f1_score(y_true, y_pred),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def find_threshold_for_sensitivity(y_true, y_prob, target=0.90):
    """Sensitivity >= target sağlayan en yüksek specificity'li threshold."""
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    valid = tpr >= target
    if not valid.any():
        return 0.5
    idx = np.where(valid)[0]
    best = idx[np.argmin(fpr[idx])]
    # roc_curve thr[0] = +inf olabilir → güvenli kontrol
    return float(thr[best]) if np.isfinite(thr[best]) else 0.5


def youden_threshold(y_true, y_prob):
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    j = tpr - fpr
    return float(thr[np.argmax(j)])


def format_block(name, m):
    return (
        f"--- {name} (threshold={m['threshold']:.3f}) ---\n"
        f"  Sensitivity: {m['sensitivity']:.4f}  (kanseri yakalama)\n"
        f"  Specificity: {m['specificity']:.4f}\n"
        f"  Precision:   {m['precision']:.4f}\n"
        f"  F1:          {m['f1']:.4f}\n"
        f"  Confusion: TN={m['tn']} FP={m['fp']} FN={m['fn']} TP={m['tp']}\n"
        f"  FN (kaçırılan kanser): {m['fn']}\n"
    )


def main():
    device = config.DEVICE
    test_ds = MammographyDataset(
        os.path.join(config.SPLITS_DIR, "test.csv"),
        transform=get_eval_transform(),
    )
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE,
                             shuffle=False, num_workers=config.NUM_WORKERS,
                             pin_memory=True)

    model = BreastCancerModel(pretrained=False).to(device)
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, "best.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} "
          f"(val AUC {ckpt['val_auc']:.4f}, stage={ckpt.get('stage','?')})")

    probs, labels = predict(model, test_loader, device)

    # Val set'te threshold belirleyelim (test'te belirlemek leakage olur)
    val_ds = MammographyDataset(
        os.path.join(config.SPLITS_DIR, "val.csv"),
        transform=get_eval_transform(),
    )
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE,
                            shuffle=False, num_workers=config.NUM_WORKERS,
                            pin_memory=True)
    val_probs, val_labels = predict(model, val_loader, device)

    auc = roc_auc_score(labels, probs)

    # 3 threshold karşılaştırması (val'da seçildi, test'te raporlandı)
    thr_default = 0.5
    thr_youden  = youden_threshold(val_labels, val_probs)
    thr_sens95  = find_threshold_for_sensitivity(val_labels, val_probs, 0.95)
    thr_sens90  = find_threshold_for_sensitivity(val_labels, val_probs, 0.90)

    m_default = metrics_at_threshold(labels, probs, thr_default)
    m_youden  = metrics_at_threshold(labels, probs, thr_youden)
    m_sens90  = metrics_at_threshold(labels, probs, thr_sens90)
    m_sens95  = metrics_at_threshold(labels, probs, thr_sens95)

    report = (
        f"=== TEST RESULTS ===\n"
        f"Test size: {len(labels)}\n"
        f"\n"
        f"ROC-AUC: {auc:.4f}  (threshold-bagimsiz genel performans)\n"
        f"\n"
        f"Threshold secimi val set uzerinde yapildi (test leakage yok).\n"
        f"\n"
        f"{format_block('Default 0.5', m_default)}"
        f"{format_block('Youden J (max sens+spec-1)', m_youden)}"
        f"{format_block('Klinik oncelik: Sensitivity >= 0.90', m_sens90)}"
        f"{format_block('Klinik oncelik: Sensitivity >= 0.95', m_sens95)}"
    )
    print(report)

    out_path = os.path.join(config.OUTPUT_DIR, "results.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)

    # Olasılıkları da kaydet (sonra Grad-CAM seçiminde işine yarar)
    np.savez(os.path.join(config.OUTPUT_DIR, "test_predictions.npz"),
             probs=probs, labels=labels)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()