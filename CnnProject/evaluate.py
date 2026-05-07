"""Test set'te best model'i değerlendir + threshold tuning (fused model)."""
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix,
                             f1_score, precision_score)

import config
from dataset import MammographyDataset, get_eval_transform, collate_with_tabular
from model import BreastCancerModel
from tabular_encoder import TabularPreprocessor


@torch.no_grad()
def predict(model, loader, device, use_tabular):
    model.eval()
    all_probs, all_labels = [], []
    for batch in loader:
        if use_tabular:
            images, labels, cat_inputs, num_inputs = batch
            cat_inputs = {k: v.to(device) for k, v in cat_inputs.items()}
            num_inputs = num_inputs.to(device)
        else:
            images, labels = batch
            cat_inputs = num_inputs = None

        images = images.to(device)
        logits = model(images, cat_inputs, num_inputs)
        probs  = torch.sigmoid(logits).cpu().numpy().ravel()
        all_probs.append(probs)
        all_labels.append(labels.numpy().ravel())
    return np.concatenate(all_probs), np.concatenate(all_labels)


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
    idx = np.where(valid)[0]
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


def main():
    device = config.DEVICE

    # ---- Load preprocessor & cat_dims from checkpoint ----
    prep_path = os.path.join(config.CHECKPOINT_DIR, "tabular_preprocessor.pkl")
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, "best.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    use_tabular = os.path.exists(prep_path)
    preprocessor = TabularPreprocessor.load(prep_path) if use_tabular else None
    cat_dims = ckpt.get("cat_dims") if use_tabular else None
    print(f"use_tabular={use_tabular}, cat_dims={cat_dims}")

    collate_fn = collate_with_tabular if use_tabular else None

    def make_loader(split_name):
        ds = MammographyDataset(
            os.path.join(config.SPLITS_DIR, f"{split_name}.csv"),
            transform=get_eval_transform(),
            preprocessor=preprocessor,
        )
        return DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=False,
                          num_workers=config.NUM_WORKERS, pin_memory=True,
                          collate_fn=collate_fn)

    test_loader = make_loader("test")
    val_loader  = make_loader("val")

    # ---- Model ----
    model = BreastCancerModel(pretrained=False, cat_dims=cat_dims).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} "
          f"(val AUC {ckpt['val_auc']:.4f}, stage={ckpt.get('stage','?')})")

    # ---- Predict ----
    probs,     labels     = predict(model, test_loader, device, use_tabular)
    val_probs, val_labels = predict(model, val_loader,  device, use_tabular)

    auc = roc_auc_score(labels, probs)

    # Threshold selection on val (no test leakage)
    thr_default = 0.5
    thr_youden  = youden_threshold(val_labels, val_probs)
    thr_sens90  = find_threshold_for_sensitivity(val_labels, val_probs, 0.90)
    thr_sens95  = find_threshold_for_sensitivity(val_labels, val_probs, 0.95)

    report = (
        f"=== TEST RESULTS (Fused: Image + Tabular) ===\n"
        f"Test size: {len(labels)}\n\n"
        f"ROC-AUC: {auc:.4f}\n\n"
        f"Threshold secimi val set uzerinde yapildi.\n\n"
        f"{format_block('Default 0.5', metrics_at_threshold(labels, probs, thr_default))}"
        f"{format_block('Youden J',    metrics_at_threshold(labels, probs, thr_youden))}"
        f"{format_block('Sensitivity >= 0.90', metrics_at_threshold(labels, probs, thr_sens90))}"
        f"{format_block('Sensitivity >= 0.95', metrics_at_threshold(labels, probs, thr_sens95))}"
    )
    print(report)

    out_path = os.path.join(config.OUTPUT_DIR, "results.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)

    np.savez(os.path.join(config.OUTPUT_DIR, "test_predictions.npz"),
             probs=probs, labels=labels)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
