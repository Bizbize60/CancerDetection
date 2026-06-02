"""
Mode-aware training:
  config.MODE = "image"   → ResNet50 + SpatialAttention, two-stage fine-tune
  config.MODE = "tabular" → MLP only, single-stage hızlı eğitim
  config.MODE = "fused"   → image + tabular fusion, two-stage fine-tune (orijinal)

Mass + Calc birleşik dataset ile çalışır.
"""
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import pandas as pd

import config
from dataset import (MammographyDataset, get_train_transform,
                     get_eval_transform, get_collate_fn)
from model import build_model
from tabular_encoder import TabularPreprocessor, normalise_columns


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Train / Eval loops — mode'a göre batch yapısını çözer
# ---------------------------------------------------------------------------
def _unpack_batch(batch, mode, device):
    """Batch'i moda göre çöz ve cihaza taşı."""
    if mode == "image":
        images, labels = batch
        images = images.to(device, non_blocking=True)
        labels = labels.to(device).unsqueeze(1)
        return images, labels, None, None

    # tabular veya fused
    images, labels, cat_inputs, num_inputs = batch
    labels     = labels.to(device).unsqueeze(1)
    cat_inputs = {k: v.to(device, non_blocking=True) for k, v in cat_inputs.items()}
    num_inputs = num_inputs.to(device, non_blocking=True)
    if mode == "fused":
        images = images.to(device, non_blocking=True)
    else:
        images = None   # tabular-only: dummy images modele verilmez
    return images, labels, cat_inputs, num_inputs


def train_epoch(model, loader, optimizer, criterion, device, mode):
    model.train()
    total_loss, all_probs, all_labels = 0.0, [], []
    for batch in tqdm(loader, desc="train", leave=False):
        images, labels, cat_inputs, num_inputs = _unpack_batch(batch, mode, device)

        optimizer.zero_grad()
        logits = model(images, cat_inputs, num_inputs)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        bs = labels.size(0)
        total_loss += loss.item() * bs
        all_probs.append(torch.sigmoid(logits).detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

    probs  = np.concatenate(all_probs).ravel()
    labels = np.concatenate(all_labels).ravel()
    return total_loss / len(loader.dataset), roc_auc_score(labels, probs)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device, mode):
    model.eval()
    total_loss, all_probs, all_labels = 0.0, [], []
    for batch in tqdm(loader, desc="val", leave=False):
        images, labels, cat_inputs, num_inputs = _unpack_batch(batch, mode, device)

        logits = model(images, cat_inputs, num_inputs)
        loss   = criterion(logits, labels)

        bs = labels.size(0)
        total_loss += loss.item() * bs
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    probs  = np.concatenate(all_probs).ravel()
    labels = np.concatenate(all_labels).ravel()
    return total_loss / len(loader.dataset), roc_auc_score(labels, probs)


# ---------------------------------------------------------------------------
# Mode-specific training routines
# ---------------------------------------------------------------------------
def train_tabular_only(model, train_loader, val_loader, criterion,
                       device, ckpt_path, cat_dims):
    """Tabular-only: tek aşama, basit AdamW + cosine schedule, erken durdurma."""
    print(f"\n--- Tabular-only training (up to {config.TABULAR_ONLY_EPOCHS} epochs) ---")
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=config.TABULAR_ONLY_LR,
                                  weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.TABULAR_ONLY_EPOCHS,
    )

    best_auc, patience_counter = 0.0, 0
    for epoch in range(1, config.TABULAR_ONLY_EPOCHS + 1):
        tr_loss, tr_auc   = train_epoch(model, train_loader, optimizer,
                                        criterion, device, "tabular")
        val_loss, val_auc = eval_epoch(model, val_loader, criterion,
                                       device, "tabular")
        scheduler.step()
        print(f"[Tabular] {epoch:02d}/{config.TABULAR_ONLY_EPOCHS} | "
              f"Train {tr_loss:.4f}/{tr_auc:.4f} | "
              f"Val {val_loss:.4f}/{val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "val_auc": val_auc, "stage": "tabular",
                        "mode": "tabular", "cat_dims": cat_dims}, ckpt_path)
            print(f"  → Saved best (val AUC {val_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"Early stopping at epoch {epoch}. Best val AUC: {best_auc:.4f}")
                break

    return best_auc


def train_with_image(model, train_loader, val_loader, criterion,
                     device, ckpt_path, cat_dims, mode):
    """
    Two-stage: head warmup → full fine-tune.
    mode: "image" veya "fused" — sadece optimizer parametre grupları değişir.
    """
    best_auc = 0.0

    # ====== Stage 1: Head warmup ======
    print(f"\n--- Stage 1: Head warmup ({config.WARMUP_EPOCHS} epochs) ---")
    model.freeze_backbone()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.HEAD_LR, weight_decay=config.WEIGHT_DECAY,
    )

    for epoch in range(1, config.WARMUP_EPOCHS + 1):
        tr_loss, tr_auc   = train_epoch(model, train_loader, optimizer,
                                        criterion, device, mode)
        val_loss, val_auc = eval_epoch(model, val_loader, criterion,
                                       device, mode)
        print(f"[Warmup] {epoch:02d}/{config.WARMUP_EPOCHS} | "
              f"Train {tr_loss:.4f}/{tr_auc:.4f} | "
              f"Val {val_loss:.4f}/{val_auc:.4f}")
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "val_auc": val_auc, "stage": "warmup",
                        "mode": mode, "cat_dims": cat_dims}, ckpt_path)
            print(f"  → Saved best (val AUC {val_auc:.4f})")

    # ====== Stage 2: Full fine-tune ======
    print(f"\n--- Stage 2: Full fine-tune (up to {config.EPOCHS} epochs) ---")
    model.unfreeze_backbone()

    # Parametre grupları moda göre kurulur
    param_groups = [
        {"params": model.image_branch.backbone.parameters(),  "lr": config.LR},
        {"params": model.image_branch.attention.parameters(), "lr": config.LR * 2},
        {"params": model.image_branch.proj.parameters(),      "lr": config.LR * 5},
    ]
    if mode == "fused":
        param_groups.append(
            {"params": model.tabular_branch.parameters(), "lr": config.LR * 5}
        )
    param_groups.append(
        {"params": model.head.parameters(), "lr": config.LR * 5}
    )

    optimizer = torch.optim.AdamW(param_groups, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.EPOCHS - config.WARMUP_EPOCHS,
    )

    patience_counter = 0
    for epoch in range(config.WARMUP_EPOCHS + 1, config.EPOCHS + 1):
        tr_loss, tr_auc   = train_epoch(model, train_loader, optimizer,
                                        criterion, device, mode)
        val_loss, val_auc = eval_epoch(model, val_loader, criterion,
                                       device, mode)
        scheduler.step()
        print(f"[FineTune] {epoch:02d}/{config.EPOCHS} | "
              f"Train {tr_loss:.4f}/{tr_auc:.4f} | "
              f"Val {val_loss:.4f}/{val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "val_auc": val_auc, "stage": "finetune",
                        "mode": mode, "cat_dims": cat_dims}, ckpt_path)
            print(f"  → Saved best (val AUC {val_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"Early stopping at epoch {epoch}. Best val AUC: {best_auc:.4f}")
                break

    return best_auc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    set_seed(config.SEED)
    device = config.DEVICE
    mode   = config.MODE
    print(f"Device: {device} | MODE: {mode}")

    # ---- Tabular preprocessor (image-only modunda da hâlâ pos_weight için CSV'yi okuyacağız) ----
    train_csv_path = os.path.join(config.SPLITS_DIR, "train.csv")
    train_raw = normalise_columns(pd.read_csv(train_csv_path))

    preprocessor = None
    cat_dims = None
    if mode in {"tabular", "fused"}:
        preprocessor = TabularPreprocessor()
        preprocessor.fit(train_raw)
        prep_path = os.path.join(config.CHECKPOINT_DIR, "tabular_preprocessor.pkl")
        preprocessor.save(prep_path)
        cat_dims = preprocessor.cat_dims
        print(f"TabularPreprocessor fitted → {prep_path}")
        print(f"  cat_dims: {cat_dims}")
    else:
        print("Image-only mode: tabular preprocessor SKIPPED.")

    # ---- Datasets ----
    train_ds = MammographyDataset(
        train_csv_path,
        transform=(get_train_transform() if mode in {"image", "fused"} else None),
        preprocessor=preprocessor,
        mode=mode,
    )
    val_ds = MammographyDataset(
        os.path.join(config.SPLITS_DIR, "val.csv"),
        transform=(get_eval_transform() if mode in {"image", "fused"} else None),
        preprocessor=preprocessor,
        mode=mode,
    )

    # Tabular-only modda batch'i büyütmek hızlı + daha stabil
    batch_size = config.TABULAR_ONLY_BATCH if mode == "tabular" else config.BATCH_SIZE
    collate_fn = get_collate_fn(mode)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=config.NUM_WORKERS,
                              pin_memory=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds, batch_size=batch_size,
                              shuffle=False, num_workers=config.NUM_WORKERS,
                              pin_memory=True, collate_fn=collate_fn)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | batch_size={batch_size}")

    # ---- Model ----
    model = build_model(mode=mode, cat_dims=cat_dims,
                        pretrained=True, dropout=config.DROPOUT).to(device)
    print(f"Model: {type(model).__name__}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    # ---- Weighted BCE ----
    pos = train_raw["label"].sum()
    neg = len(train_raw) - pos
    pos_weight = torch.tensor([neg / max(pos, 1)], dtype=torch.float32).to(device)
    print(f"pos_weight = {pos_weight.item():.3f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    ckpt_path = os.path.join(config.CHECKPOINT_DIR, config.CHECKPOINT_NAME)

    # ---- Train (mode-specific) ----
    if mode == "tabular":
        best_auc = train_tabular_only(model, train_loader, val_loader,
                                      criterion, device, ckpt_path, cat_dims)
    else:
        # image veya fused → iki aşamalı
        best_auc = train_with_image(model, train_loader, val_loader,
                                    criterion, device, ckpt_path, cat_dims, mode)

    print(f"\nTraining done. MODE={mode}. Best Val AUC: {best_auc:.4f}")
    print(f"Checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()