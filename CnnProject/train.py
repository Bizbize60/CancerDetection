"""
Two-stage fine-tuning with intermediate fusion (image + tabular).

Stage 1 — Head warmup  : backbone frozen, head + tabular MLP eğitilir.
Stage 2 — Full finetune: tüm parametreler açılır, discriminative LR.
"""
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import config
from dataset import MammographyDataset, get_train_transform, get_eval_transform, collate_with_tabular
from model import BreastCancerModel
from tabular_encoder import TabularPreprocessor, CAT_FEATURES


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Train / eval steps (handles both fused and image-only batches)
# ---------------------------------------------------------------------------
def train_epoch(model, loader, optimizer, criterion, device, use_tabular):
    model.train()
    total_loss, all_probs, all_labels = 0.0, [], []

    for batch in tqdm(loader, desc="train", leave=False):
        if use_tabular:
            images, labels, cat_inputs, num_inputs = batch
            cat_inputs = {k: v.to(device) for k, v in cat_inputs.items()}
            num_inputs = num_inputs.to(device)
        else:
            images, labels = batch
            cat_inputs = num_inputs = None

        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)

        optimizer.zero_grad()
        logits = model(images, cat_inputs, num_inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        all_probs.append(torch.sigmoid(logits).detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

    probs  = np.concatenate(all_probs).ravel()
    labels = np.concatenate(all_labels).ravel()
    return total_loss / len(loader.dataset), roc_auc_score(labels, probs)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device, use_tabular):
    model.eval()
    total_loss, all_probs, all_labels = 0.0, [], []

    for batch in tqdm(loader, desc="val", leave=False):
        if use_tabular:
            images, labels, cat_inputs, num_inputs = batch
            cat_inputs = {k: v.to(device) for k, v in cat_inputs.items()}
            num_inputs = num_inputs.to(device)
        else:
            images, labels = batch
            cat_inputs = num_inputs = None

        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)

        logits = model(images, cat_inputs, num_inputs)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    probs  = np.concatenate(all_probs).ravel()
    labels = np.concatenate(all_labels).ravel()
    return total_loss / len(loader.dataset), roc_auc_score(labels, probs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    set_seed(config.SEED)
    device = config.DEVICE
    print(f"Device: {device}")

    # ---- Tabular preprocessor: fit on train split ----
    import pandas as pd
    from tabular_encoder import normalise_columns
    train_csv_path = os.path.join(config.SPLITS_DIR, "train.csv")
    train_raw = normalise_columns(pd.read_csv(train_csv_path))

    preprocessor = TabularPreprocessor()
    preprocessor.fit(train_raw)
    prep_path = os.path.join(config.CHECKPOINT_DIR, "tabular_preprocessor.pkl")
    preprocessor.save(prep_path)
    print(f"TabularPreprocessor fitted & saved → {prep_path}")
    print(f"  cat_dims: {preprocessor.cat_dims}")

    # ---- Datasets ----
    train_ds = MammographyDataset(
        train_csv_path, transform=get_train_transform(),
        preprocessor=preprocessor,
    )
    val_ds = MammographyDataset(
        os.path.join(config.SPLITS_DIR, "val.csv"),
        transform=get_eval_transform(),
        preprocessor=preprocessor,
    )

    collate_fn = collate_with_tabular
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE,
                              shuffle=True, num_workers=config.NUM_WORKERS,
                              pin_memory=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds, batch_size=config.BATCH_SIZE,
                              shuffle=False, num_workers=config.NUM_WORKERS,
                              pin_memory=True, collate_fn=collate_fn)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # ---- Model ----
    model = BreastCancerModel(
        pretrained=True,
        cat_dims=preprocessor.cat_dims,
        dropout=config.DROPOUT,
    ).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # ---- Loss: weighted BCE ----
    pos = train_raw["label"].sum()
    neg = len(train_raw) - pos
    pos_weight = torch.tensor([neg / max(pos, 1)], dtype=torch.float32).to(device)
    print(f"pos_weight = {pos_weight.item():.3f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    ckpt_path = os.path.join(config.CHECKPOINT_DIR, "best.pt")
    best_auc   = 0.0

    # ====== STAGE 1: Head warmup ======
    print(f"\n--- Stage 1: Head warmup ({config.WARMUP_EPOCHS} epochs) ---")
    model.freeze_backbone()
    # Train head + tabular branch; backbone frozen
    trainable = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=config.HEAD_LR,
                                  weight_decay=config.WEIGHT_DECAY)

    for epoch in range(1, config.WARMUP_EPOCHS + 1):
        tr_loss, tr_auc = train_epoch(model, train_loader, optimizer, criterion,
                                      device, use_tabular=True)
        val_loss, val_auc = eval_epoch(model, val_loader, criterion,
                                       device, use_tabular=True)
        print(f"[Warmup] {epoch:02d}/{config.WARMUP_EPOCHS} | "
              f"Train {tr_loss:.4f}/{tr_auc:.4f} | Val {val_loss:.4f}/{val_auc:.4f}")
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "val_auc": val_auc, "stage": "warmup",
                        "cat_dims": preprocessor.cat_dims}, ckpt_path)
            print(f"  → Saved best (val AUC {val_auc:.4f})")

    # ====== STAGE 2: Full fine-tuning ======
    print(f"\n--- Stage 2: Full fine-tune (up to {config.EPOCHS} epochs) ---")
    model.unfreeze_backbone()
    optimizer = torch.optim.AdamW([
        {"params": model.image_branch.backbone.parameters(), "lr": config.LR},
        {"params": model.image_branch.attention.parameters(), "lr": config.LR * 2},
        {"params": model.image_branch.proj.parameters(),     "lr": config.LR * 5},
        {"params": model.tabular_branch.parameters(),        "lr": config.LR * 5},
        {"params": model.head.parameters(),                  "lr": config.LR * 5},
    ], weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.EPOCHS - config.WARMUP_EPOCHS
    )

    patience_counter = 0
    for epoch in range(config.WARMUP_EPOCHS + 1, config.EPOCHS + 1):
        tr_loss, tr_auc = train_epoch(model, train_loader, optimizer, criterion,
                                      device, use_tabular=True)
        val_loss, val_auc = eval_epoch(model, val_loader, criterion,
                                       device, use_tabular=True)
        scheduler.step()
        print(f"[FineTune] {epoch:02d}/{config.EPOCHS} | "
              f"Train {tr_loss:.4f}/{tr_auc:.4f} | Val {val_loss:.4f}/{val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "val_auc": val_auc, "stage": "finetune",
                        "cat_dims": preprocessor.cat_dims}, ckpt_path)
            print(f"  → Saved best (val AUC {val_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"Early stopping at epoch {epoch}. Best val AUC: {best_auc:.4f}")
                break

    print(f"\nTraining done. Best Val AUC: {best_auc:.4f}")


if __name__ == "__main__":
    main()
