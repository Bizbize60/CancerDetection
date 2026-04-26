"""MVP training: two-stage fine-tuning (head warmup → full fine-tune)."""
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import config
from dataset import MammographyDataset, get_train_transform, get_eval_transform
from model import BreastCancerModel


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    all_probs, all_labels = [], []

    for images, labels in tqdm(loader, desc="train", leave=False):
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        all_probs.append(torch.sigmoid(logits).detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    probs = np.concatenate(all_probs).ravel()
    labels = np.concatenate(all_labels).ravel()
    auc = roc_auc_score(labels, probs)
    return avg_loss, auc


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []

    for images, labels in tqdm(loader, desc="val", leave=False):
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    probs = np.concatenate(all_probs).ravel()
    labels = np.concatenate(all_labels).ravel()
    auc = roc_auc_score(labels, probs)
    return avg_loss, auc


def main():
    set_seed(config.SEED)
    device = config.DEVICE
    print(f"Device: {device}")

    train_ds = MammographyDataset(
        os.path.join(config.SPLITS_DIR, "train.csv"),
        transform=get_train_transform(),
    )
    val_ds = MammographyDataset(
        os.path.join(config.SPLITS_DIR, "val.csv"),
        transform=get_eval_transform(),
    )

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE,
                              shuffle=True, num_workers=config.NUM_WORKERS,
                              pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE,
                            shuffle=False, num_workers=config.NUM_WORKERS,
                            pin_memory=True)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    model = BreastCancerModel(pretrained=True).to(device)

    pos = train_ds.df["label"].sum()
    neg = len(train_ds.df) - pos
    pos_weight = torch.tensor([neg / max(pos, 1)], dtype=torch.float32).to(device)
    print(f"pos_weight = {pos_weight.item():.3f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ====== STAGE 1: Head warmup ======
    print(f"\n--- Stage 1: Head warmup ({config.WARMUP_EPOCHS} epochs) ---")
    model.freeze_backbone()
    # Sadece head parametreleri trainable
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.HEAD_LR, weight_decay=config.WEIGHT_DECAY)

    best_auc = 0.0
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, "best.pt")

    for epoch in range(1, config.WARMUP_EPOCHS + 1):
        tr_loss, tr_auc = train_epoch(model, train_loader, optimizer,
                                      criterion, device)
        val_loss, val_auc = eval_epoch(model, val_loader, criterion, device)

        print(f"[Warmup] Epoch {epoch:02d}/{config.WARMUP_EPOCHS} | "
              f"Train loss {tr_loss:.4f} AUC {tr_auc:.4f} | "
              f"Val loss {val_loss:.4f} AUC {val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "val_auc": val_auc, "stage": "warmup"}, ckpt_path)
            print(f"  → Saved best (val AUC {val_auc:.4f})")

    # ====== STAGE 2: Full fine-tuning ======
    print(f"\n--- Stage 2: Full fine-tuning (up to {config.EPOCHS} epochs) ---")
    model.unfreeze_backbone()
    # Discriminative LR: backbone düşük, head yüksek
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": config.LR},
        {"params": model.head.parameters(),    "lr": config.LR * 5},
    ], weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.EPOCHS - config.WARMUP_EPOCHS)

    patience_counter = 0
    for epoch in range(config.WARMUP_EPOCHS + 1, config.EPOCHS + 1):
        tr_loss, tr_auc = train_epoch(model, train_loader, optimizer,
                                      criterion, device)
        val_loss, val_auc = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        print(f"[FineTune] Epoch {epoch:02d}/{config.EPOCHS} | "
              f"Train loss {tr_loss:.4f} AUC {tr_auc:.4f} | "
              f"Val loss {val_loss:.4f} AUC {val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "val_auc": val_auc, "stage": "finetune"}, ckpt_path)
            print(f"  → Saved best (val AUC {val_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"Early stopping at epoch {epoch}. Best val AUC: {best_auc:.4f}")
                break

    print(f"\nTraining done. Best Val AUC: {best_auc:.4f}")
    print(f"Checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()