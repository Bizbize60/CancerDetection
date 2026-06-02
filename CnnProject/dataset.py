"""
MammographyDataset — image + tabular veri.
Agresif augmentation (train) + Test-Time Augmentation (TTA) desteği.

Mode davranışı (config.MODE):
  "image"   → sadece görüntü döner (cat/num None)
  "tabular" → sadece tabular döner (dummy image tensor)
  "fused"   → her ikisi de döner (mevcut davranış)
"""
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

import config
from tabular_encoder import (
    CAT_FEATURES, NUM_FEATURES,
    TabularPreprocessor, normalise_columns,
)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Tabular-only modda image yüklemeden kaçınmak için kullanılan boş tensor.
# Image-only ya da fused modda bu dummy hiç oluşturulmaz.
_DUMMY_IMAGE = torch.zeros(1, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
def get_train_transform():
    """Agresif augmentation — val→test gap'ini kapatmak için güçlendirildi."""
    return A.Compose([
        A.LongestMaxSize(max_size=config.IMAGE_SIZE),
        A.PadIfNeeded(min_height=config.IMAGE_SIZE, min_width=config.IMAGE_SIZE,
                      border_mode=0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.Rotate(limit=15, border_mode=0, p=0.6),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                           rotate_limit=0, border_mode=0, p=0.4),
        A.OneOf([
            A.ElasticTransform(alpha=60, sigma=6, p=1.0),
            A.GridDistortion(num_steps=5, distort_limit=0.2, p=1.0),
        ], p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.25,
                                   contrast_limit=0.25, p=0.6),
        A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.4),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.GaussNoise(p=0.3),
        A.CoarseDropout(num_holes_range=(1, 6),
                        hole_height_range=(8, 24),
                        hole_width_range=(8, 24), p=0.4),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_eval_transform():
    return A.Compose([
        A.LongestMaxSize(max_size=config.IMAGE_SIZE),
        A.PadIfNeeded(min_height=config.IMAGE_SIZE, min_width=config.IMAGE_SIZE,
                      border_mode=0),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_tta_transforms():
    """5 deterministik TTA varyantı döndürür."""
    base = [
        A.LongestMaxSize(max_size=config.IMAGE_SIZE),
        A.PadIfNeeded(min_height=config.IMAGE_SIZE, min_width=config.IMAGE_SIZE,
                      border_mode=0),
    ]
    end = [
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ]
    return [
        A.Compose(base + end),                                          # orijinal
        A.Compose(base + [A.HorizontalFlip(p=1.0)] + end),             # yatay flip
        A.Compose(base + [A.VerticalFlip(p=1.0)] + end),               # dikey flip
        A.Compose(base + [A.Rotate(limit=(10, 10), p=1.0)] + end),     # +10° rot
        A.Compose(base + [A.Rotate(limit=(-10, -10), p=1.0)] + end),   # -10° rot
    ]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class MammographyDataset(Dataset):
    """
    mode:
      "image"   → (image, label)  — preprocessor None olabilir
      "tabular" → (dummy_image, label, cat, num) — image okunmaz (hızlı)
      "fused"   → (image, label, cat, num)
    """
    def __init__(self, csv_path: str, transform=None,
                 preprocessor: TabularPreprocessor | None = None,
                 mode: str = "fused"):
        self.df = normalise_columns(pd.read_csv(csv_path))
        self.transform = transform
        self.preprocessor = preprocessor
        self.mode = mode

        if mode not in {"image", "tabular", "fused"}:
            raise ValueError(f"Unknown mode: {mode!r}")

        if mode in {"tabular", "fused"}:
            if preprocessor is None:
                raise ValueError(f"'{mode}' modu için preprocessor gerekli.")
            transformed = preprocessor.transform(self.df)
            self._cat = transformed["cat"]
            self._num = transformed["num"]
        else:
            self._cat = None
            self._num = None

    def __len__(self):
        return len(self.df)

    def _load_image(self, idx):
        img_path = self.df.iloc[idx]["image_path"]
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Cannot read: {img_path}")
        image = np.stack([image] * 3, axis=-1)
        if self.transform:
            image = self.transform(image=image)["image"]
        return image

    def _load_tabular(self, idx):
        cat_tensors = {
            col: torch.tensor(self._cat[col][idx], dtype=torch.long)
            for col in CAT_FEATURES
        }
        num_tensor = torch.tensor(self._num[idx], dtype=torch.float32)
        return cat_tensors, num_tensor

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = torch.tensor(row["label"], dtype=torch.float32)

        if self.mode == "image":
            image = self._load_image(idx)
            # collate fonksiyonu image-only batch beklediği için tuple kısa.
            return image, label

        if self.mode == "tabular":
            # Image okunmaz — disk I/O'dan kaçınılır.
            cat_tensors, num_tensor = self._load_tabular(idx)
            return _DUMMY_IMAGE, label, cat_tensors, num_tensor

        # fused
        image = self._load_image(idx)
        cat_tensors, num_tensor = self._load_tabular(idx)
        return image, label, cat_tensors, num_tensor


# ---------------------------------------------------------------------------
# TTA Dataset — tek örnek için tüm TTA varyantlarını döndürür
# (sadece image / fused modlarda anlamlı — tabular'da TTA yok)
# ---------------------------------------------------------------------------
class TTADataset(Dataset):
    def __init__(self, base_dataset: MammographyDataset,
                 tta_transforms: list):
        if base_dataset.mode == "tabular":
            raise ValueError("TTA tabular-only modda kullanılamaz.")
        self.base = base_dataset
        self.tta_transforms = tta_transforms
        self.n_tta = len(tta_transforms)

    def __len__(self):
        return len(self.base) * self.n_tta

    def __getitem__(self, idx):
        sample_idx = idx // self.n_tta
        tta_idx    = idx  % self.n_tta

        row = self.base.df.iloc[sample_idx]
        img_path = row["image_path"]
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Cannot read: {img_path}")
        image = np.stack([image] * 3, axis=-1)
        image = self.tta_transforms[tta_idx](image=image)["image"]

        label = torch.tensor(row["label"], dtype=torch.float32)

        if self.base.mode == "image":
            return image, label

        # fused
        cat_tensors = {
            col: torch.tensor(self.base._cat[col][sample_idx], dtype=torch.long)
            for col in CAT_FEATURES
        }
        num_tensor = torch.tensor(self.base._num[sample_idx], dtype=torch.float32)
        return image, label, cat_tensors, num_tensor


# ---------------------------------------------------------------------------
# Collate fonksiyonları — moda göre seçilir
# ---------------------------------------------------------------------------
def collate_image_only(batch):
    """(image, label) batch'leri için."""
    images, labels = zip(*batch)
    return torch.stack(images), torch.stack(labels)


def collate_with_tabular(batch):
    """(image, label, cat, num) batch'leri için (tabular ve fused)."""
    images, labels, cat_dicts, nums = zip(*batch)
    # Tabular modda images dummy zero-tensor; yine de stack edilebilir
    # (her biri shape=(1,)). Bu durumda batch'i model kullanmayacak.
    if images[0].dim() == 1:
        # tabular-only → dummy image, stack etmenin maliyeti ihmal edilebilir
        images_out = torch.stack(images)
    else:
        images_out = torch.stack(images)
    labels = torch.stack(labels)
    nums   = torch.stack(nums)
    cat_batch = {
        col: torch.stack([d[col] for d in cat_dicts])
        for col in cat_dicts[0].keys()
    }
    return images_out, labels, cat_batch, nums


def get_collate_fn(mode: str):
    if mode == "image":
        return collate_image_only
    return collate_with_tabular   # tabular ve fused