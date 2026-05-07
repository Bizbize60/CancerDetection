"""
MammographyDataset — görüntü + tabular veriyi birlikte döndürür.

CSV'de şu kolonlar beklenir (prepare_splits.py tarafindan üretilir):
  image_path, label, left_or_right_breast, image_view,
  mass_shape, mass_margins, abnormality_type,
  breast_density, assessment, subtlety, abnormality_id
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


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
def get_train_transform():
    return A.Compose([
        A.LongestMaxSize(max_size=config.IMAGE_SIZE),
        A.PadIfNeeded(min_height=config.IMAGE_SIZE, min_width=config.IMAGE_SIZE,
                      border_mode=0),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, border_mode=0, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2,
                                   contrast_limit=0.2, p=0.5),
        A.GaussNoise(p=0.3),
        A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(8, 16),
                        hole_width_range=(8, 16), p=0.3),
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


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class MammographyDataset(Dataset):
    """
    Parameters
    ----------
    csv_path : str
        Split CSV (train/val/test.csv) üretilmiş olmalı.
    transform : albumentations Compose
    preprocessor : TabularPreprocessor | None
        Fit edilmiş preprocessor.  None → tabular özellikler döndürülmez.
    """

    def __init__(self, csv_path: str, transform=None,
                 preprocessor: TabularPreprocessor | None = None):
        self.df = normalise_columns(pd.read_csv(csv_path))
        self.transform = transform
        self.preprocessor = preprocessor

        # Tabular özellikler varsa önceden dönüştür (hız için)
        if preprocessor is not None:
            transformed = preprocessor.transform(self.df)
            self._cat = transformed["cat"]   # dict[str, np.ndarray]
            self._num = transformed["num"]   # (N, len(NUM_FEATURES))
        else:
            self._cat = None
            self._num = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # ---- image ----
        img_path = row["image_path"]
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Cannot read: {img_path}")
        image = np.stack([image] * 3, axis=-1)
        if self.transform:
            image = self.transform(image=image)["image"]

        label = torch.tensor(row["label"], dtype=torch.float32)

        # ---- tabular ----
        if self._cat is not None:
            cat_tensors = {
                col: torch.tensor(self._cat[col][idx], dtype=torch.long)
                for col in CAT_FEATURES
            }
            num_tensor = torch.tensor(self._num[idx], dtype=torch.float32)
            return image, label, cat_tensors, num_tensor

        return image, label


# ---------------------------------------------------------------------------
# Collate helper — DataLoader için (cat_tensors listesini dict'e çevirir)
# ---------------------------------------------------------------------------
def collate_with_tabular(batch):
    """
    batch elemanı: (image, label, cat_dict, num_tensor)
    Döndürür:      images, labels, cat_batch_dict, num_batch
    """
    images, labels, cat_dicts, nums = zip(*batch)
    images = torch.stack(images)
    labels = torch.stack(labels)
    nums   = torch.stack(nums)
    cat_batch = {
        col: torch.stack([d[col] for d in cat_dicts])
        for col in cat_dicts[0].keys()
    }
    return images, labels, cat_batch, nums
