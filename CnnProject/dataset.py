import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

import config

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_train_transform():
    return A.Compose([
        A.LongestMaxSize(max_size=config.IMAGE_SIZE),
        A.PadIfNeeded(min_height=config.IMAGE_SIZE,
                      min_width=config.IMAGE_SIZE,
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
        A.PadIfNeeded(min_height=config.IMAGE_SIZE,
                      min_width=config.IMAGE_SIZE,
                      border_mode=0),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


class MammographyDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["image_path"]

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Cannot read: {img_path}")

        image = np.stack([image] * 3, axis=-1)

        if self.transform:
            image = self.transform(image=image)["image"]

        label = torch.tensor(row["label"], dtype=torch.float32)
        return image, label