"""
Tabular feature preprocessing + MLP encoder.
Mass ve Calc CSV'lerini birleşik olarak destekler.

Mass özgü : mass_shape, mass_margins
Calc özgü : calc_type, calc_distribution
Ortak     : breast_density, left_or_right_breast, image_view,
            abnormality_id, abnormality_type, assessment, subtlety

Eksik kolonlar otomatik "UNKNOWN" ile doldurulur.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

import config

# ---------------------------------------------------------------------------
# Feature schema
# ---------------------------------------------------------------------------
CAT_FEATURES = [
    "left_or_right_breast",
    "image_view",
    "abnormality_type",     # "mass" vs "calcification"
    "mass_shape",           # calc için UNKNOWN
    "mass_margins",         # calc için UNKNOWN
    "calc_type",            # mass için UNKNOWN
    "calc_distribution",    # mass için UNKNOWN
]

NUM_FEATURES = [
    "breast_density",
    "abnormality_id",      # assesment ve subtlety sızıntı sebebiyle kaldırıldı
]

TABULAR_EMB_DIM = 64


# ---------------------------------------------------------------------------
# Column normaliser
# ---------------------------------------------------------------------------
def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def fill_missing_tabular_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Mass veya Calc CSV'sinde olmayan kolonları UNKNOWN / 0 ile doldur."""
    df = df.copy()
    for col in CAT_FEATURES:
        if col not in df.columns:
            df[col] = "UNKNOWN"
        else:
            df[col] = df[col].fillna("UNKNOWN").astype(str).str.upper().str.strip()
    for col in NUM_FEATURES:
        if col not in df.columns:
            df[col] = 0
        else:
            df[col] = df[col].fillna(0)
    return df


# ---------------------------------------------------------------------------
# TabularPreprocessor
# ---------------------------------------------------------------------------
class TabularPreprocessor:
    def __init__(self):
        self.cat_encoders: dict[str, LabelEncoder] = {}
        self.num_scaler = StandardScaler()
        self.cat_dims: dict[str, int] = {}
        self._fitted = False

    def fit(self, df: pd.DataFrame):
        df = fill_missing_tabular_cols(normalise_columns(df))
        for col in CAT_FEATURES:
            le = LabelEncoder()
            le.fit(df[col])
            self.cat_encoders[col] = le
            self.cat_dims[col] = len(le.classes_) + 1   # +1 OOV bucket
        self.num_scaler.fit(df[NUM_FEATURES].values.astype(float))
        self._fitted = True

    def transform(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        assert self._fitted, "Call fit() first."
        df = fill_missing_tabular_cols(normalise_columns(df))
        cat_arrays = {}
        for col in CAT_FEATURES:
            le = self.cat_encoders[col]
            encoded = np.array([
                le.transform([v])[0] if v in le.classes_ else len(le.classes_)
                for v in df[col]
            ], dtype=np.int64)
            cat_arrays[col] = encoded
        num_array = self.num_scaler.transform(
            df[NUM_FEATURES].values.astype(float)
        ).astype(np.float32)
        return {"cat": cat_arrays, "num": num_array}

    def save(self, path: str):
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "TabularPreprocessor":
        return joblib.load(path)


# ---------------------------------------------------------------------------
# TabularMLP
# ---------------------------------------------------------------------------
class TabularMLP(nn.Module):
    def __init__(self, cat_dims: dict[str, int], dropout: float = 0.3):
        super().__init__()
        self.cat_features = list(cat_dims.keys())
        self.embeddings = nn.ModuleDict({
            col: nn.Embedding(n_cls, min(50, n_cls // 2 + 2))
            for col, n_cls in cat_dims.items()
        })
        emb_total = sum(min(50, n_cls // 2 + 2) for n_cls in cat_dims.values())
        total_in  = emb_total + len(NUM_FEATURES)
        hidden    = 128
        self.net = nn.Sequential(
            nn.Linear(total_in, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, TABULAR_EMB_DIM),
            nn.ReLU(inplace=True),
        )

    def forward(self, cat_inputs: dict[str, torch.Tensor],
                num_inputs: torch.Tensor) -> torch.Tensor:
        embs = [self.embeddings[col](cat_inputs[col]) for col in self.cat_features]
        x = torch.cat(embs + [num_inputs], dim=1)
        return self.net(x)
