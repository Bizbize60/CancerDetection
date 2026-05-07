"""
Tabular feature preprocessing + MLP encoder for CBIS-DDSM mass CSV.

Categorical features  → ordinal-encoded (unknown → -1 → mapped to last bucket)
Numerical  features   → z-score normalised

Output: 64-dim embedding, ready to concat with image branch.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

import config

# ---------------------------------------------------------------------------
# Feature schema
# ---------------------------------------------------------------------------
CAT_FEATURES = [
    "left_or_right_breast",   # LEFT / RIGHT
    "image_view",             # CC / MLO
    "mass_shape",             # IRREGULAR, OVAL, ROUND, …
    "mass_margins",           # ILL-DEFINED, SPICULATED, …
    "abnormality_type",       # mass (always, but kept for generality)
]
NUM_FEATURES = [
    "breast_density",         # 1-4
    "assessment",             # BI-RADS 0-5
    "subtlety",               # 1-5
    "abnormality_id",         # integer index (proxy for lesion count)
]

TABULAR_EMB_DIM = 64          # MLP output dimension


# ---------------------------------------------------------------------------
# Column normaliser (rename CSV cols → internal names)
# ---------------------------------------------------------------------------
COL_MAP = {
    "left or right breast": "left_or_right_breast",
    "image view":           "image_view",
    "mass shape":           "mass_shape",
    "mass margins":         "mass_margins",
    "abnormality type":     "abnormality_type",
    "abnormality id":       "abnormality_id",
}


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


# ---------------------------------------------------------------------------
# TabularPreprocessor  (fit on train, transform on val/test)
# ---------------------------------------------------------------------------
class TabularPreprocessor:
    """Fits LabelEncoders + StandardScaler on train split; serialisable."""

    def __init__(self):
        self.cat_encoders: dict[str, LabelEncoder] = {}
        self.num_scaler = StandardScaler()
        self.cat_dims: dict[str, int] = {}   # feature → num_classes (for MLP)
        self._fitted = False

    # ------------------------------------------------------------------ fit
    def fit(self, df: pd.DataFrame):
        df = normalise_columns(df)

        for col in CAT_FEATURES:
            le = LabelEncoder()
            vals = df[col].fillna("UNKNOWN").astype(str).str.upper().str.strip()
            le.fit(vals)
            self.cat_encoders[col] = le
            self.cat_dims[col] = len(le.classes_) + 1   # +1 for unseen

        num_data = df[NUM_FEATURES].fillna(0).values.astype(float)
        self.num_scaler.fit(num_data)
        self._fitted = True

    # --------------------------------------------------------------- transform
    def transform(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        assert self._fitted, "Call fit() first."
        df = normalise_columns(df)

        cat_arrays = {}
        for col in CAT_FEATURES:
            le = self.cat_encoders[col]
            vals = df[col].fillna("UNKNOWN").astype(str).str.upper().str.strip()
            # Unseen → last index (OOV bucket)
            encoded = np.array([
                le.transform([v])[0] if v in le.classes_ else len(le.classes_)
                for v in vals
            ], dtype=np.int64)
            cat_arrays[col] = encoded

        num_array = self.num_scaler.transform(
            df[NUM_FEATURES].fillna(0).values.astype(float)
        ).astype(np.float32)

        return {"cat": cat_arrays, "num": num_array}

    # ----------------------------------------------------------- persist
    def save(self, path: str):
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "TabularPreprocessor":
        return joblib.load(path)


# ---------------------------------------------------------------------------
# Tabular MLP  (embedding lookup per cat feature + dense layers for num)
# ---------------------------------------------------------------------------
class TabularMLP(nn.Module):
    """
    Input:
        cat_inputs  : dict[str, LongTensor]  (one per categorical feature)
        num_inputs  : FloatTensor  (batch, len(NUM_FEATURES))
    Output:
        FloatTensor  (batch, TABULAR_EMB_DIM)
    """

    def __init__(self, cat_dims: dict[str, int], dropout: float = 0.3):
        super().__init__()
        self.cat_features = list(cat_dims.keys())

        # Per-feature embedding: dim = min(50, (n_classes+1)//2 + 1) rule of thumb
        self.embeddings = nn.ModuleDict({
            col: nn.Embedding(n_cls, min(50, (n_cls) // 2 + 2))
            for col, n_cls in cat_dims.items()
        })
        emb_total = sum(min(50, (n_cls) // 2 + 2) for n_cls in cat_dims.values())
        total_in = emb_total + len(NUM_FEATURES)

        hidden = 128
        self.net = nn.Sequential(
            nn.Linear(total_in, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, TABULAR_EMB_DIM),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        cat_inputs: dict[str, torch.Tensor],
        num_inputs: torch.Tensor,
    ) -> torch.Tensor:
        embs = [
            self.embeddings[col](cat_inputs[col])
            for col in self.cat_features
        ]
        x = torch.cat(embs + [num_inputs], dim=1)
        return self.net(x)
