"""
CBIS-DDSM Kaggle versiyonu için patient-level train/val/test split üretir.
Tabular kolonlar da split CSV'lerine eklenir (MLP için).

Join mantığı:
  - CSV'lerdeki 'cropped image file path' / 'image file path'
      → SeriesUID = sondan 2. path segmenti
  - dicom_info.csv SeriesDescription filtresi ile jpeg path eşlenir.
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

import config

SERIES_DESC_MAP = {
    "cropped": "cropped images",
    "full":    "full mammogram images",
    "mask":    "ROI mask images",
}

# Tabular kolonlar split CSV'de saklanacak
TABULAR_KEEP_COLS = [
    "breast_density", "left_or_right_breast", "image_view",
    "abnormality_id", "abnormality_type",
    "mass_shape", "mass_margins",
    "assessment", "subtlety",
]


def extract_series_uid(path_str):
    if pd.isna(path_str):
        return None
    s = str(path_str).strip().replace("\\", "/").rstrip("/")
    parts = [p for p in s.split("/") if p]
    return parts[-2] if len(parts) >= 2 else None


def load_case_csv(csv_path):
    df = pd.read_csv(csv_path)
    # Normalize column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    df["label"] = (df["pathology"].str.upper() == "MALIGNANT").astype(int)

    path_col = (
        "cropped_image_file_path" if config.USE_IMAGE_TYPE == "cropped"
        else "image_file_path"
    )
    if path_col not in df.columns:
        raise KeyError(f"'{path_col}' not in {csv_path}. "
                       f"Available: {df.columns.tolist()}")

    keep = ["patient_id", path_col, "label", "pathology"] + [
        c for c in TABULAR_KEEP_COLS if c in df.columns
    ]
    df = df[keep].copy()
    df = df.rename(columns={path_col: "orig_path"})
    df["series_uid"] = df["orig_path"].apply(extract_series_uid)
    return df


def build_series_lookup(dicom_info_path, jpeg_dir, series_description):
    di = pd.read_csv(dicom_info_path)
    di.columns = [c.strip() for c in di.columns]

    required = {"file_path", "image_path", "SeriesDescription"}
    missing = required - set(di.columns)
    if missing:
        raise KeyError(f"dicom_info.csv missing columns: {missing}")

    di["SeriesDescription"] = di["SeriesDescription"].astype(str).str.strip()
    before = len(di)
    di = di[di["SeriesDescription"] == series_description].copy()
    print(f"  Filtered dicom_info by SeriesDescription='{series_description}': "
          f"{before} → {len(di)} rows")

    di["image_path"] = di["image_path"].astype(str).str.strip()
    di["series_uid"] = di["image_path"].apply(extract_series_uid)

    lookup, skipped = {}, 0
    for _, row in di.iterrows():
        suid = row["series_uid"]
        if not suid:
            skipped += 1
            continue
        rel = row["image_path"].replace("\\", "/")
        parts = [p for p in rel.split("/") if p]
        rel_short = os.path.join(parts[-2], parts[-1])
        full = os.path.join(jpeg_dir, rel_short)
        if suid not in lookup:
            lookup[suid] = full

    if skipped:
        print(f"  Skipped {skipped} rows (no series_uid)")
    return lookup


def attach_jpeg_path(df, lookup, label=""):
    df["image_path"] = df["series_uid"].map(lookup)
    matched = df["image_path"].notna().sum()
    print(f"  [{label}] matched: {matched}/{len(df)}")
    df = df.dropna(subset=["image_path"]).reset_index(drop=True)
    exists = df["image_path"].apply(os.path.exists)
    if (~exists).any():
        print(f"  [{label}] WARNING: {(~exists).sum()} paths not on disk")
        df = df[exists].reset_index(drop=True)
    return df


def main():
    series_desc = SERIES_DESC_MAP[config.USE_IMAGE_TYPE]
    print(f"Image type: {config.USE_IMAGE_TYPE!r} → SeriesDescription={series_desc!r}")

    dicom_info_path = os.path.join(config.CSV_DIR, "dicom_info.csv")
    print(f"\nLoading {dicom_info_path}...")
    lookup = build_series_lookup(dicom_info_path, config.JPEG_DIR, series_desc)
    print(f"  Lookup size: {len(lookup)}")

    train_files = [
        os.path.join(config.CSV_DIR, "mass_case_description_train_set.csv"),
    ]
    test_files = [
        os.path.join(config.CSV_DIR, "mass_case_description_test_set.csv"),
    ]

    print("\nLoading train CSVs...")
    train_full = pd.concat([load_case_csv(p) for p in train_files], ignore_index=True)
    train_full = attach_jpeg_path(train_full, lookup, "train+val")

    print("\nLoading test CSVs...")
    test_df = pd.concat([load_case_csv(p) for p in test_files], ignore_index=True)
    test_df = attach_jpeg_path(test_df, lookup, "test")

    if len(train_full) == 0 or len(test_df) == 0:
        raise RuntimeError("No rows matched. Check paths and dicom_info.csv format.")

    # Remove overlapping patients
    overlap_pids = set(train_full["patient_id"]) & set(test_df["patient_id"])
    if overlap_pids:
        before = len(train_full)
        train_full = train_full[
            ~train_full["patient_id"].isin(overlap_pids)
        ].reset_index(drop=True)
        print(f"\nRemoved {len(overlap_pids)} overlapping patients: "
              f"{before} → {len(train_full)}")

    # Patient-level stratified split
    n_splits = max(2, int(round(1.0 / config.VAL_RATIO)))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                random_state=config.SEED)
    train_idx, val_idx = next(sgkf.split(
        X=train_full, y=train_full["label"], groups=train_full["patient_id"]
    ))
    train_df = train_full.iloc[train_idx].reset_index(drop=True)
    val_df   = train_full.iloc[val_idx].reset_index(drop=True)

    assert len(set(train_df["patient_id"]) & set(val_df["patient_id"])) == 0

    # Save — tabular cols included
    base_cols = ["patient_id", "image_path", "label", "pathology"]
    tab_cols  = [c for c in TABULAR_KEEP_COLS if c in train_full.columns]
    keep_cols = base_cols + tab_cols

    train_df[keep_cols].to_csv(os.path.join(config.SPLITS_DIR, "train.csv"), index=False)
    val_df[keep_cols].to_csv(  os.path.join(config.SPLITS_DIR, "val.csv"),   index=False)
    test_df[keep_cols].to_csv( os.path.join(config.SPLITS_DIR, "test.csv"),  index=False)

    print(f"\nSaved splits to {config.SPLITS_DIR}")
    print(f"  Train: {len(train_df)}  malignant={train_df['label'].mean():.3f}")
    print(f"  Val:   {len(val_df)}  malignant={val_df['label'].mean():.3f}")
    print(f"  Test:  {len(test_df)}  malignant={test_df['label'].mean():.3f}")
    print(f"  Tabular columns saved: {tab_cols}")


if __name__ == "__main__":
    main()
