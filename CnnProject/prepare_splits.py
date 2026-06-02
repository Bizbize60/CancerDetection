"""
CBIS-DDSM — Mass + Calc birleşik patient-level train/val/test split.

Mass CSV'sine özgü : mass_shape, mass_margins
Calc CSV'sine özgü : calc_type, calc_distribution
Ortak              : breast_density, assessment, subtlety, vb.
Eksik kolonlar otomatik UNKNOWN/0 ile doldurulur (tabular_encoder ile uyumlu).
"""
import os
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

import config

SERIES_DESC_MAP = {
    "cropped": "cropped images",
    "full":    "full mammogram images",
}

# Tüm tabular kolonlar (mass + calc birleşimi)
TABULAR_KEEP_COLS = [
    "breast_density", "left_or_right_breast", "image_view",
    "abnormality_id", "abnormality_type",
    "mass_shape", "mass_margins",           # mass özgü
    "calc_type", "calc_distribution",       # calc özgü
    "assessment", "subtlety",               # sızıntı riski dikkat
]


def extract_series_uid(path_str):
    if pd.isna(path_str):
        return None
    s = str(path_str).strip().replace("\\", "/").rstrip("/")
    parts = [p for p in s.split("/") if p]
    return parts[-2] if len(parts) >= 2 else None


def load_case_csv(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df["label"] = (df["pathology"].str.upper() == "MALIGNANT").astype(int)

    path_col = (
        "cropped_image_file_path" if config.USE_IMAGE_TYPE == "cropped"
        else "image_file_path"
    )
    if path_col not in df.columns:
        raise KeyError(f"'{path_col}' not in {csv_path}. "
                       f"Available: {df.columns.tolist()}")

    # Mevcut tabular kolonları al, olmayanları boş bırak
    avail_tab = [c for c in TABULAR_KEEP_COLS if c in df.columns]
    keep = ["patient_id", path_col, "label", "pathology"] + avail_tab
    df = df[keep].copy()
    df = df.rename(columns={path_col: "orig_path"})
    df["series_uid"] = df["orig_path"].apply(extract_series_uid)
    return df


def build_series_lookup(dicom_info_path, jpeg_dir, series_description):
    di = pd.read_csv(dicom_info_path)
    di.columns = [c.strip() for c in di.columns]
    di["SeriesDescription"] = di["SeriesDescription"].astype(str).str.strip()
    before = len(di)
    di = di[di["SeriesDescription"] == series_description].copy()
    print(f"  Filtered by '{series_description}': {before} → {len(di)}")
    di["image_path"] = di["image_path"].astype(str).str.strip()
    di["series_uid"] = di["image_path"].apply(extract_series_uid)

    lookup, skipped = {}, 0
    for _, row in di.iterrows():
        suid = row["series_uid"]
        if not suid:
            skipped += 1
            continue
        rel   = row["image_path"].replace("\\", "/")
        parts = [p for p in rel.split("/") if p]
        full  = os.path.join(jpeg_dir, parts[-2], parts[-1])
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
    print(f"Image type: {config.USE_IMAGE_TYPE!r} → '{series_desc}'")

    dicom_info_path = os.path.join(config.CSV_DIR, "dicom_info.csv")
    lookup = build_series_lookup(dicom_info_path, config.JPEG_DIR, series_desc)
    print(f"  Lookup size: {len(lookup)}")

    # ---- Mass + Calc CSV'leri birleştir ----
    train_files = [
        os.path.join(config.CSV_DIR, "mass_case_description_train_set.csv"),
        os.path.join(config.CSV_DIR, "calc_case_description_train_set.csv"),
    ]
    test_files = [
        os.path.join(config.CSV_DIR, "mass_case_description_test_set.csv"),
        os.path.join(config.CSV_DIR, "calc_case_description_test_set.csv"),
    ]

    print("\nLoading train CSVs (mass + calc)...")
    train_full = pd.concat([load_case_csv(p) for p in train_files],
                           ignore_index=True)
    print(f"  Raw rows: {len(train_full)}")
    train_full = attach_jpeg_path(train_full, lookup, "train+val")

    print("\nLoading test CSVs (mass + calc)...")
    test_df = pd.concat([load_case_csv(p) for p in test_files],
                        ignore_index=True)
    print(f"  Raw rows: {len(test_df)}")
    test_df = attach_jpeg_path(test_df, lookup, "test")

    if len(train_full) == 0 or len(test_df) == 0:
        raise RuntimeError("No rows matched. Check paths and dicom_info.csv.")

    print(f"\nTrain+val label dist:\n{train_full['label'].value_counts()}")
    print(f"\nTest label dist:\n{test_df['label'].value_counts()}")
    print(f"\nabnormality_type dist (train):\n"
          f"{train_full.get('abnormality_type', pd.Series()).value_counts()}")

    # Hasta sızıntısını temizle
    overlap = set(train_full["patient_id"]) & set(test_df["patient_id"])
    if overlap:
        before = len(train_full)
        train_full = train_full[
            ~train_full["patient_id"].isin(overlap)
        ].reset_index(drop=True)
        print(f"\nRemoved {len(overlap)} overlapping patients: {before} → {len(train_full)}")

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

    # Eksik tabular kolonları doldur (mass'ta calc kolonları yok, tersi de)
    for col in TABULAR_KEEP_COLS:
        for df_ in [train_df, val_df, test_df]:
            if col not in df_.columns:
                df_[col] = None   # tabular_encoder fill_missing_tabular_cols halleder

    base_cols = ["patient_id", "image_path", "label", "pathology"]
    keep_cols = base_cols + TABULAR_KEEP_COLS

    train_df[keep_cols].to_csv(os.path.join(config.SPLITS_DIR, "train.csv"), index=False)
    val_df[keep_cols].to_csv(  os.path.join(config.SPLITS_DIR, "val.csv"),   index=False)
    test_df[keep_cols].to_csv( os.path.join(config.SPLITS_DIR, "test.csv"),  index=False)

    print(f"\nSaved splits → {config.SPLITS_DIR}")
    print(f"  Train: {len(train_df)}  malignant={train_df['label'].mean():.3f}")
    print(f"  Val:   {len(val_df)}  malignant={val_df['label'].mean():.3f}")
    print(f"  Test:  {len(test_df)}  malignant={test_df['label'].mean():.3f}")


if __name__ == "__main__":
    main()
