"""
CBIS-DDSM Kaggle versiyonu için patient-level train/val/test split üretir.

Join mantığı:
  - CSV'lerdeki 'cropped image file path' formatı:
      Mass-Training_P_00001_LEFT_CC_1/<StudyUID>/<SeriesUID>/000000.dcm
    Sondan 2. segment = SeriesInstanceUID.
  - dicom_info.csv'deki yollar:
      CBIS-DDSM/jpeg/<SeriesUID>/<file>.jpg
    SeriesUID'yi sondan 2. segmentten alıyoruz.
  - SeriesDescription ile filtre:
      'cropped images', 'full mammogram images', 'ROI mask images'
    USE_IMAGE_TYPE'a göre doğru olanı seçiyoruz.
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

import config


# config.USE_IMAGE_TYPE → SeriesDescription değeri
SERIES_DESC_MAP = {
    "cropped": "cropped images",
    "full":    "full mammogram images",
    "mask":    "ROI mask images",   # genelde kullanılmaz, paranoya için var
}


def extract_series_uid(path_str):
    """
    Path örn: 'Mass-Training_P_00001_LEFT_CC_1/<StudyUID>/<SeriesUID>/000000.dcm\n'
    Sondan 2. segmenti döndürür (SeriesInstanceUID).
    """
    if pd.isna(path_str):
        return None
    s = str(path_str).strip().replace("\\", "/").rstrip("/")
    parts = [p for p in s.split("/") if p]
    if len(parts) < 2:
        return None
    return parts[-2]


def load_case_csv(csv_path):
    """Mass/Calc CSV'sini oku, label çıkar, kolonları normalize et."""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().replace(" ", "_").lower() for c in df.columns]

    df["label"] = (df["pathology"].str.upper() == "MALIGNANT").astype(int)

    if config.USE_IMAGE_TYPE == "cropped":
        path_col = "cropped_image_file_path"
    elif config.USE_IMAGE_TYPE == "full":
        path_col = "image_file_path"
    else:
        raise ValueError(f"Unknown USE_IMAGE_TYPE: {config.USE_IMAGE_TYPE}")

    if path_col not in df.columns:
        raise KeyError(
            f"'{path_col}' not in CSV. Available: {df.columns.tolist()}"
        )

    df = df[["patient_id", path_col, "label", "pathology",
             "abnormality_type"]].copy()
    df = df.rename(columns={path_col: "orig_path"})
    df["series_uid"] = df["orig_path"].apply(extract_series_uid)
    return df


def build_series_lookup(dicom_info_path, jpeg_dir, series_description):
    """
    {SeriesInstanceUID: jpeg_full_path} sözlüğü.
    SeriesDescription filtresi uygulanır (örn. 'cropped images').
    """
    di = pd.read_csv(dicom_info_path)
    di.columns = [c.strip() for c in di.columns]

    required = {"file_path", "image_path", "SeriesDescription"}
    missing = required - set(di.columns)
    if missing:
        raise KeyError(f"dicom_info.csv missing columns: {missing}")

    # Doğru tip (cropped / full / mask) için filtre
    di["SeriesDescription"] = di["SeriesDescription"].astype(str).str.strip()
    before = len(di)
    di = di[di["SeriesDescription"] == series_description].copy()
    print(f"  Filtered dicom_info by SeriesDescription='{series_description}': "
          f"{before} → {len(di)} rows")

    di["image_path"] = di["image_path"].astype(str).str.strip()
    di["series_uid"] = di["image_path"].apply(extract_series_uid)

    lookup = {}
    skipped = 0
    for _, row in di.iterrows():
        suid = row["series_uid"]
        if not suid:
            skipped += 1
            continue
        # image_path = 'CBIS-DDSM/jpeg/<SeriesUID>/<file>.jpg'
        # Sadece <SeriesUID>/<file>.jpg kısmını jpeg_dir altına yapıştır.
        rel = row["image_path"].replace("\\", "/")
        parts = [p for p in rel.split("/") if p]
        rel_short = os.path.join(parts[-2], parts[-1])  # SeriesUID/file.jpg
        full = os.path.join(jpeg_dir, rel_short)

        # Aynı SeriesUID'de birden fazla dosya olabilir (nadir).
        # İlkini sakla; tipik durumda 1 dosya zaten.
        if suid not in lookup:
            lookup[suid] = full

    if skipped:
        print(f"  Skipped {skipped} rows in dicom_info (no series_uid)")
    return lookup


def attach_jpeg_path(df, lookup, label=""):
    df["image_path"] = df["series_uid"].map(lookup)
    matched = df["image_path"].notna().sum()
    print(f"  [{label}] matched: {matched}/{len(df)}")
    df = df.dropna(subset=["image_path"]).reset_index(drop=True)

    # Diskte var mı sanity check (yavaş ama bir kerelik):
    exists = df["image_path"].apply(os.path.exists)
    if (~exists).any():
        n_missing = (~exists).sum()
        print(f"  [{label}] WARNING: {n_missing} mapped paths not found on disk")
        df = df[exists].reset_index(drop=True)
    return df


def main():
    series_desc = SERIES_DESC_MAP[config.USE_IMAGE_TYPE]
    print(f"Image type: {config.USE_IMAGE_TYPE!r} → "
          f"SeriesDescription={series_desc!r}")

    # --- 1) Lookup tablo ---
    dicom_info_path = os.path.join(config.CSV_DIR, "dicom_info.csv")
    print(f"\nLoading {dicom_info_path}...")
    lookup = build_series_lookup(dicom_info_path, config.JPEG_DIR, series_desc)
    print(f"  Lookup size: {len(lookup)} unique SeriesUIDs")

    # --- 2) CSV'leri oku ---
    train_files = [
        os.path.join(config.CSV_DIR, "mass_case_description_train_set.csv"),
        os.path.join(config.CSV_DIR, "calc_case_description_train_set.csv"),
    ]
    test_files = [
        os.path.join(config.CSV_DIR, "mass_case_description_test_set.csv"),
        os.path.join(config.CSV_DIR, "calc_case_description_test_set.csv"),
    ]

    print("\nLoading train CSVs...")
    train_full = pd.concat([load_case_csv(p) for p in train_files],
                           ignore_index=True)
    print(f"  Raw train+val rows: {len(train_full)}")
    train_full = attach_jpeg_path(train_full, lookup, "train+val")

    print("\nLoading test CSVs...")
    test_df = pd.concat([load_case_csv(p) for p in test_files],
                        ignore_index=True)
    print(f"  Raw test rows: {len(test_df)}")
    test_df = attach_jpeg_path(test_df, lookup, "test")

    if len(train_full) == 0 or len(test_df) == 0:
        raise RuntimeError("No rows matched. Check paths and dicom_info.csv format.")

    # --- 3) Label dağılımı ---
    print(f"\nTrain+val label dist:\n{train_full['label'].value_counts()}")
    print(f"\nTest label dist:\n{test_df['label'].value_counts()}")

    # --- 3.5) Train↔Test hasta sızıntısını temizle ---
    # CBIS-DDSM resmi split'inde bazı hastalar (örn. aynı kişide mass+calc)
    # her iki sette de görünebiliyor. Bu hastaları train+val'den çıkarıp
    # test'i temiz tutuyoruz (test set'i değiştirmek standart pratiğe aykırı).
    overlap_pids = set(train_full["patient_id"]) & set(test_df["patient_id"])
    if overlap_pids:
        before = len(train_full)
        train_full = train_full[~train_full["patient_id"].isin(overlap_pids)] \
                                .reset_index(drop=True)
        print(f"\nRemoved {len(overlap_pids)} overlapping patients from train+val: "
              f"{before} → {len(train_full)} rows")

    # --- 4) Patient-level + stratified val split ---
    # StratifiedGroupKFold: hem patient sızıntısını engeller (groups)
    # hem sınıf oranını korur (stratify=y).
    from sklearn.model_selection import StratifiedGroupKFold
    n_splits = max(2, int(round(1.0 / config.VAL_RATIO)))   # 0.15 → ~7 fold
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                random_state=config.SEED)
    train_idx, val_idx = next(sgkf.split(
        X=train_full,
        y=train_full["label"],
        groups=train_full["patient_id"],
    ))
    train_df = train_full.iloc[train_idx].reset_index(drop=True)
    val_df   = train_full.iloc[val_idx].reset_index(drop=True)

    leak_tv = set(train_df["patient_id"]) & set(val_df["patient_id"])
    assert len(leak_tv) == 0, f"PATIENT LEAK train↔val: {len(leak_tv)}"
    leak_tt = set(train_full["patient_id"]) & set(test_df["patient_id"])
    assert len(leak_tt) == 0, f"PATIENT LEAK train↔test: {len(leak_tt)}"
    if leak_tt:
        print(f"\nNOTE: {len(leak_tt)} patients overlap train+val ↔ test "
              f"(may happen if same patient has both mass and calc lesions)")

    # --- 5) Kaydet ---
    keep_cols = ["patient_id", "image_path", "label",
                 "pathology", "abnormality_type"]
    train_df[keep_cols].to_csv(
        os.path.join(config.SPLITS_DIR, "train.csv"), index=False)
    val_df[keep_cols].to_csv(
        os.path.join(config.SPLITS_DIR, "val.csv"), index=False)
    test_df[keep_cols].to_csv(
        os.path.join(config.SPLITS_DIR, "test.csv"), index=False)

    print(f"\nSaved splits to {config.SPLITS_DIR}")
    print(f"  Train: {len(train_df)}  | malignant ratio: "
          f"{train_df['label'].mean():.3f}")
    print(f"  Val:   {len(val_df)}  | malignant ratio: "
          f"{val_df['label'].mean():.3f}")
    print(f"  Test:  {len(test_df)}  | malignant ratio: "
          f"{test_df['label'].mean():.3f}")


if __name__ == "__main__":
    main()