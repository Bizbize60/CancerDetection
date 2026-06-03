# Breast Cancer Detection MVP — CBIS-DDSM

ResNet18 + binary classification (benign vs malignant) on CBIS-DDSM.

## Setup

```bash
pip install -r requirements.txt
```

Edit `config.py`:
- `IMAGE_ROOT`: CBIS-DDSM JPEG/PNG klasörünü göster
- `RAW_CSV_DIR`: Original CBIS-DDSM CSV'lerinin klasörü

## Run

```bash
# 1) Patient-level split üret
python prepare_splits.py

# 2) Eğit
python train.py

# 3) Test
python evaluate.py
```

## Expected Output

- `data/splits/{train,val,test}.csv`
- `checkpoints/best.pt`
- `outputs/results.txt`

Beklenen baseline: Val AUC ~0.75-0.82, Test AUC ~0.72-0.80.

-------------------------------
# ARAYÜZ ÇALIŞTIRMA ADIMLARI

# 1) pip install -r requirements.txt

# put best_image.pt / best_tabular.pt / best_fused.pt / tabular_preprocessor.pkl in checkpoints/

# 2) python run_api.py          # or: uvicorn app.main:app --reload

# 3) open http://127.0.0.1:8000/

-------------------------------
# GÜNCELLENMİŞ HALİ

# 1) Script izin ver (virtual env kurduktan sonra- kuruluysa devam)
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# 2) venv aktif et
.\.venv\Scripts\activate

# 3) Bağımlılıklar
pip install -r requirements.txt

# 4) config.py içinde DATASET_ROOT'u kendi yoluna göre değiştir
#    (içinde csv/ ve jpeg/ klasörleri olan kök)

# 5) Patient-level split üret (dicom_info üzerinden join yapacak)
python prepare_splits.py

# 6) Eğit
python train.py

# 7) Test
python evaluate.py