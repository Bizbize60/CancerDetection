"""Tüm hiperparametreler ve yollar tek dosyada. MVP için yeterli."""
import os
import torch

# --- Paths ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
SPLITS_DIR = os.path.join(DATA_ROOT, "splits")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

# Kaggle CBIS-DDSM dataset kök klasörü (içinde csv/ ve jpeg/ olan)
# Örnek Kaggle yapısı:
#   <DATASET_ROOT>/
#     csv/
#       calc_case_description_train_set.csv
#       calc_case_description_test_set.csv
#       mass_case_description_train_set.csv
#       mass_case_description_test_set.csv
#       dicom_info.csv
#     jpeg/
#       1.3.6.1.4.1.9590.100.1.2.xxxxx/
#         1-xxx.jpg
DATASET_ROOT = r"C:\Users\PC\Desktop\Dataset"

CSV_DIR  = os.path.join(DATASET_ROOT, "csv")
JPEG_DIR = os.path.join(DATASET_ROOT, "jpeg")

# Hangi tip kullanılacak?
# "full"    : full mammogram (image file path) — gerçekçi ama daha zor
# "cropped" : ROI patch (cropped image file path) — MVP için ÖNERİLEN, kolay baseline
USE_IMAGE_TYPE = "cropped"

# --- Data ---
IMAGE_SIZE = 384
NUM_CHANNELS = 3
VAL_RATIO = 0.15
SEED = 42

# --- Training ---
BATCH_SIZE = 16
NUM_WORKERS = 4
EPOCHS = 50
LR = 3e-5
WEIGHT_DECAY = 1e-4
DROPOUT = 0.4
PATIENCE = 10
RADIMAGENET_WEIGHTS = r"C:\Users\PC\Desktop\Dataset\RadImageNet-ResNet50_notop.h5"
LABEL_SMOOTHING = 0.1   # 1.0 → 0.9, 0.0 → 0.1
MIXUP_ALPHA     = 0.4   # beta dağılımı parametresi
N_FOLDS         = 5 
LABEL_SMOOTHING = 0.1 
MIXUP_ALPHA     = 0.4
 
 

# Two-stage fine-tuning
WARMUP_EPOCHS = 5        # ilk 3 epoch backbone donuk, sadece head eğitilir
HEAD_LR = 5e-4           # warmup sırasında head learning rate

# --- Device ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Klasörleri oluştur
os.makedirs(SPLITS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)