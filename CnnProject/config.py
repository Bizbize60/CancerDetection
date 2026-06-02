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
DATASET_ROOT = r"C:\Users\PC\Desktop\Dataset"

CSV_DIR  = os.path.join(DATASET_ROOT, "csv")
JPEG_DIR = os.path.join(DATASET_ROOT, "jpeg")

# Hangi tip kullanılacak?
# "full"    : full mammogram (image file path) — gerçekçi ama daha zor
# "cropped" : ROI patch (cropped image file path) — MVP için ÖNERİLEN, kolay baseline
USE_IMAGE_TYPE = "cropped"

# ============================================================
# MODE — Hangi modaliteyle eğitim/değerlendirme yapılacak?
#   "image"   → sadece CNN (ResNet50 + SpatialAttention)
#   "tabular" → sadece MLP (kategorik + sayısal özellikler)
#   "fused"   → image + tabular intermediate fusion (varsayılan)
# ============================================================
MODE = "fused"

# Checkpoint dosya isimleri mode'a göre ayrılır → modeller birbirini ezmesin
CHECKPOINT_NAME = f"best_{MODE}.pt"
RESULTS_NAME    = f"results_{MODE}.txt"
PREDICTIONS_NAME = f"test_predictions_{MODE}.npz"

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
LABEL_SMOOTHING = 0.1
MIXUP_ALPHA     = 0.4
N_FOLDS         = 5

# Tabular-only mod için ayrı (daha hızlı) ayarlar
TABULAR_ONLY_EPOCHS = 80
TABULAR_ONLY_LR     = 1e-3
TABULAR_ONLY_BATCH  = 64

# Two-stage fine-tuning (image / fused mod için)
WARMUP_EPOCHS = 5        # ilk epoch'larda backbone donuk, sadece head eğitilir
HEAD_LR = 5e-4           # warmup sırasında head learning rate

# --- Device ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Klasörleri oluştur
os.makedirs(SPLITS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)