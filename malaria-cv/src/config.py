
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "cell_images"
CACHE_DIR = ROOT / "cache"
RESULTS_DIR = ROOT / "results"
SPLITS_CSV = ROOT / "splits.csv"

CLASSES = ["Uninfected", "Parasitized"]  # label 0, label 1 (positive class)
N_FOLDS = 5
SEED = 42
IMG_SIZE = 128  # traditional pipeline; the CNN resizes to 224 itself
