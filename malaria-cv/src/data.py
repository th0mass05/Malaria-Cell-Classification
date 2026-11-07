# Helpers to load images and fold indices from splits.csv.
import cv2
import numpy as np
import pandas as pd

from src.config import DATA_DIR, SPLITS_CSV


def load_splits(path=SPLITS_CSV) -> pd.DataFrame:
    return pd.read_csv(path)


def fold_indices(df: pd.DataFrame, fold: int):
    # Return (train_idx, test_idx) numpy arrays for one outer fold.
    test = df["fold"].to_numpy() == fold
    return np.flatnonzero(~test), np.flatnonzero(test)


def load_image(filename: str, size: int | None = None) -> np.ndarray:
    # Load an image, optionally resized to size x size.
    img = cv2.imread(str(DATA_DIR / filename), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if size is not None:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img
