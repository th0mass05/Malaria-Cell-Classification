
import re

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.config import CLASSES, DATA_DIR, N_FOLDS, SEED, SPLITS_CSV

# Grab everytjing in the file name before _IMG
_GROUP_RE = re.compile(r"^(.*?)_IMG", re.IGNORECASE)


def group_id(filename: str) -> str:
    # Return the slide/patient ID encoded in a filename (prefix before '_IMG')
    m = _GROUP_RE.match(filename)
    prefix = m.group(1) if m else filename.split("_")[0]
    # normalise
    return prefix.lower().replace("thinf", "").strip("_")


def build_splits(data_dir=DATA_DIR, n_folds=N_FOLDS, seed=SEED) -> pd.DataFrame:
    # List all images and assign each patient group to exactly one fold
    rows = []
    for label, cls in enumerate(CLASSES):
        for path in sorted((data_dir / cls).glob("*.png")):
            rows.append({"filename": f"{cls}/{path.name}", "label": label,
                         "patient_id": group_id(path.name)})
    df = pd.DataFrame(rows)
    if df.empty:
        raise FileNotFoundError(f"No images found in {data_dir}. Run scripts/download_data.sh")
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    df["fold"] = -1
    for fold, (_, test_idx) in enumerate(sgkf.split(df, df["label"], df["patient_id"])):
        df.loc[df.index[test_idx], "fold"] = fold
    return df


if __name__ == "__main__":
    df = build_splits()
    df.to_csv(SPLITS_CSV, index=False)
    print(f"{len(df)} images, {df['patient_id'].nunique()} patient groups -> {SPLITS_CSV}")
    print(df.groupby("fold")["label"].agg(["count", "mean"]).rename(columns={"mean": "frac_parasitized"}))
