# Compute features for every image
# For each feature group it goes through every image in splits.csv and runs that groups extract()
# Saves all the vectors as one array

import numpy as np
from src.data import load_image, load_splits
from src.config import CACHE_DIR, IMG_SIZE
import src.features.colours as colours

FEATURE_TYPES = {"colour": (colours.extract, colours.FEATURE_LENGTH)}
FEATURES_DIRECTORY = CACHE_DIR / "features"

# Smaller for tests
TEST_LIMIT = None

# For re-running durinv development. groups that already have an npy file are skipped
OVERWRITE = True

def build_group(name, extract, vector_length, file_paths):
    vectors = []
    fails = 0

    for file_path in file_paths:
        try:
            image = load_image(file_path, size = IMG_SIZE)
            vector = extract(image)
        except Exception as error:
            # Store zeros so row order does not break
            print("Could not process")
            vector = np.zeros(vector_length)
            fails += 1
        vectors.append(vector)

    features = np.stack(vectors)
    if TEST_LIMIT is None:

        output = FEATURES_DIRECTORY / f"{name}.npy"
    else:
        output = FEATURES_DIRECTORY / f"{name}_test.npy"
    np.save(output, features)

    print("Saved ", output, "with shape", features.shape, "(Failed: )", fails)

    return features

def main():

    df = load_splits()
    file_paths = list(df["filename"])

    if TEST_LIMIT is not None:
        half = TEST_LIMIT // 2
        df = df.groupby("label").head(half)
        file_paths = list(df["filename"])
        print("Limiting to ", len(file_paths), "images for testing")

    
    # In case it isnt made
    FEATURES_DIRECTORY.mkdir(parents=True, exist_ok=True)

    for name, (extract, vector_length) in FEATURE_TYPES.items():
        output = FEATURES_DIRECTORY / f"{name}.npy"
        if output.exists() and not OVERWRITE:
            print("Skipping: ", name, "this output already exists, if you want to rebuild change OVERWRITE")
            continue

        print("Building ", name, "features for ", len(file_paths), "images")

        build_group(name, extract, vector_length, file_paths)

if __name__ == "__main__":
    main()
