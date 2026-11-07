# Testing colours.py for one case 

import matplotlib.pyplot as plt
from src.data import load_splits, load_image
from src.features import colours

df = load_splits()

infected = df[df["label"] == 1]["filename"].iloc[0]
healthy  = df[df["label"] == 0]["filename"].iloc[0]

for name in [infected, healthy]:
    img = load_image(name, size=128)
    mask = colours.cell_mask(img)
    features = colours.extract(img)

    print(name)
    print("  vector length:", len(features))
    print("  histogram sums to:", features[:512].sum())
    print("  moments (R G B H S V: mean, std, skew):")
    print("  ", features[512:].round(2))

    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(img);  axes[0].set_title("image")
    axes[1].imshow(mask, cmap="gray"); axes[1].set_title("cell mask")
    plt.show()