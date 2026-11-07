import numpy as np
import pandas as pd

x = np.load("cache/features/colour_test.npy")
splits = pd.read_csv("splits.csv").groupby("label").head(25)

df = pd.DataFrame(x[:, 512:])
df.insert(0, "label", splits["label"].to_numpy())
print(df.head(10))
print(df.groupby("label").mean().round(2))
print(x.shape)