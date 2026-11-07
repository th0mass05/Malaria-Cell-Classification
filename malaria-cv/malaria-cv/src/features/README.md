One module per feature group (colour.py, interest_points.py, region.py, shape.py, lbp.py).
Each exposes `extract(img, mask=None) -> 1D np.ndarray`; build_features.py caches them to cache/features/<group>.npy.
