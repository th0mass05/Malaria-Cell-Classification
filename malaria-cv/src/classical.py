# Main system for manually classifying the images

# Uses kNN and SVM classifiers

import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
 
from src.config import CACHE_DIR, SEED
from src.data import load_splits
from src.run_cv import run_cv

FEATURES_DIRECTORY = CACHE_DIR / "features"
FEATURE_GROUPS = ["colour"]

# Can either use a search grid (default) or not.
# Using it is slower but better results, keep to false for faster but worse results
GRID_SEARCH = False

SVM_GRID = {
    "svc__C": [1, 10],
    "svc__gamma": ["scale", 0.01]
}

KNN_GRID = {
    "kneighborsclassifier__n_neighbours": [3, 5, 11, 21]
}

# Turn all feautures into one continous feature vector
def combine_features(groups):
    arrays = []
    for group in groups:
        path = FEATURES_DIRECTORY / f"{group}.npy"
        array = np.load(path)
        print("Loaded ", path, "with shape ", array.shape)
        arrays.append(array)

    allColumns = np.hstack(arrays)
    return allColumns


# Try every combination in the grid and return the best model
def tune(model, grid, X_train, y_train, patientIds_train):
    inner_folds = StratifiedGroupKFold(n_splits = 3, shuffle= True, random_state = SEED)
    search = GridSearchCV(
        estimator = model,
        param_grid = grid,
        cv = inner_folds,
        scoring = "accuracy",
        n_jobs = -1
    )

    search.fit(X_train, y_train, groups = patientIds_train)

    print("Best settings: ", search.best_params_)
    return search.best_estimator_