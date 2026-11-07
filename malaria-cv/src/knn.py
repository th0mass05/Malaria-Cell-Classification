import sklearn
from src.classical import combine_features, tune
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.run_cv import run_cv

GROUPS = ["colour"]

GRID_SEARCH = True

KNN_GRID = {"kneighborsclassifier__n_neighbors": [3, 5, 11, 21]}

def train_and_predict(X, train_index, test_index, y_train, patientIds):

    X_train = X[train_index]
    X_test = X[test_index]

    # Standardise every feature
    model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors = 5))

    if GRID_SEARCH:
        model = tune(model, KNN_GRID, X_train, y_train, patientIds[train_index])
    else:
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]
    return y_pred, y_score

def main():
    X = combine_features(GROUPS)
    feature_name = "+".join(GROUPS)
    if GRID_SEARCH:
        name = f"System_A_{feature_name}_kNN_tuned"
    else:
        name = f"System_A_{feature_name}_kNN"
    run_cv(train_and_predict, X, name = name)


if __name__ == "__main__":
    main()
