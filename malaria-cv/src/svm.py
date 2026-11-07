from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
 
import src.classical
from src.run_cv import run_cv

GROUPS = ["colour"]

GRID_SEARCH = True

SVM_GRID = {
    "svc__C": [1, 10],
    "svc__gamma": ["scale", 0.01]
}

def train_and_predict(X, train_index, test_index, y_train, patient_ids):
    # Training an SVM

    X_train = X[train_index]
    X_test = X[test_index]

    # Standardise every feature, and allows an rbf kernel (can curve)
    model = make_pipeline(StandardScaler(), SVC(kernel = "rbf", C = 1, gamma = "scale"))

    if GRID_SEARCH == True:
        model = src.classical.tune(model, SVM_GRID, X_train, y_train, patient_ids[train_index])
    else:
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    # Distance from the boundary (positive means infected)
    y_score = model.decision_function(X_test)
    return y_pred, y_score

def main():
    X = src.classical.combine_features(GROUPS)
    feature_name = "+".join(GROUPS)
    if GRID_SEARCH:
        name = f"System_A_{feature_name}_SVM_tuned"
    else:
        name = f"System_A_{feature_name}_SVM"
    run_cv(train_and_predict, X, name = name)

if __name__ == "__main__":
    main()