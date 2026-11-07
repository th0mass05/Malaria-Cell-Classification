# Shared cross-validation runner

import numpy as np
from src.data import load_splits
from src.evaluate import evaluate, summarise

# Returns a table of mean and std of each metric across folds
def run_cv(train_and_predict, X, name, experiment = "main", **extra):
    # Can take extra columns to save

    splits = load_splits()
    labels = splits["label"].to_numpy()
    folds = splits["fold"].to_numpy()
    patientIds = splits["patient_id"].to_numpy()

    num_of_folds = len(np.unique(folds))

    for fold in range(num_of_folds):

        test_index = np.where(folds == fold)[0]
        train_index = np.where(folds != fold)[0]

        y_pred, y_score = train_and_predict(X, train_index, test_index, labels[train_index], patientIds)

        result = evaluate(labels[test_index], y_pred, y_score, system = name, fold = fold, experiment = experiment, **extra)

        print(f"[{name}] fold {fold}: accuracy = {result['accuracy']:.4f}, "f"recall = {result['recall']:.4f}")

    summary = summarise(experiment)
    print(summary)
    return summary