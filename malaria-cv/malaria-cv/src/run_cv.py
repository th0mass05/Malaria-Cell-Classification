"""Generic cross-validation runner.

A "system" is any object with fit(train_idx) and predict(test_idx) that returns
(y_pred, y_score). This keeps every method on the same folds and metrics.

Example:
    from src.run_cv import run_cv
    run_cv(MySystem(), name="A_handcrafted_svm")
"""
from src.data import fold_indices, load_splits
from src.evaluate import evaluate, summarise


def run_cv(system, name: str, experiment: str = "main", folds=None, **extra):
    df = load_splits()
    y = df["label"].to_numpy()
    folds = range(df["fold"].nunique()) if folds is None else folds
    for fold in folds:
        tr, te = fold_indices(df, fold)
        system.fit(tr, y[tr])
        y_pred, y_score = system.predict(te)
        row = evaluate(y[te], y_pred, y_score, system=name, fold=fold, experiment=experiment, **extra)
        print(f"[{name}] fold {fold}: acc={row['accuracy']:.4f} recall={row['recall']:.4f}")
    return summarise(experiment)
