import numpy as np
import pandas as pd

from src.evaluate import evaluate, summarise, to_latex
from src.make_splits import group_id


def test_group_id_normalises_case():
    assert group_id("C33P1thinF_IMG_20150619_114756a_cell_179.png") == \
           group_id("C33P1ThinF_IMG_20150619_120000_cell_1.png")


def test_evaluate_appends_and_summarises(tmp_path):
    y = np.array([0, 0, 1, 1])
    for fold in range(2):
        evaluate(y, np.array([0, 1, 1, 1]), np.array([.1, .6, .8, .9]),
                 system="dummy", fold=fold, results_dir=tmp_path)
    df = pd.read_csv(tmp_path / "main.csv")
    assert len(df) == 2 and df["recall"].iloc[0] == 1.0 and df["accuracy"].iloc[0] == 0.75
    s = summarise(results_dir=tmp_path)
    assert "dummy" in to_latex(s)
