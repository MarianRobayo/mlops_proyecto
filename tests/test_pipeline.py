import os
from src.train import load_data, preprocess_data


def test_load_and_preprocess():
    path = "data/winequality-red.csv"
    assert os.path.exists(path)
    df = load_data(path)
    X, y = preprocess_data(df, "quality")
    assert len(X) == len(y)
    assert X.shape[1] > 0
