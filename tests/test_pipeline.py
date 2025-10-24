import os
from src.utils import load_data, preprocess

def test_load_and_preprocess():
    path = "data/winequality-red-sample.csv"
    assert os.path.exists(path)
    df = load_data(path)
    X, y = preprocess(df)
    assert len(X) == len(y)
    assert X.shape[1] > 0
