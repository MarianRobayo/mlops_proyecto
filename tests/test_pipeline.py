import os
from src.utils import load_data, preprocess


def test_load_and_preprocess():
    path = "data/winequality-red.csv"
    assert os.path.exists(path)
    df = load_data(path)
    X, y = preprocess(df)
    # chequear tamaño consistencia # al menos una columna
    assert len(X) == len(y)
    assert X.shape[1] > 0
