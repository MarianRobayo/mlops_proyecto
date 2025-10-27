import pandas as pd


def load_data(path: str):
    """Carga un archivo CSV en un DataFrame de pandas."""
    return pd.read_csv(path)


def preprocess_data(df, target_column="quality"):
    """Divide el DataFrame en variables independientes y dependientes."""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y
