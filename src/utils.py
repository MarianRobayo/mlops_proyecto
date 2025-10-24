import pandas as pd


def load_data(data_path):
    """Carga los datos desde un archivo CSV."""
    data = pd.read_csv(data_path)
    return data


def fill_missing(data, method="mean"):
    """Rellena valores nulos según el método indicado."""
    if method == "mean":
        return data.fillna(data.mean(numeric_only=True))
    if method == "median":
        return data.fillna(data.median(numeric_only=True))
    if method == "mode":
        return data.fillna(data.mode().iloc[0])
    return data


def encode_categorical(data):
    """Codifica variables categóricas usando one-hot encoding."""
    return pd.get_dummies(data, drop_first=True)


def scale_features(X, scaler):
    """Escala las características con el scaler dado (fit-transform)."""
    return scaler.fit_transform(X)
