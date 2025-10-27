import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    return df

def preprocess(df):
    # Ejemplo: crear target binario a partir de quality
    df = df.copy()
    df["target"] = (df["quality"] >= 7).astype(int)  # 7+ -> buena calidad
    X = df.drop(columns=["quality", "target"])
    y = df["target"]
    return X, y

def split_and_scale(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, scaler
