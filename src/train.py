import os
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import mlflow.sklearn


def load_config(config_path="config.yaml"):
    """Carga el archivo de configuración YAML."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def load_data(data_path):
    """Carga los datos desde un archivo CSV."""
    data = pd.read_csv(data_path)
    return data


def preprocess_data(data, target_column):
    """Divide los datos en características y variable objetivo."""
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return X, y


def train_model(X_train, y_train, params):
    """Entrena un modelo RandomForest con los parámetros especificados."""
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Evalúa el modelo usando Accuracy y F1 Score."""
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="weighted")
    return acc, f1


def main():
    """Pipeline principal de entrenamiento."""
    config = load_config()

    data_path = config["data_path"]
    target_column = config["target_column"]
    test_size = config["test_size"]
    random_state = config["random_state"]
    model_params = config["model_params"]

    data = load_data(data_path)
    X, y = preprocess_data(data, target_column)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    with mlflow.start_run():
        mlflow.log_params(model_params)
        model = train_model(X_train, y_train, model_params)
        acc, f1 = evaluate_model(model, X_test, y_test)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        mlflow.sklearn.log_model(model, "model")

        print(f"Accuracy: {acc:.4f}")
        print(f"F1 Score: {f1:.4f}")


if __name__ == "__main__":
    main()