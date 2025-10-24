import argparse
import yaml
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from src.utils import load_data, preprocess_data


def main():
    # ===============================
    # 1. Argumentos de línea de comando
    # ===============================
    parser = argparse.ArgumentParser(
        description="Entrenamiento de modelo con MLflow"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Ruta al archivo de configuración YAML",
    )
    args = parser.parse_args()

    # ===============================
    # 2. Cargar configuración
    # ===============================
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    data_path = config["data_path"]
    test_size = config["test_size"]
    random_state = config["random_state"]
    model_type = config["model_type"]
    model_params = config["model_params"]

    # ===============================
    # 3. Cargar y preparar los datos
    # ===============================
    df = load_data(data_path)
    X, y = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # ===============================
    # 4. Seleccionar modelo
    # ===============================
    if model_type == "logistic_regression":
        model = LogisticRegression(**model_params)
    elif model_type == "random_forest":
        model = RandomForestClassifier(**model_params)
    else:
        raise ValueError(f"Modelo no soportado: {model_type}")

    # ===============================
    # 5. Entrenamiento del modelo
    # ===============================
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)

    # ===============================
    # 6. Configurar MLflow y registrar
    # ===============================
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("default")

    with mlflow.start_run():
        mlflow.log_param("model_type", model_type)
        mlflow.log_params(model_params)
        mlflow.log_metric("accuracy", score)
        mlflow.sklearn.log_model(model, "model")

    print(f"✅ Entrenamiento completado con accuracy: {score:.4f}")


if __name__ == "__main__":
    main()
