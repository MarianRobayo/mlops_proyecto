import os
import argparse
import yaml
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from src.utils import load_data, preprocess, split_and_scale

def main(config_path: str):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    data_path = cfg["data"]["path"]
    test_size = cfg["data"].get("test_size", 0.2)
    random_state = cfg["data"].get("random_state", 42)

    mlflow_dir = cfg["mlflow"]["tracking_dir"]
    tracking_uri = f"file://{os.path.abspath(mlflow_dir)}"
    mlflow.set_tracking_uri(tracking_uri)
    os.makedirs(mlflow_dir, exist_ok=True)

    df = load_data(data_path)
    X, y = preprocess(df)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(
        X, y, test_size=test_size, random_state=random_state
    )

    model_cfg = cfg.get("model", {})
    n_estimators = model_cfg.get("n_estimators", 100)
    max_depth = model_cfg.get("max_depth", None)

    with mlflow.start_run() as run:
        model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=random_state
        )
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("accuracy", float(acc))
        mlflow.log_metric("f1", float(f1))

        example_input = X_test[:2].tolist() if hasattr(X_test, 'tolist') else X_test[:2]
        signature = infer_signature(X_train, model.predict(X_train))

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=example_input
        )

        os.makedirs("artifacts", exist_ok=True)
        scaler_path = os.path.join("artifacts", "scaler.joblib")
        joblib.dump(scaler, scaler_path)
        mlflow.log_artifact(scaler_path, artifact_path="preprocessing")

        run_id = run.info.run_id
        print(f"MLflow run_id: {run_id}")
        print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config YAML")
    args = parser.parse_args()
    main(args.config)
