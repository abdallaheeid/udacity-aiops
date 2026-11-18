"""
Evaluate sentiment model on the test set and register it
in the MLflow Model Registry.

Run:
    python scripts/evaluate.py
"""

import os

import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import pipeline

# TODO: add mlflow required dependencies
import mlflow
from mlflow import MlflowClient

load_dotenv()

MAX_LENGTH = 512
BATCH_SIZE = 32


def load_test_data(test_path: str) -> tuple[list[str], list[str]]:
    df = pd.read_csv(test_path)
    return df["text"].tolist(), df["label"].tolist()


def build_classifier(model_id: str):
    print(f"Loading {model_id}...")
    return pipeline(
        "text-classification",
        model=model_id,
        tokenizer=model_id,
        truncation=True,
        max_length=MAX_LENGTH,
    )


def run_inference(classifier, texts: list[str]) -> list[str]:
    print(f"Running inference on {len(texts)} samples...")
    results = classifier(texts, BATCH_SIZE=32)
    return [r["label"] for r in results]


def compute_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted"),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted"),
    }


def main():
    model_id = os.getenv("HF_MODEL_ID", "baptle/FinBERT_market_based")
    test_path = os.path.join("data", "test.csv")
    model_name = os.getenv("MODEL_NAME") or "finserv-sentiment-model"
    model_alias = os.getenv("MODEL_STAGE") or "dev"

    texts, y_true = load_test_data(test_path)
    classifier = build_classifier(model_id)
    y_pred = run_inference(classifier, texts)
    metrics = compute_metrics(y_true, y_pred)

    print("\nEvaluation Results:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    # TODO: Use MLflow to track the evaluation run.
    # Set the tracking URI and experiment name from environment variables,
    # start a run, log the model parameters and computed metrics, then log
    # and register the pipeline in the Model Registry.
    # Documentation: https://mlflow.org/docs/latest/tracking.html
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "finserv-sentiment"))

    with mlflow.start_run(run_name=f"evaluate-{model_id.split('/')[-1]}") as run:
        mlflow.log_params({
            "model_id": model_id,
            "max_length": MAX_LENGTH,
            "batch_size": BATCH_SIZE,
            "test_path": test_path,
            "n_test_samples": len(texts),
        })
        
        mlflow.log_metrics(metrics)
        
        model_info = mlflow.transformers.log_model(
            transformers_model=classifier,
            name="model",
            task="text-classification",
            registered_model_name=model_name,
        )

        MlflowClient().set_registered_model_alias(
            name=model_name,
            alias=model_alias,
            version=model_info.registered_model_version,
        )

        print(f"\nRun ID: {run.info.run_id}")
        print(
            f"Registered {model_name} v{model_info.registered_model_version} "
            f"with alias '@{model_alias}'"
        )
        
if __name__ == "__main__":
    main()
