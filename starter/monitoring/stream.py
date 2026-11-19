"""
Production monitoring stream for FinBERT sentiment API.

Reads headlines from data/stream.csv, sends them to the /predict endpoint,
and logs aggregated metrics to MLflow every WINDOW_SIZE predictions.

Each observation window logs:
    - Sentiment distribution (% positive, % negative, % neutral)
    - Average confidence score
    - Average latency (ms)

Run from the project root:
    python monitoring/stream.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import mlflow
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

API_URL = f"http://{os.getenv('API_HOST', 'localhost')}:{os.getenv('API_PORT', 8000)}"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "finbert-evaluation")
WINDOW_SIZE = 50  # number of predictions per observation window
SLEEP_MS = 100    # delay between requests to simulate real traffic (ms)
SENTIMENTS = ("positive", "negative", "neutral")

def predict(text: str) -> dict:
    response = requests.post(
        f"{API_URL}/predict",
        json={"text": text},
        timeout=10,
    )
    response.raise_for_status()
    return response.json()

# TODO (Task 6)
# Implement log_window() to aggregate results from one observation window and
# log them as MLflow metrics. For each window, compute and log:
# - Sentiment distribution (% positive, % negative, % neutral)
# - Average confidence score
# - Average latency in milliseconds
# Documentation: https://mlflow.org/docs/latest/tracking.html#logging-metrics
def log_window(window: list[dict], window_idx: int) -> None:
    """Log aggregated metrics for one observation window to MLflow."""
    n = len(window)
    if n == 0:
        return

    metrics = {
        f"pct_{s}": sum(r["sentiment"] == s for r in window) / n * 100
        for s in SENTIMENTS
    }
    metrics["avg_confidence"] = sum(r["confidence"] for r in window) / n
    metrics["avg_latency_ms"] = sum(r["latency_ms"] for r in window) / n
    metrics["window_size"] = n

    mlflow.log_metrics(metrics, step=window_idx)
    print(
        f"[window {window_idx}] n={n} "
        + " ".join(f"{k}={v:.2f}" for k, v in metrics.items())
    )


# TODO (Task 6)
# Implement main() to:
# 1. Load headlines from data/stream.csv
# 2. Start an MLflow run for monitoring
# 3. Loop through headlines, call predict() for each, and collect results in windows
# 4. Call log_window() every WINDOW_SIZE predictions to log metrics to MLflow
def main():
    df = pd.read_csv("data/stream.csv")
    headlines = df["text"].dropna().astype(str).tolist()
    print(f"Loaded {len(headlines)} headlines from data/stream.csv")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name="production-monitoring"):
        mlflow.log_params({"api_url": API_URL, "window_size": WINDOW_SIZE, "sleep_ms": SLEEP_MS})

        window: list[dict] = []
        window_idx = 0
        errors = 0

        for text in headlines:
            try:
                window.append(predict(text))
            except requests.RequestException as e:
                errors += 1
                print(f"Request failed: {e}")

            if len(window) == WINDOW_SIZE:
                log_window(window, window_idx)
                window = []
                window_idx += 1

            time.sleep(SLEEP_MS / 1000)

        if window:  # log the remaining partial window
            log_window(window, window_idx)

        mlflow.log_metric("total_errors", errors)
        print(f"Done. Windows logged: {window_idx + (1 if window else 0)}, errors: {errors}")


if __name__ == "__main__":
    main()
