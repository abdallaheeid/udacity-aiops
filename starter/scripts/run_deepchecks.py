"""
Model quality check (drift) using deepchecks

Run:
    python scripts/run_deepchecks.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import yaml
from app.utils import load_classifier
from deepchecks.nlp import TextData
from deepchecks.nlp.checks import PredictionDrift, PropertyDrift
from dotenv import load_dotenv

load_dotenv()

MODEL_CLASSES = ["negative", "neutral", "positive"]


def load_params() -> dict:
    with open("params.yaml") as f:
        return yaml.safe_load(f)["deepchecks"]


def run_predictions(classifier, texts: list[str]) -> list[str]:
    results = classifier(texts, batch_size=32)
    return [r["label"] for r in results]


def main():
    params = load_params()
    property_drift_threshold = params["property_drift_threshold"]
    prediction_drift_threshold = params["prediction_drift_threshold"]

    print("Loading production model...")

    classifier = load_classifier()

    stream_df = pd.read_csv("data/stream.csv")
    test_df = pd.read_csv("data/test.csv")

    stream_texts = stream_df["text"].tolist()
    test_texts = test_df["text"].tolist()

    # TODO: Create two TextData objects from the text lists above: one for stream_texts
    # and one for test_texts. Use task_type="text_classification".
    # Then call calculate_builtin_properties() on each to compute NLP properties.
    # https://docs.deepchecks.com/stable/nlp/usage_guides/text_data_object.html#nlp-textdata-object

    # TODO: Compute NLP Property Drift between test_dataset (reference) and stream_dataset (current).
    # Extract the "Sentiment" drift score from the result and compare it against
    # property_drift_threshold.
    # If it exceeds the threshold, print an error message and terminate with failure exit code
    # https://docs.deepchecks.com/stable/nlp/auto_checks/train_test_validation/plot_property_drift.html#nlp-property-drift

    # TODO: Run predictions on both datasets with run_predictions(), then compute Prediction Drift.
    # Add a condition to fail if drift score is above the prediction_drift_threshold
    # If drift exceeds threshold, print an error message and terminate with failure exit code
    # https://docs.deepchecks.com/stable/general/guides/drift_guide.html#text-nlp-checks

    # --- Datasets + NLP properties ---
    test_dataset = TextData(raw_text=test_texts, task_type="text_classification")
    stream_dataset = TextData(raw_text=stream_texts, task_type="text_classification")

    test_dataset.calculate_builtin_properties()
    stream_dataset.calculate_builtin_properties()

    # --- Property drift (Sentiment) ---
    property_result = PropertyDrift().run(
        train_dataset=test_dataset, test_dataset=stream_dataset
    )
    sentiment_drift = property_result.value["Sentiment"]["Drift score"]
    print(f"Sentiment property drift: {sentiment_drift:.4f} (threshold={property_drift_threshold})")

    if sentiment_drift > property_drift_threshold:
        print(f"ERROR: Sentiment property drift {sentiment_drift:.4f} exceeds {property_drift_threshold}")
        sys.exit(1)

    # --- Prediction drift ---
    test_preds = run_predictions(classifier, test_texts)
    stream_preds = run_predictions(classifier, stream_texts)

    prediction_check = PredictionDrift().add_condition_drift_score_less_than(
        max_allowed_drift_score=prediction_drift_threshold
    )
    prediction_result = prediction_check.run(
        train_dataset=test_dataset,
        test_dataset=stream_dataset,
        train_predictions=test_preds,
        test_predictions=stream_preds,
        model_classes=MODEL_CLASSES,
    )
    prediction_drift = prediction_result.value["Drift score"]
    print(f"Prediction drift: {prediction_drift:.4f} (threshold={prediction_drift_threshold})")

    if not prediction_result.passed_conditions():
        print(f"ERROR: Prediction drift {prediction_drift:.4f} exceeds {prediction_drift_threshold}")
        sys.exit(1)

    print("All drift checks passed.")

if __name__ == "__main__":
    main()
