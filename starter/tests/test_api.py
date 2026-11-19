"""
Integration tests for the Sentiment Analysis API.
"""

import pytest
from fastapi.testclient import TestClient

VALID_LABELS = {"positive", "negative", "neutral"}
EXPECTED_KEYS = {"text", "sentiment", "confidence", "latency_ms"}

def _assert_valid_result(item: dict, text: str) -> None:
    assert set(item.keys()) == EXPECTED_KEYS
    assert item["text"] == text
    assert item["sentiment"] in VALID_LABELS
    assert isinstance(item["confidence"], float)
    assert 0.0 <= item["confidence"] <= 1.0
    assert isinstance(item["latency_ms"], float)
    assert item["latency_ms"] >= 0.0

# TODO: Test that POST /predict returns the correct response schema
# (text, sentiment, confidence, latency_ms) and a valid sentiment label.
# Hint: use @pytest.mark.parametrize to run the same assertions across multiple headlines.
@pytest.mark.parametrize(
    "text",
    [
        "The company reported record profits and raised its dividend.",
        "The firm filed for bankruptcy after massive losses.",
        "Stocks closed flat on Friday amid low trading volume.",
    ],
)
def test_predict_returns_valid_response(client: TestClient, text):
    response = client.post("/predict", json={"text": text})
    assert response.status_code == 200
    _assert_valid_result(response.json(), text)


# TODO: Test that POST /predict/batch returns a list of results
# whose length matches the number of input texts, and each item has the correct schema.
def test_predict_batch(client: TestClient):
    texts = [
        "Quarterly earnings beat analyst expectations by wide margin.",
        "Tech giant announces major layoffs amid revenue decline.",
        "Federal Reserve signals interest rate cuts later this year.",
    ]
    response = client.post("/predict/batch", json={"texts": texts})
    assert response.status_code == 200

    body = response.json()
    assert isinstance(body, list)
    assert len(body) == len(texts)
    for item, text in zip(body, texts):
        _assert_valid_result(item, text)


# TODO: Test that POST /predict/batch returns HTTP 422 for an empty texts list.
def test_predict_batch_empty_list_returns_422(client: TestClient):
    response = client.post("/predict/batch", json={"texts": []})
    assert response.status_code == 422
