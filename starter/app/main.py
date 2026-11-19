"""
Sentiment Prediction API.

Endpoints:
    GET  /health          — service health status
    POST /predict         — single headline sentiment
    POST /predict/batch   — batch headline sentiment
    GET  /metrics         — Prometheus metrics
"""

import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi import Response
from pydantic import BaseModel
from utils import load_classifier
from time import perf_counter

# TODO (Task 6)
# Configure the Python logger using basicConfig.
import logging
import json 
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from datetime import datetime, timezone

logging.basicConfig(level=logging.DEBUG, format="%(message)s")

# TODO (Task 6)
# Implement a structured logging helper that emits each log entry as a JSON
# object containing at least a timestamp, level, and message field.
def log(level: str, message: str, **kwargs) -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "level": logging.getLevelName(numeric_level),
        "message": message,
        **kwargs,
    }
    logging.log(numeric_level, json.dumps(entry, default=str))


# TODO (Task 6)
# Define the following Prometheus metrics:
# 1. A Counter for the total number of prediction requests, labelled by sentiment.
# 2. A Histogram for prediction latency in milliseconds.
# 3. A Counter for the total number of prediction errors.
# Then instrument run_predictions() to update each metric accordingly.
# Documentation: https://prometheus.io/docs/concepts/metric_types/
PREDICTION_REQUESTS = Counter("pred_reqs", "Total Number of Prediction Requests", labelnames=["sentiment"])
PREDICTION_LATENCY = Histogram(
    "prediction_latency_ms",
    "Prediction latency in milliseconds",
    buckets=[5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000],
)
PREDICTION_ERRORS = Counter("pred_errors", "Total Number of Prediction Errors")

load_dotenv()

classifiers = {}


class PredictRequest(BaseModel):
    text: str


class PredictBatchRequest(BaseModel):
    texts: list[str]


class PredictionResult(BaseModel):
    text: str
    sentiment: str
    confidence: float
    latency_ms: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    log("INFO", "Loading model...")
    classifiers["sentiment"] = load_classifier()
    log("INFO", "Model loaded successfully")
    yield
    classifiers.clear()
    log("INFO", "Model unloaded")


app = FastAPI(title="Sentiment Analysis API", lifespan=lifespan)


def run_predictions(texts: list[str]) -> list[PredictionResult]:
    try:
        # TODO (Task 3): Log a warning using log() if any input text exceeds 2000
        # characters — the model will silently truncate it, so this makes it visible.
        
        # TODO (Task 3): Record the start time, run batch inference using
        # classifiers["sentiment"], and compute latency_ms from start to finish.
        
        # TODO (Task 3): Build and return a list of PredictionResult objects from
        # the inference results (each result has "label" and "score" keys).

        # TODO (Task 6): Observe latency_ms on PREDICTION_LATENCY and increment
        # PREDICTION_REQUESTS (labelled by sentiment) for each prediction.

        # TODO (Task 6): Log each prediction using log() with sentiment, confidence,
        # and latency_ms fields.
        
        if any(len(t) > 2000 for t in texts):
            log("WARNING", "characters - the model will silently truncate it, so this makes it visible.")

        start_time = perf_counter()
        results = classifiers["sentiment"](texts)
        end_time = perf_counter()
        latency_ms = (end_time - start_time) * 1000

        pred_results = [PredictionResult(text=t, sentiment=r["label"], latency_ms=latency_ms, confidence=r["score"]) for t, r in zip(texts, results)]
        PREDICTION_LATENCY.observe(latency_ms)

        for pr in pred_results:
            PREDICTION_REQUESTS.labels(sentiment=pr.sentiment).inc()
            log("INFO", "prediction", sentiment=pr.sentiment, confidence=pr.confidence, latency_ms=pr.latency_ms)
        
        return pred_results

    except Exception as e:
        # TODO (Task 6): Increment PREDICTION_ERRORS, log the error using log(),
        # then re-raise.
        PREDICTION_ERRORS.inc()
        log("ERROR", "prediction failed", error=str(e))
        raise e


# TODO (Task 6): Implement the /metrics endpoint.
# Return the Prometheus metrics in the correct format using generate_latest()
# and CONTENT_TYPE_LATEST.
# Documentation: https://prometheus.io/docs/instrumenting/exposition_formats/
@app.get("/metrics")
def metrics():
    Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# TODO: Implement the /health endpoint.
# Return {"status": "ok"} when the model
# is loaded, and raise an HTTP 503 error when it is not.
@app.get("/health")
def health():
    if "sentiment" not in classifiers:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return {"status": "ok"}

# TODO: Implement the POST /predict endpoint.
# Accept a PredictRequest and return a PredictionResult.
# Return HTTP 503 if the model is not loaded
# Return HTTP 422 if the text is empty
@app.post("/predict", response_model=PredictionResult)
def predict(request: PredictRequest):
    if "sentiment" not in classifiers:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    if not request.text.strip():
        raise HTTPException(status_code=422, detail="Text must not be empty")
    return run_predictions([request.text])[0]


# TODO: Implement the POST /predict/batch endpoint.
# Accept a PredictBatchRequest and return a list of PredictionResult.
# Return HTTP 503 if the model is not loaded
# and HTTP 422 if the texts list is empty.
@app.post("/predict/batch", response_model=list[PredictionResult])
def predict_batch(request: PredictBatchRequest):
    if "sentiment" not in classifiers:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    if not request.texts:
            raise HTTPException(status_code=422, detail="Text must not be empty")    
    return run_predictions(request.texts)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8000)),
    )
