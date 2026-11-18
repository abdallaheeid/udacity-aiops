"""
Promote the latest version of the registered model to Production
if the weighted F1 score from the most recent evaluation run exceeds
the required threshold.

Run:
    python scripts/promote.py
"""
# TODO: Write a script that :
# 1- reads the F1 threshold from params.yaml,
# 2- fetches the weighted F1 score from the most recent MLflow evaluation run,
# 3- promotes the latest model version to a "production" alias if the score
# meets the threshold.
# 4- Print a clear message when the model does not qualify.
# Documentation: https://mlflow.org/docs/latest/model-registry.html

import yaml
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
import os 

load_dotenv()


def load_params() -> dict:
    with open("params.yaml") as f:
        return yaml.safe_load(f)["promote"]


def get_latest_f1(client: MlflowClient, experiment_name: str) -> tuple[str, float]:
    """
    Return the run_id and f1_weighted of the most recent evaluation run.
    """
    # TODO: Look up the experiment by name, search for the most recent run,
    # and return its run_id and f1_weighted metric value.
    # Documentation: https://mlflow.org/docs/latest/api_reference/python_api/mlflow.client.html
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found")
    
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )

    run = runs[0]
    f1 = run.data.metrics.get("f1_weighted")
    if f1 is None:
        raise ValueError(f"Run {run.info.run_id} has no 'f1_weighted' metric")
    return run.info.run_id, f1


def promote(client: MlflowClient, model_name: str):
    # TODO: Fetch all registered versions of the model, identify the latest one,
    # and assign it the "production" alias using the MLflow client.
    # Documentation: https://mlflow.org/docs/latest/ml/model-registry/

    # get the latest version of the model
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        raise ValueError(f"No versions registered for '{model_name}'")

    latest_version = max(int(mv.version) for mv in versions)
    client.set_registered_model_alias(
        name=model_name, alias="production", version=str(latest_version)
    )
    print(f"Promoted {model_name} v{latest_version} to @production")


def main():
    # TODO: Write the main function:
    # 1- load the threshold from params.yaml,
    # 2- call get_latest_f1(),
    # 3- call promote() only if the threshold is met.
    
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME") or "finserv-sentiment"
    model_name = os.getenv("MODEL_NAME") or "finserv-sentiment-model"

    client = MlflowClient()  # uses MLFLOW_TRACKING_URI
    threshold = load_params()["f1_threshold"]
    run_id, f1 = get_latest_f1(client, experiment_name)

    print(f"Latest run {run_id}: f1_weighted={f1:.4f} (threshold={threshold})")
    if f1 >= threshold:
        promote(client, model_name)
    else:
        print(f"Model does not qualify: f1_weighted {f1:.4f} < {threshold}. Not promoted.")


if __name__ == "__main__":
    main()
