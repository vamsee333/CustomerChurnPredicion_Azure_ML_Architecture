"""
Reads run_info.json written by train.py, checks quality gates,
then promotes the model version to Production using MLClient

"""

import argparse
import json
import os
from xml.parsers.expat import model
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.identity import DefaultAzureCredential

#  Quality check thresholds
MIN_ROC_AUC = 0.80
MIN_RECALL  = 0.65


def get_ml_client():
    
    config_path =  "config.json"
    # config_path = os.path.join(os.path.dirname(__file__), "../azure/config.json")
    with open(config_path) as f:
        config = json.load(f)

    return MLClient(
        DefaultAzureCredential(),
        config["subscription_id"],
        config["resource_group"],
        config["workspace_name"],
    )
    


def promote_model(model_output_dir: str):

    # Load metrics written by train.py 
    run_info_path = os.path.join(model_output_dir, "run_info.json")
    if not os.path.exists(run_info_path):
        raise FileNotFoundError(
            f"run_info.json not found in {model_output_dir}. "
            "Make sure train.py ran successfully first."
        )

    with open(run_info_path) as f:
        run_info = json.load(f)

    model_name = run_info["model_name"]  
    roc_auc    = run_info["roc_auc"]
    recall     = run_info["recall"]
    run_id     = run_info["run_id"]

    print(f"\nModel promotion check")
    print(f"  Model name : {model_name}")
    print(f"  Run ID     : {run_id}")
    print(f"  ROC-AUC    : {roc_auc:.4f}  (minimum: {MIN_ROC_AUC})")
    print(f"  Recall     : {recall:.4f}  (minimum: {MIN_RECALL})")

    # Quality check
    failures = []
    if roc_auc < MIN_ROC_AUC:
        failures.append(f"ROC-AUC {roc_auc:.4f} < minimum {MIN_ROC_AUC}")
    if recall < MIN_RECALL:
        failures.append(f"Recall {recall:.4f} < minimum {MIN_RECALL}")

    if failures:
        raise ValueError(
            "\nModel REJECTED because of unmet quality criteria\n"
            + "\n".join(f"  - {f}" for f in failures)
            + "\nCurrent Production model is unchanged."
        )

    print("\nQuality check passed.")

    
    ml_client = get_ml_client()

    # ml_client.models.list() returns all versions for this model name.
    # The latest one is always what train.py just registered.
    versions = list(ml_client.models.list(name=model_name))
    if not versions:
        raise ValueError(
            f"No model named '{model_name}' found in the registry. "
            "Check that train.py ran with registered_model_name set correctly."
        )

    # versions come back newest-first from the API
    latest         = versions[0]
    version_number = latest.version
    print(f"Found version  : {version_number}")


    # MLClient does not have built-in Staging/Production stages the way
    # MlflowClient does. The MLClient equivalent is tags
    # in the Azure ML Studio Models tab under the Tags column.
    updated_model = Model(
        name=model_name,
        version=version_number,
        description=(
            f"Promoted by model_promoter.py"
            f"ROC-AUC={roc_auc:.4f} | Recall={recall:.4f}"
            f"run_id={run_id}"
        ),
        tags={
            "stage":       "production",
            "roc_auc":     str(round(roc_auc, 4)),
            "recall":      str(round(recall, 4)),
            "run_id":      run_id,
            "promoted_by": "model_promoter.py",
        },
    )
    # Get existing model
    model = ml_client.models.get(
    name=model_name,
    version=version_number
    )

# Update metadata
    model.tags = {
    "stage": "production",
    "roc_auc": str(round(roc_auc, 4)),
    "recall": str(round(recall, 4)),
    "run_id": run_id,
    "promoted_by": "model_promoter.py",
}

    model.description = (
    f"Promoted by model_promoter.py | "
    f"ROC-AUC={roc_auc:.4f} | Recall={recall:.4f} | "
    f"run_id={run_id}"
)

    ml_client.models.create_or_update(model)


    for v in versions[1:]:
        tags = v.tags or {}

        if tags.get("stage") == "production":
            old_model = ml_client.models.get(
            name=model_name,
            version=v.version
            )

            old_model.tags = {**tags, "stage": "archived"}

            ml_client.models.create_or_update(old_model)

            print(f"Version {v.version} is tagged as archived (was previous production)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_output", required=True,
                        help="Folder containing run_info.json written by train.py")
    args = parser.parse_args()
    promote_model(args.model_output)