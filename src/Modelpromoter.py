"""
Reads run_info.json written by train.py, checks quality gates,
then promotes the model version to Production using MLClient.

Config strategy (same pattern as Pipeline.py):
  - Inside an Azure ML pipeline job: MLClient.from_config() reads the workspace
    context that Azure ML automatically injects into every job environment via
    the AZUREML_ARM_* environment variables. No config file needed.
  - Running locally: falls back to .azureml/config.json as before.
"""

import argparse
import json
import os

from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.identity import DefaultAzureCredential


# Quality gate thresholds

MIN_ROC_AUC = 0.80
MIN_RECALL  = 0.65


def get_ml_client() -> MLClient:
    """
    Returns an authenticated MLClient that works both inside an Azure ML
    pipeline job and when run locally.
    """
    # ── Inside an Azure ML pipeline job ─────────────────────────────────────
    # AZUREML_ARM_WORKSPACE_NAME is always set by the Azure ML job runtime.
    # We use AzureMLOnBehalfOfCredential which reads the OBO token Azure ML
    # injects — this is the ONLY credential that reliably works inside a job
    # when the compute cluster has no user-assigned managed identity.
    if os.environ.get("AZUREML_ARM_WORKSPACE_NAME"):
        print("Credential: AzureMLOnBehalfOfCredential (pipeline job mode)")
        try:
            from azure.ai.ml.identity import AzureMLOnBehalfOfCredential
            credential = AzureMLOnBehalfOfCredential()
        except ImportError:
            # Older SDK versions — fall back to ManagedIdentityCredential
            # which works if the cluster has a system-assigned managed identity
            print("  AzureMLOnBehalfOfCredential not available, trying ManagedIdentityCredential")
            from azure.identity import ManagedIdentityCredential
            credential = ManagedIdentityCredential()
 
            
        return MLClient(
            credential=credential,
            subscription_id=os.environ["AZUREML_ARM_SUBSCRIPTION"],
            resource_group_name=os.environ["AZUREML_ARM_RESOURCEGROUP"],
            workspace_name=os.environ["AZUREML_ARM_WORKSPACE_NAME"],
            )


def promote_model(model_output_dir: str) -> None:
    """
    1. Reads run_info.json written by train.py
    2. Runs quality gate (ROC-AUC and Recall thresholds)
    3. Tags the new model version as 'production' in the Azure ML registry
    4. Tags all previously-production versions as 'archived'
    """


    # Load metrics from train.py output

    run_info_path = os.path.join(model_output_dir, "run_info.json")
    if not os.path.exists(run_info_path):
        raise FileNotFoundError(
            f"run_info.json not found in '{model_output_dir}'.\n"
            "Make sure train.py completed successfully before this step runs."
        )

    with open(run_info_path) as f:
        run_info = json.load(f)

    model_name = run_info["model_name"]
    roc_auc    = run_info["roc_auc"]
    recall     = run_info["recall"]
    run_id     = run_info["run_id"]

    print("\nModel promotion check")
    print(f"  Model name : {model_name}")
    print(f"  Run ID     : {run_id}")
    print(f"  ROC-AUC    : {roc_auc:.4f}  (minimum required: {MIN_ROC_AUC})")
    print(f"  Recall     : {recall:.4f}  (minimum required: {MIN_RECALL})")

 
    # Quality gate — fail fast, never touch the registry if thresholds missed
   
    failures = []
    if roc_auc < MIN_ROC_AUC:
        failures.append(f"ROC-AUC {roc_auc:.4f} < minimum {MIN_ROC_AUC}")
    if recall < MIN_RECALL:
        failures.append(f"Recall  {recall:.4f} < minimum {MIN_RECALL}")

    if failures:
        raise ValueError(
            "\nModel REJECTED — quality gate failed:\n"
            + "\n".join(f"  {f}" for f in failures)
            + "\n\nThe current Production model is unchanged."
        )

    print("\n  Quality gate passed — proceeding to promote.\n")

 
    ml_client = get_ml_client()

    versions = list(ml_client.models.list(name=model_name))
    if not versions:
        raise ValueError(
            f"No model named '{model_name}' found in the registry.\n"
            "Check that train.py ran with the correct registered_model_name."
        )

    # list() returns versions newest-first
    latest         = versions[0]
    version_number = latest.version
    print(f"Latest registered version: {version_number}")

    # ------------------------------------------------------------------
    # Tag the new version as production
    # ------------------------------------------------------------------
    new_model = ml_client.models.get(name=model_name, version=version_number)

    new_model.tags = {
        "stage":       "production",
        "roc_auc":     str(round(roc_auc, 4)),
        "recall":      str(round(recall, 4)),
        "run_id":      run_id,
        "promoted_by": "Modelpromoter.py",
    }
    new_model.description = (
        f"Promoted by Modelpromoter.py | "
        f"ROC-AUC={roc_auc:.4f} | Recall={recall:.4f} | "
        f"run_id={run_id}"
    )

    ml_client.models.create_or_update(new_model)
    print(f"Version {version_number} tagged as → production")


    # Archive any previously-production versions

    for v in versions[1:]:
        v_tags = v.tags or {}
        if v_tags.get("stage") == "production":
            old_model = ml_client.models.get(name=model_name, version=v.version)
            old_model.tags = {**v_tags, "stage": "archived"}
            ml_client.models.create_or_update(old_model)
            print(f"Version {v.version} tagged as → archived (previous production)")

    print(f"\nPromotion complete. '{model_name}' v{version_number} is now Production.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Quality gate + promote model to Production in Azure ML registry"
    )
    parser.add_argument(
        "--model_output",
        required=True,
        help="Folder containing run_info.json written by train.py",
    )
    args = parser.parse_args()
    promote_model(args.model_output)