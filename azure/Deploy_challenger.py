"""
Creates or updates the managed online endpoint and Challenger deployment.

Runs in two contexts:
  LOCAL  — reads .azureml/config.json, uses DefaultAzureCredential (az login)
  CI/CD  — reads AZURE_SUBSCRIPTION_ID / AZURE_RESOURCE_GROUP / AZURE_WORKSPACE
            env vars injected by cd.yml, uses DefaultAzureCredential
            (the azure/login@v2 step in the workflow handles authentication
            by writing a token that DefaultAzureCredential picks up via
            EnvironmentCredential automatically).

No AzureMLOnBehalfOfCredential needed here — this script never runs
inside an Azure ML pipeline job on a compute node.
"""

import json
import os
import sys

from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    CodeConfiguration,
    DataCollector,
    DeploymentCollection,
    Environment,
    ManagedOnlineDeployment,
    ManagedOnlineEndpoint,
    OnlineRequestSettings,
    ProbeSettings,
)
from azure.core.exceptions import ResourceNotFoundError
from azure.identity import DefaultAzureCredential


# Config — env vars in CI, .azureml/config.json locally
# Same pattern as Pipeline.py so both scripts are consistent

_HERE = os.path.dirname(os.path.abspath(__file__))

if os.environ.get("AZURE_SUBSCRIPTION_ID"):
    config = {
        "subscription_id": os.environ["AZURE_SUBSCRIPTION_ID"],
        "resource_group":  os.environ["AZURE_RESOURCE_GROUP"],
        "workspace_name":  os.environ["AZURE_ML_WORKSPACE"],
    }
    print("Config source: environment variables (CI mode)")
else:
    config_path = os.path.join(_HERE, "../.azureml/config.json")
    if not os.path.exists(config_path):
        print(f"ERROR: config not found at {config_path}")
        print("  Create .azureml/config.json with subscription_id, resource_group, workspace_name")
        sys.exit(1)
    with open(config_path) as f:
        config = json.load(f)
    print(f"Config source: {config_path} (local mode)")

# Fail fast if any value is empty — catches silent secret misses in CI
missing = [k for k, v in config.items() if not str(v).strip()]
if missing:
    print(f"ERROR: These config values are empty: {missing}")
    print("  In CI: check GitHub Secrets are named AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP, AZURE_WORKSPACE")
    sys.exit(1)


ml_client = MLClient(
    DefaultAzureCredential(),
    config["subscription_id"],
    config["resource_group"],
    config["workspace_name"],
)
print(f"\nConnected to workspace : {config['workspace_name']}")
print(f"Resource group         : {config['resource_group']}")
print(f"Subscription           : {config['subscription_id'][:8]}...\n")


ENDPOINT_NAME   = "churn-predictions-endpoint"
CHAMPION_DEPLOYMENT  = "champion"                      # already running
CHALLENGER_DEPLOYMENT = "challenger"   
MODEL_NAME      = "customer-churn-model"


# Find the production-tagged model version

print(f"Looking for challenger-tagged version of '{MODEL_NAME}'...")
all_versions = list(ml_client.models.list(name=MODEL_NAME))

Challenger_version = next(
    (v for v in all_versions if v.tags.get("stage") == "challenger"),
    None,
)

if Challenger_version is None:
    print(f"ERROR: No version of '{MODEL_NAME}' has tag stage=challenger.")
    print("  Run the training pipeline first so Modelpromoter.py can tag a version.")
    sys.exit(1)

model_ref = f"azureml:{MODEL_NAME}:{Challenger_version.version}"
print(f"Challenger model       : {model_ref}")
print(f"  ROC-AUC : {Challenger_version.tags.get('roc_auc', 'n/a')}")
print(f"  Recall  : {Challenger_version.tags.get('recall', 'n/a')}\n")


# Confirm champion is still live before we attempt a traffic split
print(f"Confirming champion deployment '{CHAMPION_DEPLOYMENT}' exists...")
try:
    ml_client.online_deployments.get(
        name=CHAMPION_DEPLOYMENT, endpoint_name=ENDPOINT_NAME)
    print("Champion confirmed.\n")
except ResourceNotFoundError:
    print(f"ERROR: '{CHAMPION_DEPLOYMENT}' not found on endpoint.")
    print("  Run Deploy_endpoint.py first.")
    sys.exit(1)

# Challenger deployment
# Resolve the environment version dynamically so this script doesn't need
# updating every time a new environment version is registered.
# It picks the latest version of churn-pipeline-env-fixed.

env_versions = list(ml_client.environments.list(name="churn-pipeline-env-fixed"))
if not env_versions:
    print("ERROR: Environment 'churn-pipeline-env-fixed' not found in registry.")
    print("  Run Pipeline.py at least once to register it.")
    sys.exit(1)

latest_env_version = env_versions[0].version          # list() returns newest-first
env_ref = f"azureml:churn-pipeline-env-fixed:{latest_env_version}"
print(f"Environment            : {env_ref}")

# src/ path is relative to the azure/ folder where this script lives
src_path = os.path.join(_HERE, "../src")
if not os.path.isdir(src_path):
    print(f"ERROR: src/ folder not found at {src_path}")
    sys.exit(1)

deployment = ManagedOnlineDeployment(
    name=CHALLENGER_DEPLOYMENT,
    endpoint_name=ENDPOINT_NAME,
    model=model_ref,
    code_configuration=CodeConfiguration(
        code=src_path,
        scoring_script="Score.py",
    ),
    environment=env_ref,
    instance_type="Standard_F2s_v2",
    instance_count=1,

    # Data collection feeds the drift monitor in setup_drift_monitor.py
    data_collector=DataCollector(
        collections={
            "model_inputs":  DeploymentCollection(enabled=True),
            "model_outputs": DeploymentCollection(enabled=True),
        }
    ),

    # Give init() enough time to load model.pkl and feature_columns.json
    liveness_probe=ProbeSettings(
        initial_delay=10, period=10, timeout=5, failure_threshold=3
    ),
    readiness_probe=ProbeSettings(
        initial_delay=30, period=10, timeout=5, failure_threshold=3
    ),

    request_settings=OnlineRequestSettings(
        request_timeout_ms=5000,
        max_concurrent_requests_per_instance=10,
    ),

    # Uncomment when ready to enable auto-scaling (Phase 4)
    # scale_settings=TargetUtilizationScaleSettings(
    #     min_instances=1,
    #     max_instances=4,
    #     target_utilization_percentage=70,
    # ),

    tags={"model_type": "high_recall_lr", "project": "customer-churn"},
)

print(f"\nDeploying '{CHALLENGER_DEPLOYMENT}'...")
ml_client.online_deployments.begin_create_or_update(deployment).result()
print("Deployment ready.")


# Route 100% of traffic to champion
# Later: add a challenger deployment and split traffic here (Phase 4)

endpoint = ml_client.online_endpoints.get(ENDPOINT_NAME)
endpoint.traffic = {CHAMPION_DEPLOYMENT: 90, CHALLENGER_DEPLOYMENT: 10}
ml_client.online_endpoints.begin_create_or_update(endpoint).result()
print(f"Traffic: {CHAMPION_DEPLOYMENT} → 90%, {CHALLENGER_DEPLOYMENT} → 10%")
print(f"\nEndpoint live at: https://{ENDPOINT_NAME}.eastus2.inference.ml.azure.com/score")