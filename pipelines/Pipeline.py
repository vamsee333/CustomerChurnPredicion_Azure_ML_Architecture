from azure.ai.ml import MLClient, Input, Output
from azure.ai.ml.entities import Environment
from azure.ai.ml.dsl import pipeline
from azure.ai.ml import command
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential
import json
import os
import sys


# Config loading — works both locally and in CI (GitHub Actions)
#
# LOCAL:  reads .azureml/config.json (gitignored, never committed)
# CI:     reads environment variables injected by the workflow via the
#         "env:" block in cd.yml — no config file needed on the runner

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
        print(f"ERROR: config file not found at {config_path}")
        print("  Run locally?  Create .azureml/config.json with subscription_id,")
        print("  resource_group, and workspace_name.")
        sys.exit(1)
    with open(config_path, "r") as f:
        config = json.load(f)
    print(f"Config source: {config_path} (local mode)")


# catch empty values immediately — fail fast with a clear message

REQUIRED_KEYS = ["subscription_id", "resource_group", "workspace_name"]
missing = [k for k in REQUIRED_KEYS if not config.get(k, "").strip()]
if missing:
    print(f"ERROR: The following config values are empty or missing: {missing}")
    print("  In CI: check that your GitHub Secrets are set and named correctly.")
    print("  Required secret names: AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP, AZURE_WORKSPACE")
    sys.exit(1)

SUBSCRIPTION_ID = config["subscription_id"]
RESOURCE_GROUP  = config["resource_group"]
WORKSPACE_NAME  = config["workspace_name"]

# Print workspace identity so it's visible in the Actions log — makes

print(f"\nConnecting to Azure ML workspace:")
print(f"  Subscription : {SUBSCRIPTION_ID[:8]}...{SUBSCRIPTION_ID[-4:]}")
print(f"  Resource group: {RESOURCE_GROUP}")
print(f"  Workspace    : {WORKSPACE_NAME}\n")


COMPUTE_NAME   = "ChurnComputeCluster"
CHURN_DATA_URI = "azureml:customer_churn_data:1"

ml_client = MLClient(
    DefaultAzureCredential(),
    SUBSCRIPTION_ID,
    RESOURCE_GROUP,
    WORKSPACE_NAME,
)

# Environment
# Resolved relative to this file's location so it works from any working dir

conda_path = os.path.join(_HERE, "../env/conda.yml")
if not os.path.exists(conda_path):
    print(f"ERROR: conda.yml not found at {conda_path}")
    print("  Expected repo layout:  env/conda.yml  (sibling of pipelines/)")
    sys.exit(1)

ENV = Environment(
    name="churn-pipeline-env-fixed",
    description="Churn pipeline environment — sklearn + mlflow + Azure ML SDK",
    conda_file=conda_path,
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
)
print("Registering / updating environment...")
ml_client.environments.create_or_update(ENV)
print("Environment ready.\n")

# Source code path — all steps share the same src/ folder

SRC = os.path.join(_HERE, "../src")
if not os.path.isdir(SRC):
    print(f"ERROR: src/ folder not found at {SRC}")
    sys.exit(1)



preprocess_step = command(
    name="preprocess",
    display_name="Preprocess churn data",
    code=SRC,
    command=(
        "python preprocessing.py "
        "--input_data  ${{inputs.raw_data}} "
        "--output_data ${{outputs.processed_data}}"
    ),
    inputs={
        "raw_data": Input(type=AssetTypes.URI_FILE),
    },
    outputs={
        "processed_data": Output(type=AssetTypes.URI_FOLDER),
    },
    environment=ENV,
    compute=COMPUTE_NAME,
)

train_step = command(
    name="train",
    display_name="Train churn models",
    code=SRC,
    command=(
        "python train.py "
        "--processed_data ${{inputs.processed_data}} "
        "--model_output   ${{outputs.model_output}} "
        "--model_type     high_recall_lr"
    ),
    inputs={
        "processed_data": Input(type=AssetTypes.URI_FOLDER),
    },
    outputs={
        "model_output": Output(type=AssetTypes.URI_FOLDER),
    },
    environment=ENV,
    compute=COMPUTE_NAME,
)

# Quality gate — if ROC-AUC or Recall miss thresholds, this step errors
# and the entire pipeline stops. The production model is never touched.
promote_step = command(
    name="promote",
    display_name="Quality gate + promote to Registry",
    code=SRC,
    command=(
        "python Modelpromoter.py "
        "--model_output ${{inputs.model_output}}"
    ),
    inputs={
        "model_output": Input(type=AssetTypes.URI_FOLDER),
    },
    environment=ENV,
    compute=COMPUTE_NAME,
)

predict_step = command(
    name="predict",
    display_name="Batch churn predictions",
    code=SRC,
    command=(
        "python predict.py "
        "--processed_data     ${{inputs.processed_data}} "
        "--model_input        ${{inputs.model_input}} "
        "--predictions_output ${{outputs.predictions_output}}"
    ),
    inputs={
        "processed_data": Input(type=AssetTypes.URI_FOLDER),
        "model_input":    Input(type=AssetTypes.URI_FOLDER),
    },
    outputs={
        "predictions_output": Output(type=AssetTypes.URI_FOLDER),
    },
    environment=ENV,
    compute=COMPUTE_NAME,
)


# Pipeline graph


@pipeline(
    name="churn_prediction_pipeline",
    description="Preprocess - Train  - Quality gate - Predict",
    default_compute=COMPUTE_NAME,
)
def churn_pipeline(raw_churn_data):
    preprocess = preprocess_step(raw_data=raw_churn_data)

    train = train_step(
        processed_data=preprocess.outputs.processed_data,
    )

    # promote runs after train; if it fails, predict never runs
    promote = promote_step( 
        model_output=train.outputs.model_output,
    )

    predict = predict_step(
        processed_data=preprocess.outputs.processed_data,
        model_input=train.outputs.model_output,
    )

    return {"predictions": predict.outputs.predictions_output}




pipeline_job = churn_pipeline(
    raw_churn_data=Input(type=AssetTypes.URI_FILE, path=CHURN_DATA_URI)
)
pipeline_job.settings.default_datastore = "workspaceblobstore"

print("Submitting pipeline job...")
submitted = ml_client.jobs.create_or_update(
    pipeline_job,
    experiment_name="churn-prediction",
)

# These two lines are parsed by cd.yml — do not change their format
print(f"Pipeline submitted!")
print(f"Job name  : {submitted.name}")
print(f"Studio URL: {submitted.studio_url}")