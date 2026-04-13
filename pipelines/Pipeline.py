from azure.ai.ml import MLClient, Input, Output
from azure.ai.ml.entities import Environment
from azure.ai.ml.dsl import pipeline
from azure.ai.ml import command
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential
import json
import os

# Load configuration from config file
config_path = os.path.join(os.path.dirname(__file__), "../azure/config.json")
with open(config_path, "r") as f:
    config = json.load(f)


# Retrieving Azure ML workspace details from config
SUBSCRIPTION_ID  = config["subscription_id"]
RESOURCE_GROUP   = config["resource_group"]
WORKSPACE_NAME   = config["workspace_name"]



COMPUTE_NAME     = "ChurnComputeInstance"
CHURN_DATA_URI   = "azureml:customer_churn_data:1"  

ml_client = MLClient(
    DefaultAzureCredential(),
    SUBSCRIPTION_ID,
    RESOURCE_GROUP,
    WORKSPACE_NAME
)

# Using Curated Azure ML environment with sklearn + mlflow pre-installed



ENV = Environment(
    name="churn-pipeline-env-fixed",
    description="Churn pipeline Environment",
    conda_file="./env/conda.yml",                 # path relative to this script
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
)
ml_client.environments.create_or_update(ENV)

#ENV = "churn-pipeline-env:2"  # Use the version created above

# Step definitions 

preprocess_step = command(
    name="preprocess",
    display_name="Preprocess churn data",
    code="./src",
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
    code="./src",
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

## If quality check fails, this step errors and the pipeline stops
promote_step = command(
    name="promote",
    display_name="Quality gate + promote to Registry",
    code="./src",
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
    code="./src",
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
    description="Preprocess - Train - Quality Check for deployment - Predict",
    default_compute=COMPUTE_NAME,
)
def churn_pipeline(raw_churn_data):
    preprocess = preprocess_step(raw_data=raw_churn_data)

    train = train_step(
        processed_data=preprocess.outputs.processed_data,
    )
    promote = promote_step(
        model_output=train.outputs.model_output,
    )

    predict = predict_step(
        processed_data=preprocess.outputs.processed_data,
        model_input=train.outputs.model_output,
    )

    return {"predictions": predict.outputs.predictions_output}

# Submit

pipeline_job = churn_pipeline(
    raw_churn_data=Input(type=AssetTypes.URI_FILE, path=CHURN_DATA_URI)
)
pipeline_job.settings.default_datastore = "workspaceblobstore"

submitted = ml_client.jobs.create_or_update(
    pipeline_job,
    experiment_name="churn-prediction",
)

print(f"Pipeline submitted!")
print(f"Job name  : {submitted.name}")
print(f"Studio URL: {submitted.studio_url}")