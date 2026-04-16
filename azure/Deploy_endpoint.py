"""
This script creates a managed online endpoint and deployment in Azure ML for real-time scoring.
"""
 
import json
import os
import time
 
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    CodeConfiguration,
    Environment,
    ProbeSettings,
    OnlineRequestSettings,
    TargetUtilizationScaleSettings,
)
from azure.identity import DefaultAzureCredential
 
 
config_path = os.path.join(os.path.dirname(__file__), "../.azureml/config.json")
# config_path = "config.json"
with open(config_path) as f:
    config = json.load(f)
 
ml_client = MLClient(
    DefaultAzureCredential(),
    config["subscription_id"],
    config["resource_group"],
    config["workspace_name"],
)
print(f"Connected to Azure ML Workspace: {config['workspace_name']}")



print("Subscription:", ml_client.subscription_id)
 
ENDPOINT_NAME   = "churn-prediction-endpoint"
DEPLOYMENT_NAME = "champion"
MODEL_NAME      = "customer-churn-model"   
          
# picking the model version with tag as production

versions = ml_client.models.list(name=MODEL_NAME)
prod_version = next(v for v in versions if v.tags.get("stage") == "production")
model_ref = f"azureml:{MODEL_NAME}:{prod_version.version}"
 
 
# Create or update the endpoint
# It has no compute — that lives in the deployment below.
endpoint = ManagedOnlineEndpoint(
    name=ENDPOINT_NAME,
    description="Real-time churn prediction endpoint for customer churn prediction project",
    auth_mode="key",
    tags={"project": "customer-churn"},
)
 
print(f"Creating / updating endpoint: {ENDPOINT_NAME}")
end_point = ml_client.online_endpoints.begin_create_or_update(endpoint)
end_point.result()   # blocks until provisioning completes
print("Endpoint ready.")
 
 
# Create or update the champion deployment
# This is where compute, the model artifact, and score.py are wired together.
# name is Champion because we will later add a Challenger deployment for A/B testing and Champion is getting 100% of the traffic initially.
deployment = ManagedOnlineDeployment(
    name=DEPLOYMENT_NAME,
    endpoint_name=ENDPOINT_NAME,
    model=model_ref,
    code_configuration=CodeConfiguration(
        code="./src",
        scoring_script="Score.py",
    ),
 
    # Reuse the same env as registered
    environment="azureml:churn-pipeline-env-fixed:4",
 
    instance_type="Standard_DS2_v2",
    instance_count=1,
 
    # Give init() time to load model.pkl and feature_columns.json
    # initial_delay of 10 seconds means do not send any requests to model for 10 seconds 
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
 
    # scale_settings=TargetUtilizationScaleSettings(
    #     min_instances=1,
    #     max_instances=4,
    #     target_utilization_percentage=70,
    # ),
 
    tags={"model_type": "high_recall_lr", "project": "customer-churn"},
)
 
print(f"Deploying {DEPLOYMENT_NAME}")
deployment_obj = ml_client.online_deployments.begin_create_or_update(deployment)
deployment_obj.result()
print("Deployment is ready")
 
 
# Send 100% of traffic to champion deployment,later we will add a challenger and update the dictionary to split traffic between them
endpoint.traffic = {DEPLOYMENT_NAME: 100}
ml_client.online_endpoints.begin_create_or_update(endpoint).result()
print("Champion Traffic : 100%")