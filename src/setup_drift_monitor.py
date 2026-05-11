"""
Sets up weekly drift monitoring for the churn prediction endpoint.
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import pandas as pd
from azure.ai.ml import Input, MLClient
from azure.ai.ml.constants import AssetTypes, MonitorTargetTasks, MonitorDatasetContext
from azure.ai.ml.entities import (AlertNotification, CronTrigger, Data,DataDriftMetricThreshold, DataDriftSignal, DataQualitySignal,
CategoricalDriftMetrics, NumericalDriftMetrics,MonitorDefinition, MonitorFeatureFilter, MonitorInputData,
MonitoringTarget, MonitorSchedule, ReferenceData)

from azure.ai.ml.entities import (
    DataQualityMetricThreshold,
    DataQualityMetricsNumerical,
    DataQualityMetricsCategorical,
)
from azure.identity import DefaultAzureCredential

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from preprocessing import preprocess_data



ml_client = MLClient.from_config(
    credential=DefaultAzureCredential())


ENDPOINT_NAME    = "churn-predictions-endpoint"
DEPLOYMENT_NAME  = "champion"
MONITOR_NAME     = "churn-drift-monitor"
BASELINE_DATASET = "churn-training-baseline-mltable"
DRIFT_THRESHOLD  = 0.1  



# The baseline is the "normal" — the feature distributions from your
# training data. The monitor compares live traffic against this every week.
# We preprocess train.csv first so the baseline matches exactly what
# the model was trained on .
def register_baseline(train_csv_path: str) -> str:
    print("\nBuilding baseline from train.csv...")
    df    = pd.read_csv(train_csv_path)
    X, _  = preprocess_data(df)
    print(f"Baseline: {X.shape[0]} rows, {X.shape[1]} features")

    tmp_dir  = tempfile.mkdtemp()
    csv_path = os.path.join(tmp_dir, "baseline.csv")
    X.to_csv(csv_path, index=False)

    # Azure ML monitoring requires MLTable format
    mltable_content = f"""paths:
  - file: ./baseline.csv
transformations:
  - read_delimited:
      delimiter: ','
      encoding: utf8
      header: all_files_same_headers
"""
    with open(os.path.join(tmp_dir, "MLTable"), "w") as f:
        f.write(mltable_content)

    result = ml_client.data.create_or_update(Data(
        name=BASELINE_DATASET,
        description="Training data baseline for churn drift monitoring.",
        path=tmp_dir,              
        type=AssetTypes.MLTABLE,
    ))
    shutil.rmtree(tmp_dir)

    uri = f"azureml:{BASELINE_DATASET}:{result.version}"
    print(f"Baseline registered: {uri}")
    return uri


#  Create the monitor
#
# The monitor runs every day at 07:00 pm set up using cron expression, looks at the last 7 days of
# requests sent to your endpoint, and compares those feature distributions
# against the baseline registered above.

# Two checks run on each feature:
# for numeric features, Normailized Wasserstein distance is used and for categorical features, Jensen-Shannon distance is used.
# If any feature score goes above 0.1, you get an email alert.
from azure.ai.ml.entities import ServerlessSparkCompute

compute = ServerlessSparkCompute(
    instance_type="standard_e4s_v3",
    runtime_version="3.4",
)

#compute = ml_client.compute.get("ComputeCluster1")  # Reuse existing compute cluster

#from azure.ai.ml.entities import AmlCompute

#compute = AmlCompute(name="ComputeCluster1")
def create_monitor(baseline_uri: str, alert_email: str) -> None:
    
    reference = ReferenceData(
        input_data=Input(
            type=AssetTypes.MLTABLE,
            path=baseline_uri,
        ),
        data_context=MonitorDatasetContext.TRAINING,
        data_column_names={"target_column": "Churn"},
    )

    metric_thresholds = DataDriftMetricThreshold(
        numerical=NumericalDriftMetrics(
            normalized_wasserstein_distance=DRIFT_THRESHOLD,
        ),
        categorical=CategoricalDriftMetrics(
            jensen_shannon_distance=DRIFT_THRESHOLD,
        ),
    )
 
    drift_signal = DataDriftSignal(
        reference_data=reference,
        features=MonitorFeatureFilter(top_n_feature_importance=20),
        metric_thresholds=metric_thresholds,
        alert_enabled=True,
    )
 
    quality_signal = DataQualitySignal(
    reference_data=reference,
    features=MonitorFeatureFilter(top_n_feature_importance=20),
    metric_thresholds=DataQualityMetricThreshold(
        numerical=DataQualityMetricsNumerical(
            null_value_rate=0.05,
            out_of_bounds_rate=0.05,
        ),
        categorical=DataQualityMetricsCategorical(
            null_value_rate=0.05,
            out_of_bounds_rate=0.05,
        ),
    ),
    alert_enabled=True,
)
    ml_client.schedules.begin_create_or_update(MonitorSchedule(
        name=MONITOR_NAME,
        trigger=CronTrigger(expression="0 19 * * *"), 
        create_monitor=MonitorDefinition(
            compute= compute,
            monitoring_target=MonitoringTarget(
                ml_task=MonitorTargetTasks.CLASSIFICATION,
                endpoint_deployment_id=f"azureml:{ENDPOINT_NAME}:{DEPLOYMENT_NAME}",
            ),
            alert_notification=AlertNotification(emails=[alert_email]),
            monitoring_signals={
                "data_drift":   drift_signal,
                "data_quality": quality_signal,
            },
        ),
    )).result()

    print(f"Monitor created - {MONITOR_NAME}")
    print(f"Alerts sent to email - {alert_email}")



# use --backfill to test it immediately against your existing traffic.

def trigger_backfill():
    print("\nTriggering test run")
    ml_client.schedules.trigger(name=MONITOR_NAME)
    print("Test Running")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data",  required=True,help="Path to train.csv")
    parser.add_argument("--alert_email", required=True, help="Email address for alerts")
    parser.add_argument("--backfill",    action="store_true", help="Trigger an test run")
    args = parser.parse_args()

    baseline_uri = register_baseline(args.train_data)
    create_monitor(baseline_uri, args.alert_email)

    if args.backfill:
        trigger_backfill()

    print("Sucessfully executed the script")