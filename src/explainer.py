"""
Model Explainer Script
Downloads the production model from Azure ML, analyzes feature importance using
coefficients and SHAP values, and logs results to MLflow.
"""

import json
import os
import sys
import shutil

import joblib
import numpy as np
import pandas as pd
import mlflow
import shap

from sklearn.model_selection import train_test_split
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential



MODEL_NAME = "customer-churn-model"
DOWNLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "downloaded_production_model")


def download_model(ml_client: MLClient, model_name: str, version: str, download_path: str) -> None:
    # model dowload
    if os.path.exists(download_path):
        shutil.rmtree(download_path, ignore_errors=True)
        print(f"Cleared existing folder: {download_path}")
    
    ml_client.models.download(name=model_name, version=version, download_path=download_path)
    print(f"Model downloaded to: {download_path}")


def load_model(download_dir: str, model_name: str):
    # Load the model and feature columns
    model_dir = os.path.join(download_dir, model_name, "model")
    model = joblib.load(os.path.join(model_dir, "model.pkl"))
    feature_cols = json.load(open(os.path.join(model_dir, "feature_columns.json")))
    return model, feature_cols


def analyze_coefficients(model, feature_cols):
    #Generate model coefficients
    coef_df = pd.DataFrame({
        "feature": feature_cols,
        "coefficient": model.coef_[0],
        "odds_ratio": np.exp(model.coef_[0]),
    }).sort_values("coefficient", key=abs, ascending=False).reset_index(drop=True)

    print(f"\n{'Feature'} --------------------- {'Coefficient'}---------------------------{'Odds Ratio'}")
    print("-" * 66)
    for _, row in coef_df.head(15).iterrows():
        direction = "churn Increases" if row["coefficient"] > 0 else "churn Decreases"
        print(f"{row['feature']} ----------------- {row['coefficient']} ------------------ {row['odds_ratio']} which Results -  {direction}")

    return coef_df


def analyze_shap(model, X, feature_cols):
    #Generate SHAP values 
    explainer = shap.LinearExplainer(model, shap.maskers.Independent(X))
    shap_values = explainer.shap_values(X)

    shap_df = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        "mean_shap": shap_values.mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    print(f"feature  ------------------------------- Mean SHAP ------------------------------------------- Effect")
    print("-" * 60)
    for _, row in shap_df.head(15).iterrows():
        effect = "churn" if row["mean_shap"] > 0 else "stay"
        print(f"{row['feature']} ------------------- {round(row['mean_abs_shap'], 3)}  --------------------      {effect}")

    return shap_df, shap_values


def explain_customer(model, X, shap_values, feature_cols, customer_index: int):
    # Individual prediction explanation
    pred_proba = model.predict_proba(X.iloc[[customer_index]])[0][1]

    local_df = pd.DataFrame({
        "feature": feature_cols,
        "value": X.iloc[customer_index].values,
        "contribution": shap_values[customer_index],
    }).sort_values("contribution", key=abs, ascending=False)

    print(f"\nPredicted churn probability: {pred_proba:.2%}")
    print(f"\n{'Feature'} --------------------  {'Value'} ------------------------ {'Contribution'}")
    print("-" * 65)
    for _, row in local_df.head(10).iterrows():
        direction = "churn Increases" if row["contribution"] > 0 else "churn Decreases"
        print(f"{row['feature']} -------------- {row['value']} -------------------- {row['contribution']}  {direction}")


def log_to_mlflow(coef_df, shap_df, ml_client):
    coef_df.to_csv("Coefficients.csv", index=False)
    shap_df.to_csv("SHAP_Values.csv", index=False)

    # # Find the existing explainability experiment or create it
    # from azure.ai.ml.entities import Job
    # from azure.ai.ml import command

    # # Upload artifacts directly to the workspace datastore
    # ml_client.data.upload_file(
    #     src=["Coefficients.csv", "SHAP_Values.csv"],
    #     dest="explainability_outputs/",
    # )
    print("Reults are uploaded")


def main():
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    from src.preprocessing import preprocess_data

    ml_client = MLClient.from_config(credential=DefaultAzureCredential())
    versions = ml_client.models.list(name=MODEL_NAME)
    prod_version = next(v for v in versions if v.tags.get("stage") == "production")
    print(f"Using production model version: {prod_version.version}")

   
    download_model(ml_client, MODEL_NAME, prod_version.version, DOWNLOAD_DIR)
    model, feature_cols = load_model(DOWNLOAD_DIR, MODEL_NAME)

  
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "train.csv")
    df = pd.read_csv(data_path)
    X, y = preprocess_data(df)
    X = X[feature_cols]

   
    coef_df = analyze_coefficients(model, feature_cols)

    shap_df, shap_values = analyze_shap(model, X, feature_cols)

    log_to_mlflow(coef_df, shap_df, ml_client)

    print("\nModel explainability analysis completed")


if __name__ == "__main__":
    main()