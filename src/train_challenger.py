import argparse
import os
import json
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score,
    precision_score, recall_score, f1_score
)
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

# preprocessing.py lives in the same folder — same import as your other scripts
import sys
sys.path.append(os.path.dirname(__file__))
from preprocessing import preprocess_data

MODEL_REGISTRY_NAME = "customer-churn-model"

# ── Connect MLflow to your Azure ML workspace ─────────────────────────────────
# When train.py runs inside the Azure ML pipeline on the compute cluster,
# MLflow is automatically connected to the workspace.
# When running locally, we have to set the tracking URI manually — otherwise
# MLflow logs to a local mlruns folder on your machine and the model never
# reaches the Azure ML Model Registry.

_HERE = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(_HERE, ".azureml/config.json")
if not os.path.exists(config_path):
    config_path = os.path.join(_HERE, "../.azureml/config.json")

with open(config_path) as f:
    _config = json.load(f)

ml_client = MLClient(
    DefaultAzureCredential(),
    _config["subscription_id"],
    _config["resource_group"],
    _config["workspace_name"],
)

# This single line is what connects MLflow to Azure ML.
# It tells MLflow: "send all runs and registered models to this workspace"
# instead of saving them locally.
mlflow.set_tracking_uri(ml_client.workspaces.get(_config["workspace_name"]).mlflow_tracking_uri)
print(f"MLflow tracking URI set → Azure ML workspace: {_config['workspace_name']}\n")


def evaluate(model, X_test, y_test, label: str):
    # Identical to train.py
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    auc     = roc_auc_score(y_test, y_proba)

    metrics = {
        "roc_auc":   auc,
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "f1":        f1_score(y_test, y_pred, zero_division=0),
    }

    print(f"\n── {label} ──────────────────────────────────────")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print(f"ROC-AUC: {auc:.4f}")

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data',   required=True,
                        help='Path to raw churn CSV (e.g. train.csv)')
    parser.add_argument('--model_output', required=True,
                        help='Folder to write challenger model and metadata')
    parser.add_argument('--test_size',    type=float, default=0.2)
    parser.add_argument('--random_state', type=int,   default=42)
    args = parser.parse_args()

    os.makedirs(args.model_output, exist_ok=True)

    # Load and preprocess raw CSV — same as preprocessing.py does in the pipeline
    # This means you can run this script locally with just train.csv
    print(f"Loading {args.input_data}...")
    df = pd.read_csv(args.input_data)
    print(f"Loaded {len(df):,} rows, {df.shape[1]} columns")

    X, y = preprocess_data(df)
    feature_cols = X.columns.tolist()

    print(f"After preprocessing: {len(X):,} rows, {len(feature_cols)} features")
    print(f"Class distribution:\n{y.value_counts()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    # StratifiedKFold — preserves the churn class ratio (~27%) in every fold.
    # Plain KFold splits randomly and can put very few churners in a fold,
    # making CV scores unreliable and non-comparable between models.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.random_state)

    # ── Hyperparameter grids ──────────────────────────────────────────────────
    # Your champion used default params. These grids test whether a better
    # configuration exists before we challenge it in production.
    #
    # C in LogisticRegression — controls regularisation strength.
    #   Low C = strong penalty, simpler model. High C = weak penalty, can overfit.
    #   With 50 features, the sweet spot is usually somewhere between 0.1 and 10.
    #
    # n_estimators / max_depth in RandomForest —
    #   More trees = more stable but slower. max_depth=None grows full trees
    #   which overfit on tabular data — constraining to 10 or 20 helps generalise.
    #
    # scoring='recall' — we optimise the search for the same metric we gate on
    #   at deployment (Recall >= 0.65). Optimising for AUC but gating on recall
    #   can give you a model that wins the search but fails the deployment check.

    lr_grid = {
        'C':            [0.01, 0.1, 1.0, 10.0],
        'max_iter':     [1000, 2000],
        'class_weight': ['balanced'],
    }

    rf_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth':    [10, 20, None],
        'class_weight': ['balanced'],
    }

    mlflow.sklearn.autolog(log_models=False)

    with mlflow.start_run(run_name="train_challenger") as run:

        mlflow.log_params({
            "test_size":        args.test_size,
            "random_state":     args.random_state,
            "n_features":       len(feature_cols),
            "n_train_samples":  len(X_train),
            "n_test_samples":   len(X_test),
            "train_churn_rate": round(float(y_train.mean()), 4),
            "search_type":      "GridSearchCV",
            "cv_folds":         5,
            "cv_scoring":       "recall",
        })

        # ── Run GridSearchCV for each candidate ───────────────────────────────
        # cv here means - cross validation strategy
        print("\nSearching LogisticRegression hyperparameters...")
        lr_search = GridSearchCV( 
            LogisticRegression(random_state=args.random_state),
            lr_grid, scoring='recall', cv=cv, n_jobs=-1, refit=True
        )
        lr_search.fit(X_train, y_train)
        print(f"  Best params : {lr_search.best_params_}")
        print(f"  CV recall   : {lr_search.best_score_:.4f}")

        print("\nSearching RandomForest hyperparameters...")
        rf_search = GridSearchCV(
            RandomForestClassifier(random_state=args.random_state),
            rf_grid, scoring='recall', cv=cv, n_jobs=-1, refit=True
        )
        rf_search.fit(X_train, y_train)
        print(f"  Best params : {rf_search.best_params_}")
        print(f"  CV recall   : {rf_search.best_score_:.4f}")

        # ── Evaluate best model from each search on held-out test set ─────────
        models = {
            'tuned_logistic_regression': lr_search.best_estimator_,
            'tuned_random_forest':       rf_search.best_estimator_,
        }

        all_metrics = {}
        for name, model in models.items():
            metrics = evaluate(model, X_test, y_test, name)
            all_metrics[name] = metrics
            mlflow.log_metrics({f"{name}/{k}": v for k, v in metrics.items()})
            joblib.dump(model, os.path.join(args.model_output, f'{name}.pkl'))

        # Log best hyperparams found by each search
        mlflow.log_params({
            "lr_best_C":            lr_search.best_params_['C'],
            "lr_best_max_iter":     lr_search.best_params_['max_iter'],
            "lr_cv_best_recall":    round(lr_search.best_score_, 4),
            "rf_best_n_estimators": rf_search.best_params_['n_estimators'],
            "rf_best_max_depth":    str(rf_search.best_params_['max_depth']),
            "rf_cv_best_recall":    round(rf_search.best_score_, 4),
        })

        # ── Pick the winner ───────────────────────────────────────────────────
        # Rank by recall first (our deployment gate), AUC as tiebreaker

        
        winner_name = max(
            all_metrics,
            key=lambda n: (all_metrics[n]['recall'], all_metrics[n]['roc_auc'])
        )
        winner_model   = models[winner_name]
        winner_metrics = all_metrics[winner_name]

        other_name = [n for n in models if n != winner_name][0]
        print(f"\n── Winner: {winner_name} ──────────────────────────────────────")
        print(f"  Recall : {winner_metrics['recall']:.4f}  (vs {all_metrics[other_name]['recall']:.4f} for {other_name})")
        print(f"  AUC    : {winner_metrics['roc_auc']:.4f}  (vs {all_metrics[other_name]['roc_auc']:.4f} for {other_name})")

        mlflow.log_params({"winning_model": winner_name})
        mlflow.log_metrics({
            "primary_roc_auc":   winner_metrics["roc_auc"],
            "primary_recall":    winner_metrics["recall"],
            "primary_precision": winner_metrics["precision"],
            "primary_f1":        winner_metrics["f1"],
        })

        # Save winner as model.pkl
        joblib.dump(winner_model, os.path.join(args.model_output, 'model.pkl'))
        print(f"\nChallenger model ({winner_name}) saved as model.pkl")

        # Save feature list
        with open(os.path.join(args.model_output, 'feature_columns.json'), 'w') as f:
            json.dump(feature_cols, f)

        mlflow.set_tags({
            "model_type":    winner_name,
            "dataset":       "customer_churn",
            "pipeline_step": "train_challenger",
            "stage_intent":  "challenger",
        })

        run_id = run.info.run_id

    # ── Register model in Azure ML Model Registry via SDK ────────────────────
    # We skip mlflow.sklearn.log_model here due to a version incompatibility
    # between mlflow and azureml-mlflow when running locally.
    # Instead we register directly using ml_client — the same approach
    # Modelpromoter.py uses successfully.
    from azure.ai.ml.entities import Model
    from azure.ai.ml.constants import AssetTypes

    registered = ml_client.models.create_or_update(
        Model(
            path=args.model_output,
            name=MODEL_REGISTRY_NAME,
            description=(
                f"Challenger | {winner_name} | "
                f"ROC-AUC={winner_metrics['roc_auc']:.4f} | "
                f"Recall={winner_metrics['recall']:.4f} | "
                f"run_id={run_id}"
            ),
            type=AssetTypes.CUSTOM_MODEL,
            tags={
                "stage":       "challenger",
                "model_type":  winner_name,
                "roc_auc":     str(round(winner_metrics["roc_auc"], 4)),
                "recall":      str(round(winner_metrics["recall"], 4)),
                "run_id":      run_id,
                "promoted_by": "train_challenger.py",
            },
        )
    )
    print(f"Registered in Azure ML Model Registry: {MODEL_REGISTRY_NAME} v{registered.version}")

    # run_info.json — read by challenger_deploy.py
    with open(os.path.join(args.model_output, 'run_info.json'), 'w') as f:
        json.dump({
            "run_id":         run_id,
            "model_name":     MODEL_REGISTRY_NAME,
            "model_version":  registered.version,
            "model_type":     winner_name,
            "roc_auc":        winner_metrics["roc_auc"],
            "recall":         winner_metrics["recall"],
            "precision":      winner_metrics["precision"],
            "f1":             winner_metrics["f1"],
            "all_candidates": {
                name: {k: round(v, 4) for k, v in m.items()}
                for name, m in all_metrics.items()
            },
        }, f, indent=2)

    print(f"\nMLflow run ID  : {run_id}")
    print(f"Challenger     : {winner_name}")
    print(f"ROC-AUC        : {winner_metrics['roc_auc']:.4f}")
    print(f"Recall         : {winner_metrics['recall']:.4f}")
    print(f"Registered as  : {MODEL_REGISTRY_NAME} v{registered.version}  (tag: challenger)")
    print(f"Next step      : python challenger_deploy.py")


if __name__ == '__main__':
    main()