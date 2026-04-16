import argparse
import os
import json
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature        
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score,
    precision_score, recall_score, f1_score   
)

# Assiging a singe name to the model in the Azure ML Model Registry so we can easily find it later.
MODEL_REGISTRY_NAME = "customer-churn-model"


def evaluate(model, X_test, y_test, label: str):
    """
    CHANGED: now returns a full metrics dict instead of just AUC.
    This dict gets logged to MLflow so every metric is searchable later.
    """
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    auc     = roc_auc_score(y_test, y_proba)

    # capture all metrics, not just AUC
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
    parser.add_argument('--processed_data', required=True,
                        help='Folder from preprocessing step (contains X.csv, y.csv, feature_columns.json)')
    parser.add_argument('--model_output',   required=True,
                        help='Folder to write trained model and feature list')
    parser.add_argument('--model_type',     default='high_recall_lr',
                        choices=['logistic_regression', 'random_forest', 'high_recall_lr'],
                        help='Which model to promote as the primary model for prediction')
    parser.add_argument('--test_size',      type=float, default=0.2)
    parser.add_argument('--random_state',   type=int,   default=42)
    args = parser.parse_args()

    os.makedirs(args.model_output, exist_ok=True)

    # Load preprocessed data and feature list
    X = pd.read_csv(os.path.join(args.processed_data, 'X.csv'))
    y = pd.read_csv(os.path.join(args.processed_data, 'y.csv')).squeeze()

    with open(os.path.join(args.processed_data, 'feature_columns.json')) as f:
        feature_cols = json.load(f)

    X = X[feature_cols]  # guarantee column order
    print(f"Training on {len(X):,} rows, {len(feature_cols)} features")
    print(f"Class distribution:\n{y.value_counts()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    # open one MLflow run that wraps ALL training below 
    # Everything inside this block is recorded as one experiment entry.
    # In Azure ML Studio in Experiments tab, you will see this run with all
    # params, metrics, and the registered model linked to it.
    mlflow.sklearn.autolog(log_models=False)  # we will log the model manually so we can add the signature and input example

    with mlflow.start_run(run_name=f"train_{args.model_type}") as run:

        # log the inputs so every run is fully reproducible 
        mlflow.log_params({
            "model_type":      args.model_type,
            "test_size":       args.test_size,
            "random_state":    args.random_state,
            "n_features":      len(feature_cols),
            "n_train_samples": len(X_train),
            "n_test_samples":  len(X_test),
            "train_churn_rate":round(float(y_train.mean()), 4),
        })

        # Train all three models
        models = {
            'logistic_regression': LogisticRegression(
                max_iter=1000, random_state=args.random_state),
            'random_forest':       RandomForestClassifier(
                n_estimators=100, random_state=args.random_state, class_weight='balanced'),
            'high_recall_lr':      LogisticRegression(
                max_iter=1000, class_weight='balanced', random_state=args.random_state),
        }

        all_metrics = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            metrics = evaluate(model, X_test, y_test, name)
            all_metrics[name] = metrics

            #log per-model metrics with a prefix so they don't collide
            mlflow.log_metrics({f"{name}/{k}": v for k, v in metrics.items()})

            joblib.dump(model, os.path.join(args.model_output, f'{name}.pkl'))

        # Select primary model 
        primary        = models[args.model_type]
        primary_m      = all_metrics[args.model_type]

        # log primary model metrics at the top level too
        # These are the ones visible when comparing runs side by side in Studio
        mlflow.log_metrics({
            "primary_roc_auc":   primary_m["roc_auc"],
            "primary_recall":    primary_m["recall"],
            "primary_precision": primary_m["precision"],
            "primary_f1":        primary_m["f1"],
        })

        # Save model.pkl 
        joblib.dump(primary, os.path.join(args.model_output, 'model.pkl'))
        print(f"\nPrimary model ({args.model_type}) saved as model.pkl")

        # Save feature list 
        with open(os.path.join(args.model_output, 'feature_columns.json'), 'w') as f:
            json.dump(feature_cols, f)
        
        mlflow.log_artifact(
            os.path.join(args.model_output, 'feature_columns.json'),
                artifact_path="model"  
            )

        # save model_config.json
        # predict.py will read this so inference uses the exact same settings
        # that were chosen at training time (threshold, feature order, model type)
        with open(os.path.join(args.model_output, 'model_config.json'), 'w') as f:
            json.dump({
                "model_type":      args.model_type,
                "feature_columns": feature_cols,
            }, f, indent=2)

        # ── NEW: register model to Azure ML Model Registry ────────────────────
        #
        # infer_signature() inspects 5 sample rows and records:
        #   - input schema  and output schema
        # Azure ML checks this contract at deployment time so you can never
        # accidentally deploy a model that expects different columns.
        # Every time this runs, it creates a new version (v1, v2, v3 ...).

        signature = infer_signature(
            X_test.iloc[:5],
            primary.predict(X_test.iloc[:5])
        )

        mlflow.sklearn.log_model(
            sk_model=primary,
            artifact_path="model",
            signature=signature,
            input_example=X_test.iloc[:5],
            registered_model_name=MODEL_REGISTRY_NAME,       
        )

        # tag the run so it's searchable in Studio
        mlflow.set_tags({
            "model_type":    args.model_type,
            "dataset":       "customer_churn",
            "pipeline_step": "train",
        })

        run_id = run.info.run_id

    # I am using below json file metrics to decide which model is good for production
    with open(os.path.join(args.model_output, 'run_info.json'), 'w') as f:
        json.dump({
            "run_id":     run_id,
            "model_name": MODEL_REGISTRY_NAME,
            "model_type": args.model_type,
            "roc_auc":    primary_m["roc_auc"],
            "recall":     primary_m["recall"],
            "precision":  primary_m["precision"],
            "f1":         primary_m["f1"],
        }, f, indent=2)

 
    print(f"MLflow run ID : {run_id}")
    print(f"Primary model : {args.model_type}")
    print(f"ROC-AUC       : {primary_m['roc_auc']:.4f}")
    print(f"Recall        : {primary_m['recall']:.4f}")
    print(f"Registered as : {MODEL_REGISTRY_NAME}")


if __name__ == '__main__':
    main()