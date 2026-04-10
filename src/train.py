import argparse
import os
import json
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


def evaluate(model, X_test, y_test, label: str):
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    auc     = roc_auc_score(y_test, y_proba)

    print(classification_report(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print(f"ROC-AUC: {auc:.4f}")
    return auc


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

    # Load preprocessed data 
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

    mlflow.sklearn.autolog(log_models=False)  # log params & metrics; we save models manually

    # Train all three models
    models = {
        'logistic_regression': LogisticRegression(max_iter=1000, random_state=args.random_state),
        'random_forest':       RandomForestClassifier(n_estimators=100, random_state=args.random_state,
                                                      class_weight='balanced'),
        'high_recall_lr':      LogisticRegression(max_iter=1000, class_weight='balanced',
                                                  random_state=args.random_state),
    }

    auc_scores = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        auc_scores[name] = evaluate(model, X_test, y_test, name)
        joblib.dump(model, os.path.join(args.model_output, f'{name}.pkl'))
        mlflow.log_metric(f'{name}_roc_auc', auc_scores[name])


    # predict.py always loads model.pkl — no hardcoded filename there
    primary = models[args.model_type]
    joblib.dump(primary, os.path.join(args.model_output, 'model.pkl'))
    print(f"\nPrimary model ({args.model_type}) saved as model.pkl")

    # predict.py reads this to align inference columns and no CSV re-read needed
    with open(os.path.join(args.model_output, 'feature_columns.json'), 'w') as f:
        json.dump(feature_cols, f)

    # Log summary using MLflow
    mlflow.log_param('primary_model', args.model_type)
    mlflow.log_metric('primary_roc_auc', auc_scores[args.model_type])

    print("\nAll models and feature list saved to:", args.model_output)
    print("AUC scores:", {k: f"{v}" for k, v in auc_scores.items()})


if __name__ == '__main__':
    main()