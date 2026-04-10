import argparse
import os
import json
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder


def build_inference_row(sample_input: dict, feature_cols: list) -> pd.DataFrame:
    """
    Transform a raw customer dict into a single-row DataFrame that matches
    the feature columns produced by preprocessing.py — without re-reading CSVs.
    """
    df = pd.DataFrame([sample_input])

    # Ordinal - Contract 
    if 'Contract' in df.columns:
        mapping = {'Month-to-Month': 0, 'Month-to-month': 0, 'One Year': 1, 'Two Year': 2}
        df['Contract'] = df['Contract'].map(mapping).fillna(0).astype(int)

    # Binary -  Gender 
    for col in ['Gender', 'gender']:
        if col in df.columns:
            df[col] = df[col].map({'Male': 1, 'Female': 0}).fillna(0).astype(int)

    # Binary -Yes/No columns 
    yes_no_cols = [
        'Partner', 'Dependents', 'Phone Service', 'Multiple Lines',
        'Online Security', 'Online Backup', 'Device Protection Plan',
        'Premium Tech Support', 'Streaming TV', 'Streaming Movies',
        'Streaming Music', 'Unlimited Data', 'Paperless Billing',
        'Referred a Friend',
    ]
    for col in yes_no_cols:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)

    # OneHot -Internet Type 
    for raw_col in ['Internet Type', 'InternetService', 'Internet Service']:
        if raw_col in df.columns:
            dummies = pd.get_dummies(df[raw_col], prefix=raw_col.replace(' ', '_'))
            df = pd.concat([df.drop(columns=[raw_col]), dummies], axis=1)

    # OneHot - Payment Method
    for raw_col in ['Payment Method', 'PaymentMethod']:
        if raw_col in df.columns:
            dummies = pd.get_dummies(df[raw_col], prefix=raw_col.replace(' ', '_'))
            df = pd.concat([df.drop(columns=[raw_col]), dummies], axis=1)

    # OneHot - Offer
    if 'Offer' in df.columns:
        dummies = pd.get_dummies(df['Offer'], prefix='Offer')
        df = pd.concat([df.drop(columns=['Offer']), dummies], axis=1)


    # Add any missing columns as 0, drop any extras, reorder to match training
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_cols]
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)

    return df


def predict_churn(sample_input: dict, model_dir: str) -> dict:
    """
    Predict churn for a single customer dict.

    Args:
        sample_input: Raw customer feature dict (same format as your original sample_input).
        model_dir:    Folder containing model.pkl and feature_columns.json.

    Returns:
        dict with churn_prediction, label, and probabilities.
    """
    model_path   = os.path.join(model_dir, 'model.pkl')
    feature_path = os.path.join(model_dir, 'feature_columns.json')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"model.pkl not found in {model_dir}")
    if not os.path.exists(feature_path):
        raise FileNotFoundError(f"feature_columns.json not found in {model_dir}")

    model        = joblib.load(model_path)
    feature_cols = json.load(open(feature_path))

    df         = build_inference_row(sample_input, feature_cols)
    prediction = model.predict(df)[0]
    proba      = model.predict_proba(df)[0]

    return {
        'churn_prediction':       int(prediction),
        'churn_prediction_label': 'Churn' if prediction == 1 else 'No Churn',
        'probability_no_churn':   float(proba[0]),
        'probability_churn':      float(proba[1]),
    }


# Azure ML pipeline entry-point for batch scoring
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed_data',      required=True,
                        help='Folder from preprocessing step (contains X.csv)')
    parser.add_argument('--model_input',         required=True,
                        help='Folder containing model.pkl and feature_columns.json')
    parser.add_argument('--predictions_output',  required=True,
                        help='Folder to write predictions.csv')
    args = parser.parse_args()

    os.makedirs(args.predictions_output, exist_ok=True)

    # Load the best saved model
    model        = joblib.load(os.path.join(args.model_input, 'model.pkl'))
    feature_cols = json.load(open(os.path.join(args.model_input, 'feature_columns.json')))

    X = pd.read_csv(os.path.join(args.processed_data, 'X.csv'))
    y = pd.read_csv(os.path.join(args.processed_data, 'y.csv')).squeeze()

    X = X[feature_cols]  # enforce column order

    #  Predict churn probabilities and labels
    y_pred  = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    # Final output 
    results = X.copy()
    results['actual_churn']      = y.values
    results['predicted_churn']   = y_pred
    results['churn_probability'] = y_proba

    out_path = os.path.join(args.predictions_output, 'predictions.csv')
    results.to_csv(out_path, index=False)

    # Summary
    from sklearn.metrics import roc_auc_score, classification_report
    print(classification_report(y, y_pred))
    print(f"ROC-AUC: {roc_auc_score(y, y_proba):.4f}")
    print(f"Predictions written → {out_path}  ({len(results):,} rows)")