import argparse
import os
import json
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def preprocess_data(df: pd.DataFrame):
    """
    Apply all transformations to a raw churn DataFrame.
    Returns processed X, y, and the ordered feature column list.
    Importable by predict.py for single-row inference without re-reading CSVs.
    """
    # Fill known nullable columns
    df['Churn Category'] = df['Churn Category'].fillna('No Churn')
    df['Churn Reason']   = df['Churn Reason'].fillna('No Churn')
    df.dropna(inplace=True)

    if 'Customer ID' in df.columns:
        df = df.drop(columns=['Customer ID'])

    y = df['Churn']
    X = df.drop(columns=['Churn'])

    # Drop leaky / irrelevant columns
    drop_cols = [
        'Country', 'Customer Status', 'Quarter', 'State',
        'Churn Category', 'Churn Reason',
        'Lat Long', 'City', 'Latitude', 'Zip Code', 'Longitude',
        'Satisfaction Score', 'Churn Score',
    ]
    X = X.drop(columns=[c for c in drop_cols if c in X.columns])

    # Ordinal for Contract types 
    contract_mapping = {'Month-to-Month': 0, 'One Year': 1, 'Two Year': 2}
    X['Contract'] = X['Contract'].map(contract_mapping).fillna(0).astype(int)

    # Binary mapping for Gender 
    le = LabelEncoder()
    X['Gender'] = le.fit_transform(X['Gender'].astype(str))

    # Binary - Yes/No columns
    yes_no_cols = [
        'Partner', 'Dependents', 'Phone Service', 'Multiple Lines',
        'Online Security', 'Online Backup', 'Device Protection Plan',
        'Premium Tech Support', 'Streaming TV', 'Streaming Movies',
        'Streaming Music', 'Unlimited Data', 'Paperless Billing',
        'Referred a Friend',
    ]
    for col in yes_no_cols:
        if col in X.columns:
            X[col] = X[col].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)

    # Frequency encoding for Column City because it has high cardinality 
    if 'City' in df.columns:
        city_freq = df['City'].value_counts()
        X['City Freq'] = df['City'].map(city_freq).fillna(0).astype(int)

    # OneHot encoding fornominal columns
    onehot_cols = ['Internet Type', 'Payment Method', 'Offer']
    X = pd.get_dummies(X, columns=[c for c in onehot_cols if c in X.columns], drop_first=True)

    # Drop single-value columns
    X = X[[c for c in X.columns if X[c].nunique() > 1]]

    # Ensure all numeric
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

    return X, y


# Azure ML pipeline entry-point 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data',  required=True,
                        help='Path to raw churn CSV')
    parser.add_argument('--output_data', required=True,
                        help='Output folder for processed files')
    args = parser.parse_args()

    os.makedirs(args.output_data, exist_ok=True)

    df = pd.read_csv(args.input_data)
    print(f"Loaded {len(df):,} rows, {df.shape[1]} columns")

    X, y = preprocess_data(df)

    # Save feature column order — predict.py reads this at inference time
    with open(os.path.join(args.output_data, 'feature_columns.json'), 'w') as f:
        json.dump(X.columns.tolist(), f)

    X.to_csv(os.path.join(args.output_data, 'X.csv'), index=False)
    y.to_csv(os.path.join(args.output_data, 'y.csv'), index=False)

    print(f"Saved {len(X.columns)} features → {args.output_data}")
    print(f"Class distribution:\n{y.value_counts()}")