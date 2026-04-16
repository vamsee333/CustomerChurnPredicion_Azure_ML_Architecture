import os
import json
import joblib
from predict import build_inference_row


def init():
    model_dir = os.environ.get("AZUREML_MODEL_DIR", ".")

    # Azure ML can mount the artifact at root or in a sub-folder named "model"
    nested = os.path.join(model_dir, "model", "model.pkl")
    root   = os.path.join(model_dir, "model.pkl")
    model_path = nested if os.path.exists(nested) else root

    feat_path = model_path.replace("model.pkl", "feature_columns.json")

    global MODEL, FEATURE_COLS
    MODEL        = joblib.load(model_path)
    FEATURE_COLS = json.load(open(feat_path))
    print(f"[init] Model loaded from: {model_path}")


def run(raw_data):
    payload = json.loads(raw_data)

    # Support both single and batch inputs
    records = payload.get("input_data", [payload])

    predictions = []
    for record in records:
        df         = build_inference_row(record, FEATURE_COLS)
        prediction = int(MODEL.predict(df)[0])
        proba      = MODEL.predict_proba(df)[0]
        predictions.append({
            "churn_prediction":       prediction,
            "churn_prediction_label": "Churn" if prediction == 1 else "No Churn",
            "probability_no_churn":   round(float(proba[0]), 4),
            "probability_churn":      round(float(proba[1]), 4),
        })

    return json.dumps({"predictions": predictions})