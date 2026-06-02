import os
import json
import joblib
from predict import build_inference_row

# declared for monkey-patching in tests
# Test_comment



MODEL        = None
FEATURE_COLS = None

def init():
    model_dir = os.environ.get("AZUREML_MODEL_DIR", ".")

    # Champion path (registered via mlflow): model/model.pkl
    # Challenger path (registered via SDK):  challenger_model/model.pkl
    # This searches all subdirectories so it works for both
    
    model_path = None
    for root, dirs, files in os.walk(model_dir):
        if "model.pkl" in files:
            model_path = os.path.join(root, "model.pkl")
            break

    if model_path is None:
        raise FileNotFoundError(f"model.pkl not found anywhere under {model_dir}")

    feat_path = os.path.join(os.path.dirname(model_path), "feature_columns.json")

    if not os.path.exists(feat_path):
        raise FileNotFoundError(f"feature_columns.json not found at {feat_path}")

    global MODEL, FEATURE_COLS
    MODEL        = joblib.load(model_path)
    FEATURE_COLS = json.load(open(feat_path))
    print(f"[init] Model loaded from: {model_path}")


def run(raw_data):
    # raw_data is a JSON string from the HTTP request body
    payload = json.loads(raw_data)
    predictions = []

    # Support both a list under "input_data" and a bare list
    records = payload.get("input_data", payload) if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        records = [records]

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