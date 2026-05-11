"""
The following tests must pass before merging any changes to the pipeline code in src:
1.  preprocessing.py for - feature engineering, columns, outputs
2.  predict.py for - inference row construction, feature alignment
3.  Score.py for - response schema, batch support, probability consistency  
4.  Modelpromoter.py for - quality gate logic, run_info.json contract

To run these tests locally, run python -m pytest tests/test_pipeline.py --cov=src --cov-report=term-missing --cov-fail-under=80 -v
"""

import json
import os
import sys

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression


SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

print(f"SRC_DIR: {SRC_DIR}")  

from preprocessing import preprocess_data
from predict import build_inference_row


# FIXTURES — scope explained
#  I have set scope="session"  because I am reading data once for the entire session  

# HOW preprocess_data() CALL COUNT IS CONTROLLED:
# processed fixture calls preprocess_data() once and shared by all 16 tests in TestPreprocessData 
# trained_model also calls preprocess_data() once and used by TestScoreRun


@pytest.fixture(scope="session")
def raw_df():
    """
    Base raw churn DataFrame — built once, shared across all tests.
    No test mutates this directly. Tests that need dirty data call
    raw_df.copy() locally before modifying.
    """
    n = 10
    return pd.DataFrame({
        "Customer ID":                       [f"C{i}" for i in range(n)],
        "Age":                               [30, 45, 52, 23, 67, 41, 55, 38, 29, 60],
        "Gender":                            ["Male", "Female"] * 5,
        "Married":                           [1, 0] * 5,
        "Partner":                           ["Yes", "No"] * 5,
        "Dependents":                        ["No"] * n,
        "Number of Dependents":              [0] * n,
        "City":                              ["Dundee", "Glasgow"] * 5,
        "Zip Code":                          [12345] * n,
        "Latitude":                          [56.46] * n,
        "Longitude":                         [-2.97] * n,
        "Lat Long":                          ["56.46, -2.97"] * n,
        "Population":                        [100000] * n,
        "Quarter":                           ["Q1"] * n,
        "Country":                           ["United Kingdom"] * n,
        "State":                             ["Scotland"] * n,
        "Phone Service":                     ["Yes"] * n,
        "Multiple Lines":                    ["No", "Yes"] * 5,
        "Internet Service":                  [1] * n,
        "Internet Type":                     ["Fiber Optic", "Cable", "DSL", "Fiber Optic", "Cable",
                                              "DSL", "Fiber Optic", "Cable", "DSL", "Fiber Optic"],
        "Avg Monthly GB Download":           [40, 20, 60, 15, 80, 35, 55, 25, 70, 10],
        "Online Security":                   ["No", "Yes"] * 5,
        "Online Backup":                     ["Yes", "No"] * 5,
        "Device Protection Plan":            ["No"] * n,
        "Premium Tech Support":              ["Yes", "No"] * 5,
        "Streaming TV":                      ["Yes"] * n,
        "Streaming Movies":                  ["No", "Yes"] * 5,
        "Streaming Music":                   ["No"] * n,
        "Unlimited Data":                    ["Yes"] * n,
        "Contract":                          ["Month-to-Month", "One Year", "Two Year",
                                              "Month-to-Month", "One Year", "Two Year",
                                              "Month-to-Month", "One Year", "Two Year",
                                              "Month-to-Month"],
        "Paperless Billing":                 ["Yes", "No"] * 5,
        "Payment Method":                    ["Credit Card (automatic)", "Bank transfer (automatic)",
                                              "Electronic check", "Mailed check",
                                              "Credit Card (automatic)", "Bank transfer (automatic)",
                                              "Electronic check", "Mailed check",
                                              "Credit Card (automatic)", "Bank transfer (automatic)"],
        "Monthly Charge":                    [70.5, 55.0, 90.0, 45.0, 85.0,
                                              60.0, 75.0, 50.0, 95.0, 40.0],
        "Total Charges":                     [846.0, 660.0, 1080.0, 540.0, 1020.0,
                                              720.0, 900.0, 600.0, 1140.0, 480.0],
        "Total Refunds":                     [0.0] * n,
        "Total Extra Data Charges":          [0] * n,
        "Total Long Distance Charges":       [50.0] * n,
        "Total Revenue":                     [896.0, 710.0, 1130.0, 590.0, 1070.0,
                                              770.0, 950.0, 650.0, 1190.0, 530.0],
        "Offer":                             ["No Offer", "Offer A", "Offer B", "Offer C", "Offer D",
                                              "No Offer", "Offer A", "Offer B", "Offer C", "Offer D"],
        "Referred a Friend":                 ["No", "Yes"] * 5,
        "Number of Referrals":               [0, 1] * 5,
        "Tenure in Months":                  [12, 24, 36, 6, 48, 18, 30, 12, 42, 6],
        "Avg Monthly Long Distance Charges": [5.0] * n,
        "CLTV":                              [3200, 4000, 5000, 2000, 6000,
                                              3500, 4500, 2500, 5500, 1800],
        "Senior Citizen":                    [0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
        "Under 30":                          [0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
        "Satisfaction Score":                [3, 4, 2, 5, 1, 3, 4, 2, 5, 1],
        "Churn Score":                       [50, 30, 80, 20, 90, 45, 35, 75, 25, 85],
        "Customer Status":                   ["Active"] * n,
        "Churn Category":                    [None] * 5 + ["Competitor"] * 3 + [None] * 2,
        "Churn Reason":                      [None] * 5 + ["Better offer"] * 3 + [None] * 2,
        "Churn":                             [0, 0, 1, 0, 1, 0, 0, 1, 0, 1],
    })


@pytest.fixture(scope="session")
def processed(raw_df):
   
    #Runs preprocess_data() ONCE for the entire session. All 16 tests in TestPreprocessData read from this shared (X, y), instead of each calling preprocess_data() themselves.
    
    X, y = preprocess_data(raw_df.copy())
    return X, y


@pytest.fixture(scope="session")
def feature_cols(processed):
  
   # Column contract from training — derived from the already-computed processed result, so no extra preprocess_data() call needed.
    
    X, _ = processed
    return X.columns.tolist()


@pytest.fixture(scope="session")
def trained_model(raw_df):
    """
    LogisticRegression trained once for the whole session.Used by TestScoreRun to inject a real model into Score.py globals without needing any .pkl files on disk.
    """
    X, y = preprocess_data(raw_df.copy())
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X, y)
    return clf, X.columns.tolist()


@pytest.fixture
def sample_customer():
    """
    Single raw customer dict which is same format as InvokeService.ipynb. Kept at function scope (default) so each test gets a fresh copy
    and mutations in one test can never affect another.
    """
    return {
        "Age": 45,
        "Contract": "Month-to-month",
        "Monthly Charge": 89.50,
        "Tenure in Months": 6,
        "Internet Type": "Fiber Optic",
        "Payment Method": "Credit Card (automatic)",
        "Offer": "No Offer",
        "Gender": "Male",
        "Partner": "Yes",
        "Dependents": "No",
        "Phone Service": "Yes",
        "Multiple Lines": "No",
        "Online Security": "No",
        "Online Backup": "No",
        "Device Protection Plan": "No",
        "Premium Tech Support": "No",
        "Streaming TV": "Yes",
        "Streaming Movies": "Yes",
        "Streaming Music": "No",
        "Unlimited Data": "Yes",
        "Paperless Billing": "Yes",
        "Referred a Friend": "No",
        "Number of Referrals": 0,
        "Avg Monthly GB Download": 45,
        "Avg Monthly Long Distance Charges": 0.0,
        "CLTV": 3200,
        "Number of Dependents": 0,
        "Satisfaction Score": 2,
    }



# 1. preprocessing.py tests

# Every method receives processed- the single shared (X, y).
# No method calls preprocess_data() itself.

# The ONE exception is test_handles_null_churn_category_filled — it must
# call preprocess_data() on deliberately dirty data (all NaNs injected),
# which the shared clean processed result cannot cover.
# It uses raw_df.copy() so the session fixture is never mutated.


class TestPreprocessData:

    def test_returns_dataframe_and_series(self, processed):
        X, y = processed
        assert isinstance(X, pd.DataFrame), "X must be a DataFrame"
        assert isinstance(y, pd.Series),    "y must be a Series"

    def test_output_row_count_matches_input(self, processed, raw_df):
        X, y = processed
        assert len(X) == len(y),      "X and y must have the same number of rows"
        assert len(X) == len(raw_df), "No rows should be dropped from a clean fixture"

    def test_leaky_columns_removed(self, processed):
        X, _ = processed
        leaky = [
            "Churn Category", "Churn Reason", "Customer Status",
            "Satisfaction Score", "Churn Score",
        ]
        for col in leaky:
            assert col not in X.columns, f"Leaky column '{col}' found in X"

    def test_geographic_columns_removed(self, processed):
        X, _ = processed
        geo = ["Lat Long", "Latitude", "Longitude", "Zip Code", "City", "Country", "State"]
        for col in geo:
            assert col not in X.columns, f"Geographic column '{col}' found in X"

    def test_customer_id_removed(self, processed):
        X, _ = processed
        assert "Customer ID" not in X.columns

    def test_churn_not_in_X(self, processed):
        X, _ = processed
        assert "Churn" not in X.columns, "'Churn' target must not appear in features"

    def test_contract_ordinal_encoding(self, processed):
        X, _ = processed
        assert "Contract" in X.columns
        assert set(X["Contract"].unique()).issubset({0, 1, 2}), (
            f"Contract must be 0/1/2, got: {X['Contract'].unique()}"
        )

    def test_gender_binary_encoding(self, processed):
        X, _ = processed
        assert "Gender" in X.columns
        assert set(X["Gender"].unique()).issubset({0, 1}), (
            f"Gender must be 0 or 1, got: {X['Gender'].unique()}"
        )

    def test_yes_no_cols_are_binary(self, processed):
        X, _ = processed
        yes_no = [
            "Partner", "Phone Service", "Multiple Lines",
            "Online Security", "Online Backup", "Streaming TV", "Paperless Billing",
        ]
        for col in yes_no:
            if col in X.columns:
                unique_vals = set(X[col].unique())
                assert unique_vals.issubset({0, 1}), (
                    f"Column '{col}' has non-binary values: {unique_vals}"
                )

    def test_internet_type_one_hot_expanded(self, processed):
        X, _ = processed
        assert "Internet Type" not in X.columns

    def test_payment_method_one_hot_expanded(self, processed):
        X, _ = processed
        assert "Payment Method" not in X.columns

    def test_offer_one_hot_expanded(self, processed):
        X, _ = processed
        assert "Offer" not in X.columns

    def test_all_columns_numeric(self, processed):
        X, _ = processed
        non_numeric = [c for c in X.columns if X[c].dtype == object]
        assert non_numeric == [], f"Non-numeric columns found: {non_numeric}"

    def test_no_nulls_in_output(self, processed):
        X, _ = processed
        assert X.isnull().sum().sum() == 0, "X contains NaN values after preprocessing"

    def test_no_single_value_columns(self, processed):
        X, _ = processed
        constant = [c for c in X.columns if X[c].nunique() <= 1]
        assert constant == [], f"Single-value columns found: {constant}"

    def test_target_is_binary(self, processed):
        _, y = processed
        assert set(y.unique()).issubset({0, 1}), f"Target has non-binary values: {y.unique()}"

    def test_handles_null_churn_category_filled(self, raw_df):
        """
        EXCEPTION — must call preprocess_data() directly.
        Tests that NaN in Churn Category / Churn Reason doesn't drop all rows.
        Uses raw_df.copy() so the shared session fixture is never mutated.
        """
        df = raw_df.copy()
        df.loc[:, "Churn Category"] = None
        df.loc[:, "Churn Reason"]   = None
        X, y = preprocess_data(df)
        assert len(X) > 0, (
            "All rows dropped when Churn Category/Reason are None. "
            "The fillna('No Churn') step may be broken."
        )



# 2. predict.py — build_inference_row tests


class TestBuildInferenceRow:

    def test_output_columns_match_feature_cols(self, sample_customer, feature_cols):
        df = build_inference_row(sample_customer, feature_cols)
        assert list(df.columns) == feature_cols, (
            "Column mismatch between inference row and training feature columns"
        )

    def test_output_shape_is_single_row(self, sample_customer, feature_cols):
        df = build_inference_row(sample_customer, feature_cols)
        assert df.shape[0] == 1, "build_inference_row must return exactly 1 row"

    def test_no_nulls_in_inference_row(self, sample_customer, feature_cols):
        df = build_inference_row(sample_customer, feature_cols)
        assert df.isnull().sum().sum() == 0, "Inference row must have no NaN values"

    def test_all_dtypes_numeric(self, sample_customer, feature_cols):
        df = build_inference_row(sample_customer, feature_cols)
        non_numeric = [c for c in df.columns if df[c].dtype == object]
        assert non_numeric == [], f"Non-numeric columns in inference row: {non_numeric}"

    def test_missing_feature_defaulted_to_zero(self, feature_cols):
        minimal = {"Age": 30, "Monthly Charge": 55.0, "Tenure in Months": 12}
        df = build_inference_row(minimal, feature_cols)
        assert df.shape == (1, len(feature_cols))

    def test_contract_month_to_month_variants(self, feature_cols):
        """Both 'Month-to-Month' and 'Month-to-month' must map to 0."""
        for variant in ("Month-to-month", "Month-to-Month"):
            row = build_inference_row({"Contract": variant}, feature_cols)
            if "Contract" in row.columns:
                assert row["Contract"].iloc[0] == 0, (
                    f"Contract variant '{variant}' did not encode to 0"
                )

    def test_gender_male_encodes_to_1(self, feature_cols):
        row = build_inference_row({"Gender": "Male"}, feature_cols)
        if "Gender" in row.columns:
            assert row["Gender"].iloc[0] == 1

    def test_gender_female_encodes_to_0(self, feature_cols):
        row = build_inference_row({"Gender": "Female"}, feature_cols)
        if "Gender" in row.columns:
            assert row["Gender"].iloc[0] == 0

    def test_yes_no_fields_encode_correctly(self, feature_cols):
        row = build_inference_row(
            {"Partner": "Yes", "Online Security": "No"}, feature_cols
        )
        if "Partner" in row.columns:
            assert row["Partner"].iloc[0] == 1
        if "Online Security" in row.columns:
            assert row["Online Security"].iloc[0] == 0



# 3. Score.py — run() function tests
# Your Score.py returns churn_prediction, churn_prediction_label,
#                    probability_churn, probability_no_churn

# _get_first_prediction() is a shared helper that unwraps predictions[0]
# so individual tests only assert on the fields they care about,
# without repeating the unwrap boilerplate in every test.
#
# monkeypatch scope stays at "function" (default) — it automatically
# undoes the patch after each test, keeping tests isolated even though
# the underlying model object is session-scoped.


class TestScoreRun:

    @pytest.fixture(autouse=True)
    def patch_score_globals(self, trained_model, monkeypatch):
        """
        Injects the session-scoped trained model into Score.py module globals.
        MODEL and FEATURE_COLS are declared as None at module level in Score.py
        so monkeypatch.setattr can find and replace them without AttributeError.
        """
        import Score
        clf, cols = trained_model
        monkeypatch.setattr(Score, "MODEL",        clf)
        monkeypatch.setattr(Score, "FEATURE_COLS", cols)

    def _call_run(self, sample_customer):
        """Calls Score.run() and returns the full parsed response dict."""
        import Score
        return json.loads(Score.run(json.dumps({"input_data": [sample_customer]})))

    def _get_first_prediction(self, sample_customer):
        """Unwraps predictions[0] — used by tests that check individual fields."""
        return self._call_run(sample_customer)["predictions"][0]

    # --- Response structure ---

    def test_run_returns_json_string(self, sample_customer):
        import Score
        result = Score.run(json.dumps({"input_data": [sample_customer]}))
        assert isinstance(result, str), "run() must return a JSON string"

    def test_run_response_has_predictions_key(self, sample_customer):
        """
        Top-level response must have a 'predictions' key containing a list.
        This is the contract your Score.py defines — any downstream system
        reading the response depends on this key existing.
        """
        result = self._call_run(sample_customer)
        assert "predictions" in result, (
            f"Top-level response must have 'predictions' key, got: {list(result.keys())}"
        )
        assert isinstance(result["predictions"], list), "'predictions' must be a list"

    def test_run_single_input_returns_one_prediction(self, sample_customer):
        """One record in → exactly one prediction dict out."""
        result = self._call_run(sample_customer)
        assert len(result["predictions"]) == 1, (
            f"One input record should return 1 prediction, got {len(result['predictions'])}"
        )

    def test_run_batch_input_returns_multiple_predictions(self, sample_customer):
        """
        Score.py loops over input_data so two records in must return two out.
        Tests the batch support in your current Score.py version.
        """
        import Score
        payload = json.dumps({"input_data": [sample_customer, sample_customer]})
        result  = json.loads(Score.run(payload))
        assert len(result["predictions"]) == 2, (
            f"Two input records should return 2 predictions, got {len(result['predictions'])}"
        )

    # --- Individual prediction fields ---

    def test_run_response_has_churn_prediction_key(self, sample_customer):
        result = self._get_first_prediction(sample_customer)
        assert "churn_prediction" in result, (
            f"'churn_prediction' key missing from prediction dict: {result}"
        )

    def test_run_prediction_is_binary(self, sample_customer):
        result = self._get_first_prediction(sample_customer)
        assert result["churn_prediction"] in (0, 1), (
            f"churn_prediction must be 0 or 1, got {result['churn_prediction']}"
        )

    def test_run_response_has_probability_fields(self, sample_customer):
        result = self._get_first_prediction(sample_customer)
        assert "probability_churn"    in result, "Missing 'probability_churn'"
        assert "probability_no_churn" in result, "Missing 'probability_no_churn'"

    def test_run_probabilities_sum_to_one(self, sample_customer):
        result = self._get_first_prediction(sample_customer)
        total  = result["probability_churn"] + result["probability_no_churn"]
        assert abs(total - 1.0) < 1e-3, f"Probabilities must sum to 1.0, got {total}"

    def test_run_label_consistent_with_prediction(self, sample_customer):
        result     = self._get_first_prediction(sample_customer)
        prediction = result["churn_prediction"]
        label      = result.get("churn_prediction_label", "")
        if prediction == 1:
            assert label == "Churn",    f"Label mismatch for prediction=1: got '{label}'"
        else:
            assert label == "No Churn", f"Label mismatch for prediction=0: got '{label}'"



# 4. Modelpromoter.py — quality gate logic tests


class TestQualityGate:

    # Must exactly match the values in Modelpromoter.py
    # If you change MIN_ROC_AUC or MIN_RECALL there, update these too
    MIN_ROC_AUC = 0.80
    MIN_RECALL  = 0.65

    def _check_gate(self, roc_auc: float, recall: float):
        """Mirrors the gate logic from promote_model() without any Azure SDK calls."""
        failures = []
        if roc_auc < self.MIN_ROC_AUC:
            failures.append(f"ROC-AUC {roc_auc:.4f} < minimum {self.MIN_ROC_AUC}")
        if recall < self.MIN_RECALL:
            failures.append(f"Recall {recall:.4f} < minimum {self.MIN_RECALL}")
        return failures

    def test_model_passes_gate_when_both_metrics_above_threshold(self):
        assert self._check_gate(roc_auc=0.85, recall=0.70) == []

    def test_model_passes_gate_at_exact_thresholds(self):
        """Gate uses strict < so a model at exactly 0.80/0.65 must pass."""
        assert self._check_gate(roc_auc=0.80, recall=0.65) == []

    def test_model_rejected_when_roc_auc_below_threshold(self):
        failures = self._check_gate(roc_auc=0.79, recall=0.70)
        assert len(failures) == 1 and "ROC-AUC" in failures[0]

    def test_model_rejected_when_recall_below_threshold(self):
        failures = self._check_gate(roc_auc=0.85, recall=0.64)
        assert len(failures) == 1 and "Recall" in failures[0]

    def test_model_rejected_when_both_metrics_fail(self):
        assert len(self._check_gate(roc_auc=0.70, recall=0.50)) == 2

    def test_gate_fails_for_zero_metrics(self):
        """A random/majority-class model with AUC=0.5, recall=0 must be rejected."""
        assert len(self._check_gate(roc_auc=0.50, recall=0.00)) == 2

    def test_recall_threshold_is_stricter_than_auc(self):
        """
        A model with great AUC but low recall must still be rejected.
        Business logic: missing a churner costs more than a false alarm.
        High AUC does not compensate for missing churners.
        """
        failures = self._check_gate(roc_auc=0.92, recall=0.60)
        assert len(failures) == 1 and "Recall" in failures[0]

    def test_run_info_json_has_required_keys(self, tmp_path):
        """
        run_info.json written by train.py must contain all keys that
        Modelpromoter.py reads. A missing key causes a KeyError mid-pipeline.
        """
        run_info = {
            "run_id":     "abc123",
            "model_name": "customer-churn-model",
            "model_type": "high_recall_lr",
            "roc_auc":    0.83,
            "recall":     0.71,
            "precision":  0.75,
            "f1":         0.73,
        }
        path = tmp_path / "run_info.json"
        path.write_text(json.dumps(run_info))
        loaded = json.loads(path.read_text())
        for key in ("run_id", "model_name", "roc_auc", "recall"):
            assert key in loaded, f"run_info.json missing required key: '{key}'"