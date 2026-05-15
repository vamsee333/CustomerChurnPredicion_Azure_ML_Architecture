# Customer Churn Prediction — End-to-End MLOps on Azure ML

**Production-grade machine learning pipeline featuring automated training, quality-gated model promotion, real-time REST inference, continuous data drift monitoring, and a full CI/CD pipeline — all orchestrated on Azure Machine Learning and GitHub Actions.**

---

## Project Overview

This project builds a complete customer churn prediction system for a telecom business, taking a raw dataset of ~7,000 records through every stage of the ML lifecycle: data preparation, model training, automated quality gating, deployment to a live REST endpoint, production monitoring, and now automated CI/CD with GitHub Actions.

The goal was never just to build a model. The goal was to build a system that can be maintained, monitored, and improved in production without manual intervention.

---

## Architecture

```
Raw Data (train.csv)
        │
        ▼
 ┌─────────────────┐
 │  preprocessing  │  Feature engineering, encoding, leakage removal
 │   .py           │
 └────────┬────────┘
          │
          ▼
 ┌─────────────────┐
 │    train.py     │  3 models trained, MLflow tracking, registry registration
 └────────┬────────┘
          │
          ▼
 ┌─────────────────┐
 │ Modelpromoter   │  Quality gate: ROC-AUC ≥ 0.80, Recall ≥ 0.65
 │     .py         │  Tags production version in Azure ML Registry
 └────────┬────────┘
          │
          ▼
 ┌─────────────────┐
 │   predict.py    │  Batch predictions written to output folder
 └────────┬────────┘
          │
          ▼
 ┌─────────────────────────────────┐
 │  Deploy_endpoint.py             │  Managed Online Endpoint (champion pattern)
 │  churn-predictions-endpoint     │  REST API — real-time inference
 └────────┬────────────────────────┘
          │
          ▼
 ┌─────────────────┐
 │ setup_drift_    │  Daily monitor — Wasserstein + Jensen-Shannon
 │ monitor.py      │  Email alerts on feature distribution shift
 └─────────────────┘

GitHub Actions (CI/CD)
 ├── ci.yml  → PR gate: pytest + coverage (runs on every pull request)
 └── cd.yml  → Auto-deploy: Pipeline + Endpoint update (runs on merge to master)
```

---

## Phase 1 — ML Pipeline (Complete)

### Step 1: Problem and Data

I started with a telecom churn dataset with around 7,000 records. It contains a mix of demographic, billing, contract, and usage features across 52 columns.

The key challenge identified early was class imbalance: far more customers stayed than churned. This meant accuracy was a misleading metric. I focused on recall, because missing a churner (false negative) is more costly to the business than triggering an unnecessary retention offer (false positive).

### Step 2: Data Preparation — `preprocessing.py`

The preprocessing pipeline handles:
- Removing data leakage columns: `Churn Score`, `Satisfaction Score`, `Customer Status`, `Churn Category`, `Churn Reason`
- Dropping geographic columns irrelevant to the model: `Lat Long`, `City`, `Latitude`, `Longitude`, `Zip Code`
- Ordinal encoding for `Contract` (Month-to-Month=0, One Year=1, Two Year=2)
- Binary encoding for `Gender` and all Yes/No columns
- One-hot encoding for `Internet Type`, `Payment Method`, `Offer`
- Saving `feature_columns.json` to guarantee training/inference column consistency

**Key design decision:** `feature_columns.json` is the single source of truth for column order. Every downstream step (training, batch prediction, real-time scoring) reads this file. This eliminates the most common silent production bug in ML systems.

### Step 3: Model Training — `train.py`

Three models trained and tracked in a single MLflow run:
- Logistic Regression (standard)
- Random Forest (class-balanced)
- High-Recall Logistic Regression (`class_weight='balanced'`) — selected as primary

MLflow logs all parameters, per-model metrics, the model signature (input/output schema), and an input example. The model is registered to Azure ML Model Registry as `customer-churn-model` on every run, automatically creating a new version.

**Why high-recall logistic regression:** The `class_weight='balanced'` setting adjusts the loss function to penalise missing churners more heavily. The business cost is asymmetric: a missed churner means lost revenue; a false alarm means a cheap retention offer.

### Step 4: Quality Gate — `Modelpromoter.py`

The quality gate is a mandatory pipeline step, not a post-hoc check. It reads `run_info.json` written by `train.py` and enforces:
- **ROC-AUC ≥ 0.80** — minimum discriminative power
- **Recall ≥ 0.65** — minimum sensitivity to churners

If either threshold is missed, the pipeline errors and stops. The production model in the registry is never touched. This makes the gate impossible to accidentally bypass.

If both pass, the latest model version is tagged `stage: production` and all previous production versions are tagged `stage: archived`.

### Step 5: Batch Predictions — `predict.py`

Batch scoring runs as the final pipeline step, writing predictions and churn probabilities to `predictions.csv`. The `build_inference_row()` function is shared with `Score.py` so training and inference always apply identical transformations.

### Step 6: Deployment — `Deploy_endpoint.py`

The model is deployed to an Azure ML Managed Online Endpoint named `churn-predictions-endpoint` with a deployment named `champion`.

- Auth mode: API key
- Instance type: `Standard_DS2_v2`
- Data collection enabled on both inputs and outputs (feeds the drift monitor)
- Liveness and readiness probes configured for zero-downtime deployments
- 100% traffic routed to `champion`

The `champion` naming is deliberate architectural intent: adding a challenger deployment and splitting traffic is a single config change (Phase 4).

**Endpoint URL:** `https://churn-predictions-endpoint.eastus2.inference.ml.azure.com/score`

**Sample request:**
```json
{
  "input_data": [{
    "Age": 25,
    "Contract": "Month-to-month",
    "Monthly Charge": 89.50,
    "Tenure in Months": 3,
    "Internet Type": "Fiber Optic",
    "Payment Method": "Credit Card (automatic)",
    "Offer": "No Offer"
  }]
}
```

**Sample response:**
```json
{
  "predictions": [{
    "churn_prediction": 1,
    "churn_prediction_label": "Churn",
    "probability_no_churn": 0.1051,
    "probability_churn": 0.8949
  }]
}
```

### Step 7: Drift Monitoring — `setup_drift_monitor.py`

A daily monitor runs at 19:00 (cron: `0 19 * * *`) and compares the last 7 days of live endpoint traffic against the training baseline.

- **Baseline:** built by running `preprocess_data()` on `train.csv` — measured in the same feature space the model operates in, not raw input space
- **Numerical features:** Normalised Wasserstein Distance (threshold: 0.10)
- **Categorical features:** Jensen-Shannon Distance (threshold: 0.10)
- **Top 20 features** monitored by importance
- **Email alerts** sent when any feature exceeds threshold

---

## Phase 3 — CI/CD with GitHub Actions (Complete)

### CI Pipeline — `ci.yml`

Triggers on every pull request targeting `master`. Blocks the PR if tests fail.

**What it does:**
1. Checks out the repository
2. Sets up Python 3.10
3. Installs ML and test dependencies (no Azure SDK needed for unit tests)
4. Runs `pytest tests/test_pipeline.py` with coverage reporting
5. Uploads the coverage XML as a downloadable artifact (available even on failure)

**Key design:** The coverage failure threshold is set at 20% for initial setup and will be raised as the test suite matures. The important thing is that the gate exists and all tests must pass.

### CD Pipeline — `cd.yml`

Triggers on every push to `master` when source files change. Does not redeploy on documentation-only changes.

**Two-job structure:**

**Job 1 — `submit-pipeline`:**
1. Authenticates to Azure using a Service Principal (`AZURE_CREDENTIALS` secret)
2. Validates all required secrets are present before running anything
3. Runs `Pipeline.py` — submits the 4-step Azure ML pipeline job
4. Polls job status every 60 seconds until completion (max 1 hour)
5. Fails the workflow if the pipeline fails, preventing endpoint update

**Job 2 — `update-endpoint`** (only runs if Job 1 succeeds):
1. Writes `.azureml/config.json` from GitHub Secrets (no file committed to repo)
2. Validates the config JSON before using it
3. Runs `Deploy_endpoint.py` to update the champion deployment
4. Runs a smoke test: sends one real HTTP request to the live endpoint and verifies the response contains expected keys

**Path filtering:** CD only triggers when these files change:
```
src/**.py
env/conda.yml
pipelines/Pipeline.py
azure/Deploy_endpoint.py
.github/workflows/cd.yml
```

**Required GitHub Secrets:**
| Secret | Value |
|--------|-------|
| `AZURE_CREDENTIALS` | JSON from `az ad sp create-for-rbac --sdk-auth` |
| `AZURE_SUBSCRIPTION_ID` | Your Azure subscription ID |
| `AZURE_RESOURCE_GROUP` | e.g. `RG1` |
| `AZURE_ML_WORKSPACE` | e.g. `Vamsee_AzureML` |
| `CHURN_ENDPOINT_KEY` | Primary key from the managed online endpoint |

**Creating the Service Principal (run once):**
```bash
az ad sp create-for-rbac \
  --name "churn-pipeline-sp" \
  --role Contributor \
  --scopes /subscriptions/<SUBSCRIPTION_ID>/resourceGroups/RG1 \
  --sdk-auth
```
Copy the full JSON output and paste it as the `AZURE_CREDENTIALS` secret.

---

## Key Design Decisions

**1. Feature column serialisation**
`feature_columns.json` is written by `preprocessing.py` and read by every downstream step. This single source of truth guarantees training and inference always use the same column order — a common source of silent production bugs.

**2. Quality gate as a pipeline step**
Model promotion is a mandatory pipeline step that errors and stops execution if thresholds are missed. It cannot be skipped accidentally. The current production model is never touched on a failed run.

**3. `class_weight='balanced'` for churn**
Churn datasets are inherently imbalanced. Balanced class weights ensure the model does not learn to predict "no churn" for everyone to game accuracy. The quality gate's Recall threshold reinforces this at promotion time.

**4. MLflow model signature**
Recording input/output schema at training time means Azure ML validates every deployment against this contract, catching feature mismatch bugs at deploy time rather than silently at inference time.

**5. Drift baseline built from preprocessed data**
The monitoring baseline is built by running `preprocess_data()` on `train.csv`, not the raw CSV. This means drift is measured in the same feature space the model operates in, giving more meaningful and actionable signals.

**6. Champion naming convention**
The first deployment is explicitly named `champion` to signal that the architecture supports A/B testing from day one. Adding a challenger deployment and splitting traffic is a one-line config change (Phase 4).

**7. CI/CD path filtering**
The CD pipeline only triggers on source code and config changes, not documentation changes. This prevents unnecessary redeployments and keeps the pipeline fast.

**8. Service Principal over personal credentials**
CI/CD uses a Service Principal with Contributor scope on the resource group only — not the full subscription. This follows the principle of least privilege and means credentials can be rotated without touching the pipeline.

---

## File Structure

```
.
├── .github/
│   └── workflows/
│       ├── ci.yml              # PR gate: tests + coverage
│       └── cd.yml              # Auto-deploy on merge to master
├── src/
│   ├── preprocessing.py        # Feature engineering pipeline
│   ├── train.py                # Model training + MLflow tracking
│   ├── Modelpromoter.py        # Quality gate + registry promotion
│   ├── predict.py              # Batch scoring + inference helper
│   └── Score.py                # Online endpoint scoring script
├── pipelines/
│   └── Pipeline.py             # Azure ML pipeline definition
├── azure/
│   └── Deploy_endpoint.py      # Managed Online Endpoint deployment
├── monitoring/
│   └── setup_drift_monitor.py  # Daily drift monitor setup
├── env/
│   └── conda.yml               # Pipeline environment dependencies
├── tests/
│   └── test_pipeline.py        # Unit tests (CI gate)
├── notebooks/
│   └── InvokeService.ipynb     # Endpoint testing + data collection
└── README.md
```

---

## Project Roadmap


| Phase 1 | Complete | ML pipeline, training, quality gate, deployment, drift monitoring |
| Phase 2 | Blocked on drift results | Auto-retraining: `retrain_trigger.py`, `data_ingest.py` | (As I am waiting for the Serverless compute to be assigned)
| Phase 3 | Complete | CI/CD: `ci.yml`, `cd.yml`, `tests/test_pipeline.py` |

---

## MLflow Experiment Tracking

Every training run logs:
- Model type and hyperparameters
- Train/test split configuration
- Per-model metrics: ROC-AUC, Precision, Recall, F1
- Primary model metrics at top level (for run comparison in Studio)
- Model signature and input example
- Feature column artifact

Runs are visible in Azure ML Studio under the `churn-prediction` experiment.

---

## Monitoring the Endpoint

After deployment, send test requests via `InvokeService.ipynb`. The notebook includes 20 varied customer profiles covering high, medium, and low churn risk — sufficient to populate the drift monitor's production window.

Check data collection in Azure ML Studio: **Endpoints → churn-predictions-endpoint → champion → Data collection**.