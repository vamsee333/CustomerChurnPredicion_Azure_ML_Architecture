<h2> Customer Churn Prediction — End-to-End MLOps on Azure ML </h3>

<h4> Production-grade machine learning pipeline featuring automated training, quality-gated model promotion, real-time REST inference, and continuous data drift monitoring which are all orchestrated on Azure Machine Learning. </h4>


I worked on an end-to-end customer churn prediction project where the goal was to identify customers likely to leave so the business can act early and reduce revenue loss.

What I focused on wasn’t just building a model, but actually taking it through the full lifecycle starting from raw data all the way to a deployed and monitored solution.

* Step 1: Problem + Data

I started with a telecom churn dataset with around 7,000 records. It had a mix of demographic, billing, and usage features.

One important thing I noticed early was class imbalance where far more customers stayed than churned, so I knew accuracy alone wouldn’t be a good metric. I focused more on recall, because missing a churner is more costly for the business.

* Step 2: Data Preparation

I built a preprocessing pipeline where I:

removed data leakage columns like churn score and satisfaction score
encoded categorical features properly
handled binary and ordinal variables
and saved the final feature list

Saving the feature columns was important because it ensures training and inference always stay consistent, which is a common issue in real deployments.

* Step 3: Model Training

I trained multiple models:

logistic regression
random forest
and a class-balanced logistic regression

I tracked everything using MLflow — metrics, parameters, and artifacts, so every experiment was reproducible.
In the end, I chose the high-recall logistic regression as the main model because it aligned better with the business goal.

* Step 4: Quality Gate

Instead of manually checking results, I built a model promotion step.

The model only gets marked as production if it meets:
ROC-AUC ≥ 0.80
Recall ≥ 0.65

If it doesn’t meet these thresholds, the pipeline just stops.

* Step 5: Deployment

I deployed the model using an Azure ML managed online endpoint.

It exposes a REST API
accepts JSON input
returns predictions with probabilities

I also structured it in a champion setup, so it can easily support challenger models later for A/B testing.

Step 6: Monitoring

After deployment, I set up data drift monitoring.

The system:

compares live request data with training data
checks if distributions are changing
and sends alerts if drift crosses a threshold

This is important because models degrade over time as real-world data changes.

* Key Design Decisions

1. Feature column serialisation: feature_columns.json is written by preprocessing and read by every downstream step. This single source of truth guarantees that training and inference always use the same column order — a common source of silent production bugs.

2. Quality gate as a pipeline step: Rather than a post-hoc manual check, model promotion is a mandatory pipeline step. If the quality gate fails, the pipeline errors and no downstream steps run. This makes the gate impossible to skip accidentally.

3. class_weight='balanced' for churn: Churn datasets are inherently imbalanced. Using balanced class weights ensures the model does not learn to simply predict "no churn" for everyone to achieve high accuracy.

4. MLflow model signature: Recording input/output schema at training time means Azure ML validates every deployment against this contract, catching feature mismatch bugs at deploy time rather than silently at inference time.

5.Drift baseline built from preprocessed data: The baseline for drift monitoring is built by running preprocess_data() on train.csv which is not the raw CSV. This means drift is measured in the same feature space the model operates in, not in the raw input space, giving more meaningful signals.

6. Champion naming convention: The first deployment is explicitly named champion to signal that the architecture supports A/B testing. Adding a challenger deployment and splitting traffic is a one-line config change.