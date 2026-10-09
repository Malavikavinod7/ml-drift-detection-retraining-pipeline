# Automated ML Drift Detection and Retraining Pipeline

An end-to-end Machine Learning monitoring pipeline that continuously detects data drift, evaluates model performance, and automatically retrains and promotes better-performing models using a Champion–Challenger strategy.

---

## Overview

Machine learning models deployed in production gradually lose performance as incoming data changes over time. This phenomenon, known as **data drift**, causes prediction quality to degrade even though the model continues making predictions.

This project demonstrates a production-style ML pipeline that:

- Trains a baseline customer churn prediction model
- Processes incoming production batches
- Detects feature drift
- Automatically triggers retraining when drift exceeds a threshold
- Compares multiple candidate models
- Promotes the best-performing model
- Logs predictions, metrics, and drift reports

The project is designed to simulate how production ML systems are monitored and maintained.

---

## Problem Statement

Suppose a customer churn model is trained using historical customer data.

Over time:

- Customer behavior changes
- Contract preferences shift
- Monthly charges increase
- Service usage patterns evolve

Although the deployed model continues making predictions, its accuracy slowly decreases because the incoming data distribution no longer matches the original training data.

This project solves that problem by continuously monitoring production data and automatically updating the model whenever necessary.

---

# System Architecture

```
Reference Dataset
        │
        ▼
Dynamic Preprocessing
        │
        ▼
Train Baseline Model
        │
        ▼
Deploy Model
        │
        ▼
Incoming Batch
        │
        ▼
Feature Transformation
        │
        ▼
Prediction
        │
        ▼
Prediction Logging
        │
        ▼
Drift Detection
        │
        ├───────────────┐
        │               │
     No Drift        Drift Detected
        │               │
        ▼               ▼
 Continue         Retraining Pipeline
                        │
                        ▼
            Train Candidate Models
                        │
                        ▼
        Champion–Challenger Comparison
                        │
                        ▼
          Promote Better Performing Model
```

---

# Features

## Dynamic Preprocessing

- Schema-driven preprocessing
- Missing value handling
- Numerical scaling
- One-hot encoding
- Feature alignment across batches

---

## Model Training

The training pipeline supports multiple models:

- Random Forest
- Logistic Regression
- Decision Tree

The pipeline automatically evaluates all candidate models and selects the one with the highest F1 score.

---

## Automatic Model Selection

Instead of hardcoding one algorithm, every retraining cycle evaluates:

- Random Forest
- Logistic Regression
- Decision Tree

The best-performing model is selected automatically.

Example:

```
Random Forest          F1 = 0.345

Logistic Regression    F1 = 0.368

Decision Tree          F1 = 0.411

Selected Model:
Decision Tree
```

---

## Drift Detection

The pipeline continuously compares:

- Reference training data
- Incoming production batches

Supported methods:

- Kolmogorov–Smirnov (KS) Test
- Population Stability Index (PSI)

Each feature is assigned:

- Drift score
- Severity level
- Drift status

Example:

| Feature | Severity | Drift |
|----------|----------|-------|
| Monthly Charges | High | ✅ |
| Contract | Medium | ✅ |
| Tenure | Low | ✅ |

---

## Retraining Policy

Retraining is triggered when the percentage of drifted features exceeds the configured threshold.

Example:

```
Drifted Features : 9

Total Features : 31

Drift Percentage : 29%

Decision:

Retraining Triggered
```

---

## Champion–Challenger Strategy

When retraining is triggered:

1. Multiple candidate models are trained.
2. Each model is evaluated.
3. The best model is selected.
4. The candidate model is compared against the current production model.
5. The candidate is promoted only if it improves the primary evaluation metric.

---

## Logging

The pipeline automatically stores:

### Predictions

```
reports/predictions/
```

### Drift Reports

```
reports/drift_reports/
```

### Evaluation Metrics

```
reports/metrics/
```

---

# Project Structure

```
drift-retrain-platform/

├── config/
│
├── data/
│
├── models/
│
├── reports/
│
├── scripts/
│
├── src/
│   ├── preprocessing/
│   ├── models/
│   ├── drift/
│   ├── retraining/
│   ├── ingestion/
│   ├── logging_utils/
│   └── main.py
│
└── requirements.txt
```

---

# Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- SciPy
- JSON Configuration
- Git
- GitHub

---

# Dataset

The project uses a synthetic customer churn dataset containing realistic telecom customer information.

Features include:

- Age
- Gender
- Region
- Contract Type
- Internet Service
- Monthly Charges
- Total Charges
- Payment Method
- Support Calls
- Payment Delay
- Usage
- Customer Tenure

The synthetic data generator can intentionally inject drift into production batches to simulate real-world deployment scenarios.

---

# Quickstart & Setup Guide

### 1. Environment Setup

```bash
cd drift-retrain-platform

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Synthetic Dataset

```bash
python scripts/generate_synthetic_customer_churn.py
```

### 3. Run Pipeline & QA Suite

```bash
# Set PYTHONPATH to src directory
# Windows PowerShell:
$env:PYTHONPATH="src"; python src/main.py

# Linux/macOS:
PYTHONPATH=src python src/main.py
```

### 4. Running Tests & Quality Verification

```bash
# Run full unit & edge-case test suite
$env:PYTHONPATH="src"; pytest

# Run Flake8 linting
$env:PYTHONPATH="src"; flake8 src tests test_batch_manager.py test_config.py test_dataset_loader.py test_feature_alignment.py test_preprocessor.py test_schema.py

# Run Mypy static type checker
$env:PYTHONPATH="src"; mypy src
```


---

# Sample Output

```
Processing Batch 004...

Drift Percentage : 29%

Retraining Triggered

Training Candidate Models...

Random Forest          F1 = 0.345

Logistic Regression    F1 = 0.368

Decision Tree          F1 = 0.411

Selected Model:

Decision Tree

Candidate Model Promoted
```

---

# Evaluation Metrics

The pipeline evaluates:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

The primary metric used for model promotion is configurable.

---

# Future Improvements

- MLflow model registry
- FastAPI prediction service
- Prefect orchestration
- Evidently AI monitoring dashboard
- Streamlit monitoring UI
- Docker deployment
- CI/CD integration
- Model explainability using SHAP

---

# Learning Outcomes

This project demonstrates:

### Machine Learning

- Classification
- Feature Engineering
- Model Evaluation
- Automatic Model Selection

### Data Engineering

- Data Ingestion
- Batch Processing
- Logging
- Configuration-driven Pipelines

### MLOps

- Drift Detection
- Retraining Automation
- Champion–Challenger Deployment
- Production Monitoring

---

# Contributors

This project was developed as a collaborative effort.

### AI/ML Responsibilities

- Model training
- Automatic model selection
- Drift detection
- Retraining policy
- Champion–Challenger model promotion
- Evaluation pipeline

### Data Engineering Responsibilities

- Data ingestion
- Batch management
- Dynamic preprocessing
- Configuration management
- Logging infrastructure
- Synthetic data generation

---

## License

This project is intended for educational and portfolio purposes.

