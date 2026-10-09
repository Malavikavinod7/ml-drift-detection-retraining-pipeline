# Contributing Guidelines

Thank you for your interest in contributing to the **Automated ML Drift Detection and Retraining Pipeline**!

---

## 1. Environment Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Malavikavinod7/ml-drift-detection-retraining-pipeline.git
   cd ml-drift-detection-retraining-pipeline/drift-retrain-platform
   ```

2. **Set up virtual environment**:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Generate synthetic data**:
   ```bash
   python scripts/generate_synthetic_customer_churn.py
   ```

---

## 2. Running Pipeline & Verification Commands

- **Run Main Pipeline**:
  ```bash
  # Windows PowerShell:
  $env:PYTHONPATH="src"; python src/main.py
  # Linux/macOS:
  PYTHONPATH=src python src/main.py
  ```

- **Run Pytest Test Suite**:
  ```bash
  $env:PYTHONPATH="src"; pytest
  ```

- **Run Flake8 Code Linting**:
  ```bash
  $env:PYTHONPATH="src"; flake8 src tests test_batch_manager.py test_config.py test_dataset_loader.py test_feature_alignment.py test_preprocessor.py test_schema.py
  ```

- **Run Mypy Type Check**:
  ```bash
  $env:PYTHONPATH="src"; mypy src
  ```

---

## 3. Pull Request Checklist

Before submitting a Pull Request:
- [ ] All new code is covered by pytest unit tests in `tests/`.
- [ ] Pytest test suite passes cleanly (`pytest`).
- [ ] Flake8 code linting passes with 0 errors (`flake8`).
- [ ] Mypy type check passes with 0 errors (`mypy src`).
- [ ] Main pipeline executes without errors (`python src/main.py`).
- [ ] No temporary files or secrets are committed.
