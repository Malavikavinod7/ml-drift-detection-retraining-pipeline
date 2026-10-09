from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from drift.detector import compute_drift_report, get_drift_severity  # noqa: E402


def test_get_drift_severity():
    assert get_drift_severity(0.35) == "High"
    assert get_drift_severity(0.20) == "Medium"
    assert get_drift_severity(0.05) == "Low"


def test_compute_drift_report_none_input():
    report = compute_drift_report(None, None)
    assert isinstance(report, pd.DataFrame)
    assert report.empty
    assert "feature" in report.columns


def test_compute_drift_report_empty_dataframes():
    X_ref = pd.DataFrame(columns=["feat1", "feat2"])
    X_batch = pd.DataFrame(columns=["feat1", "feat2"])
    report = compute_drift_report(X_ref, X_batch)
    assert isinstance(report, pd.DataFrame)
    assert report.empty or len(report) == 0


def test_compute_drift_report_mismatched_columns():
    X_ref = pd.DataFrame({"a": [1, 2, 3]})
    X_batch = pd.DataFrame({"b": [1, 2, 3]})
    with pytest.raises(ValueError, match="do not match"):
        compute_drift_report(X_ref, X_batch)


def test_compute_drift_report_valid_drift():
    np.random.seed(42)
    X_ref = pd.DataFrame({
        "age": np.random.normal(30, 5, 100),
        "income": np.random.normal(50000, 10000, 100)
    })
    X_batch = pd.DataFrame({
        "age": np.random.normal(30, 5, 100),
        "income": np.random.normal(90000, 10000, 100)  # Drifted
    })

    metadata = {"drift_threshold": 0.05}
    report = compute_drift_report(X_ref, X_batch, metadata=metadata)
    assert len(report) == 2
    assert "income" in report["feature"].values
    income_row = report[report["feature"] == "income"].iloc[0]
    assert bool(income_row["drift_detected"]) is True
    assert income_row["severity"] in ["Medium", "High"]
