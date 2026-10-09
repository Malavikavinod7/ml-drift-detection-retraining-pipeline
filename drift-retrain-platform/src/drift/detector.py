"""
Drift detection utilities.

This module compares the transformed reference dataset with an incoming
transformed batch using the Kolmogorov-Smirnov (KS) statistical test.

Since categorical variables are already One-Hot Encoded during preprocessing,
every feature is treated as numerical.
"""

import pandas as pd
from scipy.stats import ks_2samp


def get_drift_severity(statistic):
    """Assign drift severity based on KS statistic."""
    if statistic >= 0.30:
        return "High"
    elif statistic >= 0.15:
        return "Medium"
    else:
        return "Low"


def compute_drift_report(
    X_reference,
    X_batch,
    metadata=None,
    model_config=None
):
    """Compare reference and batch feature distributions."""
    if X_reference is None or X_batch is None:
        return pd.DataFrame(
            columns=[
                "feature",
                "ks_statistic",
                "p_value",
                "severity",
                "drift_detected"
            ]
        )

    if len(X_reference) == 0 or len(X_batch) == 0:
        return pd.DataFrame(
            columns=[
                "feature",
                "ks_statistic",
                "p_value",
                "severity",
                "drift_detected"
            ]
        )

    metadata = metadata or {}
    drift_threshold = metadata.get("drift_threshold", 0.05)

    if isinstance(X_reference, pd.DataFrame):
        reference_df = X_reference.copy()
    else:
        reference_df = pd.DataFrame(X_reference)

    if isinstance(X_batch, pd.DataFrame):
        batch_df = X_batch.copy()
    else:
        batch_df = pd.DataFrame(X_batch)

    if list(reference_df.columns) != list(batch_df.columns):
        raise ValueError(
            "Reference features and batch features do not match."
        )

    rows = []

    for feature in reference_df.columns:
        ref_col = reference_df[feature].values
        batch_col = batch_df[feature].values

        statistic, p_value = ks_2samp(ref_col, batch_col)

        rows.append({
            "feature": feature,
            "ks_statistic": round(float(statistic), 4),
            "p_value": round(float(p_value), 6),
            "severity": get_drift_severity(statistic),
            "drift_detected": bool(p_value < drift_threshold)
        })

    drift_report = pd.DataFrame(rows)

    drift_report = drift_report.sort_values(
        by="ks_statistic",
        ascending=False
    ).reset_index(drop=True)

    total_features = len(drift_report)
    drifted_features = int(drift_report["drift_detected"].sum())
    drift_percentage = (drifted_features / total_features) * 100 if total_features else 0

    print("\n" + "=" * 70)
    print("DRIFT DETECTION REPORT")
    print("=" * 70)

    print(
        drift_report[
            [
                "feature",
                "ks_statistic",
                "p_value",
                "severity",
                "drift_detected"
            ]
        ].to_string(index=False)
    )

    print("\n" + "=" * 70)
    print(f"Total Features      : {total_features}")
    print(f"Drifted Features    : {drifted_features}")
    print(f"Drift Percentage    : {drift_percentage:.2f}%")
    print("=" * 70 + "\n")

    return drift_report
