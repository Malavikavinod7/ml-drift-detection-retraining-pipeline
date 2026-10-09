from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from core.schema_validator import validate_reference_schema, validate_batch_schema  # noqa: E402
from preprocessing.feature_alignment import align_features  # noqa: E402
from preprocessing.preprocessor import DynamicPreprocessor  # noqa: E402


def test_schema_validator_hostile_strings():
    schema = {
        "target_column": "churn",
        "feature_columns": ["monthly_charges", "partner<script>alert(1)</script>"],
        "numerical_columns": ["monthly_charges"],
        "categorical_columns": ["partner<script>alert(1)</script>"],
        "drop_columns": []
    }

    df = pd.DataFrame({
        "monthly_charges": [50.0, 100.0],
        "partner<script>alert(1)</script>": ["Yes'; DROP TABLE users; --", "No"],
        "churn": [0, 1]
    })

    assert validate_reference_schema(df, schema) is True
    assert validate_batch_schema(df, schema) is True


def test_schema_validator_duplicate_columns():
    schema = {"feature_columns": ["a"], "target_column": "target"}
    df = pd.DataFrame([[1, 2, 3]], columns=["a", "a", "target"])
    with pytest.raises(ValueError, match="Duplicate columns found"):
        validate_reference_schema(df, schema)


def test_preprocessor_unicode_and_huge_numbers():
    metadata = {
        "target_column": "target",
        "feature_columns": ["num_val", "cat_val_🚀"],
        "numerical_columns": ["num_val"],
        "categorical_columns": ["cat_val_🚀"],
        "drop_columns": []
    }
    pipeline_config = {
        "preprocessing": {
            "missing_value_strategy": {"numerical": "median", "categorical": "most_frequent"},
            "encoding": {"categorical": "onehot"},
            "scaling": {"numerical": "standard"}
        }
    }

    df = pd.DataFrame({
        "num_val": [0, -1e9, 1e9, np.nan],
        "cat_val_🚀": ["العربية", "日本語", "Español", None],
        "target": [0, 1, 0, 1]
    })

    preprocessor = DynamicPreprocessor(metadata, pipeline_config)
    X_trans, y = preprocessor.fit_transform(df, include_target=True)

    assert X_trans.shape[0] == 4
    assert y.tolist() == [0, 1, 0, 1]
    assert any("cat_val_🚀" in col for col in X_trans.columns)


def test_feature_alignment_extra_and_missing_columns():
    training_cols = ["a", "b", "c"]
    X_batch = pd.DataFrame({
        "a": [1],
        "extra_col_💩": [999]
    })

    aligned = align_features(X_batch, training_cols)
    assert aligned.columns.tolist() == training_cols
    assert aligned["b"].iloc[0] == 0
    assert aligned["c"].iloc[0] == 0
    assert "extra_col_💩" not in aligned.columns
