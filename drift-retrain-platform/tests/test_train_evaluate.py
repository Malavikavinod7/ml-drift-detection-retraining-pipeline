from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from models.evaluate import evaluate_model  # noqa: E402
from models.predict import predict_with_model  # noqa: E402
from models.train import train_baseline_model, retrain_model  # noqa: E402


def test_train_baseline_model_normal():
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(100, 4), columns=["f1", "f2", "f3", "f4"])
    y = pd.Series(np.random.choice([0, 1], size=100))

    model = train_baseline_model(X, y)
    assert model is not None
    preds, probs = predict_with_model(model, X)
    assert len(preds) == 100
    assert len(probs) == 100


def test_train_baseline_model_single_class():
    X = pd.DataFrame(np.random.randn(20, 4), columns=["f1", "f2", "f3", "f4"])
    y = pd.Series([1] * 20)  # Only 1 class

    # Should not crash with stratify error after bug fix
    model = train_baseline_model(X, y)
    assert model is not None


def test_evaluate_model_single_class_y():
    X = pd.DataFrame(np.random.randn(20, 4), columns=["f1", "f2", "f3", "f4"])
    y = pd.Series([1] * 20)

    model = train_baseline_model(X, y)
    metrics = evaluate_model(model, X, y)
    assert "accuracy" in metrics
    assert "f1" in metrics
    assert metrics["roc_auc"] is None or isinstance(metrics["roc_auc"], float)


def test_retrain_model_with_none_batch_labels():
    X_ref = pd.DataFrame(np.random.randn(50, 4), columns=["f1", "f2", "f3", "f4"])
    y_ref = pd.Series(np.random.choice([0, 1], size=50))
    X_batch = pd.DataFrame(np.random.randn(20, 4), columns=["f1", "f2", "f3", "f4"])

    model = retrain_model(X_ref, y_ref, X_batch, None)
    assert model is not None
