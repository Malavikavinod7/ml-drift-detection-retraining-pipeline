import concurrent.futures
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from logging_utils.metrics_logger import log_metrics  # noqa: E402
from logging_utils.prediction_logger import log_predictions  # noqa: E402


def test_concurrent_metrics_logging(tmp_path):
    output_dir = tmp_path / "metrics"

    def log_task(i):
        metrics = {"accuracy": 0.8 + (i % 10) * 0.01, "f1": 0.75}
        log_metrics(metrics, f"model_{i}", output_dir=output_dir, filename="concurrent_metrics.csv")

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(log_task, i) for i in range(30)]
        concurrent.futures.wait(futures)

    log_file = output_dir / "concurrent_metrics.csv"
    assert log_file.exists()
    df = pd.read_csv(log_file)
    assert len(df) == 30


def test_concurrent_prediction_logging(tmp_path):
    output_dir = tmp_path / "predictions"

    def log_task(i):
        df = pd.DataFrame({"cust_id": [f"C{i:03d}"]})
        preds = np.array([1])
        probs = np.array([0.9])
        log_predictions(df, preds, probs, f"batch_{i}.csv", output_dir=output_dir)

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(log_task, i) for i in range(25)]
        concurrent.futures.wait(futures)

    assert len(list(output_dir.glob("*.csv"))) == 25
