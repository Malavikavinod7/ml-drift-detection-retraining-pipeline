"""Logging utilities for model metrics."""

import threading
from pathlib import Path
from datetime import datetime
import pandas as pd

_metrics_lock = threading.Lock()


def log_metrics(metrics_dict, model_name, output_dir="reports/metrics", filename="metrics_log.csv"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_entry = {
        "model_name": model_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    log_entry.update(metrics_dict)

    output_path = output_dir / filename

    with _metrics_lock:
        if output_path.exists():
            existing_df = pd.read_csv(output_path)
            updated_df = pd.concat([existing_df, pd.DataFrame([log_entry])], ignore_index=True)
        else:
            updated_df = pd.DataFrame([log_entry])

        updated_df.to_csv(output_path, index=False)

    return output_path
