from pathlib import Path
import sys
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent / "src"))

from core.config_loader import load_all_configs  # noqa: E402
from core.metadata_builder import build_metadata  # noqa: E402
from core.schema_validator import validate_reference_schema, validate_batch_schema  # noqa: E402


def test_schema_validation():
    schema, pipeline_config, model_config = load_all_configs()
    metadata = build_metadata(schema, pipeline_config, model_config)

    reference_df = pd.read_csv(metadata["reference_data_path"])
    assert validate_reference_schema(reference_df, schema) is True

    batch_path = Path(metadata["incoming_data_dir"]) / "batch_001.csv"
    batch_df = pd.read_csv(batch_path)
    assert validate_batch_schema(batch_df, schema) is True
