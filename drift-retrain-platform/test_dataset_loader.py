from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent / "src"))

from core.config_loader import load_all_configs  # noqa: E402
from core.metadata_builder import build_metadata  # noqa: E402
from ingestion.dataset_loader import load_reference_data, load_batch_data  # noqa: E402


def test_dataset_loader():
    schema, pipeline_config, model_config = load_all_configs()
    metadata = build_metadata(schema, pipeline_config, model_config)

    reference_df = load_reference_data(metadata, schema)
    assert reference_df is not None
    assert len(reference_df) > 0

    batch_file = Path(metadata["incoming_data_dir"]) / "batch_001.csv"
    batch_df = load_batch_data(str(batch_file), metadata, schema)
    assert batch_df is not None
    assert len(batch_df) > 0
