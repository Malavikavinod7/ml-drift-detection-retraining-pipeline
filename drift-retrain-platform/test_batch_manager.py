from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent / "src"))

from core.config_loader import load_all_configs  # noqa: E402
from core.metadata_builder import build_metadata  # noqa: E402
from ingestion.batch_manager import get_sorted_batches  # noqa: E402


def test_get_sorted_batches():
    schema, pipeline_config, model_config = load_all_configs()
    metadata = build_metadata(schema, pipeline_config, model_config)
    batch_files = get_sorted_batches(metadata, pipeline_config)
    assert isinstance(batch_files, list)
    assert len(batch_files) > 0
    for file in batch_files:
        assert Path(file).exists()
