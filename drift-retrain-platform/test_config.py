from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent / "src"))

from core.config_loader import load_all_configs  # noqa: E402
from core.metadata_builder import build_metadata  # noqa: E402


def test_config_loading_and_metadata():
    schema, pipeline_config, model_config = load_all_configs()
    assert isinstance(schema, dict)
    assert isinstance(pipeline_config, dict)
    assert isinstance(model_config, dict)
    assert "target_column" in schema

    metadata = build_metadata(schema=schema, pipeline_config=pipeline_config, model_config=model_config)
    assert isinstance(metadata, dict)
    assert metadata["target_column"] == schema["target_column"]
