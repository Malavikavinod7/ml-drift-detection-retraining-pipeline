from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent / "src"))

from core.config_loader import load_all_configs  # noqa: E402
from core.metadata_builder import build_metadata  # noqa: E402
from ingestion.dataset_loader import load_reference_data  # noqa: E402
from preprocessing.preprocessor import DynamicPreprocessor  # noqa: E402


def test_preprocessor():
    schema, pipeline_config, model_config = load_all_configs()
    metadata = build_metadata(schema, pipeline_config, model_config)

    reference_df = load_reference_data(metadata, schema)

    preprocessor = DynamicPreprocessor(metadata, pipeline_config)
    X_ref, y_ref = preprocessor.fit_transform(reference_df, include_target=True)

    assert X_ref is not None
    assert X_ref.shape[0] == len(reference_df)
    assert len(preprocessor.get_feature_names()) > 0
