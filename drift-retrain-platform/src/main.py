"""Orchestration entry point for the drift detection + retraining pipeline."""

from pathlib import Path

from core.config_loader import load_all_configs
from core.metadata_builder import build_metadata
from core.schema_validator import validate_reference_schema, validate_batch_schema

from ingestion.dataset_loader import load_reference_data, load_batch_data
from ingestion.batch_manager import get_sorted_batches

from preprocessing.preprocessor import DynamicPreprocessor
from preprocessing.feature_alignment import align_features

from logging_utils.prediction_logger import log_predictions
from logging_utils.drift_logger import log_drift_report
from logging_utils.metrics_logger import log_metrics

from models.train import train_baseline_model, retrain_model
from models.predict import predict_with_model
from models.evaluate import evaluate_model
from drift.detector import compute_drift_report
from retraining.policy import should_retrain


def main():
    print("Starting dynamic drift detection + retraining pipeline...")

    schema, pipeline_config, model_config = load_all_configs()
    metadata = build_metadata(schema, pipeline_config, model_config)

    reference_df = load_reference_data(metadata, schema)
    validate_reference_schema(reference_df, schema)

    preprocessor = DynamicPreprocessor(metadata, pipeline_config)
    X_ref, y_ref = preprocessor.fit_transform(reference_df, include_target=True)

    current_model = train_baseline_model(
        X_ref=X_ref,
        y_ref=y_ref,
        metadata=metadata,
        model_config=model_config
    )

    baseline_metrics = evaluate_model(
        model=current_model,
        X=X_ref,
        y=y_ref,
        metadata=metadata
    )

    log_metrics(
        metrics_dict=baseline_metrics,
        model_name="baseline_model"
    )

    training_feature_names = preprocessor.get_feature_names()
    batch_files = get_sorted_batches(metadata, pipeline_config)

    for batch_path in batch_files:
        batch_name = Path(batch_path).name
        print(f"\nProcessing batch: {batch_name}")

        batch_df = load_batch_data(batch_path, metadata, schema)
        validate_batch_schema(batch_df, schema)

        include_target = metadata.get("label_available_in_batches", True)
        X_batch, y_batch = preprocessor.transform(batch_df, include_target=include_target)

        X_batch = align_features(X_batch, training_feature_names)

        predictions, probabilities = predict_with_model(current_model, X_batch)

        prediction_log_path = log_predictions(
            batch_df=batch_df,
            predictions=predictions,
            probabilities=probabilities,
            batch_name=batch_name
        )
        print(f"Predictions logged to: {prediction_log_path}")

        drift_report_df = compute_drift_report(
            X_reference=X_ref,
            X_batch=X_batch,
            metadata=metadata,
            model_config=model_config
        )

        drift_log_path = log_drift_report(
            drift_report_df=drift_report_df,
            batch_name=batch_name
        )
        print(f"Drift report logged to: {drift_log_path}")

        retrain_flag = should_retrain(
            drift_report_df=drift_report_df,
            model_config=model_config
        )

        if not retrain_flag:
            print("No retraining triggered for this batch.")
            continue

        print("Retraining triggered.")

        if y_batch is None:
            print("Batch labels not available. Skipping retraining for this batch.")
            continue

        champion_batch_metrics = evaluate_model(
            model=current_model,
            X=X_batch,
            y=y_batch,
            metadata=metadata
        )

        candidate_model = retrain_model(
            X_ref=X_ref,
            y_ref=y_ref,
            X_batch=X_batch,
            y_batch=y_batch,
            metadata=metadata,
            model_config=model_config
        )

        candidate_metrics = evaluate_model(
            model=candidate_model,
            X=X_batch,
            y=y_batch,
            metadata=metadata
        )

        log_metrics(
            metrics_dict=candidate_metrics,
            model_name=f"candidate_model_{Path(batch_name).stem}"
        )

        primary_metric = metadata.get("primary_metric", "f1")
        champion_score = champion_batch_metrics.get(primary_metric, 0.0)
        candidate_score = candidate_metrics.get(primary_metric, 0.0)

        print(f"\n--- Champion-Challenger Promotion Evaluation ({batch_name}) ---")
        print(f"Current Champion {primary_metric.upper()} Score : {champion_score:.4f}")
        print(f"Candidate Model  {primary_metric.upper()} Score : {candidate_score:.4f}")

        if candidate_score > champion_score:
            current_model = candidate_model
            baseline_metrics = candidate_metrics
            print(f"Result: Candidate model PROMOTED to Champion for batch {batch_name}.\n")
        else:
            print(f"Result: Candidate model REJECTED. Current Champion retained for batch {batch_name}.\n")

    print("\nPipeline execution completed.")


if __name__ == "__main__":
    main()
