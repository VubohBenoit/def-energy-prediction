# =======================================================================
# **************    Projet : EDF Energy Prediction         **************
# **************    Version : 1.0.0                        **************
# =======================================================================
#
# spark/jobs/gold_to_model.py — Gold to model transformations.
# =======================================================================

from __future__ import annotations

import json
import sys
from typing import Any

from spark.common.config import MODEL_PATH, SILVER_PATH
from spark.common.job_runner import configure_job_logging, get_job_spark, utc_now
from spark.ml.constants import CANDIDATE_FEATURES, TRAIN_RATIO
from spark.ml.metrics_store import persist_metrics_to_postgres
from spark.ml.report_artifacts import (
    build_split_summary,
    compute_rf_learning_curve,
    export_best_predictions,
    save_ml_report_artifacts,
)
from spark.ml.training import (
    load_training_data,
    materialize_splits,
    save_best_model,
    time_based_split,
    train_and_compare,
)

logger = configure_job_logging("gold_to_model")


def run(
    silver_path: str | None = None,
    model_path: str | None = None,
    persist_pg: bool = True,
) -> dict[str, Any]:
    """Gold to model transformations."""
    silver_path = silver_path or SILVER_PATH
    model_path = model_path or MODEL_PATH

    started_at = utc_now()
    run_id = started_at.strftime("%Y%m%d_%H%M%S")
    logger.info("═" * 78)
    logger.info("  EDF ML -> Model (run_id=%s)", run_id)
    logger.info("═" * 78)

    spark = get_job_spark("EDF_Gold_to_Model")
    df = load_training_data(spark, silver_path)

    feature_cols = [c for c in CANDIDATE_FEATURES if c in df.columns]

    train_df, test_df = time_based_split(df, train_ratio=TRAIN_RATIO)
    train_df, test_df, n_train, n_test = materialize_splits(
        spark, train_df, test_df, run_id
    )
    spark.catalog.clearCache()

    results, saved_paths = train_and_compare(
        train_df, test_df, feature_cols, model_path
    )

    best_name = save_best_model(saved_paths, results, model_path)

    logger.info("Computing ML report artifacts (learning curve, predictions)...")
    learning_curve = compute_rf_learning_curve(train_df, test_df, feature_cols)
    predictions = export_best_predictions(saved_paths, results, test_df, best_name)
    split_summary = build_split_summary(train_df, test_df, n_train, n_test)
    try:
        save_ml_report_artifacts(
            model_path,
            predictions=predictions,
            learning_curve=learning_curve,
            split_summary=split_summary,
            best_model=best_name,
            run_id=run_id,
        )
    except Exception as exc:
        logger.error("Save ML report artifacts failed: %s", exc)

    if persist_pg:
        try:
            persist_metrics_to_postgres(results, run_id=run_id)
        except Exception as exc:
            logger.error("Persist metrics Postgres failed: %s", exc)

    try:
        from spark.common.object_storage import sync_ml_report_artifacts, sync_model_artifacts

        n_synced = sync_model_artifacts(model_path)
        n_report = sync_ml_report_artifacts()
        logger.info(
            "Models synchronized MinIO -> local: %d file(s), report artifacts: %d",
            n_synced,
            n_report,
        )
    except Exception as exc:
        logger.warning("Sync models MinIO -> local failed: %s", exc)

    serializable = results
    best_metrics = serializable.get(best_name, {})
    duration_s = (utc_now() - started_at).total_seconds()

    summary = {
        "status": "success",
        "run_id": run_id,
        "duration_s": round(duration_s, 2),
        "best_model": best_name,
        "best_metrics": best_metrics,
        "n_train": n_train,
        "n_test": n_test,
        "models": serializable,
        "n_features": len(feature_cols),
    }
    logger.info("Pipeline ML completed -> best=%s", best_name)
    logger.info(json.dumps(serializable, indent=2, default=str))
    return summary


if __name__ == "__main__":
    cli_silver = sys.argv[1] if len(sys.argv) > 1 else SILVER_PATH
    cli_model = sys.argv[2] if len(sys.argv) > 2 else MODEL_PATH
    run(cli_silver, cli_model)
