# =======================================================================
# **************    Projet : EDF Energy Prediction         **************
# **************    Version : 1.0.0                        **************
# =======================================================================
#
# spark/ml/report_artifacts.py — Artefacts ML pour rapports / dashboard Streamlit.
# =======================================================================

from __future__ import annotations

import gc
import json
import logging
from typing import Any

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.regression import RandomForestRegressor
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from spark.ml.constants import (
    LABEL_COL,
    ML_RF_MAX_DEPTH,
    ML_RF_NUM_TREES,
    RANDOM_SEED,
)
from spark.ml.training import build_feature_pipeline, evaluate_model

logger = logging.getLogger(__name__)

REPORT_SUBDIR = "_report"


def report_artifact_base(model_path: str) -> str:
    return f"{model_path.rstrip('/')}/{REPORT_SUBDIR}"


def compute_rf_learning_curve(
    train: DataFrame,
    test: DataFrame,
    feature_cols: list[str],
    fractions: tuple[float, ...] = (0.15, 0.3, 0.5, 0.65, 0.8, 1.0),
) -> DataFrame:
    """Courbe d'apprentissage RF : RMSE (MW) vs taille du jeu d'entraînement."""
    spark = train.sparkSession
    assembler = build_feature_pipeline(feature_cols)
    rf = RandomForestRegressor(
        featuresCol="features",
        labelCol=LABEL_COL,
        numTrees=ML_RF_NUM_TREES,
        maxDepth=ML_RF_MAX_DEPTH,
        seed=RANDOM_SEED,
    )

    train_ordered = train.orderBy("datetime")
    n_total = train_ordered.count()
    rows: list[dict[str, Any]] = []

    for frac in fractions:
        n_rows = max(500, int(n_total * frac))
        subset = train_ordered.limit(n_rows)
        pipeline = Pipeline(stages=[assembler, rf])
        model = pipeline.fit(subset)
        train_metrics = evaluate_model(model.transform(subset))
        val_metrics = evaluate_model(model.transform(test))
        rows.append(
            {
                "train_fraction": float(frac),
                "train_rows": int(n_rows),
                "rmse_train_mw": float(train_metrics["rmse"]),
                "rmse_validation_mw": float(val_metrics["rmse"]),
            }
        )
        logger.info(
            "   Learning curve RF — %.0f%% (%s rows) → train=%.1f MW | validation=%.1f MW",
            frac * 100,
            f"{n_rows:,}",
            train_metrics["rmse"],
            val_metrics["rmse"],
        )
        del model
        gc.collect()

    return spark.createDataFrame(rows)


def export_best_predictions(
    saved_paths: dict[str, str],
    results: dict[str, dict[str, Any]],
    test: DataFrame,
    best_name: str | None = None,
) -> DataFrame:
    """Prédictions du meilleur modèle sur le jeu de test."""
    if best_name is None:
        best_name = min(results.items(), key=lambda x: x[1]["rmse"])[0]
    source = saved_paths.get(best_name)
    if not source:
        raise ValueError(f"Missing saved model path for {best_name}")

    model = PipelineModel.load(source)
    return (
        model.transform(test)
        .select(
            F.col("datetime"),
            F.col(LABEL_COL).alias("actual_mw"),
            F.col("prediction").alias("predicted_mw"),
        )
        .orderBy("datetime")
    )


def build_split_summary(
    train: DataFrame,
    test: DataFrame,
    n_train: int,
    n_test: int,
) -> dict[str, Any]:
    """Répartition temporelle train / test pour les rapports."""
    train_bounds = (
        train.agg(F.min("datetime").alias("min_dt"), F.max("datetime").alias("max_dt"))
        .collect()[0]
    )
    test_bounds = (
        test.agg(F.min("datetime").alias("min_dt"), F.max("datetime").alias("max_dt"))
        .collect()[0]
    )
    return {
        "n_train": int(n_train),
        "n_test": int(n_test),
        "train_start": str(train_bounds["min_dt"]),
        "train_end": str(train_bounds["max_dt"]),
        "test_start": str(test_bounds["min_dt"]),
        "test_end": str(test_bounds["max_dt"]),
    }


def save_ml_report_artifacts(
    model_path: str,
    *,
    predictions: DataFrame,
    learning_curve: DataFrame,
    split_summary: dict[str, Any],
    best_model: str,
    run_id: str,
) -> str:
    """Persiste prédictions, courbe RF et métadonnées sous ``{model_path}/_report/``."""
    base = report_artifact_base(model_path)
    predictions.write.mode("overwrite").parquet(f"{base}/predictions.parquet")
    learning_curve.write.mode("overwrite").parquet(f"{base}/learning_curve.parquet")

    spark = SparkSession.getActiveSession()
    if spark is not None:
        meta_df = spark.createDataFrame(
            [
                {
                    "run_id": run_id,
                    "best_model": best_model,
                    "n_train": split_summary.get("n_train"),
                    "n_test": split_summary.get("n_test"),
                }
            ]
        )
        meta_df.write.mode("overwrite").parquet(f"{base}/run_meta.parquet")

    summary_path = f"{base}/split_summary.json"
    # Spark cannot write small JSON to S3 without Hadoop FS — use driver-side JSON string.
    payload = json.dumps(
        {"run_id": run_id, "best_model": best_model, **split_summary},
        indent=2,
        default=str,
    )
    jvm = spark._jvm if spark is not None else None
    if jvm is not None:
        hadoop_conf = spark._jsc.hadoopConfiguration()
        path = jvm.org.apache.hadoop.fs.Path(summary_path)
        fs = path.getFileSystem(hadoop_conf)
        out = fs.create(path, True)
        out.write(payload.encode("utf-8"))
        out.close()
    else:
        logger.warning("Spark session inactive — split_summary.json not written to %s", summary_path)

    logger.info("ML report artifacts → %s", base)
    return base
