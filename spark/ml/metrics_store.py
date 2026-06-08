# =======================================================================
# **************    Projet : EDF Energy Prediction         **************
# **************    Version : 1.0.0                        **************
# =======================================================================
#
# spark/ml/metrics_store.py — Persist ML metrics to PostgreSQL (psycopg2 — compatible mode local Spark).
# =======================================================================

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any
import psycopg2
from psycopg2.extras import execute_batch

# Get PostgreSQL connection
def _pg_conn() -> str:
    return os.getenv(
        "POSTGRES_CONN",
        "postgresql://edf:edf123@postgres:5432/edf_dw",
    )

logger = logging.getLogger("gold_to_model")


def _normalize_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        "rmse": round(float(metrics["rmse"]), 1),
        "mae": round(float(metrics["mae"]), 1),
        "mape": round(float(metrics["mape"]), 2),
        "r2": round(float(metrics["r2"]), 4),
        "train_time_s": round(float(metrics["train_time_s"]), 1),
    }


_MODEL_METRICS_DDL: tuple[str, ...] = (
    "CREATE SCHEMA IF NOT EXISTS etl",
    """
    CREATE TABLE IF NOT EXISTS etl.model_metrics (
        id BIGSERIAL PRIMARY KEY,
        run_id VARCHAR(50),
        model_name VARCHAR(100) NOT NULL,
        rmse DOUBLE PRECISION,
        mae DOUBLE PRECISION,
        mape_pct DOUBLE PRECISION,
        r2 DOUBLE PRECISION,
        train_time_s DOUBLE PRECISION,
        trained_at TIMESTAMP DEFAULT NOW()
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_model_metrics_trained_at "
    "ON etl.model_metrics (trained_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_model_metrics_model "
    "ON etl.model_metrics (model_name)",
    """
    CREATE OR REPLACE VIEW etl.v_model_metrics_latest AS
    WITH latest AS (
        SELECT run_id
        FROM etl.model_metrics
        ORDER BY trained_at DESC NULLS LAST
        LIMIT 1
    ),
    ranked AS (
        SELECT
            m.*,
            ROW_NUMBER() OVER (ORDER BY rmse ASC NULLS LAST) AS rang,
            MIN(rmse) OVER () AS best_rmse
        FROM etl.model_metrics m
        WHERE m.run_id = (SELECT run_id FROM latest)
    )
    SELECT
        rang,
        CASE model_name
            WHEN 'LinearRegression' THEN 'Régression linéaire'
            WHEN 'DecisionTree' THEN 'Arbre de décision'
            WHEN 'RandomForest' THEN 'Forêt aléatoire'
            WHEN 'GradientBoosting' THEN 'Gradient Boosting'
            ELSE model_name
        END AS algorithme,
        ROUND(rmse::numeric, 1)::double precision AS rmse_mw,
        ROUND(mae::numeric, 1)::double precision AS mae_mw,
        ROUND(mape_pct::numeric, 2)::double precision AS mape_pct,
        ROUND(r2::numeric, 4)::double precision AS r2,
        ROUND(train_time_s::numeric, 1)::double precision AS train_time_s,
        ROUND(
            (((rmse / NULLIF(best_rmse, 0)) - 1) * 100)::numeric,
            1
        )::double precision AS rmse_ecart_pct,
        (rang = 1) AS est_meilleur
    FROM ranked
    ORDER BY rang
    """,
)

_INSERT_SQL = """
    INSERT INTO etl.model_metrics
        (run_id, model_name, rmse, mae, mape_pct, r2, train_time_s, trained_at)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
"""

# Ensure model metrics table
def ensure_model_metrics_table() -> None:
    """Create schema/table if necessary (idempotent)."""
    conn = psycopg2.connect(_pg_conn())
    cur = conn.cursor()
    try:
        for ddl in _MODEL_METRICS_DDL:
            cur.execute(ddl.strip())
        conn.commit()
    finally:
        cur.close()
        conn.close()

# Persist metrics to PostgreSQL
def persist_metrics_to_postgres(
    results: dict[str, dict[str, Any]],
    run_id: str | None = None,
) -> None:
    """Insert metrics into ``etl.model_metrics``."""
    if not results:
        logger.warning("No metrics to persist")
        return

    ensure_model_metrics_table()
    now = datetime.now(timezone.utc)
    rid = run_id or now.strftime("%Y%m%d_%H%M%S")

    batch = []
    for name, raw in results.items():
        m = _normalize_metrics(raw)
        batch.append(
            (
                rid,
                name,
                m["rmse"],
                m["mae"],
                m["mape"],
                m["r2"],
                m["train_time_s"],
                now,
            )
        )

    conn = psycopg2.connect(_pg_conn())
    cur = conn.cursor()
    try:
        execute_batch(cur, _INSERT_SQL, batch)
        conn.commit()
        logger.info(
            "Persist metrics → etl.model_metrics (%d models, run_id=%s)",
            len(batch),
            rid,
        )
    finally:
        cur.close()
        conn.close()
