# =======================================================================
# **************    Projet : EDF Prediction Platform       **************
# **************    Version : 1.0.0                        **************
# =======================================================================
#
# spark/common/eda_report.py — EDA report generation (shared: Airflow, Makefile CLI).
# =======================================================================

from __future__ import annotations

import logging
import os
import socket
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd

from rte_pipeline.parsing.xls import iter_xls_file
from spark.common.config import DATA_DIR, REPORT_EDA_LOCAL
from spark.common.eda_style import (
    COLORS,
    add_report_footer,
    apply_production_style,
    plot_consumption_series,
    plot_missing_bars,
    plot_ml_benchmark_report,
    plot_quality_distribution,
    prepare_time_series,
    save_report_figure,
)
from spark.common.object_storage import (
    REPORT_EDA_BUCKET,
    persist_report_file,
    remove_report_from_s3,
)

logger = logging.getLogger(__name__)

DATA_CHARTS = (
    "01_consommation.png",
    "02_volume_mensuel.png",
    "03_qualite_donnees.png",
)
ML_CHART = "04_performance_ml.png"
ML_PENDING = "04_performance_ml.pending"

KEY_NUMERIC = [
    "consumption_mw",
    "nuclear_mw",
    "wind_mw",
    "solar_mw",
    "hydro_mw",
    "gas_mw",
    "coal_mw",
    "fuel_mw",
    "bioenergy_mw",
    "co2_rate",
]


def postgres_dsn() -> str:
    """PostgreSQL DSN — container (postgres) or host CLI (localhost fallback)."""
    url = os.getenv("POSTGRES_CONN", "postgresql://edf:edf123@postgres:5432/edf_dw")
    override = os.getenv("EDA_PG_DSN", "").strip()
    if override:
        return override
    if "@postgres:" not in url:
        return url
    try:
        socket.gethostbyname("postgres")
    except OSError:
        return url.replace("@postgres:", "@localhost:")
    return url


def raw_data_dir() -> Path:
    """Directory of RTE XLS source files."""
    return Path(os.getenv("DATA_DIR", DATA_DIR))


def report_output_dir() -> Path:
    """Local directory for PNG artifacts (mirrored to MinIO)."""
    configured = os.getenv("REPORT_EDA_LOCAL", REPORT_EDA_LOCAL)
    path = Path(configured)
    if not path.is_absolute():
        root = Path(os.getenv("EDF_PROJECT_ROOT", Path.cwd()))
        path = root / configured
    path.mkdir(parents=True, exist_ok=True)
    return path


def report_destination_label(out: Path | None = None) -> str:
    out = out or report_output_dir()
    return f"{out}/ + s3a://{REPORT_EDA_BUCKET}/"


def load_bronze_sample() -> pd.DataFrame:
    """Load RTE consumption rows from XLS (excludes Tempo calendar files)."""
    raw_dir = raw_data_dir()
    files = sorted(
        f
        for f in raw_dir.glob("eCO2mix*.xls")
        if "tempo" not in f.name.lower()
    )
    if not files:
        raise FileNotFoundError(f"No RTE XLS files in {raw_dir}")

    rows: list[dict[str, Any]] = []
    for path in files:
        for record in iter_xls_file(str(path)):
            rows.append(record)

    df = pd.DataFrame(rows)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    for col in KEY_NUMERIC:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["datetime"]).sort_values("datetime")


def compute_quality_score(df: pd.DataFrame) -> pd.Series:
    cols = [c for c in KEY_NUMERIC if c in df.columns]
    if not cols:
        return pd.Series(dtype=float)
    return df[cols].notna().sum(axis=1) / len(cols)


def build_monthly(df: pd.DataFrame) -> pd.DataFrame:
    monthly = (
        df.set_index("datetime")
        .resample("MS")["consumption_mw"]
        .mean()
        .dropna()
        .reset_index()
    )
    hours = monthly["datetime"].dt.days_in_month * 24
    monthly["consumption_total_twh"] = monthly["consumption_mw"] * hours / 1_000_000
    monthly["label"] = monthly["datetime"].dt.strftime("%Y-%m")
    return monthly


def load_ml_metrics() -> tuple[pd.DataFrame, str | None, str | None] | None:
    try:
        import psycopg2
    except ImportError:
        logger.warning("psycopg2 unavailable — ML chart skipped")
        return None

    run_sql = """
        SELECT run_id, MAX(trained_at) AS trained_at
        FROM etl.model_metrics
        GROUP BY run_id
        ORDER BY trained_at DESC NULLS LAST
        LIMIT 1
    """
    metrics_sql = """
        SELECT model_name, rmse, mae, mape_pct, r2, train_time_s, trained_at, run_id
        FROM etl.model_metrics
        WHERE run_id = %s
        ORDER BY rmse ASC
    """
    fallback_sql = """
        WITH latest AS (
            SELECT MAX(trained_at) AS ts FROM etl.model_metrics
        )
        SELECT model_name, rmse, mae, mape_pct, r2, train_time_s, trained_at, run_id
        FROM etl.model_metrics
        WHERE trained_at >= (SELECT ts - INTERVAL '5 minutes' FROM latest)
        ORDER BY rmse ASC
    """
    try:
        conn = psycopg2.connect(postgres_dsn())
        cur = conn.cursor()
        cur.execute(run_sql)
        run_row = cur.fetchone()
        if not run_row:
            cur.close()
            conn.close()
            return None

        run_id, trained_at = run_row[0], run_row[1]
        if run_id:
            cur.execute(metrics_sql, (run_id,))
        else:
            cur.execute(fallback_sql)
        rows = cur.fetchall()
        cols = [desc[0] for desc in cur.description]
        cur.close()
        conn.close()

        if not rows:
            return None

        df = pd.DataFrame(rows, columns=cols)
        ts_label = (
            pd.Timestamp(trained_at).strftime("%d/%m/%Y %H:%M UTC")
            if trained_at
            else None
        )
        return df, str(run_id) if run_id else None, ts_label
    except Exception as exc:
        logger.warning("Unable to load ML metrics: %s", exc)
        return None


def _persist_chart(local_path: str) -> str | None:
    uri = persist_report_file(local_path)
    if uri:
        logger.info("MinIO %s", uri)
    return uri


def chart_consumption(df: pd.DataFrame, out: Path) -> str:
    ts = prepare_time_series(df, "datetime", "consumption_mw")
    fig, ax = plt.subplots(figsize=(11, 5.5))
    plot_consumption_series(
        ax,
        ts,
        "datetime",
        "consumption_mw",
        title="Consommation électrique nationale",
        subtitle="Série horaire RTE éco2mix — tendance lissée sur 7 jours",
        ma_days=7,
    )
    add_report_footer(fig)
    return save_report_figure(fig, str(out / DATA_CHARTS[0]))


def chart_monthly(monthly: pd.DataFrame, out: Path) -> str:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    if monthly.empty:
        ax.text(
            0.5,
            0.5,
            "Données mensuelles indisponibles",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    else:
        m = monthly.sort_values("datetime")
        ax.bar(
            m["label"],
            m["consumption_total_twh"],
            color=COLORS["secondary"],
            width=0.72,
            alpha=0.85,
        )
        ax.plot(
            m["label"],
            m["consumption_total_twh"],
            color=COLORS["primary"],
            marker="o",
            markersize=4,
            linewidth=1.2,
        )
        ax.set_title("Volume mensuel de consommation", fontweight="bold", pad=10)
        ax.set_xlabel("Mois")
        ax.set_ylabel("Énergie (TWh)")
        step = max(1, len(m) // 12)
        ax.set_xticks(range(0, len(m), step))
        ax.set_xticklabels(m["label"].iloc[::step], rotation=45, ha="right")
        ax.grid(True, axis="y", alpha=0.5)
    add_report_footer(fig)
    return save_report_figure(fig, str(out / DATA_CHARTS[1]))


def chart_quality(df: pd.DataFrame, out: Path) -> str:
    missing = (df.isna().mean() * 100).sort_values(ascending=False)
    scores = compute_quality_score(df)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    plot_missing_bars(axes[0], missing, title="Variables les plus incomplètes", top_n=8)
    plot_quality_distribution(
        axes[1], scores, title="Score de qualité des enregistrements"
    )
    fig.suptitle("Qualité des données source", fontweight="bold", fontsize=13, y=1.02)
    add_report_footer(fig)
    plt.tight_layout()
    return save_report_figure(fig, str(out / DATA_CHARTS[2]))


def chart_ml_metrics(
    metrics: pd.DataFrame,
    out: Path,
    *,
    run_id: str | None,
    trained_at: str | None,
) -> str:
    fig = plt.figure(figsize=(14, 9))
    plot_ml_benchmark_report(fig, metrics, run_id=run_id, trained_at=trained_at)
    add_report_footer(fig)
    plt.tight_layout(rect=(0, 0.02, 1, 0.92))
    pending = out / ML_PENDING
    if pending.exists():
        pending.unlink()
    return save_report_figure(fig, str(out / ML_CHART))


def mark_ml_pending(out: Path | None = None) -> None:
    out = out or report_output_dir()
    ml_png = out / ML_CHART
    if ml_png.exists():
        ml_png.unlink()
    remove_report_from_s3(ML_CHART)
    pending = out / ML_PENDING
    pending.write_text(
        "Rapport ML en attente.\n"
        "Exécutez le job ML (DAG edf_ml_pipeline ou tâche ml.run_gold_to_model), "
        "puis regénérez le graphique 04.\n",
        encoding="utf-8",
    )


def generate_data_charts(out: Path | None = None) -> list[str]:
    out = out or report_output_dir()
    apply_production_style()

    logger.info("Loading RTE from %s", raw_data_dir())
    df = load_bronze_sample()
    monthly = build_monthly(df)

    paths = [
        chart_consumption(df, out),
        chart_monthly(monthly, out),
        chart_quality(df, out),
    ]
    logger.info("%d data chart(s) -> %s", len(paths), report_destination_label(out))
    for path in paths:
        _persist_chart(path)
    return paths


def generate_ml_chart(
    out: Path | None = None,
    *,
    pending_if_missing: bool = True,
) -> str | None:
    out = out or report_output_dir()
    apply_production_style()

    loaded = load_ml_metrics()
    if loaded is None:
        if pending_if_missing:
            mark_ml_pending(out)
            logger.info("ML report pending — no metrics in etl.model_metrics")
        return None

    metrics, run_id, trained_at = loaded
    path = chart_ml_metrics(metrics, out, run_id=run_id, trained_at=trained_at)
    logger.info(
        "ML report -> %s (best model: %s)",
        Path(path).name,
        metrics.iloc[0]["model_name"],
    )
    _persist_chart(path)
    return path
