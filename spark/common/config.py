# =======================================================================
# **************    Projet : EDF Prediction Platform       **************
# **************    Version : 1.0.0                        **************
# =======================================================================
#
# spark/common/config.py — Centralized configuration for Spark jobs (environment variables).
# =======================================================================

from __future__ import annotations

import os


def _env(key: str, default: str) -> str:
    """Returns the environment variable value or the default value."""
    return os.getenv(key, default)

# MinIO / S3A configuration.
MINIO_ENDPOINT: str = _env("MINIO_ENDPOINT", "http://minio:9000")
MINIO_ACCESS: str = _env(
    "MINIO_ACCESS_KEY",
    _env("MINIO_ROOT_USER", "edfadmin"),
)
MINIO_SECRET: str = _env(
    "MINIO_SECRET_KEY",
    _env("MINIO_ROOT_PASSWORD", "edfpassword123"),
)

# Spark (jobs executed on the standalone cluster) configuration.
SPARK_MASTER_URL: str = _env("SPARK_MASTER_URL", "spark://spark-master:7077")
SPARK_PACKAGES: str = _env(
    "SPARK_PACKAGES",
    "org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262,"
    "org.postgresql:postgresql:42.7.4",
)
SPARK_DRIVER_MEMORY: str = _env("SPARK_DRIVER_MEMORY", "")
SPARK_EXECUTOR_MEMORY: str = _env("SPARK_EXECUTOR_MEMORY", "")

# Bronze path configuration.
BRONZE_PATH: str = _env("BRONZE_PATH", "s3a://edf-bronze/rte/")
SILVER_PATH: str = _env("SILVER_PATH", "s3a://edf-silver/rte/")
GOLD_PATH: str = _env("GOLD_PATH", "s3a://edf-gold/rte/")
MODELS_BUCKET: str = _env("MODELS_BUCKET", "models")
MODEL_PATH: str = _env(
    "ML_MODEL_PATH",
    _env("MODEL_PATH", f"s3a://{MODELS_BUCKET}/rte/"),
)
MODEL_LOCAL_PATH: str = _env(
    "MODEL_LOCAL_PATH",
    "/opt/spark-data/models/rte",
)
REPORT_EDA_BUCKET: str = _env("REPORT_EDA_BUCKET", "rapport-eda")
REPORT_EDA_S3_PREFIX: str = _env("REPORT_EDA_S3_PREFIX", "report/")
REPORT_EDA_LOCAL: str = _env("REPORT_EDA_LOCAL", "data/eda/report")
DATA_DIR: str = _env("DATA_DIR", "/opt/airflow/data")

# Merge Bronze streaming (Kafka) into Silver when true.
BRONZE_INCLUDE_STREAMING: bool = _env(
    "BRONZE_INCLUDE_STREAMING", "true"
).lower() in ("1", "true", "yes")

# PostgreSQL (JDBC) configuration.
PG_URL: str = _env("POSTGRES_URL", "jdbc:postgresql://postgres:5432/edf_dw")
PG_USER: str = _env("POSTGRES_USER", "edf")
PG_PASS: str = _env("POSTGRES_PASSWORD", "edf123")


def bronze_raw_path(base: str | None = None) -> str:
    """Parquet batch XLS path (``.../raw/``) — excludes tempo and streaming."""
    explicit = os.getenv("BRONZE_RAW_PATH")
    if explicit:
        return explicit if explicit.endswith("/") else f"{explicit}/"
    root = (base or BRONZE_PATH).rstrip("/")
    return f"{root}/raw/"
