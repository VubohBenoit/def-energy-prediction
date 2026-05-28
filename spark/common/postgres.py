# =======================================================================
# **************    Projet : EDF Prediction Platform       **************
# **************    Version : 1.0.0                        **************
# =======================================================================
#
# spark/common/postgres.py — Write JDBC to PostgreSQL (data warehouse).
# =======================================================================

from __future__ import annotations

import logging
from pyspark.sql import DataFrame
from spark.common.config import PG_PASS, PG_URL, PG_USER

logger = logging.getLogger(__name__)

def write_to_postgres(
    df: DataFrame,
    table: str,
    *,
    columns: list[str] | None = None,
    mode: str = "append",
    batch_size: int = 5000,
) -> None:
    """Write a Spark DataFrame to PostgreSQL via JDBC.

    Parameters
    ----------
    df
        DataFrame source.
    table
        Target table (e.g. ``dw.fact_consumption_silver``).
    columns
        Subset of columns to write; by default all columns of the DF.
    mode
        Spark write mode (``append``, ``overwrite``, etc.).
    batch_size
        JDBC batch size.
    """
    if not PG_URL or not PG_USER:
        logger.warning("PostgreSQL connection not configured — skipping step")
        return

    selected = columns or df.columns
    available = [c for c in selected if c in df.columns]
    if not available:
        logger.warning("No columns to load into %s", table)
        return

    logger.info("Loading PostgreSQL -> %s (%d columns)", table, len(available))
    writer = (
        df.select(*available)
        .write
        .format("jdbc")
        .option("url", PG_URL)
        .option("dbtable", table)
        .option("user", PG_USER)
        .option("password", PG_PASS)
        .option("driver", "org.postgresql.Driver")
        .option("batchsize", str(batch_size))
    )
    # overwrite JDBC = DROP TABLE by default — incompatible with DW views (v_annual_summary…)
    if mode == "overwrite":
        writer = writer.option("truncate", "true")
    writer.mode(mode).save()
