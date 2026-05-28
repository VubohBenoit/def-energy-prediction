# =======================================================================
# **************    Projet : EDF Prediction Platform       **************
# **************    Version : 1.0.0                        **************
# =======================================================================
#
# spark/jobs/silver_to_gold.py — Silver to Gold transformations.
# =======================================================================

from __future__ import annotations

import sys
from typing import Any

from spark.common.config import GOLD_PATH, SILVER_PATH
from spark.common.job_runner import configure_job_logging, get_job_spark, utc_now
from spark.common.postgres import write_to_postgres
from spark.transform.gold import (
    compute_daily_aggregates,
    compute_monthly_aggregates,
    read_silver,
    write_gold_parquet,
)

logger = configure_job_logging("silver_to_gold")


def run(
    silver_path: str | None = None,
    gold_path: str | None = None,
    load_postgres: bool = True,
) -> dict[str, Any]:
    """Silver to Gold transformations."""
    silver_path = silver_path or SILVER_PATH
    gold_path = gold_path or GOLD_PATH

    started_at = utc_now()
    logger.info("═" * 78)
    logger.info("  EDF ETL -> Gold")
    logger.info("  Silver : %s", silver_path)
    logger.info("  Gold   : %s", gold_path)
    logger.info("═" * 78)

    spark = get_job_spark("EDF_Silver_to_Gold")
    silver_df = read_silver(spark, silver_path)
    silver_df.cache()

    daily_df = compute_daily_aggregates(silver_df)
    daily_df.cache()
    monthly_df = compute_monthly_aggregates(daily_df)

    write_gold_parquet(daily_df, f"{gold_path.rstrip('/')}/daily/", ["year", "month"])
    write_gold_parquet(monthly_df, f"{gold_path.rstrip('/')}/monthly/", ["year"])

    if load_postgres:
        write_to_postgres(daily_df, "dw.agg_daily", mode="overwrite")
        monthly_pg_cols = [
            "year",
            "month",
            "season",
            "consumption_mean_mw",
            "consumption_total_twh",
            "consumption_yoy_pct",
            "renewable_pct",
            "co2_mean_gkwh",
            "trading_days",
        ]
        write_to_postgres(
            monthly_df,
            "dw.agg_monthly",
            columns=monthly_pg_cols,
            mode="overwrite",
        )

    daily_rows = daily_df.count()
    monthly_rows = monthly_df.count()

    silver_df.unpersist()
    daily_df.unpersist()

    duration_s = (utc_now() - started_at).total_seconds()
    logger.info("Gold completed in %.1fs", duration_s)
    return {
        "status": "success",
        "daily_rows": daily_rows,
        "monthly_rows": monthly_rows,
        "rows_written": daily_rows + monthly_rows,
        "duration_s": round(duration_s, 2),
    }


if __name__ == "__main__":
    cli_silver = sys.argv[1] if len(sys.argv) > 1 else SILVER_PATH
    cli_gold = sys.argv[2] if len(sys.argv) > 2 else GOLD_PATH
    run(cli_silver, cli_gold)
