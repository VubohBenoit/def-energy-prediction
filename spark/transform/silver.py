# =======================================================================
# **************    Projet : EDF Energy Prediction         **************
# **************    Version : 1.0.0                        **************
# =======================================================================
#
# spark/transform/silver.py — Silver transformations
# =======================================================================

from __future__ import annotations

import logging
import math

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from pyspark.sql.window import Window

logger = logging.getLogger(__name__)

CONSO_MIN_MW: float = 10_000.0
CONSO_MAX_MW: float = 120_000.0

# Parquet Silver columns (includes ML features)
SILVER_OUTPUT_COLUMNS: list[str] = [
    "datetime",
    "consumption_mw", "forecast_j1_mw", "forecast_error_mw", "forecast_error_pct",
    "nuclear_mw", "wind_mw", "solar_mw", "hydro_mw",
    "gas_mw", "fuel_mw", "coal_mw", "bioenergy_mw",
    "co2_rate", "wind_onshore_mw", "wind_offshore_mw",
    "hydro_river_mw", "hydro_lake_mw", "hydro_step_mw",
    "hour", "day_of_week", "day_of_year", "week_of_year",
    "month", "year", "quarter", "season",
    "is_weekend", "is_peak_hour",
    "hour_sin", "hour_cos",
    "lag_1h_mw", "lag_24h_mw", "lag_168h_mw",
    "rolling_24h_mean", "rolling_24h_std",
    "renewable_share_pct", "nuclear_share_pct",
    "is_interpolated", "quality_score",
]

# Columns aligned with infra/postgres/schema_dw.sql (JDBC loading)
SILVER_POSTGRES_COLUMNS: list[str] = [
    "datetime",
    "consumption_mw", "forecast_j1_mw", "forecast_error_mw", "forecast_error_pct",
    "nuclear_mw", "wind_mw", "solar_mw", "hydro_mw",
    "gas_mw", "bioenergy_mw", "fuel_mw", "coal_mw",
    "wind_onshore_mw", "wind_offshore_mw",
    "physical_exchanges_mw", "co2_rate",
    "hour", "day_of_week", "day_of_year", "week_of_year",
    "month", "year", "quarter", "season",
    "is_weekend", "is_peak_hour",
    "lag_1h_mw", "lag_24h_mw", "lag_168h_mw",
    "rolling_24h_mean", "rolling_24h_std",
    "renewable_share_pct", "nuclear_share_pct",
    "is_interpolated", "quality_score",
]


def clean(df: DataFrame) -> DataFrame:
    """Business cleaning: datetime, deduplication, filtering, imputation."""
    logger.info("Cleaning Silver...")

    if "date" in df.columns and "time" in df.columns:
        df = df.withColumn(
            "datetime",
            F.to_timestamp(
                F.concat_ws(" ", F.col("date"), F.col("time")),
                "yyyy-MM-dd HH:mm",
            ),
        )
    df = df.filter(F.col("datetime").isNotNull())

    before = df.count()
    df = df.dropDuplicates(["datetime"])
    logger.info("   -> %s duplicates removed", f"{before - df.count():,}")

    df = df.filter(
        F.col("consumption_mw").isNull()
        | F.col("consumption_mw").between(CONSO_MIN_MW, CONSO_MAX_MW)
    )

    numeric_cols = [
        "consumption_mw", "forecast_j1_mw", "forecast_j_mw",
        "nuclear_mw", "wind_mw", "solar_mw", "hydro_mw",
        "gas_mw", "fuel_mw", "coal_mw", "bioenergy_mw",
        "co2_rate", "wind_onshore_mw", "wind_offshore_mw",
        "hydro_river_mw", "hydro_lake_mw", "hydro_step_mw",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df = df.withColumn(col, F.col(col).cast(DoubleType()))

    df = df.withColumn("is_interpolated", F.col("consumption_mw").isNull())
    df = df.withColumn("year", F.year("datetime"))

    w_year = Window.partitionBy("year").orderBy("datetime")
    window_fw = w_year.rowsBetween(Window.unboundedPreceding, 0)
    window_bw = w_year.rowsBetween(0, Window.unboundedFollowing)
    for col in ["consumption_mw", "nuclear_mw", "wind_mw", "solar_mw"]:
        if col in df.columns:
            df = df.withColumn(
                col,
                F.coalesce(
                    F.col(col),
                    F.last(F.col(col), ignorenulls=True).over(window_fw),
                    F.first(F.col(col), ignorenulls=True).over(window_bw),
                ),
            )

    logger.info("   -> %s lines after cleaning", f"{df.count():,}")
    return df


def engineer_features(df: DataFrame) -> DataFrame:
    """Feature engineering temporal, lags, quality score."""
    logger.info("Feature engineering Silver...")

    df = (
        df.withColumn("hour", F.hour("datetime"))
        .withColumn("day_of_week", F.dayofweek("datetime"))
        .withColumn("day_of_year", F.dayofyear("datetime"))
        .withColumn("week_of_year", F.weekofyear("datetime"))
        .withColumn("month", F.month("datetime"))
        .withColumn("year", F.year("datetime"))
        .withColumn("quarter", F.quarter("datetime"))
        .withColumn("is_weekend", F.dayofweek("datetime").isin([1, 7]))
        .withColumn(
            "is_peak_hour",
            (F.hour("datetime").between(8, 20))
            & (~F.dayofweek("datetime").isin([1, 7])),
        )
    )

    df = df.withColumn(
        "season",
        F.when(F.col("month").isin([12, 1, 2]), 0)
        .when(F.col("month").isin([3, 4, 5]), 1)
        .when(F.col("month").isin([6, 7, 8]), 2)
        .otherwise(3),
    )

    df = (
        df.withColumn("hour_sin", F.sin(F.lit(2 * math.pi) * F.col("hour") / 24))
        .withColumn("hour_cos", F.cos(F.lit(2 * math.pi) * F.col("hour") / 24))
    )

    w = Window.partitionBy("year").orderBy("datetime")
    df = (
        df.withColumn("lag_1h_mw", F.lag("consumption_mw", 2).over(w))
        .withColumn("lag_24h_mw", F.lag("consumption_mw", 48).over(w))
        .withColumn("lag_168h_mw", F.lag("consumption_mw", 336).over(w))
    )

    w_24h = Window.partitionBy("year").orderBy("datetime").rowsBetween(-48, 0)
    df = (
        df.withColumn("rolling_24h_mean", F.avg("consumption_mw").over(w_24h))
        .withColumn("rolling_24h_std", F.stddev("consumption_mw").over(w_24h))
    )

    renewable_sum = (
        F.coalesce(F.col("wind_mw"), F.lit(0))
        + F.coalesce(F.col("solar_mw"), F.lit(0))
        + F.coalesce(F.col("hydro_mw"), F.lit(0))
    )
    df = (
        df.withColumn(
            "renewable_share_pct",
            F.when(
                F.col("consumption_mw") > 0,
                renewable_sum / F.col("consumption_mw") * 100,
            ).otherwise(None),
        )
        .withColumn(
            "nuclear_share_pct",
            F.when(
                F.col("consumption_mw") > 0,
                F.col("nuclear_mw") / F.col("consumption_mw") * 100,
            ).otherwise(None),
        )
    )

    df = (
        df.withColumn(
            "forecast_error_mw",
            F.when(
                F.col("forecast_j1_mw").isNotNull()
                & F.col("consumption_mw").isNotNull(),
                F.col("consumption_mw") - F.col("forecast_j1_mw"),
            ).otherwise(None),
        )
        .withColumn(
            "forecast_error_pct",
            F.when(
                F.col("consumption_mw") > 0,
                F.abs(F.col("forecast_error_mw")) / F.col("consumption_mw") * 100,
            ).otherwise(None),
        )
    )

    key_cols = ["consumption_mw", "nuclear_mw", "wind_mw", "solar_mw", "co2_rate"]
    available_key = [c for c in key_cols if c in df.columns]
    if available_key:
        non_null_count = sum(
            F.when(F.col(c).isNotNull(), 1).otherwise(0) for c in available_key
        )
        df = df.withColumn("quality_score", non_null_count / len(available_key))

    logger.info("   -> %d columns", len(df.columns))
    return df
