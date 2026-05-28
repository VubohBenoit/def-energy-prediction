# =======================================================================
# **************    Projet : EDF Prediction Platform       **************
# **************    Version : 1.0.0                        **************
# =======================================================================
#
# spark/ml/constants.py — ML hyperparameters and features shared by the ``gold_to_model`` job.
# =======================================================================

from __future__ import annotations

import os

LABEL_COL: str = os.getenv("ML_LABEL_COL", "consumption_mw")
TRAIN_RATIO: float = float(os.getenv("ML_TRAIN_RATIO", "0.8"))
RANDOM_SEED: int = 42
ML_RF_NUM_TREES: int = int(os.getenv("ML_RF_NUM_TREES", "30"))
ML_RF_MAX_DEPTH: int = int(os.getenv("ML_RF_MAX_DEPTH", "8"))
ML_GBT_MAX_ITER: int = int(os.getenv("ML_GBT_MAX_ITER", "30"))
ML_GBT_MAX_DEPTH: int = int(os.getenv("ML_GBT_MAX_DEPTH", "6"))
ML_SPARK_PARTITIONS: int = int(os.getenv("ML_SPARK_PARTITIONS", "4"))

# Features candidates — present in Silver after ``bronze_to_silver``.
CANDIDATE_FEATURES: list[str] = [
    "hour", "day_of_week", "day_of_year", "week_of_year",
    "month", "quarter", "season",
    "hour_sin", "hour_cos",
    "is_weekend", "is_peak_hour",
    "nuclear_mw", "wind_mw", "solar_mw", "hydro_mw",
    "gas_mw", "coal_mw", "bioenergy_mw",
    "wind_onshore_mw", "wind_offshore_mw",
    "co2_rate",
    "lag_1h_mw", "lag_24h_mw", "lag_168h_mw",
    "rolling_24h_mean", "rolling_24h_std",
    "forecast_j1_mw",
]
