# =======================================================================
# **************    Projet : EDF Prediction Platform       **************
# **************    Version : 1.0.0                        **************
# =======================================================================
#
# spark/common/eda_style.py — Style EDA « rapport production » — chartes lisibles, export print-ready (300 DPI).
# =======================================================================

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import pandas as pd

# Palette sobre (energy report / corporate)
COLORS = {
    "primary": "#003DA5",
    "secondary": "#007A3D",
    "accent": "#E87722",
    "series_light": "#6B9BD1",
    "neutral": "#333333",
    "muted": "#666666",
    "grid": "#E6E6E6",
    "missing": "#B71C1C",
    "quality": "#00838F",
}

REPORT_DPI = 300
FOOTER_TEXT = "Source : RTE éco2mix — EDF ETL Platform"

# French labels for reports (avoid technical names in bars)
COLUMN_LABELS_FR: dict[str, str] = {
    "consumption_mw": "Consommation",
    "nuclear_mw": "Nucléaire",
    "wind_mw": "Éolien",
    "solar_mw": "Solaire",
    "hydro_mw": "Hydraulique",
    "gas_mw": "Gaz",
    "coal_mw": "Charbon",
    "fuel_mw": "Fioul",
    "bioenergy_mw": "Bioénergies",
    "co2_rate": "Taux CO₂",
    "forecast_j1_mw": "Prévision J-1",
    "forecast_j_mw": "Prévision J",
    "gas_cogen_mw": "Gaz — cogénération",
    "gas_tac_mw": "Gaz — TAC",
    "gas_ccg_mw": "Gaz — CCG",
    "wind_onshore_mw": "Éolien terrestre",
    "wind_offshore_mw": "Éolien offshore",
    "hydro_river_mw": "Hydraulique — fil de l'eau",
    "hydro_lake_mw": "Hydraulique — lacs",
    "hydro_step_mw": "Hydraulique — STEP",
    "battery_release_mw": "Déstockage batterie",
    "battery_storage_mw": "Stockage batterie",
    "quality_score": "Score qualité",
}


def human_label(column: str) -> str:
    """Returns a human-readable label for a column."""
    return COLUMN_LABELS_FR.get(column, column.replace("_mw", " (MW)").replace("_", " ").title())


def apply_production_style() -> None:
    """Matplotlib theme for deliverables PDF / slides."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": COLORS["grid"],
            "axes.labelcolor": COLORS["neutral"],
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.color": COLORS["muted"],
            "ytick.color": COLORS["muted"],
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.6,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
            "legend.fontsize": 9,
            "legend.frameon": False,
        }
    )


def add_report_footer(fig: plt.Figure, text: str = FOOTER_TEXT) -> None:
    """Adds a footer to a figure."""
    fig.text(
        0.99,
        0.01,
        text,
        ha="right",
        va="bottom",
        fontsize=7.5,
        color=COLORS["muted"],
        style="italic",
    )


def format_mw_axis(ax: plt.Axes, *, unit: str = "MW") -> None:
    """Formats the MW axis."""
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _pos: f"{x:,.0f}".replace(",", " "))
    )
    if ax.get_ylabel() in ("", "MW"):
        ax.set_ylabel(f"Puissance ({unit})")


def format_percent_axis(ax: plt.Axes) -> None:
    """Formats the percent axis."""
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _pos: f"{x:.0f} %"))


def prepare_time_series(
    df: pd.DataFrame,
    datetime_col: str,
    value_col: str,
    *,
    gap_hours: int = 48,
) -> pd.DataFrame:
    """Sorts, converts dates and cuts lines on large data gaps."""
    out = df[[datetime_col, value_col]].copy()
    out[datetime_col] = pd.to_datetime(out[datetime_col], errors="coerce", utc=True)
    out[value_col] = pd.to_numeric(out[value_col], errors="coerce")
    out = out.dropna(subset=[datetime_col, value_col]).sort_values(datetime_col)
    if out.empty:
        return out
    gap = out[datetime_col].diff() > pd.Timedelta(hours=gap_hours)
    out.loc[gap, value_col] = float("nan")
    return out


def plot_consumption_series(
    ax: plt.Axes,
    ts: pd.DataFrame,
    datetime_col: str,
    value_col: str,
    *,
    title: str,
    subtitle: str | None = None,
    ma_days: int = 7,
    raw_alpha: float = 0.25,
) -> None:
    """Hourly series + moving average for report."""
    if ts.empty:
        ax.text(0.5, 0.5, "Données indisponibles", ha="center", va="center", transform=ax.transAxes)
        return

    dt = ts[datetime_col]
    values = ts[value_col]
    ax.plot(dt, values, color=COLORS["series_light"], linewidth=0.6, alpha=raw_alpha, label="Mesure horaire")

    if len(ts) >= ma_days * 24:
        ma = (
            ts.set_index(datetime_col)[value_col]
            .rolling(f"{ma_days}D", min_periods=ma_days)
            .mean()
        )
        ax.plot(ma.index, ma.values, color=COLORS["primary"], linewidth=2.0, label=f"Moyenne mobile {ma_days} j")

    ax.set_xlim(dt.min(), dt.max())
    ax.set_title(title, pad=14 if subtitle else 8)
    if subtitle:
        ax.text(0.5, 1.02, subtitle, transform=ax.transAxes, ha="center", va="bottom", fontsize=9, color=COLORS["muted"])
    ax.set_xlabel("Période")
    format_mw_axis(ax)
    ax.grid(True, axis="y", alpha=0.5)
    ax.legend(loc="upper right")


def plot_missing_bars(
    ax: plt.Axes,
    missing_pct: pd.Series,
    *,
    title: str,
    top_n: int = 12,
) -> None:
    """Plots the missing bars."""
    series = missing_pct[missing_pct > 0].sort_values(ascending=True).tail(top_n)
    if series.empty:
        ax.text(0.5, 0.5, "Aucune valeur manquante significative", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return

    labels = [human_label(str(k)) for k in series.index]
    bars = ax.barh(labels, series.values, color=COLORS["missing"], height=0.65)
    ax.set_title(title)
    ax.set_xlabel("Part de valeurs manquantes (%)")
    format_percent_axis(ax)
    ax.grid(True, axis="x", alpha=0.4)
    for bar, val in zip(bars, series.values):
        ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2, f"{val:.1f} %", va="center", fontsize=8)


def plot_quality_distribution(ax: plt.Axes, scores: pd.Series, *, title: str) -> None:
    """Plots the quality distribution."""
    clean = scores.dropna()
    if clean.empty:
        ax.text(0.5, 0.5, "Score qualité indisponible", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return

    rounded = clean.round(3)
    counts = rounded.value_counts().sort_index()
    labels = [f"{v:.3f}".replace(".", ",") for v in counts.index]
    bars = ax.bar(labels, counts.values, color=COLORS["quality"], width=0.55)
    ax.set_title(title)
    ax.set_xlabel("Score de qualité (0–1)")
    ax.set_ylabel("Nombre d'enregistrements")
    ax.grid(True, axis="y", alpha=0.4)
    for bar, val in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:,}".replace(",", " "),
            ha="center",
            va="bottom",
            fontsize=8,
        )


def save_report_figure(fig: plt.Figure, path: str, *, dpi: int = REPORT_DPI) -> str:
    """Saves a report figure."""
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    return path


MODEL_LABELS_FR: dict[str, str] = {
    "LinearRegression": "Régression linéaire",
    "DecisionTree": "Arbre de décision",
    "RandomForest": "Forêt aléatoire",
    "GradientBoosting": "Gradient Boosting",
}


def human_model_name(name: str) -> str:
    """Returns a human-readable model name."""
    return MODEL_LABELS_FR.get(name, name.replace("_", " "))


def plot_ml_benchmark_report(
    fig: plt.Figure,
    metrics: pd.DataFrame,
    *,
    run_id: str | None = None,
    trained_at: str | None = None,
) -> None:
    """Panel: comparison RMSE, metrics table, selected model."""
    ordered = metrics.sort_values("rmse", ascending=True).reset_index(drop=True)
    best_row = ordered.iloc[0]
    best_name = str(best_row["model_name"])
    labels = [human_model_name(str(n)) for n in ordered["model_name"]]
    rmse_vals = ordered["rmse"].astype(float).tolist()

    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1.35, 1.0], hspace=0.38, wspace=0.28)
    ax_bars = fig.add_subplot(gs[0, :])
    ax_table = fig.add_subplot(gs[1, 0])
    ax_choice = fig.add_subplot(gs[1, 1])

    colors = [
        COLORS["secondary"] if str(n) == best_name else COLORS["primary"]
        for n in ordered["model_name"]
    ]
    bars = ax_bars.bar(labels, rmse_vals, color=colors, width=0.55, alpha=0.9, edgecolor="white")
    ax_bars.set_title("Comparaison des algorithmes — RMSE (MW)", fontweight="bold", pad=10)
    ax_bars.set_ylabel("RMSE (MW)")
    ax_bars.grid(True, axis="y", alpha=0.45)
    for bar, val in zip(bars, rmse_vals):
        ax_bars.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:,.0f}".replace(",", " "),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    title = "Benchmark prévision de consommation — jeu de test"
    fig.suptitle(title, fontweight="bold", fontsize=14, y=0.98)
    meta_parts = []
    if run_id:
        meta_parts.append(f"Run {run_id}")
    if trained_at:
        meta_parts.append(f"Entraînement : {trained_at}")
    if meta_parts:
        fig.text(0.5, 0.94, " · ".join(meta_parts), ha="center", fontsize=9, color=COLORS["muted"])

    table_cols = ["Algorithme", "RMSE", "MAE", "MAPE %", "R²", "Durée (s)"]
    table_rows: list[list[str]] = []
    for _, row in ordered.iterrows():
        table_rows.append([
            human_model_name(str(row["model_name"])),
            f"{float(row['rmse']):,.1f}".replace(",", " "),
            f"{float(row['mae']):,.1f}".replace(",", " "),
            f"{float(row['mape_pct']):.2f}",
            f"{float(row['r2']):.3f}",
            f"{float(row.get('train_time_s', 0) or 0):.1f}",
        ])

    ax_table.axis("off")
    ax_table.set_title("Métriques détaillées", fontweight="bold", loc="left", pad=8)
    tbl = ax_table.table(
        cellText=table_rows,
        colLabels=table_cols,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1.05, 1.35)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor(COLORS["primary"])
            cell.set_text_props(color="white", fontweight="bold")
        elif row > 0 and table_rows[row - 1][0] == human_model_name(best_name):
            cell.set_facecolor("#E8F5E9")

    ax_choice.axis("off")
    ax_choice.set_title("Modèle retenu", fontweight="bold", loc="left", pad=8)
    choice_text = (
        f"{human_model_name(best_name)}\n\n"
        f"Critère : RMSE minimal\n"
        f"RMSE = {float(best_row['rmse']):,.1f} MW\n"
        f"MAE = {float(best_row['mae']):,.1f} MW\n"
        f"MAPE = {float(best_row['mape_pct']):.2f} %\n"
        f"R² = {float(best_row['r2']):.3f}"
    )
    ax_choice.text(
        0.05,
        0.92,
        choice_text,
        transform=ax_choice.transAxes,
        va="top",
        ha="left",
        fontsize=11,
        linespacing=1.45,
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#E8F5E9", edgecolor=COLORS["secondary"], alpha=0.95),
    )
    ax_choice.text(
        0.05,
        0.08,
        "Artefact : s3a://models/rte/best/  ·  local : data/models/rte/best/",
        transform=ax_choice.transAxes,
        va="bottom",
        fontsize=8,
        color=COLORS["muted"],
        style="italic",
    )
