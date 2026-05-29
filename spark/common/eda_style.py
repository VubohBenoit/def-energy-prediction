# =======================================================================
# **************    Projet : EDF Energy Prediction         **************
# **************    Version : 1.0.0                        **************
# =======================================================================
#
# spark/common/eda_style.py — Style EDA « rapport production » — chartes lisibles, export print-ready (300 DPI).
# =======================================================================

from __future__ import annotations

import textwrap
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.patches import Patch

# Palette corporate — énergie / data (contraste AA, fond clair)
COLORS = {
    "primary": "#0B3D91",
    "secondary": "#0D9488",
    "accent": "#EA580C",
    "series_light": "#93C5FD",
    "neutral": "#1E293B",
    "muted": "#64748B",
    "grid": "#E2E8F0",
    "missing": "#DC2626",
    "quality": "#0891B2",
    "white": "#FFFFFF",
}

REPORT_DPI = 300
FOOTER_TEXT = "RTE éco2mix"
# Espace réservé en haut pour titre + sous-titre (fraction figure)
HEADER_BOTTOM = 0.80

# Dashboard ML — tuiles rapport / Streamlit
DASHBOARD = {
    "panel": "#F8FAFC",
    "panel_border": "#CBD5E1",
    "best_soft": "#ECFDF5",
    "bar": "#0B3D91",
    "bar_muted": "#CBD5E1",
    "line": "#0B3D91",
    "accent": "#EA580C",
    "accent_soft": "#FFF7ED",
    "text_muted": "#64748B",
    "scatter": "#3B82F6",
    "subtitle": "#475569",
}

# Performance rank — daltonien-friendly (ColorBrewer Blue / Orange + échelle vert → rouge)
PERFORMANCE_RANK_COLORS = ("#059669", "#65A30D", "#F59E0B", "#EA580C", "#DC2626")

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


def apply_dashboard_style() -> None:
    """Thème graphiques ML — rendu moderne type dashboard analytique."""
    apply_production_style()
    plt.rcParams.update(
        {
            "axes.facecolor": DASHBOARD["panel"],
            "axes.titlesize": 12,
            "axes.titlepad": 10,
            "axes.labelsize": 10,
            "axes.titleweight": "600",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": COLORS["white"],
            "legend.fontsize": 9,
            "font.size": 10,
        }
    )


def _style_ax_dashboard(ax: plt.Axes, *, grid_axis: str = "y") -> None:
    ax.set_facecolor(DASHBOARD["panel"])
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(DASHBOARD["panel_border"])
    ax.spines["bottom"].set_color(DASHBOARD["panel_border"])
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    if grid_axis == "y":
        ax.grid(True, axis="y", alpha=0.55, linestyle="-", linewidth=0.5, color=COLORS["grid"])
    elif grid_axis == "both":
        ax.grid(True, alpha=0.4, linestyle="-", linewidth=0.45, color=COLORS["grid"])
    else:
        ax.grid(False)
    ax.set_axisbelow(True)
    ax.tick_params(colors=COLORS["muted"], length=0, pad=6)


def add_accent_bar(fig: plt.Figure, *, color: str | None = None) -> None:
    """Fine bande d'accent en haut de figure (identité visuelle)."""
    fig.patches.append(
        plt.Rectangle(
            (0, 1.0),
            1,
            0.012,
            transform=fig.transFigure,
            facecolor=color or COLORS["primary"],
            clip_on=False,
            linewidth=0,
            zorder=10,
        )
    )


def add_chart_header(
    fig: plt.Figure,
    title: str,
    *,
    subtitle: str | None = None,
    y_title: float = 0.96,
) -> None:
    """Titre + sous-titre explicatif (tuile dashboard)."""
    fig.suptitle(
        title,
        fontsize=15,
        fontweight="bold",
        color=COLORS["neutral"],
        y=y_title,
        ha="center",
    )
    if subtitle:
        wrapped = "\n".join(textwrap.wrap(subtitle, width=108))
        fig.text(
            0.5,
            y_title - 0.055,
            wrapped,
            ha="center",
            va="top",
            fontsize=9.5,
            color=DASHBOARD["subtitle"],
            linespacing=1.35,
        )


def apply_figure_layout(fig: plt.Figure, *, top: float = HEADER_BOTTOM, hspace: float | None = None) -> None:
    """Réserve l'espace header/footer sans rogner titre ni sous-titre."""
    if hspace is not None:
        fig.subplots_adjust(left=0.07, right=0.98, top=top, bottom=0.08, hspace=hspace, wspace=0.28)
    else:
        fig.tight_layout(rect=(0.02, 0.05, 0.98, top))


def add_report_footer(fig: plt.Figure, text: str = FOOTER_TEXT) -> None:
    """Pied de page source (bas droite)."""
    fig.text(
        0.99,
        0.008,
        text,
        ha="right",
        va="bottom",
        fontsize=8,
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
    title: str | None = None,
    subtitle: str | None = None,
    ma_days: int = 7,
    raw_alpha: float = 0.2,
) -> None:
    """Série horaire + moyenne mobile."""
    _style_ax_dashboard(ax, grid_axis="both")
    if ts.empty:
        ax.text(0.5, 0.5, "Données indisponibles", ha="center", va="center", transform=ax.transAxes)
        return

    dt = ts[datetime_col]
    values = ts[value_col]
    ax.plot(dt, values, color=COLORS["series_light"], linewidth=0.5, alpha=raw_alpha, label="Mesure horaire")

    if len(ts) >= ma_days * 24:
        ma = (
            ts.set_index(datetime_col)[value_col]
            .rolling(f"{ma_days}D", min_periods=ma_days)
            .mean()
        )
        ax.plot(
            ma.index,
            ma.values,
            color=COLORS["primary"],
            linewidth=2.2,
            label=f"Moyenne mobile {ma_days} j",
        )

    ax.set_xlim(dt.min(), dt.max())
    if title:
        ax.set_title(title, pad=14 if subtitle else 8, fontweight="bold", loc="left")
    if subtitle:
        ax.text(0.0, 1.02, subtitle, transform=ax.transAxes, ha="left", va="bottom", fontsize=9, color=DASHBOARD["subtitle"])
    ax.set_xlabel("Période")
    format_mw_axis(ax)
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor=DASHBOARD["panel_border"])


def plot_missing_bars(
    ax: plt.Axes,
    missing_pct: pd.Series,
    *,
    title: str,
    top_n: int = 12,
) -> None:
    """Barres horizontales — taux de valeurs manquantes."""
    _style_ax_dashboard(ax, grid_axis="x")
    series = missing_pct[missing_pct > 0].sort_values(ascending=True).tail(top_n)
    if series.empty:
        ax.text(0.5, 0.5, "Aucune valeur manquante significative", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, loc="left", fontweight="bold")
        return

    labels = [human_label(str(k)) for k in series.index]
    bars = ax.barh(labels, series.values, color=COLORS["missing"], height=0.62, alpha=0.88)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel("Part de valeurs manquantes (%)")
    format_percent_axis(ax)
    for bar, val in zip(bars, series.values):
        ax.text(val + 0.25, bar.get_y() + bar.get_height() / 2, f"{val:.1f} %", va="center", fontsize=8, color=COLORS["neutral"])


def plot_quality_distribution(ax: plt.Axes, scores: pd.Series, *, title: str) -> None:
    """Distribution du score qualité."""
    _style_ax_dashboard(ax)
    clean = scores.dropna()
    if clean.empty:
        ax.text(0.5, 0.5, "Score qualité indisponible", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, loc="left", fontweight="bold")
        return

    rounded = clean.round(3)
    counts = rounded.value_counts().sort_index()
    labels = [f"{v:.3f}".replace(".", ",") for v in counts.index]
    bars = ax.bar(labels, counts.values, color=COLORS["quality"], width=0.6, alpha=0.9, edgecolor="white", linewidth=0.8)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel("Score de qualité (0–1)")
    ax.set_ylabel("Nombre d'enregistrements")
    for bar, val in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:,}".replace(",", " "),
            ha="center",
            va="bottom",
            fontsize=8,
            color=COLORS["neutral"],
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


def _ordered_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    return metrics.sort_values("rmse", ascending=True).reset_index(drop=True)


def _rank_color(rank: int, total: int) -> str:
    idx = min(rank, len(PERFORMANCE_RANK_COLORS) - 1)
    if total <= 1:
        return PERFORMANCE_RANK_COLORS[0]
    scaled = int(rank * (len(PERFORMANCE_RANK_COLORS) - 1) / max(total - 1, 1))
    return PERFORMANCE_RANK_COLORS[scaled]


def _smooth_series(x: np.ndarray, y: np.ndarray, *, points: int = 200) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    if len(xs) < 2:
        return xs, ys
    dense = np.linspace(xs.min(), xs.max(), points)
    smooth = np.interp(dense, xs, ys)
    return dense, smooth


def _learning_curve_columns(curve: pd.DataFrame) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Retourne (x, y_train, y_val) en évitant la colonne legacy rmse_mw."""
    df = curve.sort_values("train_rows")
    x = df["train_rows"].astype(float).to_numpy()
    if "rmse_validation_mw" in df.columns:
        y_val = df["rmse_validation_mw"].astype(float).to_numpy()
    elif "rmse_mw" in df.columns:
        y_val = df["rmse_mw"].astype(float).to_numpy()
    else:
        raise ValueError("learning_curve: colonne rmse_validation_mw absente")
    y_train = None
    if "rmse_train_mw" in df.columns:
        y_train = df["rmse_train_mw"].astype(float).to_numpy()
    return x, y_train, y_val


def describe_learning_curve(curve: pd.DataFrame) -> str:
    """Phrase d'interprétation pour la courbe d'apprentissage."""
    if curve.empty:
        return "Courbe indisponible — relancez le pipeline ML."
    x, y_train, y_val = _learning_curve_columns(curve)
    plateau_idx = int(np.argmin(y_val))
    plateau_rows = int(x[plateau_idx])
    plateau_rmse = float(y_val[plateau_idx])
    gap = float(y_val[plateau_idx] - y_train[plateau_idx]) if y_train is not None else float("nan")
    if y_train is not None and not np.isnan(gap):
        return (
            f"Validation minimale ~{plateau_rmse:,.0f} MW à {plateau_rows:,} lignes "
            f"(écart train/validation ~{gap:,.0f} MW)".replace(",", " ")
        )
    return f"Validation minimale ~{plateau_rmse:,.0f} MW à {plateau_rows:,} lignes".replace(",", " ")


def synthesis_footnote(metrics: pd.DataFrame, best_model: str | None = None) -> str:
    """Commentaire sous le tableau de synthèse."""
    ordered = _ordered_metrics(metrics)
    best_name = human_model_name(best_model or str(ordered.iloc[0]["model_name"]))
    gb = metrics[metrics["model_name"].str.contains("Gradient", case=False, na=False)]
    if gb.empty or "train_time_s" not in metrics.columns:
        return f"Modèle retenu : {best_name} (RMSE minimal). Validation sur période future."
    best_row = ordered.iloc[0]
    gb_row = gb.sort_values("rmse").iloc[0]
    delta = (float(gb_row["rmse"]) / float(best_row["rmse"]) - 1) * 100
    time_ratio = float(gb_row.get("train_time_s", 0) or 1) / max(float(best_row.get("train_time_s", 0) or 1), 0.1)
    return (
        f"Modèle retenu : {best_name} (RMSE minimal). "
        f"Gradient Boosting : précision proche (+{delta:.1f} %) mais ~{time_ratio:.0f}× plus lent."
    )


def _prediction_metrics(df: pd.DataFrame) -> dict[str, float]:
    actual = df["actual_mw"].astype(float)
    predicted = df["predicted_mw"].astype(float)
    err = actual - predicted
    rmse = float(np.sqrt((err**2).mean()))
    mae = float(np.abs(err).mean())
    mape = float((np.abs(err) / actual.replace(0, np.nan)).mean() * 100)
    ss_res = float((err**2).sum())
    ss_tot = float(((actual - actual.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")
    return {"rmse": rmse, "mae": mae, "mape": mape, "r2": r2}


def _metrics_table_rows(ordered: pd.DataFrame) -> tuple[list[str], list[list[str]], pd.DataFrame]:
    best_rmse = float(ordered["rmse"].min())
    table_cols = ["Algorithme", "RMSE (MW)", "MAE (MW)", "MAPE %", "R²", "Durée (s)", "Δ RMSE %"]
    table_rows: list[list[str]] = []
    raw = ordered.copy()
    raw["delta_rmse_pct"] = (raw["rmse"].astype(float) / best_rmse - 1.0) * 100.0

    for _, row in raw.iterrows():
        delta = float(row["delta_rmse_pct"])
        duration = float(row.get("train_time_s", 0) or 0)
        table_rows.append([
            human_model_name(str(row["model_name"])),
            f"{float(row['rmse']):,.0f}".replace(",", " "),
            f"{float(row['mae']):,.0f}".replace(",", " "),
            f"{float(row['mape_pct']):.1f}",
            f"{float(row['r2']):.3f}",
            f"{duration:.1f}",
            "0 %" if abs(delta) < 0.05 else f"+{delta:.1f} %",
        ])
    return table_cols, table_rows, raw


def _cell_heat_color(value: float, vmin: float, vmax: float, *, higher_is_better: bool) -> tuple[float, float, float, float]:
    if vmax <= vmin:
        t = 0.0
    else:
        t = (value - vmin) / (vmax - vmin)
    if not higher_is_better:
        t = 1.0 - t
    rgba = cm.RdYlGn(t)
    return (rgba[0], rgba[1], rgba[2], 0.35)


def plot_model_comparison_grouped(ax: plt.Axes, metrics: pd.DataFrame) -> str:
    """Barres groupées RMSE (% ref.) + pénalité (1−R²) — plus bas = meilleur."""
    _style_ax_dashboard(ax)
    ordered = _ordered_metrics(metrics)
    best_name = str(ordered.iloc[0]["model_name"])
    best_rmse = float(ordered["rmse"].min())
    best_r2 = float(ordered["r2"].max())
    best_gap = max(1.0 - best_r2, 1e-6)

    labels = [human_model_name(str(n)) for n in ordered["model_name"]]
    rmse_pct = (ordered["rmse"].astype(float) / best_rmse * 100.0).tolist()
    r2_penalty_pct = ((1.0 - ordered["r2"].astype(float)) / best_gap * 100.0).tolist()
    rmse_raw = ordered["rmse"].astype(float).tolist()
    r2_raw = ordered["r2"].astype(float).tolist()

    n = len(labels)
    x = np.arange(n)
    width = 0.36
    colors = [_rank_color(i, n) for i in range(n)]

    ax.bar(
        x - width / 2,
        rmse_pct,
        width,
        color=colors,
        alpha=0.92,
        edgecolor="white",
        linewidth=0.8,
        label="RMSE (% du meilleur = 100 %)",
        zorder=3,
    )
    ax.bar(
        x + width / 2,
        r2_penalty_pct,
        width,
        color=colors,
        alpha=0.55,
        edgecolor="white",
        linewidth=0.8,
        hatch="//",
        label="Écart R² (1−R², % du meilleur)",
        zorder=3,
    )
    ax.axhline(100, color=COLORS["neutral"], linestyle=(0, (4, 4)), linewidth=1.2, alpha=0.7, zorder=2)
    ax.text(
        n - 0.05,
        100,
        " Référence (meilleur RMSE)",
        va="bottom",
        ha="right",
        fontsize=8,
        color=COLORS["muted"],
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylabel("Indice relatif (% — 100 % = meilleur modèle)")
    ymax = max(max(rmse_pct), max(r2_penalty_pct), 100) * 1.18
    ax.set_ylim(0, ymax)

    for i, (rp, rpp, rmse, r2) in enumerate(zip(rmse_pct, r2_penalty_pct, rmse_raw, r2_raw)):
        delta = (rmse / best_rmse - 1.0) * 100.0
        delta_txt = "ref." if abs(delta) < 0.05 else f"+{delta:.1f} %"
        ax.text(
            x[i] - width / 2,
            rp + ymax * 0.015,
            f"{rmse:,.0f}\n{delta_txt}".replace(",", " "),
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            color=colors[i],
        )
        ax.text(
            x[i] + width / 2,
            rpp + ymax * 0.015,
            f"R²={r2:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=COLORS["neutral"],
        )

    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor=DASHBOARD["panel_border"], fontsize=8.5)
    return best_name


def plot_metrics_detail_table(
    ax: plt.Axes,
    metrics: pd.DataFrame,
    *,
    footnote: str | None = None,
) -> str:
    """Tableau métriques avec dégradé et meilleures valeurs par colonne."""
    ordered = _ordered_metrics(metrics)
    best_name = str(ordered.iloc[0]["model_name"])
    table_cols, table_rows, raw = _metrics_table_rows(ordered)
    ax.axis("off")

    best_row_pos = {
        "RMSE (MW)": int(raw["rmse"].astype(float).values.argmin()),
        "MAE (MW)": int(raw["mae"].astype(float).values.argmin()),
        "MAPE %": int(raw["mape_pct"].astype(float).values.argmin()),
        "R²": int(raw["r2"].astype(float).values.argmax()),
    }
    if "train_time_s" in raw.columns:
        best_row_pos["Durée (s)"] = int(raw["train_time_s"].astype(float).values.argmin())
    numeric_specs: dict[str, tuple[str, bool]] = {
        "RMSE (MW)": ("rmse", False),
        "MAE (MW)": ("mae", False),
        "MAPE %": ("mape_pct", False),
        "R²": ("r2", True),
        "Durée (s)": ("train_time_s", False),
    }

    tbl = ax.table(
        cellText=table_rows,
        colLabels=table_cols,
        loc="upper center",
        cellLoc="center",
        colWidths=[0.19, 0.11, 0.11, 0.1, 0.08, 0.1, 0.1],
        bbox=[0, 0.12 if footnote else 0, 1, 0.88 if footnote else 1],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.55)

    col_ranges: dict[str, tuple[float, float]] = {}
    for label, (field, _hb) in numeric_specs.items():
        vals = raw[field].astype(float)
        col_ranges[label] = (float(vals.min()), float(vals.max()))

    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor(DASHBOARD["panel_border"])
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor(COLORS["primary"])
            cell.set_text_props(color="white", fontweight="bold")
            continue
        col_name = table_cols[col]
        is_best_row = table_rows[row - 1][0] == human_model_name(best_name)
        if col_name in numeric_specs:
            field, higher_is_better = numeric_specs[col_name]
            val = float(raw.iloc[row - 1][field])
            vmin, vmax = col_ranges[col_name]
            cell.set_facecolor(_cell_heat_color(val, vmin, vmax, higher_is_better=higher_is_better))
            if (row - 1) == best_row_pos.get(col_name):
                cell.set_text_props(fontweight="bold")
        elif is_best_row:
            cell.set_facecolor(DASHBOARD["best_soft"])
            cell.set_text_props(fontweight="bold")
        else:
            cell.set_facecolor("white" if row % 2 else DASHBOARD["panel"])
    if footnote:
        ax.text(
            0.5,
            0.02,
            footnote,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=DASHBOARD["text_muted"],
            wrap=True,
        )
    return best_name


def plot_learning_curve_rf(
    ax: plt.Axes,
    curve: pd.DataFrame,
) -> None:
    """Courbe d'apprentissage lissée avec écart train/validation (axe X log)."""
    _style_ax_dashboard(ax, grid_axis="both")
    if curve.empty:
        ax.text(0.5, 0.5, "Courbe indisponible", ha="center", va="center", transform=ax.transAxes)
        return

    x, y_train, y_val = _learning_curve_columns(curve)

    # Bande d'incertitude optionnelle (multi-runs)
    if "rmse_validation_std_mw" in curve.columns:
        std = curve.sort_values("train_rows")["rmse_validation_std_mw"].astype(float).to_numpy()
        ax.fill_between(x, y_val - std, y_val + std, color=DASHBOARD["accent"], alpha=0.15, zorder=1)

    xs, ys_v = _smooth_series(x, y_val)
    plateau_idx = int(np.argmin(y_val))
    x_plateau = float(x[plateau_idx])
    y_val_at = float(y_val[plateau_idx])

    if y_train is not None:
        _, ys_t = _smooth_series(x, y_train)
        y_train_at = float(y_train[plateau_idx])
        gap_at_best = y_val_at - y_train_at
        y_mid = (y_train_at + y_val_at) / 2

        ax.fill_between(xs, ys_t, ys_v, alpha=0.18, color=DASHBOARD["accent"], zorder=2)
        ax.plot(xs, ys_t, color=COLORS["primary"], linewidth=2.6, label="Entraînement", zorder=4)
        ax.plot(xs, ys_v, color=DASHBOARD["accent"], linewidth=2.6, label="Validation", zorder=5)
        ax.scatter(x, y_train, s=52, color=COLORS["primary"], edgecolors="white", linewidths=1.5, zorder=6)
        ax.scatter(x, y_val, s=52, color=DASHBOARD["accent"], marker="s", edgecolors="white", linewidths=1.5, zorder=6)

        # Repère d'écart au meilleur point (entre les deux courbes, pas en bas à gauche)
        bracket_x = x_plateau * 1.06
        ax.plot(
            [x_plateau, bracket_x, bracket_x],
            [y_train_at, y_train_at, y_val_at],
            color=COLORS["secondary"],
            linewidth=1.4,
            linestyle="-",
            zorder=7,
        )
        ax.plot([x_plateau, bracket_x], [y_val_at, y_val_at], color=COLORS["secondary"], linewidth=1.4, zorder=7)
        ax.annotate(
            f"Écart\n{gap_at_best:,.0f} MW".replace(",", " "),
            xy=(bracket_x, y_mid),
            xytext=(14, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=COLORS["neutral"],
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="white",
                edgecolor=COLORS["secondary"],
                linewidth=1.2,
                alpha=0.97,
            ),
            zorder=10,
        )
    else:
        ax.plot(xs, ys_v, color=COLORS["primary"], linewidth=2.6, label="Validation")
        ax.scatter(x, y_val, s=52, color=COLORS["primary"], edgecolors="white", linewidths=1.5, zorder=6)

    y_all = np.concatenate([y_train, y_val]) if y_train is not None else y_val
    y_lo = max(500, float(np.floor(y_all.min() / 50) * 50) - 50)
    y_hi = min(2000, float(np.ceil(y_all.max() / 50) * 50) + 50)
    ax.set_ylim(y_lo, y_hi)

    ax.axvline(x_plateau, color=COLORS["secondary"], linestyle=(0, (4, 3)), linewidth=1.4, alpha=0.9, zorder=3)
    ax.text(
        x_plateau,
        y_hi * 0.97,
        "  Meilleur compromis",
        va="top",
        ha="left",
        fontsize=8,
        color=COLORS["secondary"],
    )

    x_final = float(x.max())
    if abs(x_final - x_plateau) > x_final * 0.02:
        ax.axvline(x_final, color=COLORS["muted"], linestyle=(0, (2, 4)), linewidth=1.2, alpha=0.75, zorder=3)
        ax.text(
            x_final,
            y_hi * 0.82,
            "  Taille retenue",
            va="top",
            ha="left",
            fontsize=8,
            color=COLORS["muted"],
        )

    ax.set_xscale("log")
    ax.set_xlabel("Taille du jeu d'entraînement (lignes, échelle log)")
    ax.set_ylabel("RMSE (MW)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _pos: f"{int(v):,}".replace(",", " ")))
    ax.legend(
        loc="upper left",
        frameon=True,
        facecolor="white",
        edgecolor=DASHBOARD["panel_border"],
        fontsize=9,
    )


def _contiguous_segments(
    index: pd.DatetimeIndex,
    *,
    max_gap: pd.Timedelta = pd.Timedelta(hours=24),
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Découpe une série temporelle aux trous > max_gap."""
    if len(index) == 0:
        return []
    gaps = index.to_series().diff()
    break_at = gaps[gaps > max_gap].index
    segments: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    seg_start = index[0]
    for ts in break_at:
        prev = index[index.get_loc(ts) - 1]
        segments.append((seg_start, prev))
        seg_start = ts
    segments.append((seg_start, index[-1]))
    return segments


def _plot_time_segments(
    ax: plt.Axes,
    series: pd.Series,
    *,
    color: str,
    linewidth: float,
    alpha: float,
    label: str,
    zorder: int,
    linestyle: str = "-",
) -> None:
    """Trace une série sans relier les segments séparés par un trou."""
    labeled = False
    for start, end in _contiguous_segments(series.index):
        seg = series.loc[start:end]
        if seg.empty:
            continue
        plot_label = label if not labeled else "_nolegend_"
        ax.plot(
            seg.index,
            seg.values,
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            label=plot_label,
            zorder=zorder,
            linestyle=linestyle,
        )
        labeled = True


def _segment_predictions_df(df: pd.DataFrame) -> list[pd.DataFrame]:
    """Découpe le jeu de test aux trous > 24 h."""
    if df.empty:
        return []
    ordered = df.sort_values("datetime")
    gaps = ordered["datetime"].diff() > pd.Timedelta(hours=24)
    seg_ids = gaps.cumsum()
    return [group.copy() for _, group in ordered.groupby(seg_ids)]


def _segment_caption(df: pd.DataFrame, *, index: int | None = None, total: int | None = None) -> str:
    start = df["datetime"].min().strftime("%d %b %Y")
    end = df["datetime"].max().strftime("%d %b %Y")
    n = len(df)
    if index is not None and total is not None and total > 1:
        return f"Période {index}/{total} · {n:,} pts · {start} → {end}".replace(",", " ")
    return f"{n:,} pts · {start} → {end}".replace(",", " ")


def _prepare_predictions_df(preds: pd.DataFrame) -> pd.DataFrame:
    df = preds.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    df["actual_mw"] = pd.to_numeric(df["actual_mw"], errors="coerce")
    df["predicted_mw"] = pd.to_numeric(df["predicted_mw"], errors="coerce")
    return df.dropna(subset=["actual_mw", "predicted_mw"])


def _lowess_curve(
    x: np.ndarray,
    y: np.ndarray,
    *,
    frac: float = 0.08,
    n_points: int = 300,
) -> tuple[np.ndarray, np.ndarray]:
    """Régression locale lissée (LOESS) — détection de biais systématique."""
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    if len(xs) < 4:
        return xs, ys
    x_grid = np.linspace(float(xs.min()), float(xs.max()), n_points)
    y_smooth = np.empty(n_points)
    n = len(xs)
    k = max(int(frac * n), 3)
    for i, x0 in enumerate(x_grid):
        dist = np.abs(xs - x0)
        h = float(np.sort(dist)[min(k, n - 1)])
        h = max(h, 1e-9)
        w = np.clip(1.0 - (dist / h) ** 3, 0.0, 1.0) ** 3
        wsum = float(w.sum())
        if wsum < 1e-12:
            y_smooth[i] = float(ys[np.argmin(dist)])
            continue
        x_mean = float((w * xs).sum() / wsum)
        y_mean = float((w * ys).sum() / wsum)
        num = float((w * (xs - x_mean) * (ys - y_mean)).sum())
        den = float((w * (xs - x_mean) ** 2).sum())
        if den < 1e-12:
            y_smooth[i] = y_mean
        else:
            slope = num / den
            intercept = y_mean - slope * x_mean
            y_smooth[i] = intercept + slope * x0
    return x_grid, y_smooth


def _french_public_holidays(start: pd.Timestamp, end: pd.Timestamp) -> set[pd.Timestamp]:
    """Jours fériés fixes France (UTC, normalisés à minuit)."""
    fixed = {(1, 1), (5, 1), (5, 8), (7, 14), (8, 15), (11, 1), (11, 11), (12, 25)}
    days: set[pd.Timestamp] = set()
    for year in range(start.year, end.year + 1):
        for month, day in fixed:
            ts = pd.Timestamp(year=year, month=month, day=day, tz="UTC")
            if start.normalize() <= ts <= end.normalize():
                days.add(ts.normalize())
    return days


def _shade_weekends_and_holidays(
    ax: plt.Axes,
    t_min: pd.Timestamp,
    t_max: pd.Timestamp,
    *,
    y_lo: float,
    y_hi: float,
) -> None:
    """Bandes verticales grises pour week-ends et jours fériés."""
    _ = y_lo, y_hi
    holidays = _french_public_holidays(t_min, t_max)
    day = t_min.normalize()
    end_day = t_max.normalize()
    while day <= end_day:
        if day.dayofweek >= 5 or day in holidays:
            ax.axvspan(
                day,
                day + pd.Timedelta(days=1),
                ymin=0,
                ymax=1,
                facecolor="#E2E8F0",
                alpha=0.55,
                zorder=0,
            )
        day += pd.Timedelta(days=1)


def _style_predictions_ax(ax: plt.Axes, *, grid_axis: str = "both") -> None:
    """Grille discrète pour graphiques prédictions (spec rapport)."""
    ax.set_facecolor("white")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(COLORS["grid"])
    ax.spines["bottom"].set_color(COLORS["grid"])
    ax.tick_params(labelsize=11, colors=COLORS["neutral"])
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)
    if grid_axis in ("both", "x"):
        ax.grid(True, axis="x", linestyle="--", alpha=0.5, color=COLORS["grid"], linewidth=0.6)
    if grid_axis in ("both", "y"):
        ax.grid(True, axis="y", linestyle="--", alpha=0.5, color=COLORS["grid"], linewidth=0.6)
    ax.set_axisbelow(True)


def _format_prediction_dates(ax: plt.Axes) -> None:
    """Dates lisibles avec année (période de test sur plusieurs années)."""
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %Y"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=8))
    for label in ax.get_xticklabels():
        label.set_rotation(35)
        label.set_ha("right")


def plot_predictions_dispersion(ax: plt.Axes, preds: pd.DataFrame) -> None:
    """Graphique A — dispersion réel vs prédit (jeu de test)."""
    _style_predictions_ax(ax, grid_axis="both")
    if preds.empty:
        ax.text(0.5, 0.5, "Prédictions indisponibles", ha="center", va="center", transform=ax.transAxes)
        return

    df = _prepare_predictions_df(preds)
    actual = df["actual_mw"].to_numpy(dtype=float)
    predicted = df["predicted_mw"].to_numpy(dtype=float)
    stats = _prediction_metrics(df)
    band = 2.0 * stats["rmse"]

    lo = float(min(actual.min(), predicted.min()))
    hi = float(max(actual.max(), predicted.max()))
    pad = (hi - lo) * 0.03 or 100.0
    lo_p, hi_p = lo - pad, hi + pad

    diag = np.linspace(lo_p, hi_p, 200)
    ax.fill_between(
        diag,
        diag - band,
        diag + band,
        color="#94A3B8",
        alpha=0.22,
        zorder=1,
        label=f"Bande ±2×RMSE (±{band:,.0f} MW)".replace(",", " "),
    )
    ax.plot(diag, diag, color="#0F172A", linewidth=1.0, linestyle="--", zorder=4, label="Référence y = x")

    err_abs = np.abs(actual - predicted)
    in_band = err_abs <= band
    ax.scatter(
        actual[in_band],
        predicted[in_band],
        s=14,
        c="#2563EB",
        alpha=0.35,
        linewidths=0,
        zorder=2,
        label=f"|erreur| ≤ 2×RMSE ({int(in_band.sum()):,})".replace(",", " "),
    )
    ax.scatter(
        actual[~in_band],
        predicted[~in_band],
        s=22,
        c="#DC2626",
        alpha=0.75,
        linewidths=0.3,
        edgecolors="#991B1B",
        zorder=3,
        label=f"|erreur| > 2×RMSE ({int((~in_band).sum()):,})".replace(",", " "),
    )

    loess_x, loess_y = _lowess_curve(actual, predicted)
    ax.plot(
        loess_x,
        loess_y,
        color="#EA580C",
        linewidth=2.2,
        linestyle="-",
        zorder=5,
        label="LOESS (biais local)",
    )

    ax.set_xlim(lo_p, hi_p)
    ax.set_ylim(lo_p, hi_p)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Consommation réelle (MW)")
    ax.set_ylabel("Prédiction Forêt aléatoire (MW)")
    format_mw_axis(ax, unit="MW")

    metrics_text = (
        f"R² = {stats['r2']:.3f}\n"
        f"RMSE = {stats['rmse']:,.0f} MW\n"
        f"MAE = {stats['mae']:,.0f} MW\n"
        f"MAPE = {stats['mape']:.1f} %"
    ).replace(",", " ")
    ax.text(
        0.02,
        0.98,
        metrics_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        color=COLORS["neutral"],
        bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor=COLORS["grid"], alpha=0.97),
        zorder=10,
    )
    ax.legend(
        loc="lower right",
        frameon=True,
        facecolor="white",
        edgecolor=COLORS["grid"],
        fontsize=9,
    )


def _segment_mae(df: pd.DataFrame) -> float:
    err = df["actual_mw"].astype(float) - df["predicted_mw"].astype(float)
    return float(np.abs(err).mean())


def _plot_predictions_timeseries_block(
    ax_top: plt.Axes,
    ax_bottom: plt.Axes,
    seg: pd.DataFrame,
    *,
    band: float,
    err_ylim: float,
    column_title: str,
    show_legend_top: bool,
    show_legend_bottom: bool,
    show_ylabel: bool,
    show_threshold_labels: bool,
) -> None:
    """Un bloc temporel : courbes (haut) + barres d'écart (bas)."""
    _style_predictions_ax(ax_top, grid_axis="y")
    _style_predictions_ax(ax_bottom, grid_axis="both")

    df = seg.sort_values("datetime")
    indexed = df.set_index("datetime")
    t_min, t_max = indexed.index.min(), indexed.index.max()

    roll = indexed[["actual_mw", "predicted_mw"]].rolling("6h", min_periods=1).mean()
    y_min = float(roll[["actual_mw", "predicted_mw"]].min().min())
    y_max = float(roll[["actual_mw", "predicted_mw"]].max().max())
    margin = max((y_max - y_min) * 0.05, 600.0)
    y_lo, y_hi = y_min - margin, y_max + margin

    ax_top.set_xlim(t_min, t_max)
    ax_top.set_ylim(y_lo, y_hi)
    _shade_weekends_and_holidays(ax_top, t_min, t_max, y_lo=y_lo, y_hi=y_hi)

    _plot_time_segments(
        ax_top,
        roll["predicted_mw"],
        color="#2563EB",
        linewidth=2.0,
        alpha=0.8,
        label="Prédit (MM 6 h)",
        zorder=3,
    )
    _plot_time_segments(
        ax_top,
        roll["actual_mw"],
        color="#0F172A",
        linewidth=2.4,
        alpha=1.0,
        label="Réel (MM 6 h)",
        zorder=4,
    )

    ax_top.set_title(column_title, fontsize=10, fontweight="bold", color=COLORS["neutral"], pad=8)
    if show_ylabel:
        ax_top.set_ylabel("Puissance (MW)")
        format_mw_axis(ax_top)
    else:
        ax_top.tick_params(labelleft=False)
    if show_legend_top:
        ax_top.legend(loc="upper right", frameon=True, facecolor="white", edgecolor=COLORS["grid"], fontsize=8)
    ax_top.tick_params(labelbottom=False)

    display = df.iloc[::6].copy()
    err = (display["actual_mw"] - display["predicted_mw"]).to_numpy(dtype=float)
    dates = display["datetime"]
    step = pd.Series(indexed.index).diff().median()
    bar_width = step.total_seconds() / 86400.0 * 5.0 if pd.notna(step) else 0.02
    bar_colors = np.where(err > 0, "#22C55E", "#DC2626")

    ax_bottom.bar(
        dates,
        err,
        width=bar_width,
        color=bar_colors,
        alpha=0.85,
        align="center",
        zorder=2,
    )

    ax_bottom.axhline(0, color="#0F172A", linewidth=1.0, zorder=3)
    ax_bottom.axhline(band, color="#64748B", linewidth=0.9, linestyle=(0, (4, 4)), zorder=1)
    ax_bottom.axhline(-band, color="#64748B", linewidth=0.9, linestyle=(0, (4, 4)), zorder=1)
    if show_threshold_labels:
        ax_bottom.text(
            t_max,
            band,
            f" +2×RMSE ({band:,.0f} MW)".replace(",", " "),
            ha="right",
            va="bottom",
            fontsize=8,
            color=COLORS["muted"],
        )
        ax_bottom.text(
            t_max,
            -band,
            f" −2×RMSE ({-band:,.0f} MW)".replace(",", " "),
            ha="right",
            va="top",
            fontsize=8,
            color=COLORS["muted"],
        )

    ax_bottom.set_ylim(-err_ylim, err_ylim)
    if show_ylabel:
        ax_bottom.set_ylabel("Écart réel − prédit (MW)")
    else:
        ax_bottom.tick_params(labelleft=False)
    ax_bottom.set_xlabel("Date (UTC)")
    _format_prediction_dates(ax_bottom)

    if show_legend_bottom:
        legend_patches = [
            Patch(facecolor="#22C55E", edgecolor="none", alpha=0.85, label="Sous-estimation (réel > prédit)"),
            Patch(facecolor="#DC2626", edgecolor="none", alpha=0.85, label="Sur-estimation (réel < prédit)"),
        ]
        ax_bottom.legend(
            handles=legend_patches,
            loc="upper left",
            frameon=True,
            facecolor="white",
            edgecolor=COLORS["grid"],
            fontsize=8,
        )


def build_predictions_dispersion_figure(preds: pd.DataFrame) -> tuple[plt.Figure, tuple[plt.Axes, ...]]:
    fig, ax = plt.subplots(figsize=(10.5, 10))
    plot_predictions_dispersion(ax, preds)
    return fig, (ax,)


def build_predictions_timeseries_figure(preds: pd.DataFrame) -> tuple[plt.Figure, tuple[plt.Axes, ...]]:
    """Graphique B — un panneau par bloc temporel contigu (côte à côte)."""
    full = _prepare_predictions_df(preds)
    segments = _segment_predictions_df(full)
    if not segments:
        segments = [full]

    n_seg = len(segments)
    fig_w = max(7.0 * n_seg, 14.0)
    fig, axes_grid = plt.subplots(
        2,
        n_seg,
        figsize=(fig_w, 8.8),
        sharey="row",
        gridspec_kw={"height_ratios": [2.2, 1.0], "hspace": 0.12, "wspace": 0.18},
    )
    if n_seg == 1:
        axes_grid = np.array([[axes_grid[0]], [axes_grid[1]]])

    stats = _prediction_metrics(full) if not full.empty else {"rmse": 0.0}
    band = 2.0 * stats["rmse"]
    if not full.empty:
        err_all = (full["actual_mw"] - full["predicted_mw"]).to_numpy(dtype=float)
        err_ylim = max(float(np.abs(err_all).max()), band) * 1.15
    else:
        err_ylim = band * 1.15

    axes_flat: list[plt.Axes] = []
    for col, seg in enumerate(segments):
        ax_top = axes_grid[0, col]
        ax_bottom = axes_grid[1, col]
        mae = _segment_mae(seg)
        caption = _segment_caption(seg, index=col + 1, total=n_seg)
        column_title = f"Bloc {col + 1}/{n_seg} · MAE {mae:,.0f} MW\n{caption}".replace(",", " ")
        _plot_predictions_timeseries_block(
            ax_top,
            ax_bottom,
            seg,
            band=band,
            err_ylim=err_ylim,
            column_title=column_title,
            show_legend_top=(col == n_seg - 1),
            show_legend_bottom=(col == 0),
            show_ylabel=(col == 0),
            show_threshold_labels=(col == n_seg - 1),
        )
        axes_flat.extend([ax_top, ax_bottom])

    if n_seg > 1:
        fig.text(
            0.5,
            0.01,
            "Jeu de test : blocs disjoints — absence de données RTE entre les périodes (non interpolée).",
            ha="center",
            va="bottom",
            fontsize=9,
            color=DASHBOARD["text_muted"],
            style="italic",
        )

    return fig, tuple(axes_flat)


def plot_data_split(
    ax: plt.Axes,
    split_info: dict[str, Any],
) -> None:
    """Camembert — effectifs train / test (split temporel 80/20)."""
    n_train = int(split_info.get("n_train", 0))
    n_test = int(split_info.get("n_test", 0))
    total = n_train + n_test
    if total <= 0:
        ax.text(0.5, 0.5, "Split indisponible", ha="center", va="center", transform=ax.transAxes)
        return

    ax.set_facecolor(DASHBOARD["panel"])
    pct_train = 100 * n_train / total
    pct_test = 100 * n_test / total
    sizes = [n_train, n_test]
    colors = [DASHBOARD["bar"], DASHBOARD["accent"]]
    legend_labels = [
        f"Entraînement · {n_train:,} obs. ({pct_train:.0f} %)".replace(",", " "),
        f"Test · {n_test:,} obs. ({pct_test:.0f} %)".replace(",", " "),
    ]

    def _autopct(pct: float) -> str:
        count = int(round(pct * total / 100.0))
        return f"{pct:.0f} %\n{count:,} obs.".replace(",", " ")

    wedges, _texts, autotexts = ax.pie(
        sizes,
        colors=colors,
        startangle=90,
        counterclock=False,
        autopct=_autopct,
        pctdistance=0.72,
        wedgeprops={"edgecolor": "white", "linewidth": 2.0, "alpha": 0.95},
    )
    for t in autotexts:
        t.set_fontsize(11)
        t.set_fontweight("bold")
        t.set_color(COLORS["white"])

    ax.legend(
        wedges,
        legend_labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        facecolor="white",
        edgecolor=DASHBOARD["panel_border"],
        fontsize=10,
    )
    ax.set_aspect("equal")

    def _fmt_period(start_key: str, end_key: str) -> str:
        try:
            s = pd.Timestamp(split_info.get(start_key)).strftime("%d/%m/%Y")
            e = pd.Timestamp(split_info.get(end_key)).strftime("%d/%m/%Y")
            return f"{s} → {e}"
        except Exception:
            return "—"

    ax.text(
        0.5,
        1.02,
        f"Split temporel 80/20 · {total:,} enregistrements · non aléatoire".replace(",", " "),
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color=COLORS["neutral"],
    )
    caption = (
        f"Entraînement : {_fmt_period('train_start', 'train_end')}   ·   "
        f"Test : {_fmt_period('test_start', 'test_end')}\n"
        "Découpage temporel strict : les données de test sont postérieures à l'entraînement."
    )
    ax.text(0.5, -0.08, caption, transform=ax.transAxes, ha="center", fontsize=8.5, color=DASHBOARD["text_muted"])


def build_model_comparison_figure(metrics: pd.DataFrame) -> tuple[plt.Figure, tuple[plt.Axes]]:
    """Figure unique — barres groupées RMSE + pénalité R²."""
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    plot_model_comparison_grouped(ax, metrics)
    return fig, (ax,)


def build_ml_synthesis_figure(
    metrics: pd.DataFrame,
    *,
    best_model: str | None = None,
) -> tuple[plt.Figure, tuple[plt.Axes]]:
    """Tableau comparatif unique avec note de sélection."""
    fig, ax = plt.subplots(figsize=(12, 5.8))
    footnote = synthesis_footnote(metrics, best_model=best_model)
    plot_metrics_detail_table(ax, metrics, footnote=footnote)
    return fig, (ax,)
