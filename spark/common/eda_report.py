# =======================================================================
# **************    Projet : EDF Energy Prediction         **************
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
from spark.common.config import DATA_DIR, REPORT_EDA_LOCAL, resolve_model_local_path
from spark.common.eda_style import (
    COLORS,
    HEADER_BOTTOM,
    add_accent_bar,
    add_chart_header,
    add_report_footer,
    apply_dashboard_style,
    apply_figure_layout,
    build_ml_synthesis_figure,
    build_model_comparison_figure,
    build_predictions_dispersion_figure,
    build_predictions_timeseries_figure,
    plot_consumption_series,
    plot_data_split,
    plot_learning_curve_rf,
    plot_missing_bars,
    plot_quality_distribution,
    prepare_time_series,
    save_report_figure,
    human_model_name,
    describe_learning_curve,
    synthesis_footnote,
    _style_ax_dashboard,
)
from spark.common.object_storage import (
    REPORT_EDA_BUCKET,
    persist_report_file,
    remove_report_from_s3,
    sync_ml_report_artifacts,
    sync_model_artifacts,
)

logger = logging.getLogger(__name__)

# (titre affiché, nom de fichier PNG)
DATA_CHART_SPECS: tuple[tuple[str, str], ...] = (
    ("Consommation électrique nationale", "consommation_electrique_nationale.png"),
    ("Volume mensuel de consommation", "volume_mensuel_de_consommation.png"),
    ("Qualité des données source", "qualite_des_donnees_source.png"),
)

ML_CHART_SPECS: tuple[tuple[str, str, str], ...] = (
    (
        "Courbe d'apprentissage – Forêt aléatoire (RMSE vs taille d'entraînement)",
        "courbe_apprentissage_foret_aleatoire.png",
        "Entraînement (bleu) vs validation (orange) · axe X logarithmique · trait = meilleur compromis",
    ),
    (
        "Performance des prédictions – Forêt aléatoire (jeu de test)",
        "predictions_dispersion_reel_vs_predit.png",
        "Graphique A · nuage de points coloré par |erreur| · bande ±2×RMSE · courbe LOESS",
    ),
    (
        "Performance des prédictions – Forêt aléatoire (jeu de test)",
        "predictions_serie_temporelle_ecarts.png",
        "Graphique B · 2 blocs côte à côte (données RTE incomplètes entre les périodes) · MM 6 h · MAE par bloc",
    ),
    (
        "Comparaison des modèles – Validation temporelle (80/20)",
        "comparaison_des_modeles.png",
        "RMSE (% du meilleur) + pénalité (1−R²) · palette vert → rouge = rang",
    ),
    (
        "Synthèse de performance ML",
        "synthese_performance_ml.png",
        "Tableau comparatif · meilleure valeur en gras par colonne · Δ RMSE % vs forêt aléatoire",
    ),
    (
        "Répartition des données – Split temporel",
        "repartition_des_donnees.png",
        "Camembert 80/20 · effectifs et périodes · découpage temporel strict (test = période future)",
    ),
)

ML_PENDING = "ml_dashboard.pending"

OBSOLETE_CHARTS: tuple[str, ...] = (
    # anciens noms numérotés — données
    "01_consommation.png",
    "02_volume_mensuel.png",
    "03_qualite_donnees.png",
    # anciens noms numérotés — ML
    "04_performance_ml.png",
    "04_learning_curve_rf.png",
    "05_learning_curve_rf.png",
    "05_predictions_scatter.png",
    "06_predictions_vs_actual.png",
    "06_predictions_serie.png",
    "07_r2_comparison.png",
    "08_rmse_comparison.png",
    "09_metrics_detail.png",
    "10_best_model.png",
    "11_repartition_donnees.png",
    "04_performance_ml.pending",
    # graphiques ML éclatés (remplacés par figures combinées)
    "performance_des_predictions.png",
    "predictions_vs_consommation_reelle.png",
    "serie_temporelle_jeu_de_test.png",
    "comparaison_r2.png",
    "comparaison_rmse.png",
    "metriques_detaillees.png",
    "modele_retenu.png",
)

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


def ml_artifact_dir() -> Path:
    """Répertoire local des artefacts ML (_report) pour les graphiques dashboard."""
    return resolve_model_local_path() / "_report"


def ensure_ml_artifacts_local() -> Path:
    """Télécharge les artefacts ML depuis MinIO si absents en local."""
    artifact_dir = ml_artifact_dir()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    if (artifact_dir / "predictions.parquet").exists() or (artifact_dir / "learning_curve.parquet").exists():
        return artifact_dir

    try:
        n = sync_ml_report_artifacts(artifact_dir)
        logger.info("Sync artefacts ML MinIO → %s (%d objet(s))", artifact_dir, n)
    except Exception as exc:
        logger.warning("Sync artefacts ML échouée : %s", exc)

    if not (artifact_dir / "predictions.parquet").exists():
        try:
            sync_model_artifacts()
            sync_ml_report_artifacts(artifact_dir)
        except Exception as exc:
            logger.debug("sync_model_artifacts: %s", exc)

    return artifact_dir


def load_ml_artifacts() -> dict[str, Any]:
    """Charge prédictions, courbe RF et split depuis le disque local."""
    artifact_dir = ensure_ml_artifacts_local()
    out: dict[str, Any] = {"artifact_dir": str(artifact_dir)}

    pred_path = artifact_dir / "predictions.parquet"
    if pred_path.exists():
        try:
            out["predictions"] = pd.read_parquet(pred_path)
        except Exception as exc:
            logger.warning("Unable to read predictions: %s", exc)
            out["predictions"] = pd.DataFrame()

    curve_path = artifact_dir / "learning_curve.parquet"
    if curve_path.exists():
        try:
            out["learning_curve"] = pd.read_parquet(curve_path)
        except Exception as exc:
            logger.warning("Unable to read learning curve: %s", exc)
            out["learning_curve"] = pd.DataFrame()

    split_path = artifact_dir / "split_summary.json"
    if split_path.exists():
        try:
            import json

            out["split_info"] = json.loads(split_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Unable to read split summary: %s", exc)

    return out


def _persist_chart(local_path: str) -> str | None:
    uri = persist_report_file(local_path)
    if uri:
        logger.info("MinIO %s", uri)
    return uri


def _enrich_ml_subtitle(spec_index: int, metrics: pd.DataFrame) -> str:
    """Ajoute une phrase d'interprétation contextuelle au sous-titre du graphique."""
    _, _, base = ML_CHART_SPECS[spec_index]
    if spec_index != 3 or metrics.empty:
        return base
    ordered = metrics.sort_values("rmse").reset_index(drop=True)
    best = ordered.iloc[0]
    best_name = human_model_name(str(best["model_name"]))
    best_rmse = float(best["rmse"])
    linear = metrics[metrics["model_name"].str.contains("linear", case=False, na=False)]
    if not linear.empty and float(linear.iloc[0]["rmse"]) > best_rmse:
        lr_rmse = float(linear.iloc[0]["rmse"])
        reduction = (1.0 - best_rmse / lr_rmse) * 100.0
        return (
            f"{base} · {best_name} réduit l'erreur de {reduction:.0f} % "
            "par rapport à la régression linéaire"
        )
    if len(ordered) >= 2:
        worst_rmse = float(ordered.iloc[-1]["rmse"])
        reduction = (1.0 - best_rmse / worst_rmse) * 100.0
        worst_name = human_model_name(str(ordered.iloc[-1]["model_name"]))
        return f"{base} · {best_name} réduit l'erreur de {reduction:.0f} % vs {worst_name}"
    return base


def _save_dashboard_figure(
    fig: plt.Figure,
    spec_index: int,
    out: Path,
    *,
    subtitle: str | None = None,
) -> str:
    title, filename, default_subtitle = ML_CHART_SPECS[spec_index]
    add_accent_bar(fig)
    add_chart_header(fig, title, subtitle=subtitle or default_subtitle)
    add_report_footer(fig)
    if spec_index in (1, 2):
        bottom = 0.14 if spec_index == 2 else 0.12
        fig.subplots_adjust(left=0.08, right=0.98, top=HEADER_BOTTOM - 0.01, bottom=bottom)
        if spec_index == 2:
            fig.subplots_adjust(hspace=0.12, wspace=0.18)
    elif spec_index == 4:
        apply_figure_layout(fig, top=HEADER_BOTTOM - 0.02)
        fig.subplots_adjust(bottom=0.16)
    elif spec_index == 5:
        apply_figure_layout(fig, top=HEADER_BOTTOM - 0.02)
        fig.subplots_adjust(bottom=0.18)
    else:
        apply_figure_layout(fig)
    return save_report_figure(fig, str(out / filename))


def chart_consumption(df: pd.DataFrame, out: Path) -> str:
    ts = prepare_time_series(df, "datetime", "consumption_mw")
    fig, ax = plt.subplots(figsize=(12, 5.8))
    plot_consumption_series(
        ax,
        ts,
        "datetime",
        "consumption_mw",
        ma_days=7,
    )
    add_accent_bar(fig)
    add_chart_header(
        fig,
        DATA_CHART_SPECS[0][0],
        subtitle="Série horaire RTE éco2mix — tendance lissée sur 7 jours",
    )
    add_report_footer(fig)
    apply_figure_layout(fig)
    return save_report_figure(fig, str(out / DATA_CHART_SPECS[0][1]))


def chart_monthly(monthly: pd.DataFrame, out: Path) -> str:
    fig, ax = plt.subplots(figsize=(12, 5.8))
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
        _style_ax_dashboard(ax)
        m = monthly.sort_values("datetime")
        ax.bar(
            m["label"],
            m["consumption_total_twh"],
            color=COLORS["secondary"],
            width=0.72,
            alpha=0.82,
            edgecolor="white",
            linewidth=0.6,
        )
        ax.plot(
            m["label"],
            m["consumption_total_twh"],
            color=COLORS["primary"],
            marker="o",
            markersize=5,
            linewidth=1.8,
            markerfacecolor="white",
            markeredgewidth=1.5,
        )
        ax.set_xlabel("Mois")
        ax.set_ylabel("Énergie (TWh)")
        step = max(1, len(m) // 12)
        ax.set_xticks(range(0, len(m), step))
        ax.set_xticklabels(m["label"].iloc[::step], rotation=45, ha="right")
    add_accent_bar(fig)
    add_chart_header(
        fig,
        DATA_CHART_SPECS[1][0],
        subtitle="Volume mensuel agrégé à partir des mesures horaires",
    )
    add_report_footer(fig)
    apply_figure_layout(fig)
    return save_report_figure(fig, str(out / DATA_CHART_SPECS[1][1]))


def chart_quality(df: pd.DataFrame, out: Path) -> str:
    missing = (df.isna().mean() * 100).sort_values(ascending=False)
    scores = compute_quality_score(df)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    plot_missing_bars(axes[0], missing, title="Variables les plus incomplètes", top_n=8)
    plot_quality_distribution(
        axes[1], scores, title="Score de qualité des enregistrements"
    )
    add_accent_bar(fig)
    add_chart_header(
        fig,
        DATA_CHART_SPECS[2][0],
        subtitle="Complétude des champs et distribution du score qualité (0 = faible, 1 = excellent)",
    )
    add_report_footer(fig)
    apply_figure_layout(fig)
    return save_report_figure(fig, str(out / DATA_CHART_SPECS[2][1]))


def _cleanup_obsolete_charts(out: Path) -> None:
    for name in OBSOLETE_CHARTS:
        path = out / name
        if path.exists():
            path.unlink()
        remove_report_from_s3(name)
    pending = out / ML_PENDING
    if pending.exists():
        pending.unlink()


def chart_learning_curve(curve: pd.DataFrame, out: Path) -> str:
    fig, ax = plt.subplots(figsize=(11, 5.8))
    plot_learning_curve_rf(ax, curve)
    base = ML_CHART_SPECS[0][2]
    detail = describe_learning_curve(curve)
    return _save_dashboard_figure(fig, 0, out, subtitle=f"{base} · {detail}")


def chart_predictions_dispersion(preds: pd.DataFrame, out: Path) -> str:
    fig, _axes = build_predictions_dispersion_figure(preds)
    return _save_dashboard_figure(fig, 1, out)


def chart_predictions_timeseries(preds: pd.DataFrame, out: Path) -> str:
    fig, _axes = build_predictions_timeseries_figure(preds)
    return _save_dashboard_figure(fig, 2, out)


def chart_model_comparison(metrics: pd.DataFrame, out: Path) -> str:
    fig, _axes = build_model_comparison_figure(metrics)
    return _save_dashboard_figure(
        fig, 3, out, subtitle=_enrich_ml_subtitle(3, metrics),
    )


def chart_ml_synthesis(metrics: pd.DataFrame, out: Path, *, best_model: str | None = None) -> str:
    fig, _axes = build_ml_synthesis_figure(metrics, best_model=best_model)
    foot = synthesis_footnote(metrics, best_model=best_model)
    base = ML_CHART_SPECS[4][2]
    return _save_dashboard_figure(fig, 4, out, subtitle=f"{base} · {foot}")


def chart_data_split(split_info: dict[str, Any], out: Path) -> str:
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    plot_data_split(ax, split_info)
    return _save_dashboard_figure(fig, 5, out)


def mark_ml_pending(out: Path | None = None) -> None:
    out = out or report_output_dir()
    for _title, name, _subtitle in ML_CHART_SPECS:
        path = out / name
        if path.exists():
            path.unlink()
        remove_report_from_s3(name)
    _cleanup_obsolete_charts(out)
    pending = out / ML_PENDING
    pending.write_text(
        "Dashboard ML en attente.\n"
        "Exécutez le job ML (DAG edf_ml_pipeline ou make run-gold-to-model), "
        "puis regénérez via make report-eda-ml.\n",
        encoding="utf-8",
    )


def generate_data_charts(out: Path | None = None) -> list[str]:
    out = out or report_output_dir()
    apply_dashboard_style()
    _cleanup_obsolete_charts(out)

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


def generate_ml_charts(
    out: Path | None = None,
    *,
    pending_if_missing: bool = True,
) -> list[str]:
    """Génère les tuiles ML (une métrique = un PNG, prêt pour Streamlit)."""
    out = out or report_output_dir()
    apply_dashboard_style()

    loaded = load_ml_metrics()
    if loaded is None:
        if pending_if_missing:
            mark_ml_pending(out)
            logger.info("ML dashboard pending — no metrics in etl.model_metrics")
        return []

    metrics, _run_id, _trained_at = loaded
    artifacts = load_ml_artifacts()
    preds = artifacts.get("predictions", pd.DataFrame())
    curve = artifacts.get("learning_curve", pd.DataFrame())
    split_info = artifacts.get("split_info") or {"n_train": 0, "n_test": 0}
    best_model = str(metrics.sort_values("rmse").iloc[0]["model_name"])

    _cleanup_obsolete_charts(out)

    paths: list[str] = [
        chart_model_comparison(metrics, out),
        chart_ml_synthesis(metrics, out, best_model=best_model),
    ]

    if not curve.empty:
        paths.insert(0, chart_learning_curve(curve, out))
    else:
        logger.warning(
            "Artefact courbe d'apprentissage absent (%s) — relancez make run-gold-to-model",
            ML_CHART_SPECS[0][1],
        )

    if not preds.empty:
        insert_at = 1 if not curve.empty else 0
        paths.insert(insert_at, chart_predictions_dispersion(preds, out))
        paths.insert(insert_at + 1, chart_predictions_timeseries(preds, out))
    else:
        logger.warning(
            "Artefact prédictions absent (%s, %s) — relancez make run-gold-to-model",
            ML_CHART_SPECS[1][1],
            ML_CHART_SPECS[2][1],
        )

    if split_info.get("n_train") or split_info.get("n_test"):
        paths.append(chart_data_split(split_info, out))

    logger.info("ML dashboard — %d tile(s), best model: %s", len(paths), best_model)
    for path in paths:
        _persist_chart(path)
    return paths
