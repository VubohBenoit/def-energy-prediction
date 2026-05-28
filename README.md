# EDF ETL Platform

**Version 1.0.0**

Plateforme de **data engineering** et **machine learning** pour les données **RTE éco2mix** (consommation, mix énergétique, échanges, CO₂). Elle implémente une architecture médaillon **Bronze → Silver → Gold → Modeling**, orchestrée par **Apache Airflow** et exécutable en local via **Docker Compose** et le **Makefile**.

| Domaine | Description |
|---------|-------------|
| Ingestion | Fichiers XLS batch et flux Kafka (delta quotidien) |
| Stockage | MinIO (lac de données) + PostgreSQL (entrepôt & audit) |
| Traitement | Apache Spark 3.5 (ETL et entraînement ML) |
| Livrables | Rapports PNG, modèles ML, métriques et dashboards Grafana |

> **Périmètre actuel :** environnement local Docker Compose (développement, recette, démonstration). La logique métier est portable vers un déploiement cloud (Kubernetes, MWAA, EMR, etc.) ; l'infrastructure cible reste à adapter.

---

## Sommaire

1. [Architecture](#architecture)
2. [Démarrage rapide](#démarrage-rapide)
3. [Chaîne de traitement](#chaîne-de-traitement)
4. [Modes d'exécution](#modes-dexécution)
5. [Orchestration Airflow](#orchestration-airflow)
6. [Rapports et modèles ML](#rapports-et-modèles-ml)
7. [Qualité et observabilité](#qualité-et-observabilité)
8. [Configuration](#configuration)
9. [Interfaces](#interfaces)
10. [Schéma PostgreSQL](#schéma-postgresql)
11. [Tests](#tests)
12. [Structure du projet](#structure-du-projet)
13. [Runbook opérationnel](#runbook-opérationnel)

---

## Architecture

```
                  ┌─────────────────────────────────────────────────┐
                  │             XLS RTE éco2mix (TSV)               │
                  └─────────────────┬───────────────────────────────┘
                                    │
            ┌───────────────────────┴───────────────────────┐
            │                                               │
            ▼  (streaming Kafka)              ▼  (batch Spark)
  ┌──────────────────────┐                ┌──────────────────────┐
  │  rte_producer        │                │  xls_to_bronze       │
  │  → topics Kafka      │                │                      │
  └──────────┬───────────┘                └──────────┬───────────┘
             │         rte_pipeline.parsing        │
             └────────────────────┬────────────────┘
                                  ▼
                        ┌────────────────────┐
                        │  MinIO — BRONZE    │  s3a://edf-bronze/rte/
                        └─────────┬──────────┘
                                  │  bronze_to_silver
                                  ▼
                        ┌────────────────────┐         PostgreSQL
                        │  MinIO — SILVER    │ ──────► dw.fact_consumption_silver
                        └─────────┬──────────┘
                                  │  silver_to_gold
                                  ▼
                        ┌────────────────────┐         PostgreSQL
                        │  MinIO — GOLD      │ ──────► dw.agg_daily / agg_monthly
                        └─────────┬──────────┘
                                  │  gold_to_model
                                  ▼
                        ┌────────────────────┐
                        │  MODELING (ML)     │ ──────► s3a://models/rte/best/
                        └────────────────────┘         etl.model_metrics
```

**Parsing partagé :** `rte_pipeline/parsing/xls.py` — unique pour Spark, Kafka et Airflow.

**Bronze :** la source de vérité est **MinIO** (Parquet). La table `dw.fact_consumption_bronze` n'est pas alimentée en batch ; l'analyse SQL démarre à **Silver**.

### Buckets MinIO

| Bucket | Contenu |
|--------|---------|
| `edf-bronze` | Données brutes Parquet (`rte/raw/`, `rte/tempo/`, streaming) |
| `edf-silver` | Couche nettoyée et enrichie |
| `edf-gold` | Agrégats analytiques |
| `models` | Artefacts Spark ML (`rte/`, `rte/best/`) |
| `rapport-eda` | Graphiques du rapport professionnel (`report/*.png`) |

### Stack technique

| Composant | Rôle | Port (hôte) |
|-----------|------|-------------|
| Apache Kafka | Bus de messagerie (ingestion streaming) | 29092 |
| Apache Spark | ETL et entraînement ML | 7077 / UI 8082 |
| Apache Airflow | Orchestration, retries, audit | 8081 |
| MinIO | Object storage S3-compatible | 9000 / Console 9001 |
| PostgreSQL 16 | Entrepôt DW + audit ETL/ML | 5432 |
| Prometheus | Métriques infrastructure | 9090 |
| Grafana | Dashboards ETL, qualité, ML | 3001 |
| Kafka UI | Inspection topics / messages | 8085 |

---

## Démarrage rapide

### Prérequis

| Élément | Détail |
|---------|--------|
| Docker Desktop | 8 Go RAM recommandés |
| Données source | Fichiers `eCO2mix_RTE_*.xls` dans `./data/raw/` |
| Configuration | Fichier `.env` à la racine du projet |

### Installation

```bash
cp eCO2mix_RTE_*.xls ./data/raw/
make bootstrap
```

`make bootstrap` exécute : création des répertoires locaux, build des images, démarrage de la stack, initialisation Kafka, schéma PostgreSQL et contrôle de santé.

### Premier run complet

**Via Airflow (recommandé, équivalent production) :**

```bash
make pipeline
```

Enchaîne : attente du DAG `edf_pipeline_complet` (ETL, qualité, ML, **rapports EDA dans Airflow**) → affichage des métriques ML (`make ml-metrics`).

Les graphiques 01–04 sont produits **dans le conteneur Airflow** (TaskGroup `reporting`) et persistés dans `./data/eda/report/` + MinIO `rapport-eda`. Après modification du code Airflow ou de `airflow/requirements.txt` : `make build` puis redémarrer scheduler et webserver.

**Via Spark direct (debug, sans scheduler) :**

```bash
make pipeline-spark
```

Enchaîne : ETL jobs 1→3 + rapport données (01–03) → ML + rapport ML (04) → métriques → enregistrement du run dans `etl.pipeline_runs`.

---

## Chaîne de traitement

| # | Couche | Script | Entrée | Sortie principale |
|---|--------|--------|--------|-------------------|
| 1 | Bronze | `xls_to_bronze.py` | XLS dans `data/raw/` | `s3a://edf-bronze/rte/` |
| 2 | Silver | `bronze_to_silver.py` | Bronze MinIO | Silver MinIO + `dw.fact_consumption_silver` |
| 3 | Gold | `silver_to_gold.py` | Silver | Gold MinIO + `dw.agg_daily`, `dw.agg_monthly` |
| 4 | ML | `gold_to_model.py` | Silver ML-ready | `s3a://models/rte/` + `etl.model_metrics` |

### Ingestion Bronze

| Mode | Mécanisme | Usage |
|------|-----------|-------|
| **Batch** | Spark `xls_to_bronze` | Historique multi-années, recette complète |
| **Streaming** | XLS → Kafka → consumer → Bronze | Delta quotidien J/J-1 |

Les couches Silver, Gold et ML partagent le même code Spark, quel que soit le chemin Bronze.

---

## Modes d'exécution

### Comparatif

| Critère | Spark direct (`Makefile`) | Airflow |
|---------|---------------------------|---------|
| Déclenchement | Commandes `make run-*` | Scheduler cron ou UI |
| Retries | Relance manuelle | Configurables (ex. 2 × 10 min) |
| Audit | Logs conteneur + tables DW | `etl.pipeline_runs` + logs tâches |
| Usage | Développement, debug, job isolé | Production, SLA, historique |
| Pipeline complet | `make pipeline-spark` | `make pipeline` ou DAG `edf_pipeline_complet` |
| Rapport EDA (01–04) | `make report-eda*` (CLI hôte) | Tâches `reporting.*` dans le DAG (prod & dev) |

### Commandes Makefile

| Commande | Description |
|----------|-------------|
| `make bootstrap` | Build, stack, Kafka, schéma PostgreSQL |
| `make pipeline` | Attente du DAG `edf_pipeline_complet` (EDA inclus) + `ml-metrics` |
| `make pipeline-spark` | ETL + ML via Spark REST (sans Airflow) + rapports CLI |
| `make run-etl` | Jobs 1→3, contrôles qualité, rapport données (01–03) |
| `make run-ml` | Job 4, rapport ML (04) |
| `make report-eda` | Rapport complet en CLI locale (relance manuelle) |
| `make report-eda-data` | Graphiques 01–03 en CLI locale |
| `make report-eda-ml` | Graphique 04 en CLI locale (après entraînement ML) |
| `make sync-models-local` | Télécharge `models/` MinIO → `data/models/rte/` |
| `make ml-metrics` | Affiche les dernières métriques PostgreSQL |
| `make trigger-pipeline` | Déclenche `edf_pipeline_complet` sans attendre |
| `make health` | Contrôle de santé des services |
| `make urls` | URLs et identifiants des interfaces |

**Jobs individuels :**

```bash
make run-xls-to-bronze
make run-bronze-to-silver
make run-silver-to-gold
make run-gold-to-model
```

---

## Orchestration Airflow

### Environnement dev / prod

| Variable | Dev (défaut) | Prod |
|----------|--------------|------|
| `EDF_ENVIRONMENT` | `dev` | `prod` |
| Planification DAGs | Aucune (trigger manuel) | Crons actifs |

| Variable | Cron prod (défaut) | DAG |
|----------|-------------------|-----|
| `EDF_PIPELINE_SCHEDULE` | `0 3 * * 0` (dim. 03:00 UTC) | `edf_pipeline_complet` |
| `EDF_ETL_SCHEDULE` | `0 2 * * *` (quotidien 02:00 UTC) | `edf_etl_pipeline` |
| `EDF_ML_SCHEDULE` | `0 4 * * 1` (lun. 04:00 UTC) | `edf_ml_pipeline` |
| `EDF_QUALITY_SCHEDULE` | `30 6 * * *` (quotidien 06:30 UTC) | `edf_quality_monitoring` |

Activation production :

```bash
# .env
EDF_ENVIRONMENT=prod

docker compose restart airflow-scheduler
```

### DAG `edf_pipeline_complet`

Pipeline batch complet (équivalent fonctionnel de `make pipeline`), avec retries et traçabilité.

```
start → prerequisites → bronze → silver → gold → quality
     → reporting.generate_eda_data_report (graphiques 01–03)
     → ml.check_ml_readiness
           ├─ ml.run_gold_to_model → reporting.generate_eda_ml_report (04)
           └─ ml.skip_ml_training → reporting.mark_eda_ml_pending
     → finalize_pipeline_run → end
```

| Paramètre | Valeur |
|-----------|--------|
| Retries | 2 (pipeline), 1 (tâches EDA) |
| Délai entre retries | 10 min |
| Timeout | 8 h (pipeline), 3 h (ML), 20 min (EDA données) |
| ML conditionnel | Skip si Silver < `ML_MIN_TRAINING_ROWS` (1000) |
| EDA non bloquant | Par défaut : échec rapport ≠ échec pipeline (`EDA_FAIL_PIPELINE=false`) |

**Soumission Spark :** `SparkRestSubmitOperator` via API REST du master (`http://spark-master:6066`). Scripts montés dans `/opt/spark-jobs/`. L'image Airflow n'inclut ni PySpark ni Java ; elle inclut **matplotlib** pour les rapports EDA.

**Logique EDA partagée :** `spark/common/eda_report.py` (source unique), appelée par Airflow (`edf_pipeline/eda_report.py`) et par la CLI (`scripts/generate_report_eda.py`).

### Autres DAGs

| DAG | Rôle | Trigger manuel |
|-----|------|----------------|
| `edf_etl_pipeline` | Streaming Kafka → Bronze → Silver → Gold | `make trigger-dag DAG=edf_etl_pipeline` |
| `edf_ml_pipeline` | Ré-entraînement ML + rapport 04 (sans relancer l'ETL) | `make trigger-dag DAG=edf_ml_pipeline` |
| `edf_quality_monitoring` | Contrôles qualité globaux DW | `make trigger-dag DAG=edf_quality_monitoring` |

### Calendrier type production (UTC)

| Heure | Fréquence | DAG |
|-------|-----------|-----|
| 02:00 | Quotidien | `edf_etl_pipeline` |
| 03:00 | Dimanche | `edf_pipeline_complet` |
| 04:00 | Lundi | `edf_ml_pipeline` |
| 06:30 | Quotidien | `edf_quality_monitoring` |

Interface : [http://localhost:8081](http://localhost:8081) — identifiants `admin` / `admin123`.

---

## Rapports et modèles ML

### Rapport visuel

**Orchestration (recommandé — dev & prod) :** tâches Airflow du TaskGroup `reporting` dans `edf_pipeline_complet` (et graphique 04 dans `edf_ml_pipeline`). Même logique que la production planifiée (`EDF_ENVIRONMENT=prod`).

**CLI locale (debug / régénération manuelle) :** `scripts/generate_report_eda.py` via `make report-eda*`.

Les graphiques sont persistés dans **`./data/eda/report/`** (volume Docker monté côté Airflow) et dans le bucket MinIO **`rapport-eda`**.

| Fichier | Contenu | Disponibilité |
|---------|---------|---------------|
| `01_consommation.png` | Série horaire + moyenne mobile 7 j | Fin ETL (`reporting.generate_eda_data_report`) ou `make report-eda-data` |
| `02_volume_mensuel.png` | Volume mensuel (TWh) | Idem |
| `03_qualite_donnees.png` | Complétude et score qualité | Idem |
| `04_performance_ml.png` | Benchmark des 4 algorithmes, métriques, modèle retenu | Après ML (`reporting.generate_eda_ml_report`) ou `make report-eda-ml` |

Sans entraînement ML, le graphique 04 n'est pas produit ; un fichier `04_performance_ml.pending` est créé par `reporting.mark_eda_ml_pending`.

```bash
# Production-like (EDA dans le DAG)
make pipeline

# Debug Spark + rapports CLI
make pipeline-spark

# Régénération manuelle sans relancer l'ETL
make report-eda-data
make report-eda-ml      # nécessite etl.model_metrics
make report-eda         # complet
```

### Modèles ML

| Algorithme | Estimateur Spark | Rôle |
|------------|------------------|------|
| Linear Regression | `LinearRegression` | Baseline interprétable |
| Decision Tree | `DecisionTreeRegressor` | Non-linéarités locales |
| Random Forest | `RandomForestRegressor` | Robustesse |
| Gradient Boosting | `GBTRegressor` | Performance (souvent meilleur RMSE) |

**Sélection :** modèle retenu = RMSE minimal sur le jeu de test (split temporel 80/20).

**Métriques :** RMSE, MAE, MAPE %, R² — historisées dans `etl.model_metrics`.

**Persistance :**

| Emplacement | Chemin |
|-------------|--------|
| MinIO | `s3a://models/rte/` et `s3a://models/rte/best/` |
| Local | `data/models/rte/` (sync automatique après entraînement) |

---

## Qualité et observabilité

| Niveau | Déclencheur |
|--------|-------------|
| Post-ETL batch | Fin de `edf_pipeline_complet` |
| Streaming | Fin de `edf_etl_pipeline` |
| Monitoring global | DAG `edf_quality_monitoring` (06:30 UTC en prod) |

**Persistance :** `etl.data_quality_checks`, `etl.pipeline_runs`, `etl.model_metrics`.

**Monitoring :**

| Outil | URL | Contenu |
|-------|-----|---------|
| Grafana | http://localhost:3001 (`admin` / `edf-admin`) | 2 dashboards provisionnés dans le dossier **EDF ETL** |
| Prometheus | http://localhost:9090 | Scrapes Postgres, Kafka, MinIO, Spark |

| Dashboard Grafana | UID | Contenu |
|-------------------|-----|---------|
| **Vue plateforme** (accueil) | `edf-platform-overview` | Santé infra (UP, lag Kafka, Spark, MinIO), runs ETL, qualité |
| **Analytics & ML** | `edf-data-analytics` | Consommation RTE, mix énergétique, benchmark RMSE, évolution ML |

Les graphiques métier utilisent le **dernier jeu de données DW** (fenêtre glissante relative à `MAX(datetime)`), adapté aux jeux historiques multi-années.

```bash
make health
make status
make logs
make ml-metrics
```

Variable `QUALITY_FAIL_ON_WARNING=true` dans `.env` pour faire échouer le pipeline sur les warnings qualité.

---

## Configuration

Fichier `.env` à la racine, chargé par Docker Compose, Airflow et les scripts locaux.

### Orchestration

| Variable | Défaut | Description |
|----------|--------|-------------|
| `EDF_ENVIRONMENT` | `dev` | `dev` = DAGs manuels ; `prod` = crons actifs |
| `SPARK_REST_URL` | `http://spark-master:6066` | API REST Spark (Airflow) |
| `SPARK_JOBS_DIR` | `/opt/spark-jobs` | Scripts PySpark sur le cluster |
| `POSTGRES_CONN` | `postgresql://edf:…` | DSN PostgreSQL |
| `BRONZE_INCLUDE_STREAMING` | `true` | Fusionne `bronze/streaming/` dans Silver |

### Lac de données (MinIO)

| Variable | Défaut | Description |
|----------|--------|-------------|
| `BRONZE_PATH` | `s3a://edf-bronze/rte/` | Préfixe Bronze |
| `SILVER_PATH` | `s3a://edf-silver/rte/` | Préfixe Silver |
| `GOLD_PATH` | `s3a://edf-gold/rte/` | Préfixe Gold |
| `ML_MODEL_PATH` | `s3a://models/rte/` | Modèles ML sur MinIO |
| `MODEL_LOCAL_PATH` | `data/models/rte` (hôte) | Miroir local des modèles |
| `REPORT_EDA_BUCKET` | `rapport-eda` | Bucket PNG du rapport EDA |
| `REPORT_EDA_LOCAL` | `data/eda/report` | Chemin local hôte ; Airflow : `/opt/airflow/data/eda/report` |
| `DATA_DIR` | `./data/raw` | XLS source (hôte) ; Airflow : `/opt/airflow/data/raw` |
| `EDA_FAIL_PIPELINE` | `false` | Si `true`, échec rapport EDA = échec DAG |

### Machine Learning

| Variable | Défaut | Description |
|----------|--------|-------------|
| `ML_LABEL_COL` | `consumption_mw` | Variable cible |
| `ML_TRAIN_RATIO` | `0.8` | Split temporel train/test |
| `ML_MIN_TRAINING_ROWS` | `1000` | Seuil minimal Silver pour entraîner |
| `ML_RF_NUM_TREES` / `ML_GBT_MAX_ITER` | 30 | Hyperparamètres |

---

## Interfaces

| Service | URL | Identifiants |
|---------|-----|--------------|
| Airflow | http://localhost:8081 | admin / admin123 |
| Spark Master UI | http://localhost:8082 | — |
| MinIO Console | http://localhost:9001 | edfadmin / edfpassword123 |
| Kafka UI | http://localhost:8085 | — |
| Grafana | http://localhost:3001 | admin / edf-admin |
| Prometheus | http://localhost:9090 | — |
| PostgreSQL | localhost:5432 / `edf_dw` | edf / edf123 |

Liste complète : `make urls`.

---

## Schéma PostgreSQL

| Objet | Contenu |
|-------|---------|
| MinIO `edf-bronze` | Bronze Parquet (hors Postgres) |
| `dw.fact_consumption_silver` | Faits nettoyés + features (partition année) |
| `dw.agg_daily` | Agrégats journaliers |
| `dw.agg_monthly` | Agrégats mensuels + YoY |
| `etl.pipeline_runs` | Historique des exécutions |
| `etl.data_quality_checks` | Résultats contrôles qualité |
| `etl.model_metrics` | Métriques par modèle et par run |

Schéma source : `infra/postgres/schema_dw.sql`.

---

## Tests

```bash
make dev-venv   # première fois : venv isolé (recommandé macOS)
make test       # suite pytest
```

Couverture : parsing XLS RTE, contrôles qualité, planification dev/prod, sérialisation Parquet.

**macOS — erreur `Floating-point exception` :** utiliser le venv projet (`make dev-venv`) plutôt qu'un Python système avec des wheels numpy/pyarrow incompatibles.

---

## Structure du projet

```
edf-etl-platform/
├── airflow/
│   ├── dags/                    # DAGs Airflow
│   └── plugins/edf_pipeline/    # Tâches, qualité, Spark REST, eda_report, audit
├── spark/
│   ├── common/                  # config, session, object_storage, eda_style, eda_report
│   ├── transform/               # Logique médaillon
│   ├── ml/                      # training, metrics_store
│   └── jobs/                    # Orchestrateurs (xls_to_bronze, …)
├── scripts/
│   └── generate_report_eda.py   # CLI rapport EDA (wrapper vers spark.common.eda_report)
├── rte_pipeline/
│   └── parsing/xls.py           # Parsing RTE partagé
├── infra/                       # PostgreSQL, Prometheus, Grafana
├── data/
│   ├── raw/                     # XLS source
│   ├── eda/report/              # Rapports PNG (miroir local)
│   └── models/rte/              # Modèles ML (miroir local)
├── tests/
├── Makefile
├── docker-compose.yml
└── .env
```

---

## Runbook opérationnel

### Relancer une couche isolée

```bash
make run-bronze-to-silver
make run-silver-to-gold
make run-gold-to-model
```

### Régénérer les rapports sans relancer l'ETL

**CLI locale (hôte) :**

```bash
make report-eda-data
make report-eda-ml      # nécessite des métriques dans etl.model_metrics
```

**Via Airflow :** relancer uniquement les tâches `reporting.generate_eda_data_report` et/ou `reporting.generate_eda_ml_report` depuis l'UI (Clear → Downstream).

### Après modification du plugin Airflow ou des DAGs

```bash
make build
docker compose restart airflow-scheduler airflow-webserver
```

### Rattrapage modèles locaux

```bash
make sync-models-local
```

### Réinitialisation complète

```bash
make clean      # arrêt + suppression des volumes Docker
make bootstrap
```

### Conventions de commit

Format [Conventional Commits](https://www.conventionalcommits.org/) : `feat:`, `fix:`, `chore:`, `docs:`, `refactor:`, `test:`.

---

## Référence données RTE

Plus de **40 colonnes** mappées dans `rte_pipeline/parsing/xls.py` : consommation, prévisions J/J-1, nucléaire, éolien, solaire, hydraulique, gaz, fioul, charbon, bioénergies, stockage, échanges et taux CO₂.
