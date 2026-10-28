"""
Etapa 3 — Entrenamiento con MLflow  (v4)
Modelos: XGBoost global × 2 variantes de features históricas
Target: predecir viajes_hora para CUALQUIER estación y hora
"""

# ── Imports ────────────────────────────────────────────────────────────────────
import warnings
import logging
import tempfile
import time
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost
import optuna

os.environ["GIT_PYTHON_REFRESH"] = "quiet"
optuna.logging.set_verbosity(optuna.logging.WARNING)

TMP = tempfile.gettempdir()

from sqlalchemy import create_engine
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score, mean_poisson_deviance,
)
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
from xgboost import XGBRegressor

# Con device="cuda", predecir sobre arrays en CPU funciona pero XGBoost avisa
# "mismatched devices" en cada predict
xgb.set_config(verbosity=0)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ── Config ─────────────────────────────────────────────────────────────────────
# Todo se puede sobreescribir con variables de entorno (docker-compose las define).
DB_URL        = os.getenv("DB_URL", "postgresql+psycopg://postgres:CAMBIAR@127.0.0.1:5432/ecobici")
MLFLOW_URI    = os.getenv("MLFLOW_URI", "http://127.0.0.1:5000")
EXPERIMENT    = os.getenv("EXPERIMENT", "ecobici_viajes_hora")

RANDOM_STATE   = 42
TEST_FRAC      = 0.2
OPTUNA_TRIALS  = int(os.getenv("OPTUNA_TRIALS", "30"))
OPTUNA_TIMEOUT = int(os.getenv("OPTUNA_TIMEOUT", str(60 * 60)))  # corta el tuning a la hora aunque falten trials
TUNE_ROWS      = 800_000   # filas usadas SOLO para el tuning (submuestra en orden temporal)
MAX_TREES      = 2000      # techo; el early stopping decide cuántos usar
DEVICE         = os.getenv("DEVICE", "cuda")  # "cpu" dentro de Docker salvo que expongas la GPU
LAG_RED_HORAS  = 1         # horas hacia atrás para la demanda de la red (= horizonte de predicción)
ROLLING_WEEKS  = 4         # ventana deslizante: cuántas semanas hacia atrás


# ── SQL ────────────────────────────────────────────────────────────────────────
# Se cargan por separado: conteos, estaciones y clima. La grilla con ceros y los
# joins se arman en pandas.
QUERY_CONTEOS = """
SELECT
    DATE_TRUNC('hour', v.fecha_origen)  AS hora_ts,
    v.estacion_origen_id                AS estacion_id,
    COUNT(*)                            AS viajes_hora
FROM viajes v
WHERE v.anio IN (2025, 2026)
GROUP BY 1, 2;
"""

QUERY_ESTACIONES = """
SELECT estacion_id, capacidad
FROM estaciones;
"""

# ::timestamp reproduce la misma conversión que hacía Postgres en el JOIN de v3
# (sin efecto si timestamp_utc ya es timestamp sin zona).
QUERY_CLIMA = """
SELECT
    c.timestamp_utc::timestamp  AS hora_ts,
    c.temperatura_c,
    c.sensacion_termica_c,
    c.humedad_pct,
    c.precipitacion_mm,
    c.es_lluvia::INT            AS es_lluvia,
    c.velocidad_viento_kmh
FROM clima c;
"""

TARGET = "viajes_hora"

KEYS_SLOT = ["estacion_id", "hora", "dia_semana"]
KEYS_HORA = ["estacion_id", "hora"]
KEYS_MES  = ["estacion_id", "hora", "dia_semana", "mes"]

FEATURE_COLS_BASE = [
    "hora", "dia_semana", "mes", "es_finde",
    "semana_anio", "parte_dia",
    "hora_parte_dia",        # interacción hora×franja sin colisiones numéricas
    "hora_finde",            # diferencia hábil vs finde por hora
    "demanda_red_prev",      # viajes totales de la red LAG_RED_HORAS horas antes
    "estacion_id", "capacidad",
    "temperatura_c", "sensacion_termica_c",
    "humedad_pct", "precipitacion_mm",
    "es_lluvia", "velocidad_viento_kmh",
    "lluvia_leve", "lluvia_fuerte",
]
FEATURES_ROLLING = FEATURE_COLS_BASE + ["hist_mean", "hist_std"]
FEATURES_MENSUAL = FEATURE_COLS_BASE + ["hist_mean", "hist_std", "hist_mean_hora", "hist_mean_mes"]


# ── Carga y grilla ─────────────────────────────────────────────────────────────
def _sin_tz(s: pd.Series) -> pd.Series:
    """Quita la zona horaria si la hubiera, para que los merges por hora_ts coincidan."""
    s = pd.to_datetime(s)
    return s.dt.tz_localize(None) if s.dt.tz is not None else s


def build_grid(conteos: pd.DataFrame) -> pd.DataFrame:
    """
    Una fila por estación y por hora, desde el primer hasta el último viaje de cada
    estación (así no se inventan ceros antes de que una estación existiera).
    Las horas sin viajes quedan con viajes_hora = 0.
    """
    rangos = conteos.groupby("estacion_id")["hora_ts"].agg(["min", "max"])
    partes = [
        pd.DataFrame({
            "estacion_id": sid,
            "hora_ts": pd.date_range(r["min"], r["max"], freq="h"),
        })
        for sid, r in rangos.iterrows()
    ]
    grid = pd.concat(partes, ignore_index=True)
    grid = grid.merge(conteos, on=["estacion_id", "hora_ts"], how="left")
    grid[TARGET] = grid[TARGET].fillna(0).astype("float32")
    return grid


def load_data() -> pd.DataFrame:
    logger.info("Conectando a la base de datos...")
    engine = create_engine(DB_URL)

    conteos = pd.read_sql(QUERY_CONTEOS, engine)
    conteos["hora_ts"] = _sin_tz(conteos["hora_ts"])
    estaciones = pd.read_sql(QUERY_ESTACIONES, engine)
    clima = pd.read_sql(QUERY_CLIMA, engine)
    clima["hora_ts"] = _sin_tz(clima["hora_ts"])
    clima = clima.drop_duplicates("hora_ts")

    logger.info(f"Conteos: {len(conteos):,} filas con viajes | "
                f"{conteos['estacion_id'].nunique()} estaciones")

    df = build_grid(conteos)
    # inner: igual que v3, solo estaciones presentes en la tabla estaciones
    df = df.merge(estaciones, on="estacion_id", how="inner")
    # left: si falta el clima de una hora, la fila queda con NaN (XGBoost los maneja)
    df = df.merge(clima, on="hora_ts", how="left")

    pct_ceros = (df[TARGET] == 0).mean()
    pct_sin_clima = df["temperatura_c"].isna().mean()
    logger.info(f"Grilla: {len(df):,} filas | {pct_ceros:.1%} horas con 0 viajes | "
                f"{pct_sin_clima:.1%} sin clima")
    return df


# ── Features ───────────────────────────────────────────────────────────────────
def enrich_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features de calendario, clima y demanda de la red."""
    df = df.copy()
    ts = df["hora_ts"]

    df["hora"] = ts.dt.hour
    # Convención de Postgres (EXTRACT DOW): 0 = domingo ... 6 = sábado,
    df["dia_semana"] = (ts.dt.dayofweek + 1) % 7
    df["mes"]        = ts.dt.month
    df["es_finde"]   = df["dia_semana"].isin([0, 6]).astype(int)

    df["semana_anio"] = ts.dt.isocalendar().week.astype(int).values
    df["parte_dia"]   = pd.cut(
        df["hora"], bins=[0, 6, 12, 18, 24], labels=[0, 1, 2, 3], right=False
    ).astype(int)
    df["hora_parte_dia"] = df["hora"] * 10 + df["parte_dia"]
    df["hora_finde"]     = df["hora"] * df["es_finde"]

    df["lluvia_leve"]   = df["precipitacion_mm"].between(0.1, 2).astype(int)
    df["lluvia_fuerte"] = (df["precipitacion_mm"] > 2).astype(int)

    # Demanda de la red LAG_RED_HORAS antes, desplazada por timestamp (no por fila).
    red_por_hora = df.groupby("hora_ts")[TARGET].sum()
    red_prev = red_por_hora.copy()
    red_prev.index = red_prev.index + pd.Timedelta(hours=LAG_RED_HORAS)
    df["demanda_red_prev"] = df["hora_ts"].map(red_prev).astype("float32")
    # Con la grilla completa, un NaN acá solo aparece al inicio del dataset.
    return df


def _expanding_mean_prev(df: pd.DataFrame, keys: list) -> pd.Series:
    """Promedio de TODAS las ocurrencias anteriores del grupo (sin incluir la fila)."""
    prev = df.groupby(keys, sort=False)[TARGET].shift(1)
    grp  = [df[k] for k in keys]
    suma = prev.fillna(0).groupby(grp, sort=False).cumsum()
    n    = prev.notna().astype("int32").groupby(grp, sort=False).cumsum()
    return (suma / n.replace(0, np.nan)).astype("float32")


def add_hist_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features históricas calculadas sobre el dataset completo, en orden cronológico.
    Cada fila solo ve información anterior a su propia hora, así que train y test se
    construyen con la misma lógica que se usaría en producción actualizando las
    lookups periódicamente.

      · hist_mean / hist_std : últimas ROLLING_WEEKS semanas del mismo slot
                               (estación + hora + día de semana)
      · hist_mean_hora       : promedio histórico de estación + hora
      · hist_mean_mes        : promedio histórico de estación + hora + día + mes

    Las primeras semanas quedan en NaN (no hay pasado); XGBoost los maneja.
    """
    t0 = time.time()
    df = df.sort_values(["hora_ts", "estacion_id"]).reset_index(drop=True)

    prev = df.groupby(KEYS_SLOT, sort=False)[TARGET].shift(1)
    roll = prev.groupby([df[k] for k in KEYS_SLOT], sort=False).rolling(
        ROLLING_WEEKS, min_periods=1
    )
    niveles = list(range(len(KEYS_SLOT)))
    df["hist_mean"] = roll.mean().reset_index(level=niveles, drop=True).astype("float32")
    roll_std = prev.groupby([df[k] for k in KEYS_SLOT], sort=False).rolling(
        ROLLING_WEEKS, min_periods=2
    ).std()
    df["hist_std"] = roll_std.reset_index(level=niveles, drop=True).astype("float32")

    df["hist_mean_hora"] = _expanding_mean_prev(df, KEYS_HORA)
    df["hist_mean_mes"]  = _expanding_mean_prev(df, KEYS_MES)

    logger.info(f"Features históricas calculadas en {time.time() - t0:.0f}s")
    return df


def build_lookups(df: pd.DataFrame) -> dict:
    """
    Lookup tables para producción, con TODOS los datos disponibles (train + test),
    porque en producción se predice después del último dato conocido.
    Hay que regenerarlas periódicamente (ej. cada semana) para que el rolling
    no quede viejo.
    """
    ultimas = df.sort_values("hora_ts").groupby(KEYS_SLOT).tail(ROLLING_WEEKS)
    lk_rolling = (
        ultimas.groupby(KEYS_SLOT)[TARGET]
        .agg(hist_mean="mean", hist_std="std")
        .reset_index()
    )
    lk_hora = df.groupby(KEYS_HORA)[TARGET].mean().rename("hist_mean_hora").reset_index()
    lk_mes  = df.groupby(KEYS_MES)[TARGET].mean().rename("hist_mean_mes").reset_index()

    paths = {
        "lookup_rolling": os.path.join(TMP, "lookup_rolling.parquet"),
        "lookup_hora":    os.path.join(TMP, "lookup_hora.parquet"),
        "lookup_mes":     os.path.join(TMP, "lookup_mes.parquet"),
    }
    lk_rolling.to_parquet(paths["lookup_rolling"], index=False)
    lk_hora.to_parquet(paths["lookup_hora"], index=False)
    lk_mes.to_parquet(paths["lookup_mes"], index=False)
    logger.info(f"Lookups: rolling={len(lk_rolling):,} | hora={len(lk_hora):,} | "
                f"mes={len(lk_mes):,}")
    return paths


def temporal_split(df: pd.DataFrame, test_frac: float = TEST_FRAC):
    """Split con una única fecha de corte para todas las estaciones."""
    horas = np.sort(df["hora_ts"].unique())
    corte = horas[int(len(horas) * (1 - test_frac))]
    train = df[df["hora_ts"] < corte].reset_index(drop=True)
    test  = df[df["hora_ts"] >= corte].reset_index(drop=True)
    logger.info(f"Split en {pd.Timestamp(corte)} → train: {len(train):,} | test: {len(test):,}")
    return train, test


# ── Métricas y artefactos ──────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 1e-6, None)
    return {
        "MAE":        mean_absolute_error(y_true, y_pred),
        "RMSE":       float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2":         r2_score(y_true, y_pred),
        # WAPE: error absoluto total / viajes totales. Robusto a ceros.
        "WAPE":       float(np.abs(y_true - y_pred).sum() / max(y_true.sum(), 1e-6) * 100),
        "PoissonDev": mean_poisson_deviance(y_true, y_pred),
    }


def save_red_plot(test_df: pd.DataFrame, y_pred, title: str, path: str, dias: int = 14):
    """Demanda total de la red por hora (real vs predicha) en las primeras semanas de test."""
    red = (
        test_df.assign(y_pred=y_pred)
        .groupby("hora_ts")[[TARGET, "y_pred"]].sum()
        .iloc[: 24 * dias]
    )
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(red.index, red[TARGET], label="Real", alpha=0.8)
    ax.plot(red.index, red["y_pred"], label="Predicho", alpha=0.8, linestyle="--")
    ax.set_title(title)
    ax.set_ylabel("viajes en la red por hora")
    ax.legend()
    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_feature_importance(model, feature_cols: list, model_name: str):
    fi = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    fi_csv = os.path.join(TMP, f"{model_name}_fi.csv")
    fi.to_csv(fi_csv, header=["importance"])
    mlflow.log_artifact(fi_csv, "feature_importance")

    fig, ax = plt.subplots(figsize=(8, max(4, len(feature_cols) * 0.35)))
    fi.plot(kind="barh", ax=ax)
    ax.set_title(f"{model_name} — Feature Importance")
    ax.invert_yaxis()
    plt.tight_layout()
    fi_png = os.path.join(TMP, f"{model_name}_fi.png")
    fig.savefig(fi_png)
    plt.close(fig)
    mlflow.log_artifact(fi_png, "plots")
    return fi


# ── Tuning Optuna ──────────────────────────────────────────────────────────────
def base_params() -> dict:
    return {
        "objective":    "count:poisson",
        "tree_method":  "hist",
        "device":       DEVICE,
        "max_bin":      256,
        "random_state": RANDOM_STATE,
        "n_jobs":       -1,
    }


def tune_xgboost(X_train: pd.DataFrame, y_train: pd.Series, run_prefix: str) -> dict:
    """
    CV temporal sobre el train (que llega ordenado por hora_ts). El test no se toca.
    Métrica: deviance Poisson (coherente con el objetivo del modelo).
    """
    logger.info(f"Optuna tuning ({OPTUNA_TRIALS} trials, CV temporal) para {run_prefix}...")

    X_tr = X_train.values.astype(np.float32)
    y_tr = y_train.values.astype(np.float32)

    # Submuestra aleatoria que CONSERVA el orden temporal (índices ordenados)
    if len(X_tr) > TUNE_ROWS:
        rng = np.random.default_rng(RANDOM_STATE)
        idx = np.sort(rng.choice(len(X_tr), TUNE_ROWS, replace=False))
        X_tr, y_tr = X_tr[idx], y_tr[idx]
    logger.info(f"  Tuning sobre {len(X_tr):,} filas")

    tscv = TimeSeriesSplit(n_splits=3)

    def objective(trial):
        params = {
            **base_params(),
            "n_estimators":          MAX_TREES,
            "early_stopping_rounds": 50,
            "learning_rate":    trial.suggest_float("learning_rate", 0.03, 0.2, log=True),
            "max_depth":        trial.suggest_int("max_depth", 3, 8),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 5, 50),
            "reg_alpha":        trial.suggest_float("reg_alpha", 0.01, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 0.01, 10.0, log=True),
        }
        scores, iters = [], []
        for step, (fold_tr, fold_val) in enumerate(tscv.split(X_tr)):
            m = XGBRegressor(**params)
            m.fit(X_tr[fold_tr], y_tr[fold_tr],
                  eval_set=[(X_tr[fold_val], y_tr[fold_val])], verbose=False)
            pred = np.clip(m.predict(X_tr[fold_val]), 1e-6, None)
            scores.append(float(mean_poisson_deviance(y_tr[fold_val], pred)))
            iters.append(m.best_iteration)
            trial.report(float(np.mean(scores)), step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("best_iteration", int(np.mean(iters)))
        return float(np.mean(scores))

    t0 = time.time()

    def log_progress(study, trial):
        estado = f"PoissonDev={trial.value:.4f}" if trial.value is not None else trial.state.name
        completos = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        mejor = f"{study.best_value:.4f}" if completos else "-"
        logger.info(f"  Trial {trial.number:>3} | {estado} | mejor={mejor} | "
                    f"{(time.time() - t0) / 60:.1f} min")

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )
    study.optimize(objective, n_trials=OPTUNA_TRIALS, timeout=OPTUNA_TIMEOUT,
                   callbacks=[log_progress], show_progress_bar=False)
    logger.info(f"  Mejor CV-PoissonDev: {study.best_value:.4f} | params: {study.best_params} | "
                f"best_iteration≈{study.best_trial.user_attrs['best_iteration']}")
    return study.best_params


# ══════════════════════════════════════════════════════════════════════════════
# Entrenador genérico — reutilizado por ambas variantes
# ══════════════════════════════════════════════════════════════════════════════
def train_variante(
    model_name:   str,
    feature_cols: list,
    train_df:     pd.DataFrame,
    test_df:      pd.DataFrame,
    exp_id:       str,
    lookup_paths: dict,
):
    train_df = train_df.sort_values(["hora_ts", "estacion_id"]).reset_index(drop=True)
    X_train, y_train = train_df[feature_cols], train_df[TARGET]
    X_test,  y_test  = test_df[feature_cols],  test_df[TARGET]

    logger.info(f"Features ({len(feature_cols)}): {feature_cols}")

    with mlflow.start_run(experiment_id=exp_id, run_name=model_name):
        mlflow.set_tag("model_type", "XGBoost")
        mlflow.set_tag("objective",  "count:poisson")
        mlflow.set_tag("variante",   model_name)
        mlflow.log_param("n_estaciones",  train_df["estacion_id"].nunique())
        mlflow.log_param("train_rows",    len(X_train))
        mlflow.log_param("test_rows",     len(X_test))
        mlflow.log_param("pct_ceros_train", round(float((y_train == 0).mean()), 4))
        mlflow.log_param("features",      str(feature_cols))
        mlflow.log_param("optuna_trials", OPTUNA_TRIALS)
        mlflow.log_param("lag_red_horas", LAG_RED_HORAS)

        best_params = tune_xgboost(X_train, y_train, model_name)
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})

        # Modelo final: early stopping sobre el 10% más reciente del train
        logger.info(f"Entrenando modelo final {model_name}...")
        cut = int(len(X_train) * 0.9)
        X_tr_final, X_val_final = X_train.iloc[:cut], X_train.iloc[cut:]
        y_tr_final, y_val_final = y_train.iloc[:cut], y_train.iloc[cut:]

        model = XGBRegressor(
            **base_params(), **best_params,
            n_estimators=MAX_TREES, early_stopping_rounds=50,
        )
        model.fit(X_tr_final, y_tr_final,
                  eval_set=[(X_val_final, y_val_final)], verbose=False)
        logger.info(f"  Early stopping en iteracion {model.best_iteration}")

        y_pred  = model.predict(X_test)
        metrics = compute_metrics(y_test, y_pred)
        mlflow.log_metrics(metrics)

        y_pred_train  = model.predict(X_train)
        metrics_train = compute_metrics(y_train, y_pred_train)
        mlflow.log_metrics({f"train_{k}": v for k, v in metrics_train.items()})
        logger.info(f"  TRAIN → R2: {metrics_train['R2']:.4f} | RMSE: {metrics_train['RMSE']:.4f} "
                    f"| WAPE: {metrics_train['WAPE']:.1f}%")
        logger.info(f"  TEST  → R2: {metrics['R2']:.4f} | RMSE: {metrics['RMSE']:.4f} "
                    f"| WAPE: {metrics['WAPE']:.1f}%")
        gap = metrics_train["R2"] - metrics["R2"]
        if gap > 0.15:
            logger.warning(f"  ⚠️  Brecha train-test R2 = {gap:.3f}")

        # Artefactos
        plot_path = os.path.join(TMP, f"{model_name}_red.png")
        save_red_plot(test_df, y_pred,
                      f"{model_name} — Demanda total de la red (primeras 2 semanas de test)",
                      plot_path)
        mlflow.log_artifact(plot_path, "plots")

        save_feature_importance(model, feature_cols, model_name)

        per_station = (
            test_df.assign(y_pred=y_pred)
            .groupby("estacion_id")
            .apply(lambda g: pd.Series(compute_metrics(g[TARGET], g["y_pred"])))
            .reset_index()
        )
        ps_path = os.path.join(TMP, f"{model_name}_metrics_per_station.csv")
        per_station.to_csv(ps_path, index=False)
        mlflow.log_artifact(ps_path, "metrics")

        for artifact_name, local_path in lookup_paths.items():
            mlflow.log_artifact(local_path, "lookup_tables")
            logger.info(f"  Lookup '{artifact_name}' guardada en MLflow")

        mlflow.xgboost.log_model(
            model,
            name=model_name,
            input_example=X_train.iloc[:5],
            registered_model_name=model_name,
        )

    logger.info(f"✅ {model_name} registrado en MLflow")
    return model, metrics


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    mlflow.set_tracking_uri(MLFLOW_URI)
    exp    = mlflow.set_experiment(EXPERIMENT)
    exp_id = exp.experiment_id
    logger.info(f"Experiment: '{EXPERIMENT}' (id={exp_id})")

    # 1. Carga, grilla con ceros y features (sobre el dataset completo)
    df = load_data()
    df = enrich_features(df)
    df = add_hist_features(df)

    # 2. Lookups para producción (todos los datos)
    lookup_paths = build_lookups(df)

    # 3. Split temporal global
    train, test = temporal_split(df)

    logger.info("=" * 60)
    logger.info("VARIANTE A: xgboost_v4_rolling (ventana deslizante 4 semanas)")
    logger.info("=" * 60)
    _, metrics_rolling = train_variante(
        model_name   = "xgboost_v4_rolling",
        feature_cols = FEATURES_ROLLING,
        train_df     = train,
        test_df      = test,
        exp_id       = exp_id,
        lookup_paths = {"lookup_rolling": lookup_paths["lookup_rolling"]},
    )

    logger.info("=" * 60)
    logger.info("VARIANTE B: xgboost_v4_mensual (rolling + promedios hora y mes)")
    logger.info("=" * 60)
    _, metrics_mensual = train_variante(
        model_name   = "xgboost_v4_mensual",
        feature_cols = FEATURES_MENSUAL,
        train_df     = train,
        test_df      = test,
        exp_id       = exp_id,
        lookup_paths = lookup_paths,
    )

    logger.info("")
    logger.info("══════════════════════════════════════════")
    logger.info("COMPARATIVA FINAL")
    logger.info("══════════════════════════════════════════")
    for nombre, m in [("xgboost_v4_rolling", metrics_rolling),
                      ("xgboost_v4_mensual", metrics_mensual)]:
        logger.info(f"  {nombre:<20} → R2: {m['R2']:.4f} | RMSE: {m['RMSE']:.4f} | "
                    f"WAPE: {m['WAPE']:.1f}% | PoissonDev: {m['PoissonDev']:.4f}")
    logger.info("")
    logger.info(f"UI: {MLFLOW_URI}  |  Experimento: '{EXPERIMENT}'")


if __name__ == "__main__":
    main()
