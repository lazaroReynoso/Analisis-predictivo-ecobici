"""
Monitor de drift — re-evalúa el modelo en producción con los datos de la última semana.

Qué hace en cada ejecución:
  1. Toma el mismo modelo que sirve la API (menor PoissonDev entre las versiones
     registradas, o el forzado con MODEL_NAME / MODEL_VERSION).
  2. Arma las features exactamente como el entrenamiento (grilla con ceros, calendario,
     clima, demanda de la red, históricas sin leakage) sobre todos los datos de Postgres.
  3. Evalúa sobre los últimos EVAL_DAYS días y compara el RMSE contra el RMSE de test
     que se registró al entrenar (baseline).
  4. Calcula además el PSI de algunas features (drift en los datos de entrada).
  5. Guarda todo como un run en el experimento MONITOR_EXPERIMENT de MLflow
     (así ves la evolución semana a semana en la UI).
  6. Si el RMSE sube más que RMSE_THRESHOLD_PCT → ALERTA: log, tag en MLflow,
     webhook opcional (Slack / Discord) y, si AUTO_RETRAIN=true, re-entrena y
     le pide a la API que recargue el modelo.

Antes del chequeo corre el ETL (etl.py) para traer viajes, clima y estaciones nuevos
(desactivable con RUN_ETL=false).

Uso:
    python drift_monitor.py              # una sola vez (ideal para cron / pruebas)
    python drift_monitor.py --schedule   # queda corriendo y ejecuta cada semana
"""
import argparse
import json
import logging
import os
import tempfile
import time
import urllib.request
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import mlflow
import mlflow.xgboost
from mlflow import MlflowClient

import mlflow_training_v4 as tr
from predict_api import select_best_version   # mismo criterio de elección que la API

logger = logging.getLogger("drift_monitor")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

MLFLOW_URI         = os.getenv("MLFLOW_URI", "http://127.0.0.1:5000")
API_URL            = os.getenv("API_URL", "http://127.0.0.1:8000")
MONITOR_EXPERIMENT = os.getenv("MONITOR_EXPERIMENT", "ecobici_drift_monitoring")
EVAL_DAYS          = int(os.getenv("EVAL_DAYS", "7"))
RMSE_THRESHOLD_PCT = float(os.getenv("RMSE_THRESHOLD_PCT", "0.20"))   # 0.20 = +20 %
PSI_THRESHOLD      = float(os.getenv("PSI_THRESHOLD", "0.25"))
PSI_REF_DAYS       = int(os.getenv("PSI_REF_DAYS", "90"))
PSI_FEATURES       = os.getenv(
    "PSI_FEATURES", "temperatura_c,precipitacion_mm,demanda_red_prev,hist_mean"
).split(",")
ALERT_WEBHOOK_URL  = os.getenv("ALERT_WEBHOOK_URL", "")
AUTO_RETRAIN       = os.getenv("AUTO_RETRAIN", "false").lower() == "true"
# Programación semanal (hora local del contenedor, ver TZ en docker-compose)
DRIFT_WEEKDAY      = int(os.getenv("DRIFT_WEEKDAY", "0"))   # 0 = lunes ... 6 = domingo
DRIFT_HOUR         = int(os.getenv("DRIFT_HOUR", "6"))
RUN_ON_START       = os.getenv("RUN_ON_START", "false").lower() == "true"
RUN_ETL            = os.getenv("RUN_ETL", "true").lower() == "true"   # cargar datos antes


# ── Utilidades ─────────────────────────────────────────────────────────────────
def psi(ref: pd.Series, cur: pd.Series, bins: int = 10) -> float:
    """
    Population Stability Index. Regla práctica: < 0.1 estable, 0.1–0.25 cambio
    moderado, > 0.25 cambio fuerte. Los NaN cuentan como una categoría aparte.
    """
    ref, cur = ref.astype(float), cur.astype(float)
    cortes = np.unique(np.nanquantile(ref.dropna(), np.linspace(0, 1, bins + 1)))
    if len(cortes) < 2:            # variable casi constante en la referencia
        return 0.0
    cortes[0], cortes[-1] = -np.inf, np.inf

    def dist(s):
        cuentas = pd.cut(s.dropna(), cortes).value_counts(sort=False).to_numpy()
        cuentas = np.append(cuentas, s.isna().sum())
        return np.clip(cuentas / max(len(s), 1), 1e-4, None)

    p, q = dist(ref), dist(cur)
    return float(np.sum((q - p) * np.log(q / p)))


def send_alert(mensaje: str):
    logger.warning(mensaje)
    if not ALERT_WEBHOOK_URL:
        return
    # "text" lo usa Slack, "content" lo usa Discord
    body = json.dumps({"text": mensaje, "content": mensaje[:1900]}).encode()
    req = urllib.request.Request(ALERT_WEBHOOK_URL, data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        logger.error(f"No se pudo enviar el webhook: {e}")


def reload_api():
    req = urllib.request.Request(f"{API_URL}/reload", method="POST")
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            logger.info(f"API recargada: {r.read().decode()}")
    except Exception as e:
        logger.error(f"No se pudo recargar la API ({API_URL}): {e}")


def load_production_model(client: MlflowClient):
    mv, _ = select_best_version(client)
    run = client.get_run(mv.run_id)
    model = mlflow.xgboost.load_model(f"models:/{mv.name}/{mv.version}")
    try:
        model.set_params(device="cpu")
    except Exception:
        model.get_booster().set_param({"device": "cpu"})
    return mv, run, model


# ── Una ejecución del chequeo ──────────────────────────────────────────────────
def run_check() -> dict:
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient()

    mv, train_run, model = load_production_model(client)
    feature_cols = list(model.get_booster().feature_names)
    baseline_rmse = train_run.data.metrics.get("RMSE")
    if baseline_rmse is None:
        raise RuntimeError(f"El run {mv.run_id} no tiene la métrica RMSE")
    entrenado_el = pd.Timestamp(train_run.info.start_time, unit="ms")
    logger.info(f"Modelo en producción: {mv.name} v{mv.version} | "
                f"RMSE baseline (test) = {baseline_rmse:.4f}")

    # Mismas features que en entrenamiento. Las históricas solo miran el pasado de
    # cada fila, así que evaluar la última semana es igual a lo que se vería en vivo.
    df = tr.add_hist_features(tr.enrich_features(tr.load_data()))

    # La última hora puede estar incompleta (los viajes se siguen cargando): se excluye.
    fin = df["hora_ts"].max()
    ini = fin - pd.Timedelta(days=EVAL_DAYS)
    semana = df[(df["hora_ts"] >= ini) & (df["hora_ts"] < fin)]
    if semana.empty:
        raise RuntimeError("No hay datos en la ventana de evaluación")
    datos_nuevos = bool(ini >= entrenado_el)
    if not datos_nuevos:
        logger.warning("La ventana evaluada es anterior al entrenamiento: el modelo ya pudo "
                       "haber visto esos datos. ¿Se cargaron los viajes nuevos en Postgres?")

    X = semana[feature_cols].astype("float32")
    y_pred = np.clip(model.predict(X), 0, None)
    m = tr.compute_metrics(semana[tr.TARGET], y_pred)
    cambio = m["RMSE"] / baseline_rmse - 1
    alerta = cambio > RMSE_THRESHOLD_PCT

    # RMSE por día: sirve para ver si el problema es un día puntual (feriado, paro...)
    por_dia = (
        semana.assign(y_pred=y_pred, dia=semana["hora_ts"].dt.date)
        .groupby("dia")
        .apply(lambda g: pd.Series(tr.compute_metrics(g[tr.TARGET], g["y_pred"])))
    )

    # Drift de datos de entrada: semana actual vs. los PSI_REF_DAYS días anteriores
    # al entrenamiento (o a la ventana, si es anterior).
    ref_fin = min(ini, entrenado_el)
    ref = df[(df["hora_ts"] < ref_fin) &
             (df["hora_ts"] >= ref_fin - pd.Timedelta(days=PSI_REF_DAYS))]
    psis = {f"psi_{c}": psi(ref[c], semana[c]) for c in PSI_FEATURES
            if c in df.columns and not ref.empty}
    features_con_drift = [k[4:] for k, v in psis.items() if v > PSI_THRESHOLD]

    logger.info(f"Ventana {ini} → {fin} | RMSE {m['RMSE']:.4f} vs baseline "
                f"{baseline_rmse:.4f} ({cambio:+.1%}) | WAPE {m['WAPE']:.1f}% | PSI {psis}")

    exp = mlflow.set_experiment(MONITOR_EXPERIMENT)
    with mlflow.start_run(experiment_id=exp.experiment_id,
                          run_name=f"drift_{fin:%Y-%m-%d}"):
        mlflow.set_tags({
            "modelo": mv.name, "modelo_version": mv.version, "modelo_run_id": mv.run_id,
            "alerta": str(alerta), "datos_nuevos": str(datos_nuevos),
            "features_con_drift": ",".join(features_con_drift) or "-",
        })
        mlflow.log_params({"ventana_inicio": str(ini), "ventana_fin": str(fin),
                           "eval_days": EVAL_DAYS, "rmse_threshold_pct": RMSE_THRESHOLD_PCT,
                           "filas_evaluadas": len(semana)})
        mlflow.log_metrics({**{f"semana_{k}": v for k, v in m.items()},
                            "baseline_RMSE": baseline_rmse,
                            "rmse_cambio_pct": cambio * 100,
                            "alerta": int(alerta), **psis})
        path = os.path.join(tempfile.gettempdir(), "metricas_por_dia.csv")
        por_dia.to_csv(path)
        mlflow.log_artifact(path, "drift")

    resultado = {"modelo": f"{mv.name} v{mv.version}", "rmse": m["RMSE"],
                 "baseline_rmse": baseline_rmse, "cambio_pct": cambio * 100,
                 "alerta": alerta, "psi": psis}

    if alerta:
        send_alert(
            f"🚨 Ecobici drift: RMSE de la última semana = {m['RMSE']:.3f} vs "
            f"{baseline_rmse:.3f} al entrenar ({cambio:+.1%}, umbral "
            f"+{RMSE_THRESHOLD_PCT:.0%}). Modelo {mv.name} v{mv.version}. "
            f"Features con drift (PSI>{PSI_THRESHOLD}): {features_con_drift or 'ninguna'}."
        )
        if AUTO_RETRAIN:
            logger.info("AUTO_RETRAIN activo → re-entrenando...")
            tr.main()
            reload_api()
    else:
        logger.info(f"✅ Sin alerta (cambio {cambio:+.1%} ≤ +{RMSE_THRESHOLD_PCT:.0%})")
    return resultado


# ── Programación semanal ───────────────────────────────────────────────────────
def proxima_ejecucion(ahora: datetime) -> datetime:
    objetivo = ahora.replace(hour=DRIFT_HOUR, minute=0, second=0, microsecond=0)
    objetivo += timedelta(days=(DRIFT_WEEKDAY - ahora.weekday()) % 7)
    if objetivo <= ahora:
        objetivo += timedelta(days=7)
    return objetivo


def ejecutar_seguro():
    """Pipeline semanal: cargar datos nuevos → recargar API → chequear drift."""
    if RUN_ETL:
        try:
            import etl
            resumen = etl.run()
            if resumen["estaciones_nuevas"]:
                reload_api()        # la API toma las capacidades de las estaciones nuevas
        except Exception as e:
            logger.exception("Falló el ETL; se evalúa con los datos que ya hay")
            send_alert(f"⚠️ Ecobici: falló la carga semanal de datos: {e}")
    try:
        run_check()
    except Exception as e:
        logger.exception("Falló el chequeo de drift")
        send_alert(f"⚠️ Ecobici: falló el chequeo semanal de drift: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule", action="store_true",
                        help="quedar corriendo y ejecutar una vez por semana")
    args = parser.parse_args()

    if not args.schedule:
        ejecutar_seguro()
        return

    if RUN_ON_START:
        ejecutar_seguro()
    while True:
        prox = proxima_ejecucion(datetime.now())
        logger.info(f"Próximo chequeo de drift: {prox:%A %d/%m/%Y %H:%M}")
        time.sleep(max((prox - datetime.now()).total_seconds(), 0))
        ejecutar_seguro()


if __name__ == "__main__":
    main()
