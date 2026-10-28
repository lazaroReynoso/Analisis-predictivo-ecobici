"""
API de predicción — Ecobici viajes_hora (compatible con mlflow_training_v4.py)

    POST /predict   {"station_id": 14, "datetime": "2026-10-01T08:00:00"}
    GET  /health    modelo cargado y su métrica
    POST /reload    vuelve a elegir el mejor modelo (tras re-entrenar)

Al arrancar:
  1. Busca entre las versiones registradas de xgboost_v4_rolling y xgboost_v4_mensual
     la de MENOR PoissonDev en test (métrica configurable) y la carga.
  2. Descarga las lookup_tables del run de esa versión (hist_mean, hist_std, ...).
  3. Carga desde Postgres la capacidad de las estaciones y dos tablas de respaldo:
     clima promedio por (mes, hora) y demanda promedio de la red por (día, hora).

Por request arma exactamente las mismas features que usó el entrenamiento.

Ejecutar:
    uvicorn predict_api:app --host 0.0.0.0 --port 8000
"""
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime as DateTime, timedelta
from threading import Lock
from typing import Optional

import numpy as np
import pandas as pd
import mlflow
import mlflow.xgboost
from mlflow import MlflowClient
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("predict_api")


DB_URL      = os.getenv("DB_URL", "postgresql+psycopg://postgres:CAMBIAR@127.0.0.1:5432/ecobici")
MLFLOW_URI  = os.getenv("MLFLOW_URI", "http://127.0.0.1:5000")
MODEL_NAMES = os.getenv("MODEL_NAMES", "xgboost_v4_rolling,xgboost_v4_mensual").split(",")
SELECTION_METRIC = os.getenv("SELECTION_METRIC", "PoissonDev")   # menor = mejor
# Opcional: forzar una versión concreta, ej. MODEL_NAME=xgboost_v4_mensual MODEL_VERSION=3
FORCE_NAME    = os.getenv("MODEL_NAME")
FORCE_VERSION = os.getenv("MODEL_VERSION")

# Deben coincidir con el script de entrenamiento
LAG_RED_HORAS = 1
KEYS_SLOT = ["estacion_id", "hora", "dia_semana"]
KEYS_HORA = ["estacion_id", "hora"]
KEYS_MES  = ["estacion_id", "hora", "dia_semana", "mes"]
LOOKUPS = {  # archivo -> claves
    "lookup_rolling.parquet": KEYS_SLOT,
    "lookup_hora.parquet":    KEYS_HORA,
    "lookup_mes.parquet":     KEYS_MES,
}
CLIMA_COLS = ["temperatura_c", "sensacion_termica_c", "humedad_pct",
              "precipitacion_mm", "es_lluvia", "velocidad_viento_kmh"]


# ── Estado del servicio ────────────────────────────────────────────────────────
@dataclass
class ModelBundle:
    model: object
    feature_cols: list
    name: str
    version: str
    run_id: str
    metric: Optional[float]
    lookups: dict = field(default_factory=dict)   # nombre -> dict{clave_tupla: {col: val}}


@dataclass
class ReferenceData:
    capacidad: dict            # estacion_id -> capacidad
    clima_medio: dict          # (mes, hora) -> {col: val}
    red_media: dict            # (dia_semana, hora) -> viajes promedio de la red


STATE: dict = {}
_lock = Lock()
engine = create_engine(DB_URL, pool_pre_ping=True)


# ── Selección y carga del modelo ───────────────────────────────────────────────
def select_best_version(client: MlflowClient):
    if FORCE_NAME and FORCE_VERSION:
        mv = client.get_model_version(FORCE_NAME, FORCE_VERSION)
        run = client.get_run(mv.run_id)
        return mv, run.data.metrics.get(SELECTION_METRIC)

    candidatos = []
    for name in MODEL_NAMES:
        for mv in client.search_model_versions(f"name='{name}'"):
            if not mv.run_id:
                continue
            valor = client.get_run(mv.run_id).data.metrics.get(SELECTION_METRIC)
            if valor is not None:
                candidatos.append((valor, mv))
    if not candidatos:
        raise RuntimeError(f"No hay versiones registradas de {MODEL_NAMES} con la métrica "
                           f"'{SELECTION_METRIC}'")
    candidatos.sort(key=lambda c: c[0])
    for valor, mv in candidatos:
        logger.info(f"  candidato {mv.name} v{mv.version}: {SELECTION_METRIC}={valor:.4f}")
    valor, mv = candidatos[0]
    return mv, valor


def _to_lookup_dict(df: pd.DataFrame, keys: list) -> dict:
    df = df.copy()
    for k in keys:
        df[k] = df[k].astype(int)
    return df.set_index(keys).to_dict("index")


def load_bundle() -> ModelBundle:
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient()
    mv, metric = select_best_version(client)
    logger.info(f"Modelo elegido: {mv.name} v{mv.version} (run {mv.run_id})")

    model = mlflow.xgboost.load_model(f"models:/{mv.name}/{mv.version}")
    # Se entrenó con device="cuda"; para servir alcanza con CPU y evita el aviso
    # de "mismatched devices" en cada predict.
    try:
        model.set_params(device="cpu")
    except Exception:
        model.get_booster().set_param({"device": "cpu"})
    feature_cols = list(model.get_booster().feature_names)

    lk_dir = mlflow.artifacts.download_artifacts(run_id=mv.run_id, artifact_path="lookup_tables")
    lookups = {}
    for fname, keys in LOOKUPS.items():
        path = os.path.join(lk_dir, fname)
        if os.path.exists(path):
            lookups[fname] = _to_lookup_dict(pd.read_parquet(path), keys)
            logger.info(f"  lookup {fname}: {len(lookups[fname]):,} filas")

    return ModelBundle(model, feature_cols, mv.name, str(mv.version), mv.run_id, metric, lookups)


def load_reference_data() -> ReferenceData:
    with engine.connect() as conn:
        est = pd.read_sql(text("SELECT estacion_id, capacidad FROM estaciones"), conn)
        clima_medio = pd.read_sql(text(f"""
            SELECT EXTRACT(MONTH FROM c.timestamp_utc::timestamp)::INT AS mes,
                   EXTRACT(HOUR  FROM c.timestamp_utc::timestamp)::INT AS hora,
                   AVG(c.temperatura_c)        AS temperatura_c,
                   AVG(c.sensacion_termica_c)  AS sensacion_termica_c,
                   AVG(c.humedad_pct)          AS humedad_pct,
                   AVG(c.precipitacion_mm)     AS precipitacion_mm,
                   AVG(c.es_lluvia::INT)       AS es_lluvia,
                   AVG(c.velocidad_viento_kmh) AS velocidad_viento_kmh
            FROM clima c GROUP BY 1, 2
        """), conn)
        # Misma definición de "red" que en entrenamiento: solo estaciones de la tabla
        # estaciones y años 2025-2026. DOW de Postgres: 0 = domingo.
        red = pd.read_sql(text("""
            SELECT EXTRACT(DOW  FROM h)::INT AS dia_semana,
                   EXTRACT(HOUR FROM h)::INT AS hora,
                   AVG(n)                    AS viajes
            FROM (
                SELECT DATE_TRUNC('hour', v.fecha_origen) AS h, COUNT(*) AS n
                FROM viajes v
                JOIN estaciones e ON e.estacion_id = v.estacion_origen_id
                WHERE v.anio IN (2025, 2026)
                GROUP BY 1
            ) t GROUP BY 1, 2
        """), conn)

    return ReferenceData(
        capacidad=dict(zip(est["estacion_id"].astype(int), est["capacidad"])),
        clima_medio=clima_medio.set_index(["mes", "hora"])[CLIMA_COLS].to_dict("index"),
        red_media={(int(r.dia_semana), int(r.hora)): float(r.viajes) for r in red.itertuples()},
    )


def reload_all():
    bundle = load_bundle()
    ref = load_reference_data()
    with _lock:
        STATE["bundle"], STATE["ref"] = bundle, ref


# ── Features (misma lógica que enrich_features del entrenamiento) ─────────────
def calendar_features(ts: pd.Timestamp) -> dict:
    hora = ts.hour
    dia_semana = (ts.dayofweek + 1) % 7            # 0 = domingo ... 6 = sábado
    es_finde = int(dia_semana in (0, 6))
    parte_dia = hora // 6                          # = pd.cut bins [0,6,12,18,24)
    return {
        "hora": hora,
        "dia_semana": dia_semana,
        "mes": ts.month,
        "es_finde": es_finde,
        "semana_anio": int(ts.isocalendar()[1]),
        "parte_dia": parte_dia,
        "hora_parte_dia": hora * 10 + parte_dia,
        "hora_finde": hora * es_finde,
    }


def fetch_clima(conn, ts: pd.Timestamp, ref: ReferenceData, cal: dict):
    row = conn.execute(text(f"""
        SELECT {", ".join(CLIMA_COLS[:4])}, es_lluvia::INT AS es_lluvia, velocidad_viento_kmh
        FROM clima c
        WHERE c.timestamp_utc::timestamp = :ts
        LIMIT 1
    """), {"ts": ts.to_pydatetime()}).mappings().first()
    if row is not None:
        return {k: (None if row[k] is None else float(row[k])) for k in CLIMA_COLS}, "clima_db"
    medio = ref.clima_medio.get((cal["mes"], cal["hora"]))
    if medio is not None:
        return dict(medio), "clima_promedio_mes_hora"
    return {k: np.nan for k in CLIMA_COLS}, "sin_clima"


def fetch_demanda_red_prev(conn, ts: pd.Timestamp, ref: ReferenceData):
    desde = ts - timedelta(hours=LAG_RED_HORAS)
    hasta = desde + timedelta(hours=1)
    ultimo = conn.execute(text("SELECT MAX(fecha_origen) FROM viajes")).scalar()
    if ultimo is not None and pd.Timestamp(ultimo).tz_localize(None) >= hasta:
        n = conn.execute(text("""
            SELECT COUNT(*) FROM viajes v
            JOIN estaciones e ON e.estacion_id = v.estacion_origen_id
            WHERE v.fecha_origen >= :desde AND v.fecha_origen < :hasta
        """), {"desde": desde.to_pydatetime(), "hasta": hasta.to_pydatetime()}).scalar()
        return float(n), "red_db"
    # Hora sin datos todavía (futuro): promedio histórico de ese día/hora
    dia_prev = (desde.dayofweek + 1) % 7
    valor = ref.red_media.get((dia_prev, desde.hour))
    return (float(valor) if valor is not None else np.nan), "red_promedio_dia_hora"


def hist_features(bundle: ModelBundle, station_id: int, cal: dict) -> dict:
    out = {}
    lk = bundle.lookups
    if "lookup_rolling.parquet" in lk:
        out.update(lk["lookup_rolling.parquet"].get(
            (station_id, cal["hora"], cal["dia_semana"]), {}))
    if "lookup_hora.parquet" in lk:
        out.update(lk["lookup_hora.parquet"].get((station_id, cal["hora"]), {}))
    if "lookup_mes.parquet" in lk:
        out.update(lk["lookup_mes.parquet"].get(
            (station_id, cal["hora"], cal["dia_semana"], cal["mes"]), {}))
    return out


def build_feature_row(station_id: int, ts: pd.Timestamp, bundle: ModelBundle,
                      ref: ReferenceData):
    cal = calendar_features(ts)
    with engine.connect() as conn:
        clima, fuente_clima = fetch_clima(conn, ts, ref, cal)
        red_prev, fuente_red = fetch_demanda_red_prev(conn, ts, ref)

    precip = clima["precipitacion_mm"]
    precip = np.nan if precip is None else precip
    row = {
        **cal,
        "estacion_id": station_id,
        "capacidad": ref.capacidad[station_id],
        **clima,
        "lluvia_leve":   int(0.1 <= precip <= 2) if not np.isnan(precip) else 0,
        "lluvia_fuerte": int(precip > 2) if not np.isnan(precip) else 0,
        "demanda_red_prev": red_prev,
        **hist_features(bundle, station_id, cal),
    }
    # Orden y columnas exactas que espera el modelo; lo que falte queda NaN
    X = pd.DataFrame([row]).reindex(columns=bundle.feature_cols).astype("float32")
    fuentes = {
        "clima": fuente_clima,
        "demanda_red_prev": fuente_red,
        "historico_estacion": "lookup" if "hist_mean" in row else "sin_historia",
    }
    return X, fuentes


# ── API ────────────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, json_schema_extra={
        "example": {"station_id": 14, "datetime": "2026-10-01T08:00:00"}})
    station_id: int
    fecha: DateTime = Field(alias="datetime")


class PredictResponse(BaseModel):
    station_id: int
    datetime: DateTime
    demanda_predicha: float
    modelo: dict
    fuentes: dict


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Si todavía no hay modelo registrado (ej. primer arranque), la API levanta igual
    # y responde 503 hasta que se llame a POST /reload.
    try:
        reload_all()
    except Exception as e:
        logger.error(f"No se pudo cargar el modelo al arrancar: {e}")
    yield


app = FastAPI(title="Ecobici — predicción de demanda por hora", lifespan=lifespan)


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    with _lock:
        bundle, ref = STATE.get("bundle"), STATE.get("ref")
    if bundle is None or ref is None:
        raise HTTPException(503, "Modelo no cargado todavía. Entrená uno y llamá a POST /reload")

    if req.station_id not in ref.capacidad:
        raise HTTPException(404, f"Estación {req.station_id} no existe en la tabla estaciones")

    ts = pd.Timestamp(req.fecha)
    if ts.tz is not None:                      # el entrenamiento usa timestamps sin zona
        ts = ts.tz_localize(None)
    ts = ts.floor("h")

    X, fuentes = build_feature_row(req.station_id, ts, bundle, ref)
    y = float(np.clip(bundle.model.predict(X)[0], 0, None))

    return PredictResponse(
        station_id=req.station_id,
        datetime=ts.to_pydatetime(),
        demanda_predicha=round(y, 3),
        modelo={"name": bundle.name, "version": bundle.version, "run_id": bundle.run_id,
                SELECTION_METRIC: bundle.metric},
        fuentes=fuentes,
    )


@app.get("/health")
def health():
    b = STATE.get("bundle")
    if b is None:
        raise HTTPException(503, "Modelo no cargado")
    return {"status": "ok", "modelo": b.name, "version": b.version,
            SELECTION_METRIC: b.metric, "n_features": len(b.feature_cols)}


@app.post("/reload")
def reload():
    reload_all()
    return health()
