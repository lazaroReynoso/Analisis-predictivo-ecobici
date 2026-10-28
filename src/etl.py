"""
ETL Ecobici — carga INCREMENTAL de estaciones, clima y viajes (versión script del
notebook "Carga de Datos").

Se puede correr todas las veces que quieras: solo agrega lo que falta.
  · estaciones : agrega las nuevas, actualiza la capacidad y asigna barrio/comuna
                 a las que no lo tienen.
  · clima      : pide a Open-Meteo desde la última hora cargada hasta hace CLIMA_LAG_DIAS.
  · viajes     : descarga el CSV del año en curso de BA Data e inserta solo los viajes
                 posteriores al último que ya está en la base.

Uso:
    python etl.py
"""
import io
import json
import logging
import os
import sys
import tempfile
import zipfile
from datetime import date, timedelta

import pandas as pd
import requests
from shapely.geometry import Point, shape
from sqlalchemy import create_engine, text

logger = logging.getLogger("etl")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

DB_URL = os.getenv("DB_URL", "postgresql+psycopg://postgres:CAMBIAR@127.0.0.1:5432/ecobici")

# Un archivo por año en BA Data. Cuando publiquen el del año nuevo, agregalo acá
# o sin tocar código: VIAJES_URLS_EXTRA='{"2027": "https://..."}'
VIAJES_URLS = {
    2025: "https://data.buenosaires.gob.ar/dataset/bicicletas-publicas/resource/"
          "d860362d-99f9-49a2-9ea0-1afd65581a97/download",
    2026: "https://data.buenosaires.gob.ar/dataset/bicicletas-publicas/resource/"
          "c6f027bb-5a39-4197-b6d0-9b1e0dcf81da/download",
}
VIAJES_URLS.update({int(k): v for k, v in
                    json.loads(os.getenv("VIAJES_URLS_EXTRA") or "{}").items()})

COLUMNAS = {
    "ID_recorrido": "viaje_id_origen",
    "id_usuario": "usuario_id",
    "género": "genero",
    "id_estacion_origen": "estacion_origen_id",
    "id_estacion_destino": "estacion_destino_id",
    "fecha_origen_recorrido": "fecha_origen",
    "fecha_destino_recorrido": "fecha_destino",
}
COLS_VIAJES = ["usuario_id", "genero", "estacion_origen_id",
               "estacion_destino_id", "fecha_origen", "fecha_destino"]

LAT, LON       = -34.5597, -58.4156
CLIMA_DESDE    = os.getenv("CLIMA_DESDE", "2024-01-01")
CLIMA_LAG_DIAS = int(os.getenv("CLIMA_LAG_DIAS", "2"))   # el archivo histórico tarda unos días
CLIMA_COLS     = ["timestamp_utc", "temperatura_c", "sensacion_termica_c", "humedad_pct",
                  "precipitacion_mm", "velocidad_viento_kmh", "codigo_clima"]

ESTACIONES_URL = os.getenv(
    "ESTACIONES_URL",
    "https://apitransporte.buenosaires.gob.ar/ecobici/gbfs/stationInformation")
CLIENT_ID      = os.getenv("ECOBICI_CLIENT_ID", "")
CLIENT_SECRET  = os.getenv("ECOBICI_CLIENT_SECRET", "")
BARRIOS_URL    = ("https://cdn.buenosaires.gob.ar/datosabiertos/datasets/"
                  "ministerio-de-educacion/barrios/barrios.geojson")


# ── Utilidades ─────────────────────────────────────────────────────────────────
def copy_df(engine, df: pd.DataFrame, tabla: str, chunk: int = 500_000):
    """Inserta con COPY (mucho más rápido que to_sql para millones de filas)."""
    cols = ", ".join(df.columns)
    raw = engine.raw_connection()
    try:
        with raw.driver_connection.cursor() as cur:
            for i in range(0, len(df), chunk):
                buf = io.StringIO()
                df.iloc[i:i + chunk].to_csv(buf, index=False, header=False)
                buf.seek(0)
                with cur.copy(f"COPY {tabla} ({cols}) FROM STDIN WITH (FORMAT csv)") as cp:
                    while data := buf.read(1 << 20):
                        cp.write(data)
        raw.driver_connection.commit()
    finally:
        raw.close()


def scalar(engine, sql: str):
    with engine.connect() as conn:
        return conn.execute(text(sql)).scalar()


# ── Estaciones ─────────────────────────────────────────────────────────────────
def descargar_estaciones() -> pd.DataFrame:
    r = requests.get(ESTACIONES_URL, timeout=60,
                     params={"client_id": CLIENT_ID, "client_secret": CLIENT_SECRET})
    r.raise_for_status()
    df = pd.DataFrame(r.json()["data"]["stations"])
    df = df[["station_id", "name", "address", "lat", "lon", "capacity"]].rename(columns={
        "station_id": "estacion_id", "name": "nombre",
        "address": "direccion", "capacity": "capacidad"})
    df["estacion_id"] = df["estacion_id"].astype(int)
    return df


def asignar_barrios(engine) -> int:
    """Barrio y comuna para las estaciones que todavía no los tienen."""
    pendientes = pd.read_sql(
        "SELECT estacion_id, lat, lon FROM estaciones WHERE barrio IS NULL", engine)
    if pendientes.empty:
        return 0
    r = requests.get(BARRIOS_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
    r.raise_for_status()
    barrios = [(shape(f["geometry"]), f["properties"]["nombre"], f["properties"]["comuna"])
               for f in r.json()["features"]]

    filas = []
    for e in pendientes.itertuples():
        punto = Point(e.lon, e.lat)
        hit = next(((n, c) for geom, n, c in barrios if geom.intersects(punto)), None)
        if hit:
            filas.append({"id": int(e.estacion_id), "barrio": hit[0], "comuna": int(hit[1])})
    if filas:
        with engine.begin() as conn:
            conn.execute(text("UPDATE estaciones SET barrio = :barrio, comuna = :comuna "
                              "WHERE estacion_id = :id"), filas)
    logger.info(f"Barrios asignados: {len(filas)} de {len(pendientes)} pendientes")
    return len(filas)


def cargar_estaciones(engine) -> int:
    if not CLIENT_ID or not CLIENT_SECRET:
        logger.warning("Sin ECOBICI_CLIENT_ID / ECOBICI_CLIENT_SECRET: no se actualizan estaciones")
        return 0
    api = descargar_estaciones()
    existentes = pd.read_sql("SELECT estacion_id, capacidad FROM estaciones", engine)

    nuevas = api[~api["estacion_id"].isin(existentes["estacion_id"])]
    if not nuevas.empty:
        nuevas.to_sql("estaciones", engine, if_exists="append", index=False)

    # Capacidad actualizada (las estaciones que ya no están en la API se conservan)
    cambios = api.merge(existentes, on="estacion_id", suffixes=("", "_db"))
    cambios = cambios[cambios["capacidad"] != cambios["capacidad_db"]]
    if not cambios.empty:
        with engine.begin() as conn:
            conn.execute(text("UPDATE estaciones SET capacidad = :cap WHERE estacion_id = :id"),
                         [{"cap": int(c.capacidad), "id": int(c.estacion_id)}
                          for c in cambios.itertuples()])
    logger.info(f"Estaciones: {len(nuevas)} nuevas | {len(cambios)} con capacidad actualizada")
    asignar_barrios(engine)
    return len(nuevas)


# ── Clima ──────────────────────────────────────────────────────────────────────
def descargar_clima(desde: date, hasta: date) -> pd.DataFrame:
    r = requests.get("https://archive-api.open-meteo.com/v1/archive", timeout=60, params={
        "latitude": LAT, "longitude": LON,
        "start_date": desde.isoformat(), "end_date": hasta.isoformat(),
        "hourly": "temperature_2m,apparent_temperature,relativehumidity_2m,"
                  "precipitation,windspeed_10m,weathercode",
        "timezone": "America/Argentina/Buenos_Aires",
    })
    r.raise_for_status()
    h = r.json()["hourly"]
    return pd.DataFrame({
        "timestamp_utc":        pd.to_datetime(h["time"]),
        "temperatura_c":        h["temperature_2m"],
        "sensacion_termica_c":  h["apparent_temperature"],
        "humedad_pct":          h["relativehumidity_2m"],
        "precipitacion_mm":     h["precipitation"],
        "velocidad_viento_kmh": h["windspeed_10m"],
        "codigo_clima":         h["weathercode"],
    })


def cargar_clima(engine) -> int:
    # Solo cuentan las horas con dato real: si una hora quedó en NULL se vuelve a pedir
    ultimo = scalar(engine, "SELECT MAX(timestamp_utc) FROM clima WHERE temperatura_c IS NOT NULL")
    ultimo = pd.Timestamp(ultimo) if ultimo is not None else None
    desde = ultimo.date() if ultimo is not None else date.fromisoformat(CLIMA_DESDE)
    hasta = date.today() - timedelta(days=CLIMA_LAG_DIAS)
    if desde > hasta:
        logger.info("Clima: al día")
        return 0

    total = 0
    while desde <= hasta:                       # de a un año por request
        fin = min(desde + timedelta(days=365), hasta)
        df = descargar_clima(desde, fin).dropna(subset=["temperatura_c"])
        if ultimo is not None:
            df = df[df["timestamp_utc"] > ultimo]
        if not df.empty:
            filas = df[CLIMA_COLS].astype(object).where(df[CLIMA_COLS].notna(), None)
            with engine.begin() as conn:
                # Borra las horas que hubieran quedado en NULL y reinserta
                conn.execute(text("DELETE FROM clima WHERE timestamp_utc >= :d"),
                             {"d": df["timestamp_utc"].min().to_pydatetime()})
                conn.execute(text(f"""
                    INSERT INTO clima ({", ".join(CLIMA_COLS)})
                    VALUES ({", ".join(":" + c for c in CLIMA_COLS)})
                """), filas.to_dict("records"))
            total += len(df)
        desde = fin + timedelta(days=1)
    logger.info(f"Clima: {total:,} horas nuevas (hasta {hasta})")
    return total


# ── Viajes ─────────────────────────────────────────────────────────────────────
def descargar_viajes(url: str) -> pd.DataFrame:
    # A disco y no a memoria: el zip pesa cientos de MB
    with tempfile.TemporaryFile() as tmp:
        with requests.get(url, stream=True, timeout=300) as r:
            r.raise_for_status()
            for parte in r.iter_content(chunk_size=1 << 20):
                tmp.write(parte)
        tmp.seek(0)
        with zipfile.ZipFile(tmp) as z:
            csv_name = next(n for n in z.namelist() if n.endswith(".csv"))
            df = pd.read_csv(z.open(csv_name), encoding="utf-8", low_memory=False)

    df = df.rename(columns=COLUMNAS)
    faltan = set(COLS_VIAJES) - set(df.columns)
    if faltan:
        raise ValueError(f"El CSV cambió de formato, faltan columnas: {faltan}. "
                         f"Columnas recibidas: {list(df.columns)}")

    for c in ["estacion_origen_id", "estacion_destino_id", "usuario_id"]:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace("BAEcobici", "", regex=False),
                              errors="coerce").astype("Int64")
    for c in ["fecha_origen", "fecha_destino"]:
        df[c] = pd.to_datetime(df[c], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    return df.dropna(subset=["fecha_origen", "estacion_origen_id", "estacion_destino_id"])


def cargar_viajes(engine) -> int:
    ultimo = scalar(engine, "SELECT MAX(fecha_origen) FROM viajes")
    ultimo = pd.Timestamp(ultimo) if ultimo is not None else None
    logger.info(f"Último viaje en la base: {ultimo}")

    total = 0
    for anio, url in sorted(VIAJES_URLS.items()):
        if ultimo is not None and anio < ultimo.year:
            continue                            # años ya completos: no se descargan
        logger.info(f"Descargando viajes {anio}...")
        df = descargar_viajes(url)
        if ultimo is not None:
            df = df[df["fecha_origen"] > ultimo]
        if df.empty:
            logger.info(f"  {anio}: sin viajes nuevos (BA Data todavía no actualizó el archivo)")
            continue
        copy_df(engine, df[COLS_VIAJES], "viajes")
        total += len(df)
        logger.info(f"  {anio}: {len(df):,} viajes nuevos "
                    f"({df['fecha_origen'].min()} → {df['fecha_origen'].max()})")
    if ultimo is not None and date.today().year not in VIAJES_URLS:
        logger.warning(f"No hay URL de viajes para {date.today().year}: agregala en "
                       f"VIAJES_URLS_EXTRA cuando BA Data publique el archivo")
    return total


# ── Main ───────────────────────────────────────────────────────────────────────
def run(engine=None) -> dict:
    engine = engine or create_engine(DB_URL, pool_pre_ping=True)
    resumen = {
        "estaciones_nuevas": cargar_estaciones(engine),
        "horas_clima": cargar_clima(engine),
        "viajes_nuevos": cargar_viajes(engine),
    }
    logger.info(f"ETL terminado: {resumen}")
    return resumen


def main():
    engine = create_engine(DB_URL, pool_pre_ping=True)
    try:
        run(engine)
    except Exception:
        logger.exception("Falló el ETL")
        hay_datos = (scalar(engine, "SELECT COUNT(*) FROM estaciones") and
                     scalar(engine, "SELECT EXISTS (SELECT 1 FROM viajes)"))
        if hay_datos:
            # Ej. BA Data caído: se sigue con los datos que ya hay en la base
            logger.warning("Se continúa con los datos existentes")
            return
        sys.exit(1)


if __name__ == "__main__":
    main()
