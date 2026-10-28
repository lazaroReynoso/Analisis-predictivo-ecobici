-- Esquema de la base ecobici.
-- Si tu base local tiene otro esquema, generalo con:
--   pg_dump --schema-only -h 127.0.0.1 -U postgres -d ecobici > schema.sql
-- y reemplazá este archivo.

CREATE TABLE IF NOT EXISTS estaciones (
    estacion_id INTEGER PRIMARY KEY,
    nombre      TEXT,
    direccion   TEXT,
    lat         DOUBLE PRECISION,
    lon         DOUBLE PRECISION,
    capacidad   INTEGER,
    barrio      TEXT,
    comuna      INTEGER
);

CREATE TABLE IF NOT EXISTS viajes (
    viaje_id            BIGSERIAL PRIMARY KEY,
    usuario_id          BIGINT,
    genero              TEXT,
    estacion_origen_id  INTEGER NOT NULL,
    estacion_destino_id INTEGER,
    fecha_origen        TIMESTAMP NOT NULL,
    fecha_destino       TIMESTAMP,
    -- Lo calcula Postgres solo; el entrenamiento filtra por anio.
    anio                INTEGER GENERATED ALWAYS AS (EXTRACT(YEAR FROM fecha_origen)::INTEGER) STORED
);
CREATE INDEX IF NOT EXISTS ix_viajes_fecha_origen ON viajes (fecha_origen);
CREATE INDEX IF NOT EXISTS ix_viajes_anio         ON viajes (anio);
CREATE INDEX IF NOT EXISTS ix_viajes_origen       ON viajes (estacion_origen_id);

-- Ojo: a pesar del nombre, timestamp_utc guarda hora de Buenos Aires
-- (Open-Meteo se pide con timezone=America/Argentina/Buenos_Aires), igual que los viajes.
CREATE TABLE IF NOT EXISTS clima (
    timestamp_utc        TIMESTAMP PRIMARY KEY,
    temperatura_c        REAL,
    sensacion_termica_c  REAL,
    humedad_pct          REAL,
    precipitacion_mm     REAL,
    velocidad_viento_kmh REAL,
    codigo_clima         INTEGER,
    es_lluvia            BOOLEAN GENERATED ALWAYS AS (precipitacion_mm > 0) STORED
);
