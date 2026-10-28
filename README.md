# Ecobici — predicción de demanda por hora (MLOps)

Predice cuántos viajes salen de cada estación de Ecobici (Buenos Aires) en una hora dada.
Todo el stack corre con un solo comando:

```bash
cp .env.example .env        # completar POSTGRES_PASSWORD
docker compose up --build
```

| Servicio        | Qué hace                                                                 | URL                          |
|-----------------|--------------------------------------------------------------------------|------------------------------|
| `postgres`      | Datos de viajes, estaciones y clima + base interna de MLflow             | `localhost:5433`             |
| `etl`           | Carga lo que falte: viajes (BA Data), clima (Open-Meteo), estaciones     | —                            |
| `mlflow`        | Tracking de experimentos, registro de modelos y artefactos               | http://localhost:5000        |
| `trainer`       | Entrena XGBoost (2 variantes + Optuna) si no hay modelos registrados     | —                            |
| `api`           | FastAPI: elige el modelo con menor PoissonDev y sirve predicciones       | http://localhost:8000/docs   |
| `drift-monitor` | Cada semana: carga datos nuevos, re-evalúa el modelo y alerta si el RMSE sube | —                       |

## Datos

`src/etl.py` es incremental: se puede correr cuantas veces quieras y solo agrega lo nuevo.
Con la base vacía descarga todo desde cero (tarda); para arrancar más rápido podés dejar un
dump en `dumps/ecobici.dump` y se restaura la primera vez:

```bash
pg_dump -Fc -h 127.0.0.1 -U postgres -d ecobici -f dumps/ecobici.dump
```

Las estaciones salen de la API de transporte del GCBA, que pide credenciales
(`ECOBICI_CLIENT_ID` / `ECOBICI_CLIENT_SECRET` en el `.env`).

## Detección de drift

`src/drift_monitor.py`, cada lunes a las 6:00:

1. Corre el ETL para traer viajes, clima y estaciones nuevos.
2. Carga el modelo en producción (el mismo que usa la API).
3. Arma las features de la última semana con la misma lógica que el entrenamiento.
4. Compara el RMSE de esa semana con el RMSE de test registrado al entrenar.
5. Calcula el PSI de features clave (drift en los datos de entrada).
6. Registra todo en el experimento `ecobici_drift_monitoring` de MLflow.
7. Si el RMSE sube más de `RMSE_THRESHOLD_PCT` → alerta (log + webhook de Slack/Discord)
   y, opcionalmente, re-entrena y recarga la API (`AUTO_RETRAIN=true`).

```bash
docker compose run --rm drift-monitor python drift_monitor.py   # correrlo ahora
```

## Comandos útiles

```bash
docker compose run --rm etl                           # cargar datos nuevos ahora
FORCE_TRAIN=true docker compose run --rm trainer      # re-entrenar
curl -X POST localhost:8000/reload                    # que la API tome el mejor modelo
curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
     -d '{"station_id": 14, "datetime": "2026-10-01T08:00:00"}'
docker compose logs -f drift-monitor
docker compose down        # apaga (los datos quedan en los volúmenes)
docker compose down -v     # apaga y BORRA datos y modelos
```
