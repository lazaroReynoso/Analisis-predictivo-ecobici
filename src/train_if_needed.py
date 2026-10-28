"""
Entrena solo si todavía no hay modelos registrados en MLflow.

Lo usa el servicio `trainer` de docker-compose: en el primer `docker compose up`
entrena; en los siguientes termina enseguida y la API arranca con el modelo existente.
Para forzar un re-entrenamiento:  FORCE_TRAIN=true docker compose run --rm trainer
"""
import logging
import os

import mlflow
from mlflow import MlflowClient

import mlflow_training_v4 as tr

logger = logging.getLogger("train_if_needed")
MODEL_NAMES = os.getenv("MODEL_NAMES", "xgboost_v4_rolling,xgboost_v4_mensual").split(",")
FORCE_TRAIN = os.getenv("FORCE_TRAIN", "false").lower() == "true"


def main():
    mlflow.set_tracking_uri(tr.MLFLOW_URI)
    client = MlflowClient()
    existentes = [n for n in MODEL_NAMES if client.search_model_versions(f"name='{n}'")]
    if existentes and not FORCE_TRAIN:
        logger.info(f"Ya hay modelos registrados ({existentes}); no se entrena.")
        return
    tr.main()


if __name__ == "__main__":
    main()
