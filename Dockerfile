# Una sola imagen para la API, el entrenamiento y el monitor de drift:
# mismo código y mismas librerías → las features se calculan igual en todos lados.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GIT_PYTHON_REFRESH=quiet

# libgomp1: la necesita XGBoost para usar varios núcleos
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ .

EXPOSE 8000
CMD ["uvicorn", "predict_api:app", "--host", "0.0.0.0", "--port", "8000"]
