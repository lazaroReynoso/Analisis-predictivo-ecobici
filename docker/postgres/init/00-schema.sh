#!/bin/bash
# Crea las tablas de ecobici la primera vez que arranca el contenedor (volumen vacío).
# Si hay un dump en /dumps/ecobici.dump no hace nada: el dump trae su propio esquema
# y lo restaura 02-restore-ecobici.sh.
set -e
if [ -f /dumps/ecobici.dump ]; then
  echo ">> Hay dump: el esquema lo trae el dump"
  exit 0
fi
psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$POSTGRES_DB" -f /docker-entrypoint-initdb.d/sql/schema.sql
