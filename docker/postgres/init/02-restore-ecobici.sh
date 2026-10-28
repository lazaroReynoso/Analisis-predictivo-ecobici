#!/bin/bash
# Si dejaste un dump en ./dumps/ecobici.dump, se restaura en la base "ecobici"
# la primera vez que arranca el contenedor (volumen vacío).
set -e
DUMP=/dumps/ecobici.dump
if [ -f "$DUMP" ]; then
  echo ">> Restaurando $DUMP en la base ecobici (puede tardar varios minutos)..."
  pg_restore --no-owner --no-privileges -U "$POSTGRES_USER" -d "$POSTGRES_DB" "$DUMP"
  echo ">> Restauración terminada"
else
  echo ">> No hay $DUMP: la base ecobici queda vacía (cargala con tu ETL)"
fi
