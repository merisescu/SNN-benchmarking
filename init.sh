#!/bin/bash
set -e

psql -v ON_ERROR_STOP=1 \
  --username "$POSTGRES_USER" \
  --dbname postgres <<-EOSQL

CREATE DATABASE "$LIF_DB";
CREATE DATABASE "$ALIF_DB";
CREATE DATABASE "$IZH_DB";

EOSQL