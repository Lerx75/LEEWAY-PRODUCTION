#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-/root/LEEWAY-PRODUCTION}
COMPOSE=${COMPOSE:-"docker compose -f infra/docker-compose.prod.yml"}

echo "[deploy] Using repo dir: $REPO_DIR"
cd "$REPO_DIR"

echo "[deploy] Ensuring docker network leeway-net exists"
docker network create leeway-net || true

if [ ! -f infra/.env.prod ]; then
  if [ -f infra/.env.production ]; then
    echo "[deploy][WARN] infra/.env.prod not found, but infra/.env.production exists. Copying as fallback."
    cp infra/.env.production infra/.env.prod
  else
    echo "[deploy][ERROR] Missing infra/.env.prod (and no infra/.env.production to fallback)"
    exit 1
  fi
fi

echo "[deploy] Rebuilding backend, worker, frontend"
$COMPOSE up -d --build --no-deps backend worker frontend

echo "[deploy] Health checks"
curl -s http://localhost:8000/healthz || true
echo
curl -s http://localhost:8000/healthz/osrm || true
echo
curl -s http://localhost:8000/api/debug-config || true
echo
echo "[deploy] Done"
