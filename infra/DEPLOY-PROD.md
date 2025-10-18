# Production deployment (droplet)

This uses container names: `leeway-backend`, `leeway-worker`, `leeway-frontend`, `leeway-redis`, and `osrm-routing`.

## One-time setup

1) On the droplet:
```
sudo apt-get update && sudo apt-get install -y docker.io docker-compose-plugin
docker network create leeway-net || true
```

2) Copy your env file based on the template in this repo:
```
cp infra/.env.prod.example infra/.env.prod
# Edit infra/.env.prod and set real secrets + OSRM_URL
# REQUIRED: Set PROJECT_SERVICE_TOKEN to a strong random value. The frontend will forward this via X-Service-Token.
```

## Build and run

From this repo root:
```
docker compose -f infra/docker-compose.prod.yml build
docker compose -f infra/docker-compose.prod.yml up -d
```

## Verify

```
curl -s http://localhost:8000/healthz
curl -s http://localhost:8000/osrm/debug
```
Expect `base` to be your OSRM (`http://osrm-routing:5000` or droplet IP) and `strict: true`.

## Rolling update

```
docker compose -f infra/docker-compose.prod.yml pull
docker compose -f infra/docker-compose.prod.yml up -d --build backend worker frontend
```

## Troubleshooting

- If you already run OSRM separately, comment out the `osrm-routing` service in the compose file.
- Ensure all services are on the `leeway-net` network (compose does this via default).
- If backend cannot reach OSRM by name, set `OSRM_URL=http://<droplet-ip>:5000` in `.env.prod`.
- Backend and worker share `/srv/data` via a named Docker volume `leeway-data` defined in compose. If you previously ran containers without this, remove and recreate to attach the volume.
- After code changes to `backend/app.py`, rebuild BOTH backend and worker images so the worker picks up changes (IndentationError or stale code can occur otherwise):
	- `docker compose -f infra/docker-compose.prod.yml up -d --build backend worker`

## Production env checklist (to avoid "fetch failed")

Create `infra/.env.prod` from `infra/.env.prod.example` and ensure:

- `PROJECT_SERVICE_TOKEN` is set to a strong random value. Frontend forwards this via `X-Service-Token` and backend validates it.
- Set either `NEXT_PUBLIC_API_URL` (preferred) or `NEXT_PUBLIC_API_BASE` to your public backend URL, e.g. `https://api.leewayroute.com`.
- `ALLOWED_ORIGINS` includes your frontend origin, e.g. `https://www.leewayroute.com`.
- `OSRM_URL` points to your OSRM, e.g. `http://osrm-routing:5000` when attached to `leeway-net`.

Example snippet:

```
PROJECT_SERVICE_TOKEN=REPLACE_WITH_STRONG_RANDOM
NEXT_PUBLIC_API_URL=https://api.leewayroute.com
ALLOWED_ORIGINS=https://www.leewayroute.com
REDIS_URL=redis://leeway-redis:6379/0
CELERY_BROKER_URL=redis://leeway-redis:6379/0
CELERY_RESULT_BACKEND=redis://leeway-redis:6379/1
OSRM_URL=http://osrm-routing:5000
OSRM_STRICT=1
```

After updating the env file, rebuild and restart services:

```
docker compose -f infra/docker-compose.prod.yml up -d --build frontend backend worker
```
