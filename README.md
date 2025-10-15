# LeeWay production

Repo recovery scaffold for backend (FastAPI + Celery), frontend (Next.js), and infra (Docker Compose dev).

Quickstart (Dev):

1) Prereqs: Docker Desktop (Windows), Git
2) Copy env template

	cp infra/.env.example infra/.env
	cp backend/.env.example backend/.env

3) Start stack

	docker compose -f infra/docker-compose.dev.yml up --build

4) URLs

	Backend: http://localhost:8000/healthz
	Frontend: http://localhost:3000/

Notes:
- Celery worker uses app path: celery_app:celery_app
- OSRM is external; not included in this repo.
