up:
	docker compose -f infra/docker-compose.dev.yml up --build

down:
	docker compose -f infra/docker-compose.dev.yml down

logs:
	docker compose -f infra/docker-compose.dev.yml logs -f

be-shell:
	docker exec -it leeway-backend sh

fe-shell:
	docker exec -it leeway-frontend sh
