.PHONY: up down build logs test clean

up:
	docker compose up --build -d
	@echo "Backend:  http://localhost:8000"
	@echo "Frontend: http://localhost:3000"

down:
	docker compose down

logs:
	docker compose logs -f

test:
	cd backend && pytest -v

clean:
	docker compose down -v
