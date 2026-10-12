# Deployment helpers for docker01. Run `make help` for the list.
#
# Override the tools if needed, for example with Podman:
#   make check COMPOSE="podman compose" DOCKER=podman

COMPOSE ?= docker compose
DOCKER ?= docker
NETWORK ?= cloudflare
FRONTEND_IMAGE ?= phishing-url-detector-frontend:latest

SMOKE_BODY = {"url": "example.com"}

.DEFAULT_GOAL := help
.PHONY: help check network build up deploy down restart ps logs update smoke smoke-tunnel clean

help: ## Show available targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "} {printf "  %-14s %s\n", $$1, $$2}'

check: ## Validate compose.yaml
	$(COMPOSE) config --quiet && echo "compose.yaml is valid"

network: ## Verify that the external Cloudflare network exists
	@$(DOCKER) network inspect $(NETWORK) >/dev/null 2>&1 \
		|| { echo "Network '$(NETWORK)' not found. It is created by the Cloudflare Tunnel setup."; exit 1; }
	@echo "Network '$(NETWORK)' exists"

build: ## Build both images
	$(COMPOSE) build

up: network ## Start the stack in the background (builds if needed)
	$(COMPOSE) up -d

deploy: network build ## Build and start, then run the smoke test
	$(COMPOSE) up -d --wait
	$(MAKE) smoke

down: ## Stop and remove the containers
	$(COMPOSE) down

restart: ## Restart both containers
	$(COMPOSE) restart

ps: ## Show container status and health
	$(COMPOSE) ps

logs: ## Follow logs from both services
	$(COMPOSE) logs -f --tail=100

update: ## Pull the latest code, rebuild and restart
	git pull --ff-only
	$(MAKE) deploy

smoke: ## Send a test prediction through Nginx inside the frontend container
	$(COMPOSE) exec -T frontend wget -qO- \
		--header 'Content-Type: application/json' \
		--post-data '$(SMOKE_BODY)' http://127.0.0.1/api/predict
	@echo

smoke-tunnel: ## Test http://phishing:80 the way the Cloudflare Tunnel reaches it
	$(DOCKER) run --rm --network $(NETWORK) --entrypoint wget $(FRONTEND_IMAGE) -qO- \
		--header 'Content-Type: application/json' \
		--post-data '$(SMOKE_BODY)' http://phishing:80/api/predict
	@echo

clean: ## Remove dangling images left over from rebuilds
	$(DOCKER) image prune -f
