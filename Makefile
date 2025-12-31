# Makefile for PinescriptAutogenLab

# Default command
.DEFAULT_GOAL := help

# Environment variables
PROJECT_NAME = pinelab
DOCKER_COMPOSE_DEV = docker-compose -f docker-compose.dev.yml
DOCKER_COMPOSE_PROD = docker-compose -f docker-compose.prod.yml

# ==============================================================================
# Development Commands
# ==============================================================================

.PHONY: dev-up
dev-up: ## Start the development environment
	@echo "Starting development environment..."
	$(DOCKER_COMPOSE_DEV) up -d --build

.PHONY: dev-down
dev-down: ## Stop the development environment
	@echo "Stopping development environment..."
	$(DOCKER_COMPOSE_DEV) down

.PHONY: dev-logs
dev-logs: ## Tail the logs for the development environment
	@echo "Tailing development logs..."
	$(DOCKER_COMPOSE_DEV) logs -f

.PHONY: dev-ps
dev-ps: ## List running containers for the development environment
	@echo "Development containers:"
	$(DOCKER_COMPOSE_DEV) ps

# ==============================================================================
# Production Commands
# ==============================================================================

.PHONY: prod-up
prod-up: ## Start the production environment
	@echo "Starting production environment..."
	$(DOCKER_COMPOSE_PROD) --profile production up -d --build

.PHONY: prod-down
prod-down: ## Stop the production environment
	@echo "Stopping production environment..."
	$(DOCKER_COMPOSE_PROD) down

.PHONY: prod-logs
prod-logs: ## Tail the logs for the production environment
	@echo "Tailing production logs..."
	$(DOCKER_COMPOSE_PROD) logs -f

.PHONY: prod-ps
prod-ps: ## List running containers for the production environment
	@echo "Production containers:"
	$(DOCKER_COMPOSE_PROD) ps

# ==============================================================================
# General Commands
# ==============================================================================

.PHONY: clean
clean: ## Remove all stopped containers and dangling images
	@echo "Cleaning up Docker..."
	docker system prune -f

.PHONY: help
help: ## Display this help message
	@echo "Usage: make [command]"
	@echo ""
	@echo "Development Commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'
	@echo ""
	@echo "Production Commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'
	@echo ""
	@echo "General Commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

.PHONY: test
test: ## Run the test suite
	@echo "Running tests..."
	$(DOCKER_COMPOSE_DEV) exec backend pytest
