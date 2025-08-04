# MLOps Best Practices Makefile
# Provides common development tasks and automation

.PHONY: help install install-dev clean test test-unit test-integration test-coverage lint format quality build deploy deploy-staging

# Default target
help:
	@echo "MLOps Best Practices - Available Commands:"
	@echo ""
	@echo "Setup:"
	@echo "  install        Install production dependencies"
	@echo "  install-dev    Install development dependencies"
	@echo "  setup          Setup pre-commit hooks"
	@echo ""
	@echo "Testing:"
	@echo "  test           Run all tests"
	@echo "  test-unit      Run unit tests only"
	@echo "  test-integration Run integration tests only"
	@echo "  test-coverage  Run tests with coverage report"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint           Run linting checks"
	@echo "  format         Format code with black and isort"
	@echo "  quality        Run all quality checks (lint + format + test)"
	@echo ""
	@echo "Build & Deploy:"
	@echo "  build          Build the project"
	@echo "  deploy         Deploy to production"
	@echo "  deploy-staging Deploy to staging"
	@echo ""
	@echo "Maintenance:"
	@echo "  clean          Clean up generated files"
	@echo "  docs           Generate documentation"

# Setup
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

setup: install-dev
	pre-commit install

# Testing
test:
	pytest

test-unit:
	pytest tests/ -m "not integration"

test-integration:
	pytest tests/ -m integration

test-coverage:
	pytest --cov=src --cov-report=html --cov-report=term-missing

# Code Quality
lint:
	flake8 src/ tests/
	mypy src/

format:
	black src/ tests/
	isort src/ tests/

quality: format lint test

# Build & Deploy
build:
	python -m build

deploy:
	@echo "Deploying to production..."
	@echo "This would typically involve:"
	@echo "1. Running all tests"
	@echo "2. Building Docker image"
	@echo "3. Pushing to container registry"
	@echo "4. Deploying to production environment"
	@echo "Not implemented in this example"

deploy-staging:
	@echo "Deploying to staging..."
	@echo "This would typically involve:"
	@echo "1. Running all tests"
	@echo "2. Building Docker image"
	@echo "3. Pushing to container registry"
	@echo "4. Deploying to staging environment"
	@echo "Not implemented in this example"

# Documentation
docs:
	@echo "Generating documentation..."
	@echo "This would typically involve:"
	@echo "1. Running Sphinx"
	@echo "2. Building HTML docs"
	@echo "Not implemented in this example"

# Maintenance
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	find . -type f -name "bandit-report.json" -delete
	rm -rf build/
	rm -rf dist/
	rm -rf .tox/

# Development helpers
check-all: quality test-coverage
	@echo "All checks completed!"

quick-test:
	pytest tests/ -x -v --tb=short

# Docker helpers (if using Docker)
docker-build:
	docker build -t mlops-spine-disease .

docker-run:
	docker run -p 8000:8000 mlops-spine-disease

docker-test:
	docker run mlops-spine-disease pytest

# CI/CD helpers
ci-test:
	pytest --cov=src --cov-report=xml --cov-report=term-missing

ci-lint:
	flake8 src/ tests/
	mypy src/

ci-format-check:
	black --check src/ tests/
	isort --check-only src/ tests/

# Security
security-check:
	bandit -r src/ -f json -o bandit-report.json

# Performance
benchmark:
	@echo "Running performance benchmarks..."
	@echo "Not implemented in this example"

# Database helpers (if using databases)
db-migrate:
	@echo "Running database migrations..."
	@echo "Not implemented in this example"

db-seed:
	@echo "Seeding database with test data..."
	@echo "Not implemented in this example"

# Environment setup
env-create:
	python -m venv .venv
	@echo "Virtual environment created. Activate with:"
	@echo "source .venv/bin/activate  # Linux/Mac"
	@echo ".venv\\Scripts\\activate     # Windows"
	@echo ""
	@echo "⚠️  IMPORTANT: This module requires isolated virtual environment"
	@echo "   due to conflicting dependencies with other MLOps modules."

env-activate:
	@echo "Activate virtual environment:"
	@echo "source .venv/bin/activate  # Linux/Mac"
	@echo ".venv\\Scripts\\activate     # Windows"
	@echo ""
	@echo "⚠️  Remember: Each module should have its own virtual environment"

clean-env:
	@echo "Cleaning virtual environment..."
	rm -rf .venv/
	@echo "Virtual environment removed. Run 'make env-create' to recreate."

# Git helpers
git-hooks: setup
	@echo "Git hooks installed via pre-commit"

git-clean:
	git clean -fd
	git reset --hard HEAD

# Monitoring and logging
logs:
	@echo "Viewing application logs..."
	@echo "Not implemented in this example"

monitor:
	@echo "Starting monitoring dashboard..."
	@echo "Not implemented in this example"

# Backup and restore
backup:
	@echo "Creating backup..."
	@echo "Not implemented in this example"

restore:
	@echo "Restoring from backup..."
	@echo "Not implemented in this example" 