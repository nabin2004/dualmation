# ╔══════════════════════════════════════════════════════════════╗
# ║  DualAnimate Makefile                                       ║
# ║  Common commands for development, testing, and experiments  ║
# ╚══════════════════════════════════════════════════════════════╝

.PHONY: help install install-dev install-all test lint format \
        run train evaluate info config-show clean docs \
        tensorboard env-info

PYTHON     ?= python
PIP        ?= pip
CONFIG     ?= configs/default.yaml
CONCEPT    ?= "Explain gradient descent visually"
SEED       ?= 42

# ── Help ────────────────────────────────────────────────────────

help: ## Show this help message
	@echo ""
	@echo "  DualAnimate — Available Commands"
	@echo "  ═══════════════════════════════════════════"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
	@echo ""

# ── Installation ────────────────────────────────────────────────

install: ## Install core dependencies
	$(PIP) install -e .

install-dev: ## Install with development dependencies
	$(PIP) install -e ".[dev]"

install-all: ## Install all dependencies including experiment extras
	$(PIP) install -e ".[dev,experiment]"

# ── Testing ─────────────────────────────────────────────────────

test: ## Run all tests
	PYTHONPATH=src $(PYTHON) -m pytest tests/ -v

test-cov: ## Run tests with coverage report
	PYTHONPATH=src $(PYTHON) -m pytest tests/ -v --cov=dualmation --cov-report=term-missing --cov-report=html:htmlcov

test-fast: ## Run tests without slow markers
	PYTHONPATH=src $(PYTHON) -m pytest tests/ -v -m "not slow"

# ── Code Quality ────────────────────────────────────────────────

lint: ## Run linter (ruff)
	$(PYTHON) -m ruff check src/ tests/

format: ## Auto-format code (ruff)
	$(PYTHON) -m ruff format src/ tests/
	$(PYTHON) -m ruff check --fix src/ tests/

typecheck: ## Run type checker (mypy)
	$(PYTHON) -m mypy src/dualmation/

# ── Pipeline Commands ───────────────────────────────────────────

run: ## Run the pipeline (use CONFIG=... CONCEPT=...)
	PYTHONPATH=src $(PYTHON) -m dualmation.cli --config $(CONFIG) run \
		--concept $(CONCEPT) \
		--seed $(SEED)

train: ## Start training (use CONFIG=... EPOCHS=...)
	PYTHONPATH=src $(PYTHON) -m dualmation.cli train \
		--config $(CONFIG) \
		$(if $(EPOCHS),--epochs $(EPOCHS),) \
		$(if $(LR),--lr $(LR),) \
		$(if $(BATCH_SIZE),--batch-size $(BATCH_SIZE),)

evaluate: ## Evaluate a checkpoint (use CHECKPOINT=...)
	PYTHONPATH=src $(PYTHON) -m dualmation.cli evaluate \
		--config $(CONFIG) \
		$(CHECKPOINT)

# ── Configuration ───────────────────────────────────────────────

config-show: ## Display current config (use CONFIG=...)
	PYTHONPATH=src $(PYTHON) -m dualmation.cli config show $(CONFIG)

config-init: ## Create a new config (use OUTPUT=... NAME=...)
	PYTHONPATH=src $(PYTHON) -m dualmation.cli config init \
		$(or $(OUTPUT),configs/my_experiment.yaml) \
		--experiment-name $(or $(NAME),"my_experiment")

config-diff: ## Compare two configs (use CONFIG_A=... CONFIG_B=...)
	PYTHONPATH=src $(PYTHON) -m dualmation.cli config diff $(CONFIG_A) $(CONFIG_B)

# ── Experiment Tools ────────────────────────────────────────────

info: ## Show system and environment info
	PYTHONPATH=src $(PYTHON) -m dualmation.cli info

tensorboard: ## Launch TensorBoard (use LOGDIR=...)
	tensorboard --logdir $(or $(LOGDIR),experiments/) --port $(or $(TB_PORT),6006)

env-snapshot: ## Save environment snapshot for reproducibility
	PYTHONPATH=src $(PYTHON) -c "from dualmation.experiment.reproducibility import save_environment_snapshot; save_environment_snapshot('experiments/')"
	@echo "✅ Environment snapshot saved to experiments/environment/"

# ── Cleanup ─────────────────────────────────────────────────────

clean: ## Remove build artifacts and caches
	rm -rf build/ dist/ *.egg-info src/*.egg-info
	rm -rf .pytest_cache htmlcov .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "✅ Cleaned build artifacts"

clean-experiments: ## Remove all experiment output (CAREFUL!)
	@echo "⚠️  This will delete experiments/ and outputs/ directories"
	@read -p "Are you sure? [y/N] " confirm && [ "$$confirm" = "y" ] && \
		rm -rf experiments/ outputs/ || echo "Cancelled."

# ── Default ─────────────────────────────────────────────────────

.DEFAULT_GOAL := help
