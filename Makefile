.PHONY: help sync data features train evaluate predict notebook test clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

sync:  ## Install dependencies and the remdetect package (editable)
	uv sync --extra test

data:  ## Report the raw dataset (subjects, scored epochs, REM %)
	uv run python -m remdetect.dataset

features:  ## Build data/processed/featurematrix.npz from the raw recordings
	uv run python -m remdetect.splits

train:  ## Fit the model on the full feature set, serialize to models/
	uv run python -m remdetect.modeling.train

evaluate:  ## Leave-one-subject-out scoring; writes reports/ + reports/figures/
	uv run python -m remdetect.modeling.evaluate

train-logreg:  ## Nested-CV train the logistic model headless; writes reports/logreg_nested.json
	uv run python -m remdetect.modeling.tune logreg

train-xgboost:  ## Nested-CV train XGBoost headless; writes reports/xgboost_nested.json
	uv run python -m remdetect.modeling.tune xgboost

compare:  ## Combine the two per-model reports + paired test; writes reports/comparison_nested.json
	uv run python -m remdetect.modeling.compare

predict:  ## Load the trained model from models/ and predict on the feature set
	uv run python -m remdetect.modeling.predict

notebook:  ## Open the EDA notebook in marimo (pairing-ready: --no-token so an agent can discover it)
	uv run marimo edit notebooks/1.0-mm-eda.py --no-token

test:  ## Run the test suite
	uv run --extra test python -m pytest

clean:  ## Remove Python caches
	find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} +
	rm -rf .pytest_cache
