.PHONY: help sync train-logreg train-xgboost train predict notebooks test

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

sync:  ## Install dependencies and the remdetect package (editable)
	uv sync --extra test

train-logreg:  ## Nested-CV the logistic model; writes reports/logreg_nested.json
	uv run python -m remdetect.modeling.tune logreg

train-xgboost:  ## Nested-CV XGBoost; writes reports/xgboost_nested.json
	uv run python -m remdetect.modeling.tune xgboost

train:  ## Fit the deployment model on all data, serialize to models/
	uv run python -m remdetect.modeling.train

predict:  ## Load the trained model from models/ and predict
	uv run python -m remdetect.modeling.predict

notebooks:  ## Open the marimo server on notebooks/ (browse and open all notebooks)
	uv run marimo edit notebooks/ --no-token

test:  ## Run the test suite
	uv run --extra test python -m pytest
