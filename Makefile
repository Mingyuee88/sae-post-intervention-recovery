PYTHON ?= python3

.PHONY: help test safety-check inspect-results release-check clean-generated

help:
	@echo "Targets:"
	@echo "  test             Run lightweight unit and release-safety tests"
	@echo "  safety-check     Scan for raw outputs, private paths, credentials, and placeholders"
	@echo "  inspect-results  Print a compact summary of sanitized paper artifacts"
	@echo "  release-check    Run all checks expected before publishing"
	@echo "  clean-generated  Remove Python caches and pytest caches"

test:
	$(PYTHON) -m pytest tests -q

safety-check:
	$(PYTHON) scripts/safety_check_release.py

inspect-results:
	$(PYTHON) scripts/inspect_results.py

release-check: test safety-check inspect-results

clean-generated:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name .pytest_cache -prune -exec rm -rf {} +
