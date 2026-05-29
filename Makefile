.PHONY: grace-deep grace-deep-dry grace-deep-auto grace-deep-layer lint audit

grace-deep:
	scripts/grace-full-refresh.sh

grace-deep-dry:
	scripts/grace-full-refresh.sh --dry-run

grace-deep-auto:
	scripts/grace-full-refresh.sh --auto

grace-deep-layer:
	scripts/grace-full-refresh.sh --layer $(LAYER)

audit:
	python3 scripts/audit_grace_coverage.py --summary

lint:
	ruff check src/
