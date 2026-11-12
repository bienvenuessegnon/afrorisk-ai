PYTHON ?= python3

.PHONY: fixture eda train test api web

fixture:
	$(PYTHON) -m afrorisk.synthetic

eda:
	$(PYTHON) -m afrorisk.eda

train:
	$(PYTHON) -m afrorisk.train

test:
	$(PYTHON) -m pytest

api:
	$(PYTHON) -m uvicorn afrorisk.api.main:app --host 0.0.0.0 --port 8000

web:
	cd frontend && npm run dev
