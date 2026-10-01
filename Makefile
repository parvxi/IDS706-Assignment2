# Shortcuts for common tasks. Run e.g. `make test`.

install:
	pip install -r requirements.txt

format:
	black EDA_Dataset.py tests/

lint:
	black --check EDA_Dataset.py tests/
	flake8 EDA_Dataset.py tests/

test:
	pytest

run:
	python EDA_Dataset.py

docker-build:
	docker build -t sleep-analysis .

docker-run:
	docker run --rm -v "$$(pwd)/images:/app/images" sleep-analysis

docker-test:
	docker run --rm sleep-analysis pytest -q -o addopts=""

all: lint test

.PHONY: install format lint test run docker-build docker-run docker-test all
