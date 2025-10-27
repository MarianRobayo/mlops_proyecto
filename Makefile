.PHONY: install train test lint clean

install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

train:
	python -m src.train --config config.yaml

test:
	PYTHONPATH=. pytest -q

lint:
	flake8 src tests

clean:
	rm -rf mlruns/ artifacts/ __pycache__ 
