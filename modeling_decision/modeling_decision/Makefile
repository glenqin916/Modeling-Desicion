.PHONY: all data run extra figures test smoke

all: run extra figures

data:
	python data/fetch_data.py

run:
	python -m src.run

extra:
	python -m src.extra

figures:
	python -m src.figures

test:
	python -m pytest -q

smoke:
	python tests/make_fake_data.py .smoke/diabetic_data.csv --n 30000
	python -m src.run --quick --data .smoke/diabetic_data.csv --split-file .smoke/split.csv --out .smoke/results
