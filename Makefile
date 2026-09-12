.PHONY: install run test lint

install:
	python -m pip install -r requirements-dev.txt

run:
	streamlit run streamlit_app.py

test:
	pytest

lint:
	ruff check .
