.PHONY: setup load-data prepare-data train evaluate test app api all

setup:
	pip install -r requirements.txt

load-data:
	python scripts/00_load_data.py

prepare-data:
	python scripts/01_prepare_data.py

train:
	python scripts/02_train_models.py

evaluate:
	python scripts/03_evaluate_models.py

test:
	pytest tests/

app:
	streamlit run app/streamlit_app.py

api:
	uvicorn app.api:app --reload

all: load-data prepare-data train evaluate test
