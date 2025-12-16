.PHONY: train api ui test

train:
	python -m src.pipelines.aqi_flow

api:
	uvicorn app.main:app --reload

ui:
	streamlit run ui_app.py

test:
	pytest
