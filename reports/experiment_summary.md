# Experiment Summary

This log captures baseline runs for SmogGuard PK. Metrics are from the Prefect training flow on the AQI forecasting task (3-hour horizon). Each flow run also appends a row to `reports/experiments_log.csv`.

## Latest Baseline (RandomForest)
- Regression (AQI): RMSE ~0.46, MAE ~0.30, R² ~0.82
- Classification (hazard): Accuracy ~0.90, F1 ~0.93, ROC-AUC ~0.97
- Recommender (severity bucket): ~deterministic tree fitted on regression labels

## Observations
- Data balance: Hazard label is skewed toward safe/moderate; balanced class weights help for classification.
- Overfitting risk: Tree ensembles are stable on validation split; monitor ROC-AUC/F1 per city when new data arrives.
- Drift: Reference stats stored in `data/reference_stats.json`; CI drift checks guard feature distributions.
- Deployment speed: CI (~pytest + ML checks) completes quickly; images are built in CD. Prefect flow retries on fetch/validation to reduce flakes.

## How to Log New Runs
1) Run training: `python -m src.pipelines.aqi_flow` (or `make train`).
2) On success, a CSV row is appended to `reports/experiments_log.csv` with metrics, model paths, and run_id.
3) Add brief notes if needed by editing the CSV or extending the call in `src/pipelines/aqi_flow.py`.

## Next Experiments (suggested)
- Compare lighter/faster models vs current RF for latency.
- City-specific fine-tuning or per-city calibration of hazard thresholds.
- Add ROC-AUC/F1 per city to the tracker to surface segment-specific gaps.
