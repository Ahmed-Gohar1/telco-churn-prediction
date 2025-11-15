
Telco Churn 

data-science Telco churn experiments: preprocessing, model training, and comparison.

Usage
- Clean data: `python src/clean_data.py` (writes `data/telco_cleaned.csv`).
- Train models: `python src/train_model.py` (creates `artifacts/` with models and scaler).
- Compare models / plots: `python src/compare_models.py` and `python src/comparison_plots.py`.

What’s in this folder
- `data/` — source datasets (`telco_raw.csv`, `telco_cleaned.csv`, streamlit test CSVs).
- `src/` — scripts: `clean_data.py`, `train_model.py`, `compare_models.py`, `comparison_plots.py`.
- `notebooks/` — analysis notebooks (executed and cleaned versions).
- `demo/` — `demo_streamlit.py` for a minimal Streamlit demo.
- `artifacts/` — model files and scalers (usually not tracked).
- `tests/` — basic unit tests (`tests/test_model.py`).
