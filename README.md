# HOEP Forecasting Project

A multi-model forecasting system for the **Hourly Ontario Electricity Price (HOEP)**, comparing statistical, machine learning, and deep learning approaches across 1-hour, 2-hour, and 3-hour prediction horizons.

## Overview

This project forecasts Ontario electricity market clearing prices using data from the Independent Electricity System Operator (IESO). The final models are benchmarked against IESO's own official predispatch forecasts. The GRU deep learning model achieves the best performance.

**Data range:** January 2023 – April 2025  
**Granularity:** Hourly observations  
**Horizons:** h=1, h=2, h=3 hours ahead  

---

## Project Structure

```
HOEP-forcasting-project/
├── data/
│   ├── raw/            # Raw IESO CSVs (HOEP, demand, generation)
│   ├── interim/        # Cleaned Parquet files
│   └── processed/      # Model-ready datasets (stat_dir/, ml_dir/, dl_dir/)
├── models/             # Trained model artifacts (.pkl, .joblib, .pt)
├── artifacts/scalers/  # MinMax scalers (scaler_X.pkl, scaler_y.pkl)
├── notebooks/          # Jupyter notebooks (01–07, run in order)
├── results/            # Output figures and GRU predictions
├── src/                # Modular source code (features, models, utils, etc.)
├── configs/            # Configuration files
├── data.dvc            # DVC data tracking
└── requirements.txt
```

---

## Data Sources

All data is sourced from the [IESO](https://reports-public.ieso.ca/public/):

| Dataset | Description |
|---------|-------------|
| `PUB_PriceHOEPPredispOR_*.csv` | Hourly Ontario Electricity Prices (HOEP) |
| `PUB_Demand_*.csv` | Hourly Ontario electricity demand |
| `generation_*.csv` | Hourly generation by fuel type |

---

## Models

| Category | Models |
|----------|--------|
| Statistical | ARIMA, SARIMA |
| Machine Learning | Linear Regression, Random Forest, XGBoost |
| Deep Learning | LSTM, GRU (PyTorch) |

Each model is trained separately for each forecast horizon (h=1, h=2, h=3).

**Evaluation metrics:** MAE, RMSE, MAPE

---

## Notebooks

Run in order:

| Notebook | Description |
|----------|-------------|
| `01_data_ingestion.ipynb` | Load and clean raw IESO CSVs using PySpark; output cleaned Parquet files |
| `02_data_merging_and_eda.ipynb` | Merge datasets; ACF/PACF analysis; stationarity tests |
| `03_feature_engineering.ipynb` | Lag features, rolling stats, calendar features; train/val/test splits |
| `04_statistical_models.ipynb` | Train and evaluate ARIMA and SARIMA models |
| `05_ml_models.ipynb` | Train and tune Linear Regression, Random Forest, XGBoost |
| `06_dl_models.ipynb` | Train LSTM and GRU models with hyperparameter search |
| `07_comparison_with_IESO.ipynb` | Benchmark all models against the IESO official predispatch forecast |

---

## Setup

**Requirements:** Python 3.10+, ~20 GB disk space, 8 GB+ RAM

```bash
# Clone the repository
git clone <repo-url>
cd HOEP-forcasting-project

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Data (DVC)

Large data files are tracked with [DVC](https://dvc.org) using a **Google Cloud Storage** remote (`gs://hoep_data6300/dvc-store`).

Install the DVC GCS plugin:
```bash
pip install "dvc-gs"
```

Authenticate using a GCP service account key:
```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-key.json
```

Then pull the data:
```bash
dvc pull
```

Check sync status:
```bash
dvc status --cloud
```

### PySpark

Notebooks 01 and 02 use PySpark in local mode. No Hadoop cluster is required — PySpark is installed via `requirements.txt`.

### GPU (Optional)

TensorFlow and PyTorch are configured to use NVIDIA CUDA 12.x if available. CPU training works without any additional setup.

---

## Train/Val/Test Split

| Split | Period |
|-------|--------|
| Train | Jan 2023 – Dec 2024 |
| Validation | Jan 2025 – Mar 2025 |
| Test | Final 4 weeks of April 2025 |

Splits are strictly temporal to prevent data leakage.

---

## Results

The GRU model outperforms all other models and IESO's own official predispatch forecasts across all three horizons. Predictions and comparison figures are stored in `results/`.

---

## Tech Stack

| Area | Libraries |
|------|-----------|
| Data processing | pandas, numpy, PySpark |
| Statistical modeling | statsmodels |
| Machine learning | scikit-learn, XGBoost |
| Deep learning | PyTorch, TensorFlow/Keras |
| Visualization | matplotlib, seaborn |
| Data versioning | DVC |
