# HOEP Forecasting

A production-grade MLOps system for forecasting the **Hourly Ontario Electricity Price (HOEP)**, comparing statistical, machine learning, and deep learning approaches across 1-hour, 2-hour, and 3-hour prediction horizons — served via a REST API with full CI/CD and cloud deployment.

---

## Overview

This project forecasts Ontario electricity market clearing prices using data from the Independent Electricity System Operator (IESO). The final models are benchmarked against IESO's own official predispatch forecasts. The GRU deep learning model achieves the best performance and is served in production via a FastAPI REST API deployed on Google Cloud Run.

**Data range:** January 2023 – April 2025  
**Granularity:** Hourly observations  
**Horizons:** h=1, h=2, h=3 hours ahead

---

## Architecture

```
Experimentation (notebooks)
        ↓
src/ modules (clean Python code)
        ↓
MLflow experiment tracking → compare runs → set @champion alias
        ↓
scripts/export_champion.py → artifacts/model/
        ↓
DVC + GCS → version and share data + model weights
        ↓
FastAPI REST API → Docker container
        ↓
GitHub Actions CI/CD → Google Artifact Registry → Google Cloud Run
```

---

## Project Structure

```
HOEP-forecasting-project/
├── src/
│   ├── config.py           # centralized config (DataConfig, TrainConfig)
│   ├── model.py            # GRURegressor architecture
│   ├── preprocess.py       # split, scale, sequence creation
│   ├── predict.py          # inference pipeline
│   └── train.py            # training loop with MLflow tracking
│
├── api/
│   ├── schemas.py          # Pydantic request/response models
│   └── main.py             # FastAPI endpoints
│
├── scripts/
│   └── export_champion.py  # export @champion model from MLflow to artifacts/
│
├── tests/
│   ├── test_predict.py     # inference pipeline tests
│   └── test_api.py         # API endpoint tests
│
├── notebooks/              # experimentation (run in order 01–07)
│
├── artifacts/
│   ├── model/              # champion model weights (DVC tracked)
│   │   ├── gru_h1.pt
│   │   ├── gru_h2.pt
│   │   └── gru_h3.pt
│   └── scalers/            # fitted scalers (DVC tracked)
│       ├── scaler_X.pkl
│       └── scaler_y.pkl
│
├── data/                   # raw and processed data (DVC tracked)
│   ├── raw/                # raw IESO CSVs
│   ├── interim/            # cleaned Parquet files
│   └── processed/          # model-ready datasets
│
├── .github/workflows/
│   └── ci_cd.yml           # CI/CD: test + deploy on every push to main
│
├── Dockerfile              # two-stage build using uv
├── docker-compose.yml      # local orchestration
├── pyproject.toml          # project packaging and tool configuration
├── data.dvc                # DVC pointer to data folder
└── artifacts.dvc           # DVC pointer to model weights and scalers
```

---

## Models

| Category | Models |
|----------|--------|
| Statistical | ARIMA, SARIMA |
| Machine Learning | Linear Regression, Random Forest, XGBoost |
| Deep Learning | LSTM, GRU (PyTorch) |

Each model is trained separately for each forecast horizon (h=1, h=2, h=3).

**Evaluation metrics:** MAE, RMSE, MAPE

The GRU model outperforms all other models and IESO's own official predispatch forecasts across all three horizons.

---

## Notebooks

Run in order for full experimentation:

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

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/MiladEbrahimiAbyzandi/Ontario-electricity-price-forecasting.git
cd HOEP-forecasting-project

python -m venv .venv
source .venv/bin/activate
```

Install PyTorch for your hardware first:

```bash
# CPU only
pip install torch --index-url https://download.pytorch.org/whl/cpu

# GPU (CUDA 12.1)
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

For other CUDA versions visit: https://pytorch.org/get-started/locally/

Then install the project:

```bash
pip install -e ".[dev]"
```

### 2. Get data and model weights

All data and model artifacts are versioned with DVC and stored in Google Cloud Storage (`gs://hoep_data6300/dvc-store`). A single `dvc pull` fetches everything — raw data, processed datasets, model weights, and scalers.

```bash
dvc pull
```

This restores:
- `data/` — raw and processed IESO datasets
- `artifacts/model/` — champion GRU weights for all three horizons
- `artifacts/scalers/` — fitted StandardScalers

> The GCS bucket is public read-only — no credentials needed to pull.

### 3. Run the API

**Option A — locally:**

```bash
uvicorn api.main:app --reload
```

**Option B — Docker Compose:**

```bash
docker compose up --build
```

**Option C — Docker directly:**

```bash
docker build -t hoep-forecasting .
docker run -p 8000:8000 hoep-forecasting
```

### 4. Test it

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{"status": "ok"}
```

Interactive API docs available at: `http://localhost:8000/docs`

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/predict` | Forecast HOEP for a given horizon |

### `/predict` request body

```json
{
  "recent_data": [[...], [...], ...],
  "horizon": 1
}
```

- `recent_data`: 168 rows × 9 columns of recent hourly data (not scaled), ordered oldest to newest
- `horizon`: 1, 2, or 3 hours ahead
- Feature column order: `hoep, market_demand, ontario_demand, nuclear, gas, hydro, wind, solar, biofuel`

### `/predict` response

```json
{
  "horizon": 1,
  "predicted_hoep": 34.71
}
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

## Train/Val/Test Split

| Split | Period |
|-------|--------|
| Train | Jan 2023 – Dec 2024 |
| Validation | Jan 2025 – Mar 2025 |
| Test | Apr 2025 |

Splits are strictly temporal to prevent data leakage.

---

## Retraining

### Hyperparameters

All hyperparameters can be overridden from the command line:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--learning-rate` | 0.01 | Adam optimizer learning rate |
| `--epochs` | 100 | Maximum training epochs |
| `--batch-size` | 32 | Training batch size |
| `--patience` | 10 | Early stopping patience (epochs) |
| `--hidden-size` | 32 | GRU hidden dimension |
| `--num-layers` | 1 | Number of GRU layers |
| `--dropout` | 0.2 | Dropout rate |
| `--seq-length` | 168 | Input sequence length in hours (1 week) |

### Workflow

```bash
# 1. train one or more runs with different hyperparameters
python -m src.train --data-path data/processed/master_dataset.parquet
python -m src.train --data-path data/processed/master_dataset.parquet \
    --learning-rate 1e-3 \
    --epochs 50 \
    --patience 15

# 2. compare runs in MLflow UI
mlflow ui   # open http://localhost:5000

# 3. in MLflow UI: register the best version for each horizon
#    Models → hoep-gru-h1 → click best version → Add alias → champion
#    repeat for hoep-gru-h2 and hoep-gru-h3

# 4. export champion models to artifacts/model/
python scripts/export_champion.py

# 5. version and share updated weights via DVC
dvc push
git add artifacts.dvc
git commit -m "chore: update champion models"
git push
```

> **Note:** `dvc push` requires write access to `gs://hoep_data6300/dvc-store`.
> If you want to maintain your own version of the models, configure your own DVC remote first:
> ```bash
> dvc remote add -d myremote gs://your-bucket/dvc-store
> dvc push
> ```

---

## MLflow Experiment Tracking

Every training run automatically logs:
- all hyperparameters
- per-epoch train/val loss curves
- test MAE, RMSE, MAPE
- model artifact

```bash
mlflow ui   # open http://localhost:5000
```

---

## CI/CD Pipeline

Every push to `main` automatically:

1. Sets up Python and installs dependencies
2. Authenticates with GCP and pulls model artifacts via DVC
3. Runs all tests (`pytest tests/ -v`)
4. Builds Docker image and pushes to Google Artifact Registry
5. Deploys updated image to Google Cloud Run

Pushes to the `dev` branch run tests only — no deployment.

---

## Live API

The API is publicly accessible at:

```
https://hoep-forecasting-api-778218642867.us-central1.run.app
```

---

## Tech Stack

| Area | Tool |
|------|------|
| Deep learning | PyTorch (GRU, LSTM) |
| Data processing | pandas, numpy, PySpark |
| Statistical modeling | statsmodels |
| Machine learning | scikit-learn, XGBoost |
| API | FastAPI + Pydantic |
| Experiment tracking | MLflow |
| Data & model versioning | DVC + Google Cloud Storage |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Cloud deployment | Google Cloud Run + Artifact Registry |
| Visualization | matplotlib, seaborn |

---

## PySpark & HDFS (Notebooks only)

PySpark is used for data ingestion and EDA in notebooks 01 and 02. HDFS is used as an intermediate storage layer for educational purposes. PySpark runs in local mode — no cluster needed. HDFS requires a local Hadoop installation with namenode at `hdfs://localhost:9000`.

```bash
# start HDFS
start-dfs.sh

# create required directory
hdfs dfs -mkdir -p hdfs://localhost:9000/hoep_project/interim
```