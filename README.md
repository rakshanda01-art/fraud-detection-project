# Fraud Detection (End-to-End: Training → API → Docker → CI)

A complete, **publish-ready** fraud detection project you can push to GitHub.  
It covers: synthetic data generation, feature engineering, model training & evaluation, FastAPI for serving predictions, Dockerization, tests, and CI (pytest).

## Project structure

```
fraud-detection-project/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── config.yaml
├── data/
│   ├── raw/                  # input data (generated)
│   ├── interim/              # intermediate features
│   └── processed/            # train/val/test splits
├── models/
│   └── artifacts/            # saved pipelines/models
├── notebooks/
│   └── 01_quick_eda.ipynb
├── src/
│   ├── app/
│   │   └── main.py           # FastAPI app
│   ├── data/
│   │   └── make_dataset.py   # synthetic data
│   ├── features/
│   │   └── build_features.py
│   ├── models/
│   │   ├── model.py          # pipeline definitions
│   │   ├── train.py          # training entrypoint
│   │   └── predict.py        # batch predict
│   └── utils.py
├── tests/
│   ├── test_data.py
│   └── test_model.py
├── Dockerfile
└── .github/workflows/ci.yml
```

## Quickstart (local)

```bash
# 1) Create & activate virtual env (optional but recommended)
python -m venv .venv && . .venv/Scripts/activate  # Windows PowerShell: .\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate

# 2) Install deps
pip install -r requirements.txt

# 3) Generate synthetic data
python -m src.data.make_dataset --n-samples 20000 --fraud-rate 0.015

# 4) Build features
python -m src.features.build_features

# 5) Train model
python -m src.models.train

# 6) Run API
uvicorn src.app.main:app --reload --port 8000
```

Open: `http://127.0.0.1:8000/docs` for interactive Swagger UI.

## Docker (serve API)

```bash
docker build -t fraud-api .
docker run -p 8000:8000 fraud-api
# Swagger: http://localhost:8000/docs
```

## Batch predict

```bash
python -m src.models.predict --input data/processed/test.csv --output predictions.csv
```

## Configuration

Edit hyperparameters in `configs/config.yaml`.

## Tests

```bash
pytest -q
```

## Notes
- Data is **synthetic** (no external downloads) with heavy class imbalance to reflect real fraud (default 1.5% fraud rate).
- The model is a scikit-learn Pipeline (scaler + LogisticRegression) with class weights and threshold tuning to prioritize **recall**.