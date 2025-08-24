from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd
from ..utils import load_config

app = FastAPI(title="Fraud Detection API", version="1.0.0")
pipe = None
cfg = None

class Txn(BaseModel):
    # Minimal feature set for live scoring
    f0: float = 0.0
    f1: float = 0.0
    f2: float = 0.0
    f3: float = 0.0
    f4: float = 0.0
    f5: float = 0.0
    f6: float = 0.0
    f7: float = 0.0
    f8: float = 0.0
    f9: float = 0.0
    f10: float = 0.0
    f11: float = 0.0
    amount: float = Field(100.0, ge=0)
    hour: int = Field(12, ge=0, le=23)
    is_international: int = 0
    channel: str = "ecom"            # POS|ecom|atm|p2p
    merchant_risk: str = "low"       # low|med|high
    is_weekend: int = 0

def load_model():
    global pipe, cfg
    if pipe is None:
        cfg = load_config("configs/config.yaml")
        pipe = joblib.load("models/artifacts/model.joblib")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(txn: Txn):
    load_model()
    df = pd.DataFrame([txn.model_dump()])
    proba = pipe.predict_proba(df)[0,1]
    is_fraud = int(proba >= cfg["threshold"])
    return {"fraud_probability": float(proba), "predict_is_fraud": is_fraud, "threshold": cfg["threshold"]}