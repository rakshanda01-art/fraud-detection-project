import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from ..utils import ensure_dir

def generate_synthetic(n_samples=20000, fraud_rate=0.015, random_state=42):
    # Highly imbalanced binary classification
    weights = [1 - fraud_rate, fraud_rate]
    X, y = make_classification(
        n_samples=n_samples,
        n_features=12,
        n_informative=8,
        n_redundant=2,
        n_repeated=0,
        n_classes=2,
        weights=weights,
        flip_y=0.01,
        class_sep=1.5,
        random_state=random_state,
    )
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df["amount"] = np.abs(np.random.lognormal(mean=3.2, sigma=1.0, size=n_samples)).round(2)
    df["is_international"] = np.random.binomial(1, 0.15, size=n_samples)
    df["channel"] = np.random.choice(["POS","ecom","atm","p2p"], size=n_samples, p=[0.45,0.35,0.15,0.05])
    df["merchant_risk"] = np.random.choice(["low","med","high"], size=n_samples, p=[0.7,0.25,0.05])
    df["hour"] = np.random.randint(0,24,size=n_samples)
    df["is_weekend"] = np.random.binomial(1, 0.28, size=n_samples)
    df["is_fraud"] = y
    return df

def main(n_samples: int, fraud_rate: float):
    raw_dir = ensure_dir("data/raw")
    df = generate_synthetic(n_samples=n_samples, fraud_rate=fraud_rate)
    df.to_csv(raw_dir / "synthetic_fraud.csv", index=False)
    print(f"Saved {len(df):,} rows to {raw_dir / 'synthetic_fraud.csv'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=20000)
    parser.add_argument("--fraud-rate", type=float, default=0.015)
    args = parser.parse_args()
    main(args.n_samples, args.fraud_rate)