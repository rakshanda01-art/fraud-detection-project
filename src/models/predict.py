import argparse, json, joblib
import pandas as pd
from ..utils import load_config

def main(input_path: str, output: str = "predictions.csv"):
    cfg = load_config("configs/config.yaml")
    pipe = joblib.load("models/artifacts/model.joblib")
    df = pd.read_csv(input_path)
    feature_cols = [c for c in df.columns if c != cfg["target"]]
    proba = pipe.predict_proba(df[feature_cols])[:,1]
    df_out = df.copy()
    df_out["fraud_probability"] = proba
    df_out["predict_is_fraud"] = (proba >= cfg["threshold"]).astype(int)
    df_out.to_csv(output, index=False)
    print(f"Saved predictions to {output}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV with features (e.g. data/processed/test.csv)")
    ap.add_argument("--output", default="predictions.csv")
    args = ap.parse_args()
    main(args.input, args.output)