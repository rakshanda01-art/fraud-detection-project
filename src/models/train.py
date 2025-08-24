from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, precision_recall_curve
from sklearn.pipeline import Pipeline
from ..utils import load_config, ensure_dir
from .model import Schema, build_preprocessor, build_model

def evaluate(y_true, y_proba, threshold):
    y_pred = (y_proba >= threshold).astype(int)
    roc = roc_auc_score(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)
    report = classification_report(y_true, y_pred, output_dict=True)
    return {"threshold": threshold, "roc_auc": roc, "pr_auc": pr_auc, "report": report}

def main():
    cfg = load_config("configs/config.yaml")
    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/val.csv")

    schema = Schema().infer(train_df)
    X_train = train_df[schema.numeric + schema.categorical]
    y_train = train_df[cfg["target"]]
    X_val = val_df[schema.numeric + schema.categorical]
    y_val = val_df[cfg["target"]]

    pre = build_preprocessor(schema.numeric, schema.categorical)
    clf = build_model(cfg["model"]["type"])
    pipe = Pipeline([("pre", pre), ("clf", clf)])

    pipe.fit(X_train, y_train)
    val_proba = pipe.predict_proba(X_val)[:,1]
    metrics = evaluate(y_val, val_proba, cfg["threshold"])

    # Save artifacts
    art_dir = ensure_dir("models/artifacts")
    joblib.dump(pipe, art_dir / "model.joblib")
    with open(art_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Saved model to models/artifacts/model.joblib")
    print("Validation ROC-AUC:", round(metrics["roc_auc"], 4))
    print("Validation PR-AUC:", round(metrics["pr_auc"], 4))
    print("F1 (fraud class):", round(metrics["report"]["1"]["f1-score"], 4))

if __name__ == "__main__":
    main()