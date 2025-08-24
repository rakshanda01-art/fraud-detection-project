from dataclasses import dataclass
from typing import Literal, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

Num = list[str]
Cat = list[str]

@dataclass
class Schema:
    target: str = "is_fraud"
    numeric: Num = None
    categorical: Cat = None

    def infer(self, df: pd.DataFrame) -> "Schema":
        cols = df.columns.tolist()
        num = [c for c in cols if c.startswith("f")] + ["amount", "hour"]
        cat = ["is_international", "channel", "merchant_risk", "is_weekend"]
        self.numeric = num
        self.categorical = cat
        return self

def build_preprocessor(num: Num, cat: Cat) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(with_mean=True, with_std=True), num),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ]
    )

def build_model(model_type: Literal["logistic_regression","random_forest"]="logistic_regression"):
    if model_type == "logistic_regression":
        return LogisticRegression(C=0.5, max_iter=200, class_weight="balanced")
    elif model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=300, max_depth=12, n_jobs=-1, class_weight="balanced"
        )
    else:
        raise ValueError("Unknown model type")