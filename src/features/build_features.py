from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from ..utils import ensure_dir, load_config

def main():
    cfg = load_config("configs/config.yaml")
    raw_path = Path("data/raw/synthetic_fraud.csv")
    assert raw_path.exists(), "Run: python -m src.data.make_dataset first."

    df = pd.read_csv(raw_path)

    # Simple cleaning: clip extreme amounts, derive few ratios
    df["amount"] = df["amount"].clip(0, df["amount"].quantile(0.999))

    # Split
    train_df, test_df = train_test_split(df, test_size=cfg["test_size"], random_state=cfg["random_state"], stratify=df[cfg["target"]])
    train_df, val_df = train_test_split(train_df, test_size=cfg["val_size"], random_state=cfg["random_state"], stratify=train_df[cfg["target"]])

    out_dir = ensure_dir("data/processed")
    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)
    print("Saved processed splits to data/processed/")

if __name__ == "__main__":
    main()