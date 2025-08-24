from pathlib import Path
import subprocess, sys, json

def test_end_to_end_training(tmp_path):
    # Build features
    subprocess.check_call([sys.executable, "-m", "src.features.build_features"], cwd=Path.cwd())
    # Train
    subprocess.check_call([sys.executable, "-m", "src.models.train"], cwd=Path.cwd())
    assert Path("models/artifacts/model.joblib").exists()
    metrics = json.loads(Path("models/artifacts/metrics.json").read_text())
    assert metrics["roc_auc"] > 0.85