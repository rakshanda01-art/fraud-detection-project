from pathlib import Path
import subprocess, sys

def test_can_generate_data(tmp_path):
    # Run the data generation script and check output exists
    cmd = [sys.executable, "-m", "src.data.make_dataset", "--n-samples", "1000", "--fraud-rate", "0.02"]
    subprocess.check_call(cmd, cwd=Path.cwd())
    assert Path("data/raw/synthetic_fraud.csv").exists()