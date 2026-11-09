#!/usr/bin/env bash
# Set up and run the ICB federated/DP pipeline on a fresh Ubuntu EC2 (CPU only).
# Run from the folder holding pipeline.py and the *_clean.csv files.
set -euo pipefail
sudo apt-get update -y
sudo apt-get install -y python3-venv python3-pip
python3 -m venv ~/icbenv
source ~/icbenv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python - <<'PY'
import glob, importlib
for m in ["torch","opacus","flwr","ray","sklearn","pandas","numpy","scipy"]:
    importlib.import_module(m); print("ok:", m)
csvs = sorted(glob.glob("*_clean.csv"))
print("cohort CSVs found:", csvs)
assert len(csvs) == 5, "Expected 5 *_clean.csv files."
PY
echo "Starting pipeline at $(date)"
nohup python -u pipeline.py > run.log 2>&1 &
echo "PID $! started. Follow with:  tail -f run.log"
