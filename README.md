# Thesis PoC — JIT Defect Prediction (Apache Camel)

## Quick start

**Requires Python 3.11 or 3.12** (PyTorch does not support 3.14 yet).

On macOS, install OpenMP for XGBoost first:

```bash
brew install python@3.12 libomp
```

```bash
cd thesis-jit-poc
# Use 3.12 if your default python3 is 3.14:
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 1. Download dataset (82 MB) — already done if you ran download_data.sh
#    File: data/apachejit/dataset/apachejit_total.csv

# 2. Run PoC
python run_poc.py
```

Results appear in `results/poc_results.csv`.

For the **full** PoC (all Camel commits), edit `src/config.py`:
`EMBEDDING_SAMPLE_SIZE = None`
