# Proof of concept: hybrid JIT defect prediction on Apache Camel

Feasibility study for the master's thesis *Enhancing Just-in-Time Defect Prediction by Fusing Code Embeddings from Pre-trained and LLM-based Models with Change Metrics*.
It runs the pipeline of the earlier study ([jit-llm-embeddings](https://github.com/dzelilatin/jit-llm-embeddings)) on a second benchmark, ApacheJIT, using the Apache Camel project.

## What it does

1. Loads the 22,700 Camel commits of ApacheJIT (13.6% defect-inducing) with the 12 released change metrics.
2. Splits them chronologically 60/20/20 (train / validation / test). The validation part is kept for tuning in the full study and is not used here.
3. Mines the diff of every commit from a local clone with PyDriller (Java files only) and renders it as `[ADD] added lines [DEL] removed lines`.
4. Embeds the text with frozen CodeBERT: chunks of at most 510 tokens (at most 64 per commit), the final-layer `[CLS]` vector per chunk, mean-pooled per commit.
5. Trains Logistic Regression, Random Forest and XGBoost on three configurations (metrics only, embeddings only, hybrid), each standardized on the training split, with three seeds and a fixed threshold of 0.5.
6. Reports F1, AUC-ROC, G-Mean, precision, recall, MCC and the confusion matrix on the full test split, with 95% bootstrap intervals.

## Setup

Python 3.11 or 3.12 is required (PyTorch 2.2 does not support newer versions). On macOS, XGBoost needs OpenMP:

```bash
brew install python@3.12 libomp
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download the ApacheJIT dataset (about 80 MB) from Zenodo:

```bash
bash scripts/download_data.sh
```

This places `data/apachejit/dataset/apachejit_total.csv`. The first run also clones `apache/camel` into `data/repos/camel` (about 1.5 GB, 10 to 30 minutes).

## Run

```bash
python run_poc.py --smoke      # small check, writes to results/smoke/, not for reporting
python run_poc.py              # all train and test commits
python src/paired_bootstrap.py # paired differences between configurations
```

Options for `run_poc.py`: `--train-sample N`, `--test-sample N` (stratified subsample), `--device cuda|mps|cpu`.

Mined diffs and embeddings are cached in `data/cache/`, so an interrupted run resumes. On an Apple M4 Pro (MPS), mining took about 6 minutes per split, embedding about 36 minutes for 18,160 commits (7.6 to 8.7 commits per second), and the full run about 50 minutes.

## Results

| File | Content |
|---|---|
| `results/poc_results.csv` | mean over seeds per classifier and configuration, with 95% bootstrap intervals |
| `results/poc_results_by_seed.csv` | one row per seed, with TP, FP, FN, TN |
| `results/poc_predictions.csv` | probability and label of every test prediction |
| `results/poc_paired_bootstrap.csv` | differences between configurations with 95% intervals |
| `results/poc_run_info.json` | split statistics, extraction counts and timing |

## Limitations

- One project, one chronological split, three classifiers at default settings, no tuning.
- The share of defect-inducing commits falls from 18.1% in the training period to 3.5% in the test period, while the 0.5 threshold and the class weights come from the training period. Precision is therefore low.
- Bootstrap intervals describe sampling variability of the test commits only; the paired comparisons are not corrected for multiple testing.
- 62 training and 61 test commits exceed the limit of 64 chunks and are truncated.
