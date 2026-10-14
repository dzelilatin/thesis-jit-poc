from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
REPO_DIR = DATA_DIR / "repos" / "camel"

PROJECT = "apache/camel"
TRAIN_RATIO = 0.70
VAL_RATIO = 0.10
TEST_RATIO = 0.20

METRIC_COLS = ["ns", "nd", "nf", "ent", "la", "ld", "ndev", "age", "nuc", "aexp", "arexp", "asexp"]
LABEL_COL = "buggy"
DATE_COL = "author_date"
COMMIT_COL = "commit_id"

CODEBERT_MODEL = "microsoft/codebert-base"
MAX_TOKENS = 512
RANDOM_SEED = 42

CLASSIFIERS = ["logistic_regression", "random_forest", "xgboost"]
CONFIGS = ["metrics_only", "embeddings_only", "hybrid"]

# PoC: cap embeddings for faster first run (set None for full dataset)
EMBEDDING_SAMPLE_SIZE = 500
