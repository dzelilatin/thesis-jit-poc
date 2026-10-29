from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
CACHE_DIR = DATA_DIR / "cache"
REPO_DIR = DATA_DIR / "repos" / "camel"

PROJECT = "apache/camel"

# Chronological split within the project, same proportions as the paper
# (JIT-Defects4J published split) and Section 4.3 of the proposal.
# The validation part is reserved for hyperparameter tuning in the full study;
# the PoC does no tuning, so it is neither mined nor embedded here.
TRAIN_RATIO = 0.60
VAL_RATIO = 0.20
TEST_RATIO = 0.20

METRIC_COLS = ["ns", "nd", "nf", "ent", "la", "ld", "ndev", "age", "nuc", "aexp", "arexp", "asexp"]
LABEL_COL = "buggy"
DATE_COL = "author_date"
COMMIT_COL = "commit_id"

CODEBERT_MODEL = "microsoft/codebert-base"
MAX_BODY = 510          # 512 - [CLS] - [SEP]; same chunk size as the paper
MAX_CHUNKS = 64         # cap on chunks per commit (proposal Section 5.1); commits longer
                        # than MAX_CHUNKS * MAX_BODY tokens are cut and counted as truncated
BATCH_SIZE = 16
JAVA_SUFFIX = ".java"   # only Java source files contribute to the diff text

# Seeds as in the paper (src/train.py). LogisticRegression is deterministic.
SEEDS = [42, 123, 456]
SAMPLE_SEED = 42
THRESHOLD = 0.5

CLASSIFIERS = ["logistic_regression", "random_forest", "xgboost"]
CONFIGS = ["metrics_only", "embeddings_only", "hybrid"]

# Subsampling for quick runs; None = use every commit of the split.
# Override on the command line: python run_poc.py --train-sample 3000 --test-sample 1000
TRAIN_SAMPLE = None
TEST_SAMPLE = None

# Bootstrap confidence intervals on the test set
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 42
