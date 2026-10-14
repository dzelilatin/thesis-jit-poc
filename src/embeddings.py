"""Extract CodeBERT [CLS] embeddings from commit diffs."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pydriller import Repository
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from config import CODEBERT_MODEL, COMMIT_COL, DATA_DIR, DATE_COL, LABEL_COL, MAX_TOKENS, REPO_DIR
from data_loader import sample_commits


def ensure_camel_repo() -> Path:
    if REPO_DIR.exists() and (REPO_DIR / ".git").exists():
        return REPO_DIR
    REPO_DIR.parent.mkdir(parents=True, exist_ok=True)
    import subprocess

    print("Cloning apache/camel (needs full history — may take 10–30 min)...")
    subprocess.run(
        ["git", "clone", "https://github.com/apache/camel.git", str(REPO_DIR)],
        check=True,
    )
    return REPO_DIR


def _diff_text(commit) -> str:
    parts = []
    for m in commit.modified_files:
        if m.diff:
            parts.append(m.diff)
        else:
            parts.append(f"+{m.added_lines}\n-{m.deleted_lines}")
    return "\n".join(parts)[:8000]


def fetch_diffs(commit_ids: list[str]) -> dict[str, str]:
    repo_path = ensure_camel_repo()
    wanted = set(commit_ids)
    found: dict[str, str] = {}

    for commit in Repository(str(repo_path), only_commits=commit_ids).traverse_commits():
        found[commit.hash] = _diff_text(commit)
    return found


def _cache_path(commit_ids: list[str]) -> Path:
    key = hashlib.md5(",".join(sorted(commit_ids)).encode()).hexdigest()[:12]
    return DATA_DIR / f"embeddings_{key}.npy"


def extract_embeddings(commit_ids: list[str], batch_size: int = 8) -> np.ndarray:
    cache = _cache_path(commit_ids)
    if cache.exists():
        return np.load(cache)

    diffs_map = fetch_diffs(commit_ids)
    texts = [diffs_map.get(cid, "") or " " for cid in commit_ids]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(CODEBERT_MODEL)
    model = AutoModel.from_pretrained(CODEBERT_MODEL, use_safetensors=True).to(device)
    model.eval()

    all_emb = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="CodeBERT"):
            batch = texts[i : i + batch_size]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_TOKENS,
                return_tensors="pt",
            ).to(device)
            out = model(**enc)
            cls = out.last_hidden_state[:, 0, :].cpu().numpy()
            all_emb.append(cls)

    embeddings = np.vstack(all_emb)
    np.save(cache, embeddings)
    return embeddings


def attach_embeddings(df: pd.DataFrame, sample_size: int | None = None) -> pd.DataFrame:
    out = df.copy()
    if sample_size and len(out) > sample_size:
        out = sample_commits(out, sample_size)

    ids = out[COMMIT_COL].tolist()
    out["embedding"] = list(extract_embeddings(ids))
    return out
