"""Mine commit diffs with PyDriller and compute CodeBERT embeddings.

The embedding step follows the paper pipeline (jit-llm-embeddings, src/embed.py):

  text   = "[ADD] " + added lines + " [DEL] " + removed lines   (code only, no message)
  tokens = tokenizer(text) without truncation, split into consecutive 510-token bodies
  vector = final-layer [CLS] of "[CLS] body [SEP]" per chunk, mean-pooled over chunks

Differences from the paper, forced by the data source:
  * ApacheJIT ships only commit ids, so the diff is mined from the repository.
    Added/removed lines keep their order in the diff (the paper had to sort them
    because JIT-Defects4J stores them as unordered sets).
  * Only .java files contribute to the text.
  * Commits longer than MAX_CHUNKS * MAX_BODY tokens are cut (proposal, Section 5.1).
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from config import (
    BATCH_SIZE, CACHE_DIR, CODEBERT_MODEL, COMMIT_COL, JAVA_SUFFIX, MAX_BODY, MAX_CHUNKS,
    REPO_DIR,
)
from data_loader import sample_commits

logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

EMPTY_TEXT = "[ADD]  [DEL] "


def pick_device(requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Mining
# ---------------------------------------------------------------------------

def ensure_camel_repo() -> Path:
    if REPO_DIR.exists() and (REPO_DIR / ".git").exists():
        return REPO_DIR
    REPO_DIR.parent.mkdir(parents=True, exist_ok=True)
    print("Cloning apache/camel (needs full history, may take 10-30 min)...")
    subprocess.run(
        ["git", "clone", "https://github.com/apache/camel.git", str(REPO_DIR)],
        check=True,
    )
    return REPO_DIR


def render_diff(commit) -> tuple[str, int]:
    """Return (text, number of Java files) for one PyDriller commit."""
    added: list[str] = []
    removed: list[str] = []
    n_java = 0
    for m in commit.modified_files:
        path = m.new_path or m.old_path or ""
        if not path.endswith(JAVA_SUFFIX):
            continue
        n_java += 1
        parsed = m.diff_parsed
        added.extend(line for _, line in parsed["added"])
        removed.extend(line for _, line in parsed["deleted"])
    return "[ADD] " + "\n".join(added) + " [DEL] " + "\n".join(removed), n_java


def mine_diffs(commit_ids: list[str], store: Path) -> pd.DataFrame:
    """Diff text per commit id, cached in a jsonl file so an interrupted run resumes."""
    from pydriller import Git

    store.parent.mkdir(parents=True, exist_ok=True)
    done: dict[str, dict] = {}
    if store.exists():
        with store.open() as fh:
            for line in fh:
                rec = json.loads(line)
                done[rec[COMMIT_COL]] = rec

    todo = [c for c in commit_ids if c not in done]
    if todo:
        git = Git(str(ensure_camel_repo()))
        with store.open("a") as fh:
            for i, cid in enumerate(tqdm(todo, desc=f"mining {store.stem}")):
                try:
                    text, n_java = render_diff(git.get_commit(cid))
                    rec = {COMMIT_COL: cid, "text": text, "n_java": n_java, "missing": False}
                except Exception as exc:  # commit absent from the clone, unreadable diff, ...
                    rec = {COMMIT_COL: cid, "text": EMPTY_TEXT, "n_java": 0, "missing": True,
                           "error": type(exc).__name__}
                fh.write(json.dumps(rec) + "\n")
                done[cid] = rec
                if i % 500 == 499:
                    fh.flush()
                    if hasattr(git, "clear"):
                        git.clear()

    return pd.DataFrame([done[c] for c in commit_ids])


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def _batch_forward(seqs, pad_id, model, device, batch_size) -> np.ndarray:
    """[CLS] vectors for complete token sequences (already wrapped in [CLS]/[SEP])."""
    hidden = model.config.hidden_size
    out_arr = np.empty((len(seqs), hidden), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(seqs), batch_size):
            batch = seqs[start : start + batch_size]
            max_len = max(len(s) for s in batch)
            input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
            attn_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
            for j, seq in enumerate(batch):
                input_ids[j, : len(seq)] = torch.tensor(seq, dtype=torch.long)
                attn_mask[j, : len(seq)] = 1
            out = model(input_ids=input_ids.to(device), attention_mask=attn_mask.to(device))
            out_arr[start : start + len(batch)] = out.last_hidden_state[:, 0, :].cpu().numpy()
    return out_arr


def embed_texts(texts: list[str], device: torch.device, batch_size: int = BATCH_SIZE):
    """Chunked CodeBERT embeddings, mean-pooled per commit. Returns (matrix, stats)."""
    tokenizer = AutoTokenizer.from_pretrained(CODEBERT_MODEL)
    model = AutoModel.from_pretrained(CODEBERT_MODEL).to(device).eval()
    cls_id, sep_id, pad_id = tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id
    n, hidden = len(texts), model.config.hidden_size
    cap = MAX_CHUNKS * MAX_BODY

    bodies = []
    n_truncated = 0
    for text in tqdm(texts, desc="tokenizing"):
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) > cap:
            ids = ids[:cap]
            n_truncated += 1
        bodies.append(ids)

    t0 = time.time()
    short = [i for i, b in enumerate(bodies) if len(b) <= MAX_BODY]
    long_ = [i for i, b in enumerate(bodies) if len(b) > MAX_BODY]
    emb = np.empty((n, hidden), dtype=np.float32)

    if short:  # sort by length to minimise padding, restore original order afterwards
        order = sorted(short, key=lambda i: len(bodies[i]))
        seqs = [[cls_id] + bodies[i] + [sep_id] for i in order]
        vecs = _batch_forward(seqs, pad_id, model, device, batch_size)
        for pos, i in enumerate(order):
            emb[i] = vecs[pos]

    if long_:  # flatten all chunks of all long commits, then mean-pool back per commit
        seqs, owner = [], []
        for i in long_:
            for s in range(0, len(bodies[i]), MAX_BODY):
                seqs.append([cls_id] + bodies[i][s : s + MAX_BODY] + [sep_id])
                owner.append(i)
        vecs = _batch_forward(seqs, pad_id, model, device, batch_size)
        sums = np.zeros((n, hidden), dtype=np.float64)
        counts = np.zeros(n, dtype=np.float64)
        np.add.at(sums, np.array(owner), vecs.astype(np.float64))
        np.add.at(counts, np.array(owner), 1.0)
        for i in long_:
            emb[i] = (sums[i] / counts[i]).astype(np.float32)

    seconds = time.time() - t0
    stats = {
        "n_commits": n, "n_single_window": len(short), "n_chunked": len(long_),
        "n_chunks_total": len(short) + (len(seqs) if long_ else 0),
        "n_truncated": n_truncated, "embed_seconds": round(seconds, 1),
        "commits_per_sec": round(n / seconds, 2) if seconds > 0 else None,
        "device": str(device),
    }
    return emb, stats


def attach_embeddings(
    df: pd.DataFrame, split: str, device: torch.device, sample_size: int | None = None,
    tag: str = "",
):
    """Mine diffs, embed them (cached) and add an 'embedding' column. Returns (df, stats)."""
    out = df.copy()
    if sample_size and len(out) > sample_size:
        out = sample_commits(out, sample_size)
    ids = out[COMMIT_COL].tolist()

    mined = mine_diffs(ids, CACHE_DIR / f"diffs{tag}_{split}.jsonl")
    cache = CACHE_DIR / f"codebert{tag}_{split}_{len(ids)}.npz"
    meta = np.array([CODEBERT_MODEL, str(MAX_BODY), str(MAX_CHUNKS)])

    stats: dict = {}
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        if list(z["ids"]) == ids and list(z["meta"]) == list(meta):
            emb, stats = z["emb"], json.loads(str(z["stats"]))
            print(f"[{split}] embedding cache hit: {cache.name}")
        else:
            emb = None
    else:
        emb = None

    if emb is None:
        print(f"[{split}] embedding {len(ids):,} commits on {device} ...")
        emb, stats = embed_texts(mined["text"].tolist(), device)
        stats.update({
            "n_missing_commits": int(mined["missing"].sum()),
            "n_without_java": int((mined["n_java"] == 0).sum()),
        })
        np.savez(cache, ids=np.array(ids), emb=emb, meta=meta, stats=json.dumps(stats))

    out["embedding"] = list(emb)
    return out, stats
