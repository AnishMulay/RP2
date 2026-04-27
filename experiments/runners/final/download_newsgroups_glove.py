#!/usr/bin/env python3

import gzip
import json
import pathlib
import pickle
import re
import sys
import time

import numpy as np
from sklearn.datasets import fetch_20newsgroups


BASE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent.parent
GLOVE_NPY = BASE_DIR / "data" / "glove" / "glove.6B.300d.npy"
GLOVE_TXT = BASE_DIR / "data" / "glove" / "glove.6B.300d.txt"
DATA_DIR = BASE_DIR / "data" / "newsgroups_glove"
DATA_DIR.mkdir(parents=True, exist_ok=True)
SKLEARN_DIR = BASE_DIR / "data"

EMBEDDINGS_PATH = DATA_DIR / "newsgroups_embeddings.pkl.gz"
LABELS_PATH = DATA_DIR / "newsgroups_labels.npy"
META_PATH = DATA_DIR / "newsgroups_meta.json"

STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "with",
    "by",
    "from",
    "as",
    "is",
    "was",
    "are",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "shall",
    "can",
    "not",
    "no",
    "nor",
    "so",
    "yet",
    "both",
    "either",
    "neither",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "than",
    "then",
    "that",
    "this",
    "these",
    "those",
    "it",
    "its",
    "they",
    "them",
    "their",
    "he",
    "she",
    "his",
    "her",
    "we",
    "our",
    "you",
    "your",
    "i",
    "me",
    "my",
    "us",
    "what",
    "which",
    "who",
    "whom",
    "how",
    "when",
    "where",
    "why",
    "all",
    "any",
    "there",
    "here",
    "just",
    "also",
    "into",
    "over",
    "after",
    "up",
    "about",
    "out",
    "if",
    "re",
    "s",
    "t",
    "ll",
    "ve",
    "d",
    "m",
}


def _file_size_mb(path):
    return path.stat().st_size / (1024 * 1024)


def main():
    print("=" * 60, flush=True)
    print("20 Newsgroups + GloVe Embedding Preparation", flush=True)
    print("=" * 60, flush=True)

    if EMBEDDINGS_PATH.exists():
        print("Embeddings file already exists. Delete to regenerate.", flush=True)
        return

    for path in [GLOVE_NPY, GLOVE_TXT]:
        if not path.exists():
            print(f"ERROR: {path} not found.", flush=True)
            return

    print("\n[Step 1/4] Loading GloVe vocabulary...", flush=True)
    word_to_idx = {}
    with open(GLOVE_TXT, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            word = line.split(" ", 1)[0]
            word_to_idx[word] = idx
    print(f"  GloVe vocabulary size: {len(word_to_idx):,}", flush=True)

    vectors = np.load(GLOVE_NPY)
    print(f"  GloVe matrix shape: {vectors.shape}", flush=True)

    print("\n[Step 2/4] Downloading 20 Newsgroups...", flush=True)
    train = fetch_20newsgroups(
        data_home=str(SKLEARN_DIR),
        subset="train",
        remove=("headers", "footers", "quotes"),
        download_if_missing=True,
    )
    test = fetch_20newsgroups(
        data_home=str(SKLEARN_DIR),
        subset="test",
        remove=("headers", "footers", "quotes"),
        download_if_missing=True,
    )

    all_texts = train.data + test.data
    all_labels = list(train.target) + list(test.target)

    print(f"  Train: {len(train.data):,} documents", flush=True)
    print(f"  Test:  {len(test.data):,} documents", flush=True)
    print(f"  Total: {len(all_texts):,} documents", flush=True)
    print(f"  Classes: {len(train.target_names)}", flush=True)

    print("\n[Step 3/4] Building per-document GloVe embedding sets...", flush=True)
    total = len(all_texts)
    all_embeddings = []
    valid_word_counts = []
    zero_word_docs = 0

    t0 = time.perf_counter()
    for i, text in enumerate(all_texts):
        tokens = re.findall(r"[a-z]+", text.lower())
        unique_words = set(tokens) - STOPWORDS
        valid_words = [word for word in unique_words if word in word_to_idx]

        valid_word_counts.append(len(valid_words))
        if len(valid_words) == 0:
            zero_word_docs += 1
            doc_embeddings = np.zeros((1, 300), dtype=np.float32)
        else:
            row_indices = [word_to_idx[word] for word in valid_words]
            doc_embeddings = vectors[row_indices].astype(np.float32, copy=False)

        all_embeddings.append(doc_embeddings)

        if (i + 1) % 2000 == 0 or (i + 1) == total:
            print(f"  Processed {i + 1}/{total} documents...", flush=True)

    elapsed_s = time.perf_counter() - t0
    valid_word_counts = np.asarray(valid_word_counts, dtype=np.int64)
    avg_words = float(valid_word_counts.mean()) if total > 0 else 0.0
    min_words = int(valid_word_counts.min()) if total > 0 else 0
    max_words = int(valid_word_counts.max()) if total > 0 else 0

    print(
        f"  Documents with zero valid words: {zero_word_docs}",
        flush=True,
    )
    print(f"  Average words per document: {avg_words:.2f}", flush=True)
    print(
        f"  Min words: {min_words}  Max words: {max_words}",
        flush=True,
    )
    print(f"  Build time: {elapsed_s:.1f} s", flush=True)

    print("\n[Step 4/4] Saving files...", flush=True)
    all_labels_array = np.asarray(all_labels, dtype=np.int32)
    meta = {
        "num_documents": len(all_embeddings),
        "num_classes": len(train.target_names),
        "avg_words_per_doc": avg_words,
        "min_words": min_words,
        "max_words": max_words,
        "zero_word_docs": zero_word_docs,
        "glove_vocab_size": len(word_to_idx),
    }

    with gzip.open(EMBEDDINGS_PATH, "wb") as f:
        pickle.dump(all_embeddings, f, protocol=4)
    np.save(LABELS_PATH, all_labels_array)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(
        f"  Saved {EMBEDDINGS_PATH.name} "
        f"({_file_size_mb(EMBEDDINGS_PATH):.2f} MB)",
        flush=True,
    )
    print(
        f"  Saved {LABELS_PATH.name} ({_file_size_mb(LABELS_PATH):.2f} MB)",
        flush=True,
    )
    print(
        f"  Saved {META_PATH.name} ({_file_size_mb(META_PATH):.2f} MB)",
        flush=True,
    )
    print(f"Done! Files saved to {DATA_DIR}", flush=True)

    with gzip.open(EMBEDDINGS_PATH, "rb") as f:
        sanity_embeddings = pickle.load(f)
    arr = sanity_embeddings[0]
    print(
        f"  Sanity check doc 0: shape={arr.shape}, label={all_labels_array[0]}",
        flush=True,
    )
    print("Sanity check passed.", flush=True)


if __name__ == "__main__":
    main()
