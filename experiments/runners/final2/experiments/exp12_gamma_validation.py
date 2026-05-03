#!/usr/bin/env python3
"""
Experiment 12 — Gamma validation across datasets.

Measures gamma, the fraction of exact matched transport cost that flows through
non-local 2-level matched pairs.
"""

import gzip
import math
import pathlib
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torchvision

FINAL2_DIR = Path(__file__).resolve().parent.parent
BASE_DIR = FINAL2_DIR.parent.parent.parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(FINAL2_DIR) not in sys.path:
    sys.path.insert(0, str(FINAL2_DIR))

try:
    import ot
except ImportError:
    ot = None

from clustered_push_relabel.clustering.simple_l1 import SimpleL1Clustering
from clustered_push_relabel.clustering.simple_three_level_l1 import ThreeLevelL1Clustering
from shared import (
    compute_ratio, fmt_time, fmt_cost, fmt_ratio,
    build_two_level_proxy_matrix, build_three_level_proxy_matrix,
)

EXP_ID   = 12
EXP_NAME = "Gamma Validation — All Datasets"
DATASET  = "Multi"

DATA_DIR = FINAL2_DIR / "data"
EMNIST_DATA_DIR = BASE_DIR / "data"
EMNIST_SPLIT = "byclass"
CIFAR_DATA_DIR = BASE_DIR / "data" / "cifar_sift"
TRAIN_DESC_PATH = CIFAR_DATA_DIR / "cifar10_sift_train.pkl.gz"
NEWSGROUPS_DATA_DIR = BASE_DIR / "data" / "newsgroups_glove"
EMBEDDINGS_PATH = NEWSGROUPS_DATA_DIR / "newsgroups_embeddings.pkl.gz"
MAX_WORDS = 300

N_VALUES      = list(range(1000, 11000, 1000))   # [1000, 2000, ..., 10000]
EXACT_N_LIMIT = 10_000
EPSILON       = 0.01
BATCH_SIZE    = 512
SEED          = 42

BLUE_DIGITS = list(range(5))   # digits 0-4 → blue set
RED_DIGITS  = list(range(5, 10))  # digits 5-9 → red set

# EMNIST byclass has 62 classes (0-9 digits, then A-Z, a-z)
BLUE_CLASS_END = 31   # classes 0-30 (inclusive) → blue set
RED_CLASS_START = 31  # classes 31-61 (inclusive) → red set

COL_SPECS = [
    ("Dataset", 14),
    ("N", 7),
    ("γ", 8),
    ("1+2γ", 8),
    ("2L Ratio", 10),
    ("3L Ratio", 10),
    ("Gap 2L", 8),
    ("Gap 3L", 8),
]

FMT_FNS = {
    "Dataset":  lambda r: f"{r['dataset']:<14}",
    "N":        lambda r: f"{r['n']:,}",
    "γ":        lambda r: fmt_ratio(r["gamma"]),
    "1+2γ":     lambda r: fmt_ratio(r["bound"]),
    "2L Ratio": lambda r: fmt_ratio(r["r2"]),
    "3L Ratio": lambda r: fmt_ratio(r["r3"]),
    "Gap 2L":   lambda r: fmt_ratio(r["gap2"]),
    "Gap 3L":   lambda r: fmt_ratio(r["gap3"]),
}


def load_mnist_equal(n_samples, seed):
    train = torchvision.datasets.MNIST(root=str(DATA_DIR), train=True, download=False)
    test  = torchvision.datasets.MNIST(root=str(DATA_DIR), train=False, download=False)
    images = torch.cat([train.data, test.data], dim=0).numpy().reshape(-1, 784)
    labels = torch.cat([train.targets, test.targets], dim=0).numpy()

    classes = np.unique(labels)
    spc = n_samples // len(classes)
    if spc == 0:
        raise ValueError(f"n_samples={n_samples} too small for {len(classes)} classes")

    rng_r = np.random.RandomState(seed)
    rng_b = np.random.RandomState(seed + 1)
    red_parts, blue_parts = [], []
    for cls in classes:
        idx = np.flatnonzero(labels == cls).copy()
        needed = 2 * spc
        if idx.size < needed:
            warnings.warn(f"Class {cls}: only {idx.size} samples, need {needed}. Skipping.")
            continue
        rng_r.shuffle(idx)
        chosen = idx[:needed]
        red_parts.append(images[chosen[:spc]])
        blue_parts.append(images[chosen[spc:needed]])

    red = np.concatenate(red_parts).astype(np.float32) / 255.0
    blue = np.concatenate(blue_parts).astype(np.float32) / 255.0
    for arr in (red, blue):
        s = arr.sum(axis=1, keepdims=True)
        np.maximum(s, 1e-8, out=s)
        # Treat each MNIST image as a probability measure over the pixel grid.
        # This is the standard OT image-histogram scaling; do not divide by 2,
        # because these proxy experiments report costs in natural [0, 2] L1 units.
        arr /= s
    return torch.from_numpy(red), torch.from_numpy(blue)


def _sample_from_digits(images, labels, digit_set, n_total, rng):
    """Sample n_total images equally from the specified digits."""
    classes = sorted(digit_set)
    spc = n_total // len(classes)
    if spc == 0:
        raise ValueError(f"n_total={n_total} too small for {len(classes)} classes")
    parts = []
    for cls in classes:
        idx = np.flatnonzero(labels == cls).copy()
        if idx.size < spc:
            warnings.warn(f"Digit {cls}: only {idx.size} available, need {spc}. Skipping.")
            continue
        rng.shuffle(idx)
        parts.append(images[idx[:spc]])
    return np.concatenate(parts)


def load_mnist_biased(n_samples, seed):
    train = torchvision.datasets.MNIST(root=str(DATA_DIR), train=True, download=False)
    test  = torchvision.datasets.MNIST(root=str(DATA_DIR), train=False, download=False)
    images = torch.cat([train.data, test.data], dim=0).numpy().reshape(-1, 784)
    labels = torch.cat([train.targets, test.targets], dim=0).numpy()

    rng_r = np.random.RandomState(seed)
    rng_b = np.random.RandomState(seed + 1)
    red_arr  = _sample_from_digits(images, labels, RED_DIGITS,  n_samples, rng_r).astype(np.float32) / 255.0
    blue_arr = _sample_from_digits(images, labels, BLUE_DIGITS, n_samples, rng_b).astype(np.float32) / 255.0

    for arr in (red_arr, blue_arr):
        s = arr.sum(axis=1, keepdims=True)
        np.maximum(s, 1e-8, out=s)
        # Treat each MNIST image as a probability measure over the pixel grid.
        # This is the standard OT image-histogram scaling; do not divide by 2,
        # because these proxy experiments report costs in natural [0, 2] L1 units.
        arr /= s

    return torch.from_numpy(red_arr), torch.from_numpy(blue_arr)


def load_emnist_equal(n_samples, seed, split=EMNIST_SPLIT):
    train = torchvision.datasets.EMNIST(root=str(EMNIST_DATA_DIR), split=split, train=True,  download=False)
    test  = torchvision.datasets.EMNIST(root=str(EMNIST_DATA_DIR), split=split, train=False, download=False)
    images = torch.cat([train.data, test.data], dim=0).numpy()
    labels = torch.cat([train.targets, test.targets], dim=0).numpy()
    images = images.reshape(-1, 28, 28).transpose(0, 2, 1).reshape(-1, 784)

    classes = np.unique(labels)
    spc = n_samples // len(classes)
    if spc == 0:
        raise ValueError(f"n_samples={n_samples} too small for {len(classes)} EMNIST classes")

    rng_r = np.random.RandomState(seed)
    rng_b = np.random.RandomState(seed + 1)
    red_parts, blue_parts = [], []
    for cls in classes:
        idx = np.flatnonzero(labels == cls).copy()
        needed = 2 * spc
        if idx.size < needed:
            warnings.warn(f"Class {cls}: {idx.size} available, need {needed}. Skipping.")
            continue
        rng_r.shuffle(idx)
        chosen = idx[:needed]
        red_parts.append(images[chosen[:spc]])
        blue_parts.append(images[chosen[spc:needed]])

    red  = np.concatenate(red_parts).astype(np.float32)  / 255.0
    blue = np.concatenate(blue_parts).astype(np.float32) / 255.0
    for arr in (red, blue):
        s = arr.sum(axis=1, keepdims=True)
        np.maximum(s, 1e-8, out=s)
        # Treat each EMNIST image as a probability measure over the pixel grid.
        # This is the standard OT image-histogram scaling; do not divide by 2,
        # because these proxy experiments report costs in natural [0, 2] L1 units.
        arr /= s

    return torch.from_numpy(red), torch.from_numpy(blue)


def _sample_classes(images, labels, class_indices, n_total, rng):
    classes = sorted(class_indices)
    spc = n_total // len(classes)
    if spc == 0:
        raise ValueError(f"n_total={n_total} too small for {len(classes)} classes")
    parts = []
    for cls in classes:
        idx = np.flatnonzero(labels == cls).copy()
        if idx.size < spc:
            warnings.warn(f"Class {cls}: {idx.size} available, need {spc}. Skipping.")
            continue
        rng.shuffle(idx)
        parts.append(images[idx[:spc]])
    return np.concatenate(parts)


def load_emnist_biased(n_samples, seed, split=EMNIST_SPLIT):
    train = torchvision.datasets.EMNIST(root=str(EMNIST_DATA_DIR), split=split, train=True,  download=False)
    test  = torchvision.datasets.EMNIST(root=str(EMNIST_DATA_DIR), split=split, train=False, download=False)
    images = torch.cat([train.data, test.data], dim=0).numpy()
    labels = torch.cat([train.targets, test.targets], dim=0).numpy()
    images = images.reshape(-1, 28, 28).transpose(0, 2, 1).reshape(-1, 784)

    all_classes = np.unique(labels).tolist()
    blue_classes = [c for c in all_classes if c < BLUE_CLASS_END]
    red_classes  = [c for c in all_classes if c >= RED_CLASS_START]

    rng_r = np.random.RandomState(seed)
    rng_b = np.random.RandomState(seed + 1)
    red_arr  = _sample_classes(images, labels, red_classes,  n_samples, rng_r).astype(np.float32) / 255.0
    blue_arr = _sample_classes(images, labels, blue_classes, n_samples, rng_b).astype(np.float32) / 255.0

    for arr in (red_arr, blue_arr):
        s = arr.sum(axis=1, keepdims=True)
        np.maximum(s, 1e-8, out=s)
        # Treat each EMNIST image as a probability measure over the pixel grid.
        # This is the standard OT image-histogram scaling; do not divide by 2,
        # because these proxy experiments report costs in natural [0, 2] L1 units.
        arr /= s

    return torch.from_numpy(red_arr), torch.from_numpy(blue_arr)


def load_descriptors(path):
    with gzip.open(path, "rb") as f:
        descs = pickle.load(f)
    print(f"  Loaded {len(descs):,} descriptor sets from {pathlib.Path(path).name}", flush=True)
    return descs


def sample_pair(all_descs, n, seed):
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(all_descs))
    return ([all_descs[i] for i in perm[:n]],
            [all_descs[i] for i in perm[n:2*n]])


def _descriptor_means(descs):
    rows = []
    for d in descs:
        arr = np.asarray(d, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] == 0:
            raise ValueError("descriptor set must be a non-empty 2D array")
        rows.append(arr.mean(axis=0))
    return torch.from_numpy(np.stack(rows).astype(np.float32))


def load_cifar_sift(n, seed):
    if not TRAIN_DESC_PATH.exists():
        raise FileNotFoundError(
            f"{TRAIN_DESC_PATH} not found. Run download_cifar_sift.py first."
        )
    all_descs = load_descriptors(TRAIN_DESC_PATH)
    if 2 * n > len(all_descs):
        raise ValueError(f"not enough CIFAR SIFT descriptor sets: need {2*n:,}, have {len(all_descs):,}")
    red_descs, blue_descs = sample_pair(all_descs, n, seed)
    return _descriptor_means(red_descs), _descriptor_means(blue_descs)


def load_embeddings(path):
    with gzip.open(path, "rb") as f:
        embs = pickle.load(f)
    print(f"  Loaded {len(embs):,} document embeddings from {Path(path).name}", flush=True)
    return embs


def _embedding_means(descs):
    rows = []
    for d in descs:
        arr = np.asarray(d[:MAX_WORDS] if len(d) > MAX_WORDS else d, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] == 0:
            raise ValueError("embedding set must be a non-empty 2D array")
        rows.append(arr.mean(axis=0))
    return torch.from_numpy(np.stack(rows).astype(np.float32))


def load_newsgroups(n, seed):
    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(
            f"{EMBEDDINGS_PATH} not found. Run download_newsgroups_glove.py first."
        )
    all_embs = load_embeddings(EMBEDDINGS_PATH)
    if 2 * n > len(all_embs):
        raise ValueError(f"not enough Newsgroups documents: need {2*n:,}, have {len(all_embs):,}")
    red_descs, blue_descs = sample_pair(all_embs, n, seed)
    return _embedding_means(red_descs), _embedding_means(blue_descs)


DATASET_CONFIGS = [
    {"name": "MNIST-Equal",   "loader": load_mnist_equal},
    {"name": "MNIST-Biased",  "loader": load_mnist_biased},
    {"name": "EMNIST-Equal",  "loader": load_emnist_equal},
    {"name": "EMNIST-Biased", "loader": load_emnist_biased},
    {"name": "CIFAR-SIFT",    "loader": load_cifar_sift},
    {"name": "Newsgroups",    "loader": load_newsgroups},
]


def compute_l1_matrix(red, blue):
    X = red.cpu().to(torch.float64).contiguous()
    Y = blue.cpu().to(torch.float64).contiguous()
    return torch.cdist(X, Y, p=1).numpy()


def _run_exact(red, blue):
    if red.shape[0] > EXACT_N_LIMIT:
        raise RuntimeError(f"Exact skipped: N > {EXACT_N_LIMIT:,}")
    if ot is None:
        raise RuntimeError("POT not installed")
    n = red.shape[0]
    a = np.full(n, 1.0 / n, dtype=np.float64)
    b = np.full(n, 1.0 / n, dtype=np.float64)
    C = compute_l1_matrix(red.cpu(), blue.cpu())
    t0 = time.perf_counter()
    plan = ot.emd(a, b, C, numItermax=10**6)
    elapsed = (time.perf_counter() - t0) * 1000.0
    match = torch.from_numpy(plan.argmax(axis=0).astype(np.int64))
    rc = red.cpu()
    bc = blue.cpu()
    cost = (bc - rc[match]).abs().sum(dim=1).mean().item()
    return elapsed, cost, match


def _run_proxy2(red, blue, device):
    if red.shape[0] > EXACT_N_LIMIT:
        raise RuntimeError(f"2L-Proxy skipped: N > {EXACT_N_LIMIT:,}")
    if ot is None:
        raise RuntimeError("POT not installed")
    engine = SimpleL1Clustering(epsilon=EPSILON, tile_size=BATCH_SIZE)
    c = engine.run(red.to(device), blue.to(device))
    _adj_ptr = c["adj_ptr"].cpu()
    _adj_col = c["adj_col"].cpu()
    n = red.shape[0]
    a = np.full(n, 1.0 / n, np.float64)
    b = np.full(n, 1.0 / n, np.float64)
    C = build_two_level_proxy_matrix(c, n, device)
    t0 = time.perf_counter()
    plan = ot.emd(a, b, C, numItermax=10**6)
    elapsed = (time.perf_counter() - t0) * 1000.0
    match = torch.from_numpy(plan.argmax(axis=1).astype(np.int64))
    cost = (blue.to(device) - red.to(device)[match.to(device)]).abs().sum(dim=1).mean().item()
    return elapsed, cost, _adj_ptr, _adj_col


def _run_proxy3(red, blue, device):
    if red.shape[0] > EXACT_N_LIMIT:
        raise RuntimeError(f"3L-Proxy skipped: N > {EXACT_N_LIMIT:,}")
    if ot is None:
        raise RuntimeError("POT not installed")
    engine = ThreeLevelL1Clustering(epsilon=EPSILON, tile_size=BATCH_SIZE)
    c = engine.run(red.to(device), blue.to(device))
    n = red.shape[0]
    a = np.full(n, 1.0 / n, np.float64)
    b = np.full(n, 1.0 / n, np.float64)
    C = build_three_level_proxy_matrix(c, n, device)
    t0 = time.perf_counter()
    plan = ot.emd(a, b, C, numItermax=10**6)
    elapsed = (time.perf_counter() - t0) * 1000.0
    match = torch.from_numpy(plan.argmax(axis=1).astype(np.int64))
    cost = (blue.to(device) - red.to(device)[match.to(device)]).abs().sum(dim=1).mean().item()
    return elapsed, cost


def _compute_gamma(match, red, blue, adj_ptr, adj_col):
    """
    match    : (N,) int64 CPU tensor — match[b] = red index matched to blue b
    red      : (N, d) float32 CPU tensor
    blue     : (N, d) float32 CPU tensor
    adj_ptr  : (N+1,) int64 CPU tensor — CSR row pointers
    adj_col  : (M,)   int64 CPU tensor — local red indices per blue
    Returns gamma in [0, 1].
    """
    N = blue.shape[0]
    # Vectorised per-pair L1 costs
    pair_costs = (red[match] - blue).abs().sum(dim=1)   # (N,) float32
    total_cost = pair_costs.sum().item()
    if total_cost < 1e-12:
        return 0.0
    # Check locality for each blue via tensor comparison
    is_local = torch.zeros(N, dtype=torch.bool)
    for b in range(N):
        local_reds = adj_col[adj_ptr[b].item() : adj_ptr[b + 1].item()]
        if local_reds.numel() > 0:
            is_local[b] = (local_reds == match[b]).any()
    non_local_cost = pair_costs[~is_local].sum().item()
    return non_local_cost / total_cost


def _make_na_row(ds_name, n):
    return {
        "dataset": ds_name, "n": n,
        "gamma": math.nan, "bound": math.nan,
        "r2": math.nan, "r3": math.nan,
        "gap2": math.nan, "gap3": math.nan,
        "exact_cost": math.nan, "prx2_cost": math.nan, "prx3_cost": math.nan,
        "exact_time": math.nan, "prx2_time": math.nan, "prx3_time": math.nan,
    }


def run(device, **kwargs):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    print(f"\n{'─'*70}", flush=True)
    print(f"  Exp {EXP_ID}: {EXP_NAME}", flush=True)
    print(f"  Device: {device}  ε={EPSILON}  N={N_VALUES[0]}..{N_VALUES[-1]}", flush=True)
    print(f"{'─'*70}", flush=True)

    rows = []

    for ds in DATASET_CONFIGS:
        ds_name = ds["name"]
        loader  = ds["loader"]
        print(f"\n  ── Dataset: {ds_name} ──", flush=True)

        for n in N_VALUES:
            print(f"\n    N = {n:,}", flush=True)

            # --- Load data ---
            try:
                red, blue = loader(n, SEED)
            except FileNotFoundError as exc:
                print(f"      Data not found, skipping: {exc}", flush=True)
                rows.append(_make_na_row(ds_name, n))
                continue
            except Exception as exc:
                print(f"      Load failed: {exc}", flush=True)
                rows.append(_make_na_row(ds_name, n))
                continue

            # --- Exact OT ---
            exact_time = math.nan
            exact_cost = math.nan
            match      = None
            if n <= EXACT_N_LIMIT and ot is not None:
                try:
                    exact_time, exact_cost, match = _run_exact(red, blue)
                    print(f"      Exact: cost={exact_cost:.4f}  time={exact_time:.0f}ms", flush=True)
                except Exception as exc:
                    print(f"      Exact skip: {exc}", flush=True)
            else:
                print(f"      Exact: skipped (n > EXACT_N_LIMIT)", flush=True)

            # --- 2-Level Proxy ---
            prx2_time = math.nan
            prx2_cost = math.nan
            adj_ptr   = None
            adj_col   = None
            try:
                prx2_time, prx2_cost, adj_ptr, adj_col = _run_proxy2(red, blue, device)
                print(f"      2L: cost={prx2_cost:.4f}  time={prx2_time:.0f}ms", flush=True)
            except Exception as exc:
                print(f"      2L skip: {exc}", flush=True)

            # --- 3-Level Proxy ---
            prx3_time = math.nan
            prx3_cost = math.nan
            try:
                prx3_time, prx3_cost = _run_proxy3(red, blue, device)
                print(f"      3L: cost={prx3_cost:.4f}  time={prx3_time:.0f}ms", flush=True)
            except Exception as exc:
                print(f"      3L skip: {exc}", flush=True)

            # --- Gamma ---
            gamma = math.nan
            if match is not None and adj_ptr is not None and adj_col is not None:
                try:
                    gamma = _compute_gamma(
                        match.cpu(), red.cpu(), blue.cpu(),
                        adj_ptr.cpu(), adj_col.cpu()
                    )
                    print(f"      γ={gamma:.4f}", flush=True)
                except Exception as exc:
                    print(f"      gamma skip: {exc}", flush=True)

            # --- Derived quantities ---
            r2     = compute_ratio(exact_cost, prx2_cost)
            r3     = compute_ratio(exact_cost, prx3_cost)
            bound  = (1.0 + 2.0 * gamma) if not math.isnan(gamma) else math.nan
            gap2   = (bound - r2)         if not (math.isnan(bound) or math.isnan(r2)) else math.nan
            gap3   = (bound - r3)         if not (math.isnan(bound) or math.isnan(r3)) else math.nan

            rows.append({
                "dataset":    ds_name,
                "n":          n,
                "gamma":      gamma,
                "bound":      bound,
                "r2":         r2,
                "r3":         r3,
                "gap2":       gap2,
                "gap3":       gap3,
                "exact_cost": exact_cost,
                "prx2_cost":  prx2_cost,
                "prx3_cost":  prx3_cost,
                "exact_time": exact_time,
                "prx2_time":  prx2_time,
                "prx3_time":  prx3_time,
            })

            if device.type == "cuda":
                torch.cuda.empty_cache()

    return rows


def _fmt_row_terminal(row):
    return [
        f"{row['dataset']:<14}",
        f"{row['n']:>7,}",
        f"{fmt_ratio(row['gamma']):>8}",
        f"{fmt_ratio(row['bound']):>8}",
        f"{fmt_ratio(row['r2']):>10}",
        f"{fmt_ratio(row['r3']):>10}",
        f"{fmt_ratio(row['gap2']):>8}",
        f"{fmt_ratio(row['gap3']):>8}",
    ]


def print_table(rows):
    headers = ["Dataset       ", "      N", "       γ", "    1+2γ",
               "  2L Ratio", "  3L Ratio", "  Gap 2L", "  Gap 3L"]
    sep = "-+-".join("-" * len(h) for h in headers)
    print("\n" + " | ".join(headers))
    print(sep)
    prev_dataset = None
    for row in rows:
        if prev_dataset is not None and row["dataset"] != prev_dataset:
            print()
        print(" | ".join(_fmt_row_terminal(row)))
        prev_dataset = row["dataset"]


if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = run(dev)
    print_table(results)
