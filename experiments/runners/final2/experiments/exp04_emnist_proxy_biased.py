#!/usr/bin/env python3
"""
Experiment 4 — EMNIST: Exact OT vs 2-Level Proxy vs 3-Level Proxy
Sampling: B (blue) from EMNIST classes 0–30, A (red) from classes 31–61.
Distance: L1 (Manhattan) on normalised 784-dim pixel vectors.
"""

import math
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

EXP_ID = 4
EXP_NAME = "EMNIST — Exact vs 2L-Proxy vs 3L-Proxy (Biased: B=cls 0–30, A=cls 31–61)"
DATASET = "EMNIST"
DATA_DIR = BASE_DIR / "data"
EMNIST_SPLIT = "byclass"

N_VALUES = [1_000, 5_000, 10_000]
EPSILON = 0.01
BATCH_SIZE = 512
SEED = 42
EXACT_N_LIMIT = 10_000

# EMNIST byclass has 62 classes (0-9 digits, then A-Z, a-z)
BLUE_CLASS_END = 31   # classes 0-30 (inclusive) → blue set
RED_CLASS_START = 31  # classes 31-61 (inclusive) → red set

COL_SPECS = [
    ("N",            7),
    ("Exact Time",  14),
    ("2L-Prx Time", 14),
    ("3L-Prx Time", 14),
    ("Exact Cost",  12),
    ("2L-Prx Cost", 12),
    ("3L-Prx Cost", 12),
    ("2L Ratio",     9),
    ("3L Ratio",     9),
]

FMT_FNS = {
    "N":            lambda r: f"{r['n']:,}",
    "Exact Time":   lambda r: fmt_time(r["exact"]["time_ms"]),
    "2L-Prx Time":  lambda r: fmt_time(r["prx2"]["time_ms"]),
    "3L-Prx Time":  lambda r: fmt_time(r["prx3"]["time_ms"]),
    "Exact Cost":   lambda r: fmt_cost(r["exact"]["cost"]),
    "2L-Prx Cost":  lambda r: fmt_cost(r["prx2"]["cost"]),
    "3L-Prx Cost":  lambda r: fmt_cost(r["prx3"]["cost"]),
    "2L Ratio":     lambda r: fmt_ratio(compute_ratio(r["exact"]["cost"], r["prx2"]["cost"])),
    "3L Ratio":     lambda r: fmt_ratio(compute_ratio(r["exact"]["cost"], r["prx3"]["cost"])),
}


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
    train = torchvision.datasets.EMNIST(root=str(DATA_DIR), split=split, train=True,  download=False)
    test  = torchvision.datasets.EMNIST(root=str(DATA_DIR), split=split, train=False, download=False)
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
        arr /= s
        arr /= 2.0

    return torch.from_numpy(red_arr), torch.from_numpy(blue_arr)


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
    a = np.full(n, 1.0 / n, np.float64)
    b = np.full(n, 1.0 / n, np.float64)
    C = compute_l1_matrix(red, blue)
    t0 = time.perf_counter()
    plan = ot.emd(a, b, C, numItermax=10**6)
    elapsed = (time.perf_counter() - t0) * 1000.0
    match = torch.from_numpy(plan.argmax(axis=0).astype(np.int64))
    cost = (blue.cpu() - red.cpu()[match]).abs().sum(dim=1).mean().item()
    return elapsed, cost


def _run_proxy2(red, blue, device):
    if red.shape[0] > EXACT_N_LIMIT:
        raise RuntimeError(f"2L-Proxy skipped: N > {EXACT_N_LIMIT:,}")
    if ot is None:
        raise RuntimeError("POT not installed")
    engine = SimpleL1Clustering(epsilon=EPSILON, tile_size=BATCH_SIZE)
    c = engine.run(red.to(device), blue.to(device))
    n = red.shape[0]
    a = np.full(n, 1.0 / n, np.float64)
    b = np.full(n, 1.0 / n, np.float64)
    C = build_two_level_proxy_matrix(c, n, device)
    t0 = time.perf_counter()
    plan = ot.emd(a, b, C, numItermax=10**6)
    elapsed = (time.perf_counter() - t0) * 1000.0
    match = torch.from_numpy(plan.argmax(axis=1).astype(np.int64))
    cost = (blue.to(device) - red.to(device)[match.to(device)]).abs().sum(dim=1).mean().item()
    return elapsed, cost


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


def _safe(fn, label):
    try:
        t, c = fn()
        return {"time_ms": t, "cost": c, "status": "ok"}
    except Exception as exc:
        print(f"    [{label}] skipped: {exc}", flush=True)
        return {"time_ms": math.nan, "cost": math.nan, "status": "skip"}


def run(device, **kwargs):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    print(f"\n{'─'*60}", flush=True)
    print(f"  Exp {EXP_ID}: {EXP_NAME}", flush=True)
    print(f"  Device: {device}  ε={EPSILON}  split={EMNIST_SPLIT}", flush=True)
    print(f"  Blue classes 0–{BLUE_CLASS_END-1}, Red classes {RED_CLASS_START}–61", flush=True)
    print(f"{'─'*60}", flush=True)

    rows = []
    for n in N_VALUES:
        print(f"\n  N = {n:,}", flush=True)
        try:
            red, blue = load_emnist_biased(n, SEED)
        except Exception as exc:
            print(f"    Data loading failed: {exc}", flush=True)
            continue

        print(f"    [1/3] Exact OT ...", flush=True)
        exact = _safe(lambda: _run_exact(red, blue), "Exact")

        print(f"    [2/3] 2-Level Proxy ...", flush=True)
        prx2 = _safe(lambda: _run_proxy2(red, blue, device), "2L-Proxy")

        print(f"    [3/3] 3-Level Proxy ...", flush=True)
        prx3 = _safe(lambda: _run_proxy3(red, blue, device), "3L-Proxy")

        rows.append({"n": n, "exact": exact, "prx2": prx2, "prx3": prx3})

        if device.type == "cuda":
            torch.cuda.empty_cache()

    return rows


if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = run(dev)
    for r in results:
        e, p2, p3 = r["exact"], r["prx2"], r["prx3"]
        print(f"N={r['n']:>7,} | exact={fmt_time(e['time_ms'])} cost={fmt_cost(e['cost'])} "
              f"| 2L ratio={fmt_ratio(compute_ratio(e['cost'], p2['cost']))} "
              f"| 3L ratio={fmt_ratio(compute_ratio(e['cost'], p3['cost']))}")
