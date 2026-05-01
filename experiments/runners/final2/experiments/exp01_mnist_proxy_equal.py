#!/usr/bin/env python3
"""
Experiment 1 — MNIST: Exact OT vs 2-Level Proxy vs 3-Level Proxy
Sampling: Equal from all 10 digits for both B (blue) and A (red).
Distance: L1 (Manhattan) on probability-normalized 784-dim pixel histograms.
          Each image sums to 1, so pairwise L1 costs lie in [0, 2].
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
    run_three_level_precomputed,
)

EXP_ID = 1
EXP_NAME = "MNIST — Exact vs 2L-Proxy vs 3L-Proxy (Equal Sampling)"
DATASET = "MNIST"
DATA_DIR = FINAL2_DIR / "data"

N_VALUES = [5_000, 10_000, 15_000, 20_000, 25_000]
EPSILON = 0.01
BATCH_SIZE = 512
SEED = 42
EXACT_N_LIMIT = 25_000

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


def compute_l1_matrix(red, blue):
    X = red.cpu().to(torch.float64).contiguous()
    Y = blue.cpu().to(torch.float64).contiguous()
    return torch.cdist(X, Y, p=1).numpy()


def clustering_gpu_bytes(c):
    return sum(v.numel() * v.element_size()
               for v in c.values()
               if isinstance(v, torch.Tensor) and v.device.type == "cuda")


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
    print(f"  Device: {device}  ε={EPSILON}  batch={BATCH_SIZE}  L1 range=[0,2]", flush=True)
    print(f"{'─'*60}", flush=True)

    rows = []
    for n in N_VALUES:
        print(f"\n  N = {n:,}", flush=True)
        try:
            red, blue = load_mnist_equal(n, SEED)
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


def _fmt_row_terminal(row):
    e, p2, p3 = row["exact"], row["prx2"], row["prx3"]
    r2 = compute_ratio(e["cost"], p2["cost"])
    r3 = compute_ratio(e["cost"], p3["cost"])
    return [
        f"{row['n']:>7,}",
        f"{fmt_time(e['time_ms']):>14}",
        f"{fmt_time(p2['time_ms']):>14}",
        f"{fmt_time(p3['time_ms']):>14}",
        f"{fmt_cost(e['cost']):>12}",
        f"{fmt_cost(p2['cost']):>12}",
        f"{fmt_cost(p3['cost']):>12}",
        f"{fmt_ratio(r2):>9}",
        f"{fmt_ratio(r3):>9}",
    ]


def print_table(rows):
    headers = ["       N", "    Exact Time", "  2L-Prx Time",
               "  3L-Prx Time", "  Exact Cost", " 2L-Prx Cost",
               " 3L-Prx Cost", " 2L Ratio", " 3L Ratio"]
    sep = "-+-".join("-" * len(h) for h in headers)
    print("\n" + " | ".join(headers))
    print(sep)
    for row in rows:
        print(" | ".join(_fmt_row_terminal(row)))


if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = run(dev)
    print_table(results)
