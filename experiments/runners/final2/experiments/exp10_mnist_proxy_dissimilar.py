#!/usr/bin/env python3
"""
Experiment 10 — MNIST: Exact OT vs 2-Level Proxy vs 3-Level Proxy
Sampling: B (blue) from digits {1,2,4,7}, A (red) from digits {8,6,9,3}.
Distance: L1 (Manhattan) on probability-normalized 784-dim pixel histograms.
          Each image sums to 1, so pairwise L1 costs lie in [0, 2].
Diagnostic: Per-pair matched-cost distribution for visually dissimilar splits.
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

EXP_ID = 10
EXP_NAME = "MNIST — Exact vs 2L-Proxy vs 3L-Proxy (Dissimilar: B=1,2,4,7; A=8,6,9,3)"
DATASET = "MNIST"
DATA_DIR = FINAL2_DIR / "data"

N_VALUES = [500, 1_000, 2_000, 3_000, 5_000]
EPSILON = 0.01
BATCH_SIZE = 512
SEED = 42
EXACT_N_LIMIT = 10_000

BLUE_DIGITS = [1, 2, 4, 7]
RED_DIGITS = [8, 6, 9, 3]

COL_SPECS = [
    ("N",              7),
    ("Exact Time",    14),
    ("2L-Prx Time",   14),
    ("3L-Prx Time",   14),
    ("Exact Cost",    12),
    ("2L-Prx Cost",   12),
    ("3L-Prx Cost",   12),
    ("2L Ratio",       9),
    ("3L Ratio",       9),
    ("Exact >1",      10),
    ("Exact Min/Med/Max", 20),
    ("2L >1",         10),
    ("2L Min/Med/Max", 20),
    ("3L >1",         10),
    ("3L Min/Med/Max", 20),
]


def _fmt_triplet(result):
    return (
        f"{fmt_cost(result['cost_min'])}/"
        f"{fmt_cost(result['cost_median'])}/"
        f"{fmt_cost(result['cost_max'])}"
    )


FMT_FNS = {
    "N":                lambda r: f"{r['n']:,}",
    "Exact Time":       lambda r: fmt_time(r["exact"]["time_ms"]),
    "2L-Prx Time":      lambda r: fmt_time(r["prx2"]["time_ms"]),
    "3L-Prx Time":      lambda r: fmt_time(r["prx3"]["time_ms"]),
    "Exact Cost":       lambda r: fmt_cost(r["exact"]["cost"]),
    "2L-Prx Cost":      lambda r: fmt_cost(r["prx2"]["cost"]),
    "3L-Prx Cost":      lambda r: fmt_cost(r["prx3"]["cost"]),
    "2L Ratio":         lambda r: fmt_ratio(compute_ratio(r["exact"]["cost"], r["prx2"]["cost"])),
    "3L Ratio":         lambda r: fmt_ratio(compute_ratio(r["exact"]["cost"], r["prx3"]["cost"])),
    "Exact >1":         lambda r: fmt_ratio(r["exact"]["frac_gt_1"]),
    "Exact Min/Med/Max": lambda r: _fmt_triplet(r["exact"]),
    "2L >1":            lambda r: fmt_ratio(r["prx2"]["frac_gt_1"]),
    "2L Min/Med/Max":   lambda r: _fmt_triplet(r["prx2"]),
    "3L >1":            lambda r: fmt_ratio(r["prx3"]["frac_gt_1"]),
    "3L Min/Med/Max":   lambda r: _fmt_triplet(r["prx3"]),
}


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


def load_mnist_dissimilar(n_samples, seed):
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


def compute_l1_matrix(red, blue):
    X = red.cpu().to(torch.float64).contiguous()
    Y = blue.cpu().to(torch.float64).contiguous()
    return torch.cdist(X, Y, p=1).numpy()


def _cost_stats(pair_costs):
    values = pair_costs.detach().cpu().numpy()
    return {
        "frac_gt_1": float((values > 1.0).mean()),
        "cost_min": float(values.min()),
        "cost_median": float(np.median(values)),
        "cost_max": float(values.max()),
    }


def _result(elapsed, pair_costs):
    stats = _cost_stats(pair_costs)
    return elapsed, float(pair_costs.mean().item()), stats


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
    pair_costs = (blue.cpu() - red.cpu()[match]).abs().sum(dim=1)
    return _result(elapsed, pair_costs)


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
    pair_costs = (blue.to(device) - red.to(device)[match.to(device)]).abs().sum(dim=1)
    return _result(elapsed, pair_costs)


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
    pair_costs = (blue.to(device) - red.to(device)[match.to(device)]).abs().sum(dim=1)
    return _result(elapsed, pair_costs)


def _safe(fn, label):
    try:
        t, c, stats = fn()
        return {"time_ms": t, "cost": c, "status": "ok", **stats}
    except Exception as exc:
        print(f"    [{label}] skipped: {exc}", flush=True)
        return {
            "time_ms": math.nan,
            "cost": math.nan,
            "frac_gt_1": math.nan,
            "cost_min": math.nan,
            "cost_median": math.nan,
            "cost_max": math.nan,
            "status": "skip",
        }


def _print_diagnostics(label, result):
    if result["status"] != "ok":
        return
    print(
        f"      {label}: cost={fmt_cost(result['cost'])} "
        f"frac(cost>1)={fmt_ratio(result['frac_gt_1'])} "
        f"min={fmt_cost(result['cost_min'])} "
        f"median={fmt_cost(result['cost_median'])} "
        f"max={fmt_cost(result['cost_max'])}",
        flush=True,
    )


def run(device, **kwargs):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    print(f"\n{'─'*60}", flush=True)
    print(f"  Exp {EXP_ID}: {EXP_NAME}", flush=True)
    print(f"  Device: {device}  ε={EPSILON}  batch={BATCH_SIZE}  L1 range=[0,2]", flush=True)
    print(f"  Blue from digits {BLUE_DIGITS}, Red from digits {RED_DIGITS}", flush=True)
    print(f"{'─'*60}", flush=True)

    rows = []
    for n in N_VALUES:
        print(f"\n  N = {n:,}", flush=True)
        try:
            red, blue = load_mnist_dissimilar(n, SEED)
        except Exception as exc:
            print(f"    Data loading failed: {exc}", flush=True)
            continue

        print(f"    [1/3] Exact OT ...", flush=True)
        exact = _safe(lambda: _run_exact(red, blue), "Exact")
        _print_diagnostics("Exact", exact)

        print(f"    [2/3] 2-Level Proxy ...", flush=True)
        prx2 = _safe(lambda: _run_proxy2(red, blue, device), "2L-Proxy")
        _print_diagnostics("2L-Proxy", prx2)

        print(f"    [3/3] 3-Level Proxy ...", flush=True)
        prx3 = _safe(lambda: _run_proxy3(red, blue, device), "3L-Proxy")
        _print_diagnostics("3L-Proxy", prx3)

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
        f"{fmt_ratio(e['frac_gt_1']):>10}",
        f"{_fmt_triplet(e):>20}",
        f"{fmt_ratio(p2['frac_gt_1']):>10}",
        f"{_fmt_triplet(p2):>20}",
        f"{fmt_ratio(p3['frac_gt_1']):>10}",
        f"{_fmt_triplet(p3):>20}",
    ]


def print_table(rows):
    headers = ["       N", "    Exact Time", "  2L-Prx Time",
               "  3L-Prx Time", "  Exact Cost", " 2L-Prx Cost",
               " 3L-Prx Cost", " 2L Ratio", " 3L Ratio",
               "  Exact >1", "   Exact Min/Med/Max",
               "     2L >1", "      2L Min/Med/Max",
               "     3L >1", "      3L Min/Med/Max"]
    sep = "-+-".join("-" * len(h) for h in headers)
    print("\n" + " | ".join(headers))
    print(sep)
    for row in rows:
        print(" | ".join(_fmt_row_terminal(row)))


if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = run(dev)
    print_table(results)
