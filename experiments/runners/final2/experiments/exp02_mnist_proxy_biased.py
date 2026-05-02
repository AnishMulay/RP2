#!/usr/bin/env python3
"""
Experiment 2 — MNIST: Exact OT vs 2-Level Proxy vs 3-Level Proxy
Sampling: B (blue) from digits 0–4, A (red) from digits 5–9.
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
    agg_mean, agg_median, agg_std, agg_p90, agg_max, fmt_stat, compute_gamma,
    compute_ratio, fmt_time, fmt_cost, fmt_ratio,
    build_two_level_proxy_matrix, build_three_level_proxy_matrix,
)

EXP_ID = 2
EXP_NAME = "MNIST — Exact vs 2L-Proxy vs 3L-Proxy (Biased: B=0–4, A=5–9)"
DATASET = "MNIST"
DATA_DIR = FINAL2_DIR / "data"

N_VALUES = [5_000, 10_000, 15_000, 20_000, 25_000]
EPSILON = 0.01
BATCH_SIZE = 512
SEED = 42
EXACT_N_LIMIT = 25_000
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
GAMMA_N_LIMIT = 5000

BLUE_DIGITS = list(range(5))   # digits 0-4 → blue set
RED_DIGITS  = list(range(5, 10))  # digits 5-9 → red set

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

STAT_COL_SPECS = [
    ("N",          7),
    ("2L Mean",   10),
    ("2L Median", 10),
    ("2L Std",    10),
    ("2L P90",    10),
    ("2L Max",    10),
    ("3L Mean",   10),
    ("3L Median", 10),
    ("3L Std",    10),
    ("3L P90",    10),
    ("3L Max",    10),
]

STAT_FMT_FNS = {
    "N":          lambda r: f"{r['n']:,}",
    "2L Mean":   lambda r: fmt_stat(r["r2_mean"]),
    "2L Median": lambda r: fmt_stat(r["r2_median"]),
    "2L Std":    lambda r: fmt_stat(r["r2_std"]),
    "2L P90":    lambda r: fmt_stat(r["r2_p90"]),
    "2L Max":    lambda r: fmt_stat(r["r2_max"]),
    "3L Mean":   lambda r: fmt_stat(r["r3_mean"]),
    "3L Median": lambda r: fmt_stat(r["r3_median"]),
    "3L Std":    lambda r: fmt_stat(r["r3_std"]),
    "3L P90":    lambda r: fmt_stat(r["r3_p90"]),
    "3L Max":    lambda r: fmt_stat(r["r3_max"]),
}

GAMMA_COL_SPECS = [
    ("N",             7),
    ("γ Mean",       14),
    ("γ Median",     14),
    ("Bound (1+2γ)", 14),
    ("2L Ratio",     14),
    ("3L Ratio",     14),
    ("Gap 2L",       14),
    ("Gap 3L",       14),
]

GAMMA_FMT_FNS = {
    "N":             lambda r: f"{r['n']:,}",
    "γ Mean":       lambda r: fmt_stat(r["gamma_mean"]),
    "γ Median":     lambda r: fmt_stat(r["gamma_median"]),
    "Bound (1+2γ)": lambda r: fmt_stat(r["bound_mean"]),
    "2L Ratio":     lambda r: fmt_stat(r["r2_mean"]),
    "3L Ratio":     lambda r: fmt_stat(r["r3_mean"]),
    "Gap 2L":       lambda r: fmt_stat(r["gap2_mean"]),
    "Gap 3L":       lambda r: fmt_stat(r["gap3_mean"]),
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

        exact_costs, exact_times = [], []
        prx2_costs, prx2_times = [], []
        prx3_costs, prx3_times = [], []
        gammas = []

        for trial_idx, seed in enumerate(SEEDS):
            print(f"    Trial {trial_idx+1}/{len(SEEDS)} (seed={seed}) ...", flush=True)
            try:
                red, blue = load_mnist_biased(n, seed)
            except Exception as exc:
                print(f"      Data load failed: {exc}", flush=True)
                continue

            # Exact OT
            _match = None
            if n <= EXACT_N_LIMIT and ot is not None:
                try:
                    _t, _c, _match = _run_exact(red, blue)
                    exact_costs.append(_c)
                    exact_times.append(_t)
                    print(f"      Exact: cost={_c:.4f}", flush=True)
                except Exception as exc:
                    print(f"      Exact skip: {exc}", flush=True)
                    exact_costs.append(math.nan)
                    exact_times.append(math.nan)
            else:
                exact_costs.append(math.nan)
                exact_times.append(math.nan)

            # 2-Level Proxy
            _adj_ptr, _adj_col = None, None
            try:
                _t, _c, _adj_ptr, _adj_col = _run_proxy2(red, blue, device)
                prx2_costs.append(_c)
                prx2_times.append(_t)
                print(f"      2L: cost={_c:.4f}", flush=True)
            except Exception as exc:
                print(f"      2L skip: {exc}", flush=True)
                prx2_costs.append(math.nan)
                prx2_times.append(math.nan)

            # 3-Level Proxy
            try:
                _t, _c = _run_proxy3(red, blue, device)
                prx3_costs.append(_c)
                prx3_times.append(_t)
                print(f"      3L: cost={_c:.4f}", flush=True)
            except Exception as exc:
                print(f"      3L skip: {exc}", flush=True)
                prx3_costs.append(math.nan)
                prx3_times.append(math.nan)

            # gamma (only if N <= GAMMA_N_LIMIT and both exact and 2L succeeded)
            if n <= GAMMA_N_LIMIT and _match is not None and _adj_ptr is not None:
                try:
                    g = compute_gamma(_match.cpu(), red.cpu(), blue.cpu(), _adj_ptr, _adj_col)
                    gammas.append(g)
                except Exception as exc:
                    print(f"      gamma skip: {exc}", flush=True)

            if device.type == "cuda":
                torch.cuda.empty_cache()

        # Aggregate
        r2_vals = [compute_ratio(e, p) for e, p in zip(exact_costs, prx2_costs)]
        r3_vals = [compute_ratio(e, p) for e, p in zip(exact_costs, prx3_costs)]

        gamma_mean = agg_mean(gammas)
        bound_mean = (1.0 + 2.0 * gamma_mean) if not math.isnan(gamma_mean) else math.nan

        rows.append({
            "n":            n,
            "exact":        {"time_ms": agg_mean(exact_times), "cost": agg_mean(exact_costs), "status": "ok"},
            "prx2":         {"time_ms": agg_mean(prx2_times),  "cost": agg_mean(prx2_costs),  "status": "ok"},
            "prx3":         {"time_ms": agg_mean(prx3_times),  "cost": agg_mean(prx3_costs),  "status": "ok"},
            "r2_mean":      agg_mean(r2_vals),    "r2_median": agg_median(r2_vals),
            "r2_std":       agg_std(r2_vals),     "r2_p90":    agg_p90(r2_vals),    "r2_max": agg_max(r2_vals),
            "r3_mean":      agg_mean(r3_vals),    "r3_median": agg_median(r3_vals),
            "r3_std":       agg_std(r3_vals),     "r3_p90":    agg_p90(r3_vals),    "r3_max": agg_max(r3_vals),
            "gamma_mean":   gamma_mean,           "gamma_median": agg_median(gammas),
            "bound_mean":   bound_mean,
            "gap2_mean":    (bound_mean - agg_mean(r2_vals)) if not math.isnan(bound_mean) else math.nan,
            "gap3_mean":    (bound_mean - agg_mean(r3_vals)) if not math.isnan(bound_mean) else math.nan,
        })

    return rows


if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = run(dev)
    for r in results:
        e, p2, p3 = r["exact"], r["prx2"], r["prx3"]
        r2 = compute_ratio(e["cost"], p2["cost"])
        r3 = compute_ratio(e["cost"], p3["cost"])
        print(f"N={r['n']:>6,} | exact={fmt_time(e['time_ms'])} cost={fmt_cost(e['cost'])} "
              f"| 2L={fmt_time(p2['time_ms'])} cost={fmt_cost(p2['cost'])} ratio={fmt_ratio(r2)} "
              f"| 3L={fmt_time(p3['time_ms'])} cost={fmt_cost(p3['cost'])} ratio={fmt_ratio(r3)}")
