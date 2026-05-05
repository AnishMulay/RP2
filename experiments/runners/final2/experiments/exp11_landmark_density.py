#!/usr/bin/env python3
"""
Experiment 11 — MNIST: OT Ratio vs Landmark Density (c)
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
    agg_mean, agg_median, agg_std, agg_p90, agg_max, fmt_stat,
    compute_ratio, fmt_time, fmt_cost, fmt_ratio,
    build_two_level_proxy_matrix, build_three_level_proxy_matrix,
)

EXP_ID = 11
EXP_NAME = "MNIST — OT Ratio vs Landmark Density (c)"
DATASET = "MNIST"
DATA_DIR = FINAL2_DIR / "data"

N_FIXED = 3000
EXACT_N_LIMIT = N_FIXED
C_VALUES = [0.5, 1.0, 2.0, 4.0, 8.0]
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
EPSILON = 0.01
BATCH_SIZE = 512

COL_SPECS = [
    ("c",               5),
    ("2L Landmarks",   14),
    ("3L A1 Landmarks", 16),
    ("2L Local Pairs", 14),
    ("DR Matrix Size",   14),
    ("Total Repr Size",  14),
    ("2L Mean",         9),
    ("2L Median",      10),
    ("2L P90",          8),
    ("3L Mean",         9),
    ("3L Median",      10),
    ("3L P90",          8),
    ("2L Time",        12),
    ("3L Time",        12),
]

FMT_FNS = {
    "c":               lambda r: f"{r['c']:.1f}",
    "2L Landmarks":   lambda r: f"{int(r['landmarks2l_mean']):,}" if not math.isnan(r["landmarks2l_mean"]) else "N/A",
    "3L A1 Landmarks": lambda r: f"{int(r['landmarks3l_mean']):,}" if not math.isnan(r["landmarks3l_mean"]) else "N/A",
    "2L Local Pairs": lambda r: f"{int(r['local_pairs_mean']):,}" if not math.isnan(r["local_pairs_mean"]) else "N/A",
    "DR Matrix Size":  lambda r: f"{int(r['dr_matrix_size_mean']):,}" if not math.isnan(r.get("dr_matrix_size_mean", math.nan)) else "N/A",
    "Total Repr Size": lambda r: f"{int(r['total_repr_size_mean']):,}" if not math.isnan(r.get("total_repr_size_mean", math.nan)) else "N/A",
    "2L Mean":        lambda r: fmt_stat(r["r2_mean"]),
    "2L Median":      lambda r: fmt_stat(r["r2_median"]),
    "2L P90":         lambda r: fmt_stat(r["r2_p90"]),
    "3L Mean":        lambda r: fmt_stat(r["r3_mean"]),
    "3L Median":      lambda r: fmt_stat(r["r3_median"]),
    "3L P90":         lambda r: fmt_stat(r["r3_p90"]),
    "2L Time":        lambda r: fmt_time(r["prx2_time_mean"]),
    "3L Time":        lambda r: fmt_time(r["prx3_time_mean"]),
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


def run(device, **kwargs):
    torch.manual_seed(SEEDS[0])
    np.random.seed(SEEDS[0])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEEDS[0])

    print(f"\n{'─'*60}", flush=True)
    print(f"  Exp {EXP_ID}: {EXP_NAME}", flush=True)
    print(f"  Device: {device}  N={N_FIXED:,}  ε={EPSILON}  batch={BATCH_SIZE}", flush=True)
    print(f"{'─'*60}", flush=True)

    accum = {
        c: {
            "landmarks2l": [],
            "local_pairs": [],
            "landmarks3l": [],
            "r2": [],
            "r3": [],
            "prx2_times": [],
            "prx3_times": [],
        }
        for c in C_VALUES
    }

    for trial_idx, seed in enumerate(SEEDS):
        print(f"\n  Trial {trial_idx+1}/{len(SEEDS)} (seed={seed})", flush=True)
        try:
            red, blue = load_mnist_equal(N_FIXED, seed)
            _exact_t, exact_cost, exact_match = _run_exact(red, blue)
            print(f"    Exact: cost={exact_cost:.4f}", flush=True)
        except Exception as exc:
            print(f"    Exact/data skip: {exc}", flush=True)
            continue

        for c in C_VALUES:
            print(f"    c = {c:.1f}", flush=True)

            _trial_c_seed = int(seed * 1000 + int(c * 10))
            torch.manual_seed(_trial_c_seed)
            prx2_cost = math.nan
            prx3_cost = math.nan
            try:
                engine2 = SimpleL1Clustering(epsilon=EPSILON, tile_size=BATCH_SIZE, sample_factor=c)
                c_result = engine2.run(red.to(device), blue.to(device))
                n = red.shape[0]
                a = np.full(n, 1.0 / n, np.float64)
                b = np.full(n, 1.0 / n, np.float64)
                C = build_two_level_proxy_matrix(c_result, n, device)
                t0 = time.perf_counter()
                plan = ot.emd(a, b, C, numItermax=10**6)
                elapsed = (time.perf_counter() - t0) * 1000.0
                match = torch.from_numpy(plan.argmax(axis=0).astype(np.int64))
                prx2_cost = (blue.to(device) - red.to(device)[match.to(device)]).abs().sum(dim=1).mean().item()
                accum[c]["landmarks2l"].append(float(c_result["sampled_idx"].numel()))
                accum[c]["local_pairs"].append(float(c_result["adj_col"].numel()))
                _dr_entries    = float(c_result["DR"].numel())
                _local_entries = float(c_result["adj_col"].numel())
                accum[c].setdefault("dr_matrix_size", []).append(_dr_entries)
                accum[c].setdefault("total_repr_size", []).append(_dr_entries + _local_entries)
                accum[c]["prx2_times"].append(elapsed)
                print(f"      2L: cost={prx2_cost:.4f}", flush=True)
            except Exception as exc:
                print(f"      2L skip: {exc}", flush=True)
                accum[c]["landmarks2l"].append(math.nan)
                accum[c]["local_pairs"].append(math.nan)
                accum[c]["prx2_times"].append(math.nan)

            try:
                engine3 = ThreeLevelL1Clustering(epsilon=EPSILON, tile_size=BATCH_SIZE, sample_factor=c)
                c3_result = engine3.run(red.to(device), blue.to(device))
                n = red.shape[0]
                a = np.full(n, 1.0 / n, np.float64)
                b = np.full(n, 1.0 / n, np.float64)
                C = build_three_level_proxy_matrix(c3_result, n, device)
                t0 = time.perf_counter()
                plan = ot.emd(a, b, C, numItermax=10**6)
                elapsed = (time.perf_counter() - t0) * 1000.0
                match = torch.from_numpy(plan.argmax(axis=0).astype(np.int64))
                prx3_cost = (blue.to(device) - red.to(device)[match.to(device)]).abs().sum(dim=1).mean().item()
                accum[c]["landmarks3l"].append(float(c3_result["sampled_idx_A1"].numel()))
                accum[c]["prx3_times"].append(elapsed)
                print(f"      3L: cost={prx3_cost:.4f}", flush=True)
            except Exception as exc:
                print(f"      3L skip: {exc}", flush=True)
                accum[c]["landmarks3l"].append(math.nan)
                accum[c]["prx3_times"].append(math.nan)

            accum[c]["r2"].append(compute_ratio(exact_cost, prx2_cost))
            accum[c]["r3"].append(compute_ratio(exact_cost, prx3_cost))

            if device.type == "cuda":
                torch.cuda.empty_cache()

        del exact_match

    rows = []
    for c in C_VALUES:
        vals = accum[c]
        r2_vals = vals["r2"]
        r3_vals = vals["r3"]
        rows.append({
            "c": c,
            "landmarks2l_mean": agg_mean(vals["landmarks2l"]),
            "local_pairs_mean": agg_mean(vals["local_pairs"]),
            "dr_matrix_size_mean": agg_mean(vals.get("dr_matrix_size", [])),
            "total_repr_size_mean": agg_mean(vals.get("total_repr_size", [])),
            "landmarks3l_mean": agg_mean(vals["landmarks3l"]),
            "r2_mean": agg_mean(r2_vals),
            "r2_median": agg_median(r2_vals),
            "r2_p90": agg_p90(r2_vals),
            "r3_mean": agg_mean(r3_vals),
            "r3_median": agg_median(r3_vals),
            "r3_p90": agg_p90(r3_vals),
            "prx2_time_mean": agg_mean(vals["prx2_times"]),
            "prx3_time_mean": agg_mean(vals["prx3_times"]),
        })

    return rows


def _fmt_row_terminal(row):
    return [
        f"{row['c']:>5.1f}",
        f"{FMT_FNS['2L Landmarks'](row):>14}",
        f"{FMT_FNS['3L A1 Landmarks'](row):>16}",
        f"{FMT_FNS['2L Local Pairs'](row):>14}",
        f"{FMT_FNS['DR Matrix Size'](row):>14}",
        f"{FMT_FNS['Total Repr Size'](row):>14}",
        f"{fmt_stat(row['r2_mean']):>9}",
        f"{fmt_stat(row['r2_median']):>10}",
        f"{fmt_stat(row['r2_p90']):>8}",
        f"{fmt_stat(row['r3_mean']):>9}",
        f"{fmt_stat(row['r3_median']):>10}",
        f"{fmt_stat(row['r3_p90']):>8}",
        f"{fmt_time(row['prx2_time_mean']):>12}",
        f"{fmt_time(row['prx3_time_mean']):>12}",
    ]


def print_table(rows):
    headers = ["    c", " 2L Landmarks", "3L A1 Landmarks", "2L Local Pairs",
               "DR Matrix Size", "Total Repr Size",
               "  2L Mean", " 2L Median", " 2L P90", "  3L Mean",
               " 3L Median", " 3L P90", "     2L Time", "     3L Time"]
    sep = "-+-".join("-" * len(h) for h in headers)
    print("\n" + " | ".join(headers))
    print(sep)
    for row in rows:
        print(" | ".join(_fmt_row_terminal(row)))


if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = run(dev)
    print_table(results)
