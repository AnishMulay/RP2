#!/usr/bin/env python3
"""
Experiment: Exact OT vs Three-Level Proxy-Exact OT on EMNIST.

Mirrors experiment_emnist_proxy.py exactly in structure and data loading.
This experiment uses L1 + ThreeLevelL1Clustering so the proxy cost
approximates the same L1 distance that the exact solver is run on.
"""

import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torchvision


BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
SRC_DIR  = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    import ot
except ImportError:
    ot = None

from clustered_push_relabel.clustering.simple_three_level_l1 import ThreeLevelL1Clustering


N_VALUES    = [1_000, 5_000, 10_000]
EPSILON     = 0.01
BATCH_SIZE  = 512
SEED        = 42
EMNIST_SPLIT = "byclass"

DATA_DIR = BASE_DIR / "data"


# ── Data loading (identical to experiment_emnist_proxy.py) ───────────────────

def load_emnist_balanced(n_samples, seed, split):
    train_dataset = torchvision.datasets.EMNIST(
        root=str(DATA_DIR), split=split, train=True,  download=False,
    )
    test_dataset = torchvision.datasets.EMNIST(
        root=str(DATA_DIR), split=split, train=False, download=False,
    )

    images = torch.cat([train_dataset.data, test_dataset.data], dim=0).cpu().numpy()
    labels = torch.cat([train_dataset.targets, test_dataset.targets], dim=0).cpu().numpy()
    images = images.reshape(-1, 28, 28).transpose(0, 2, 1).reshape(-1, 784)

    classes          = np.unique(labels)
    num_classes      = int(classes.size)
    samples_per_class = n_samples // num_classes
    if samples_per_class == 0:
        raise ValueError(
            f"Requested n_samples={n_samples} but EMNIST split '{split}' has "
            f"{num_classes} classes, so floor(n_samples / num_classes) == 0."
        )

    red_rng  = np.random.RandomState(seed)
    blue_rng = np.random.RandomState(seed + 1)
    red_parts, blue_parts = [], []

    for class_label in classes:
        class_indices = np.flatnonzero(labels == class_label).copy()
        needed        = 2 * samples_per_class
        if class_indices.size < needed:
            warnings.warn(
                f"Skipping class {int(class_label)}: only {class_indices.size} "
                f"samples available, need {needed}.",
                stacklevel=2,
            )
            continue
        red_rng.shuffle(class_indices)
        chosen_indices = class_indices[:needed]
        red_indices    = chosen_indices[:samples_per_class].copy()
        blue_indices   = chosen_indices[samples_per_class:needed].copy()
        blue_rng.shuffle(blue_indices)
        red_parts.append(images[red_indices])
        blue_parts.append(images[blue_indices])

    if not red_parts or not blue_parts:
        raise ValueError(
            f"Unable to sample any EMNIST classes from split '{split}' with "
            f"{samples_per_class} samples per class."
        )

    red_images  = np.concatenate(red_parts,  axis=0)
    blue_images = np.concatenate(blue_parts, axis=0)

    if red_images.shape[0] != n_samples:
        warnings.warn(
            f"Collected {red_images.shape[0]} samples per set instead of the "
            f"requested {n_samples} because some classes were skipped.",
            stacklevel=2,
        )

    red_images = red_images.astype(np.float32, copy=False) / 255.0
    blue_images = blue_images.astype(np.float32, copy=False) / 255.0
    red_row_sums  = np.maximum(red_images.sum(axis=1,  keepdims=True), 1e-8)
    blue_row_sums = np.maximum(blue_images.sum(axis=1, keepdims=True), 1e-8)
    red_images    = (red_images  / red_row_sums)  / 2.0
    blue_images   = (blue_images / blue_row_sums) / 2.0

    red  = torch.from_numpy(red_images.astype(np.float32))
    blue = torch.from_numpy(blue_images.astype(np.float32))
    return red, blue


# ── Cost-matrix helpers ───────────────────────────────────────────────────────

def compute_cost_matrix_l1(red: torch.Tensor, blue: torch.Tensor) -> np.ndarray:
    """Full N×N pairwise L1 cost matrix as float64 numpy array."""
    X = red.cpu().to(torch.float64).contiguous()
    Y = blue.cpu().to(torch.float64).contiguous()
    return torch.cdist(X, Y, p=1).numpy()


def build_proxy_cost_matrix_three_level(
    clustering: dict,
    N: int,
    device: torch.device,
) -> np.ndarray:
    """
    Build the full N×N float proxy cost matrix from a ThreeLevelL1Clustering result.

    Proxy priority (each level overwrites the previous):
        Level 2 (base)    C[b,a]  = d_min_b_A2[b]  + DR[nearest_s2[b], a]
        Level 1 (via A1)  C[b,a]  = d_min_b_A1[b]  + d(s1_b, a)    for a ∈ Adj_A1(s1_b)
        Level 0 (direct)  C[b,a]  = d(b, a)                         for a ∈ Adj_B(b)

    Blues are grouped by their nearest A1 center once (argsort + bincount),
    avoiding a per-a1 nonzero call inside the loop.

    Returns a (N, N) float64 numpy array for ot.emd().
    """
    DR                = clustering["DR"]                  # (S2, N) float32
    d_min_b_A2        = clustering["d_min_b_A2"]          # (N,)
    nearest_s2        = clustering["nearest_s2"]          # (N,) → A2 indices
    d_min_b_A1        = clustering["d_min_b_A1"]          # (N,)
    nearest_s1        = clustering["nearest_s1"]          # (N,) → A1 indices
    adj_B_ptr         = clustering["adj_B_ptr"]           # (N+1,)
    adj_B_col         = clustering["adj_B_col"]           # (MB,)
    adj_B_dist_float  = clustering["adj_B_dist_float"]    # (MB,)
    adj_A1_ptr        = clustering["adj_A1_ptr"]          # (S1+1,)
    adj_A1_col        = clustering["adj_A1_col"]          # (MA1,)
    adj_A1_dist_float = clustering["adj_A1_dist_float"]   # (MA1,)
    S1 = int(adj_A1_ptr.shape[0]) - 1

    # ── Level 2: base proxy for every (b, a) pair ────────────────────────────
    # DR[nearest_s2, :] gathers one row per blue → (N, N)
    C = d_min_b_A2.unsqueeze(1) + DR[nearest_s2, :]      # (N, N) float32

    # ── Level 1: overwrite where a ∈ Adj_A1(nearest_s1[b]) ──────────────────
    # Pre-group blues by their nearest A1 center so the inner loop does no GPU
    # nonzero calls — only cheap CPU-side ptr slicing + vectorised GPU scatter.
    #
    # sorted_order[group_ptr[k] : group_ptr[k+1]]  gives all blues whose
    # nearest A1 center is k.
    sorted_order = torch.argsort(nearest_s1)                          # (N,)
    group_counts = torch.bincount(nearest_s1, minlength=S1)           # (S1,)
    group_ptr    = torch.zeros(S1 + 1, dtype=torch.long, device=device)
    group_ptr[1:] = group_counts.cumsum(0)

    for a1_idx in range(S1):
        a1_start = int(adj_A1_ptr[a1_idx].item())
        a1_end   = int(adj_A1_ptr[a1_idx + 1].item())
        if a1_start == a1_end:
            continue   # this A1 center has no neighbours in Adj_A1 → skip

        a_cols  = adj_A1_col[a1_start:a1_end]          # (K,) red indices
        a_dists = adj_A1_dist_float[a1_start:a1_end]   # (K,) d(a1, a)

        g_start = int(group_ptr[a1_idx].item())
        g_end   = int(group_ptr[a1_idx + 1].item())
        if g_start == g_end:
            continue   # no blue has this A1 center as nearest → skip

        blues = sorted_order[g_start:g_end]             # (B_k,) blue indices

        # Outer-product overwrite: C[blues[i], a_cols[j]] = d_min_b_A1[blues[i]] + a_dists[j]
        # blues.unsqueeze(1) → (B_k, 1),  a_cols.unsqueeze(0) → (1, K)
        C[blues.unsqueeze(1), a_cols.unsqueeze(0)] = (
            d_min_b_A1[blues].unsqueeze(1) + a_dists.unsqueeze(0)
        )

    # ── Level 0: overwrite with direct distances for a ∈ Adj_B(b) ───────────
    if adj_B_col.numel() > 0:
        # Expand b indices to match the flat CSR layout of adj_B_col
        b_indices = torch.repeat_interleave(
            torch.arange(N, device=device, dtype=torch.long),
            adj_B_ptr[1:] - adj_B_ptr[:-1],
        )
        C[b_indices, adj_B_col] = adj_B_dist_float

    return C.cpu().to(torch.float64).numpy()


# ── Benchmarks ────────────────────────────────────────────────────────────────

def benchmark_exact(P_red: torch.Tensor, P_blue: torch.Tensor):
    if P_red.shape[0] > 10_000:
        raise RuntimeError("Exact OT skipped: N > 10,000")
    if ot is None:
        raise RuntimeError("POT not installed")

    red_cpu  = P_red.detach().cpu()
    blue_cpu = P_blue.detach().cpu()
    n        = red_cpu.shape[0]
    a        = np.full(n, 1.0 / n, dtype=np.float64)
    b        = np.full(n, 1.0 / n, dtype=np.float64)
    C        = compute_cost_matrix_l1(red_cpu, blue_cpu)

    t0         = time.perf_counter()
    plan       = ot.emd(a, b, C, numItermax=10 ** 6)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    matching = torch.from_numpy(plan.argmax(axis=0).astype(np.int64))
    matched  = red_cpu[matching.to(dtype=torch.long)]
    cost     = (blue_cpu - matched).abs().sum(dim=1).mean().item()
    return elapsed_ms, cost


def benchmark_three_level_proxy_exact(
    P_red: torch.Tensor,
    P_blue: torch.Tensor,
    device: torch.device,
):
    if P_red.shape[0] > 10_000:
        raise RuntimeError("Three-Level Proxy-Exact OT skipped: N > 10,000")
    if ot is None:
        raise RuntimeError("POT not installed")

    cluster_engine = ThreeLevelL1Clustering(
        epsilon=EPSILON,
        tile_size=BATCH_SIZE,
    )
    clustering = cluster_engine.run(P_red.to(device), P_blue.to(device))

    n  = P_red.shape[0]
    a  = np.full(n, 1.0 / n, dtype=np.float64)
    b  = np.full(n, 1.0 / n, dtype=np.float64)
    C  = build_proxy_cost_matrix_three_level(clustering, n, device)

    t0         = time.perf_counter()
    plan       = ot.emd(a, b, C, numItermax=10 ** 6)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # Evaluate the matching under the TRUE L1 cost (not the proxy)
    matching = torch.from_numpy(plan.argmax(axis=1).astype(np.int64))
    matched  = P_red[matching.to(device=P_red.device, dtype=torch.long)]
    cost     = (P_blue - matched).abs().sum(dim=1).mean().item()
    return elapsed_ms, cost


# ── Run helpers ───────────────────────────────────────────────────────────────

def run_exact(P_red, P_blue):
    try:
        time_ms, cost = benchmark_exact(P_red, P_blue)
        return {"time_ms": time_ms, "cost": cost, "status": "ok"}
    except Exception as exc:
        print(f"  [Exact] skipped: {exc}", flush=True)
        return {"time_ms": math.nan, "cost": math.nan, "status": "skip"}


def run_three_level_proxy_exact(P_red, P_blue, device):
    try:
        time_ms, cost = benchmark_three_level_proxy_exact(P_red, P_blue, device)
        return {"time_ms": time_ms, "cost": cost, "status": "ok"}
    except Exception as exc:
        print(f"  [ThreeLevelProxy] failed: {exc}", flush=True)
        return {"time_ms": math.nan, "cost": math.nan, "status": "fail"}


# ── Formatting helpers (identical to experiment_emnist_proxy.py) ─────────────

def is_available(value):
    return value == value

def fmt_time(value):
    return "N/A" if not is_available(value) else f"{value:.1f} ms"

def fmt_cost(value):
    return "N/A" if not is_available(value) else f"{value:.4f}"

def fmt_ratio(value):
    return "N/A" if not is_available(value) else f"{value:.4f}"

def compute_cost_ratio(exact_cost, proxy_cost):
    if math.isnan(exact_cost) or math.isnan(proxy_cost):
        return math.nan
    return proxy_cost / exact_cost


# ── Table printer ─────────────────────────────────────────────────────────────

def print_results_table(rows):
    col_widths = {
        "n"          : 7,
        "exact_time" : 14,
        "proxy_time" : 24,
        "exact_cost" : 18,
        "proxy_cost" : 28,
        "cost_ratio" : 10,
    }
    headers = [
        ("N",                     col_widths["n"],          ">"),
        ("Exact Time",            col_widths["exact_time"], ">"),
        ("3-Lvl Proxy Time",      col_widths["proxy_time"], ">"),
        ("Exact Avg L1 Cost",     col_widths["exact_cost"], ">"),
        ("3-Lvl Proxy Avg L1 Cost", col_widths["proxy_cost"], ">"),
        ("Cost Ratio",            col_widths["cost_ratio"], ">"),
    ]

    header_line = " | ".join(
        f"{label:{align}{width}}" for label, width, align in headers
    )
    separator = "-+-".join("-" * width for _, width, _ in headers)

    print()
    print(header_line)
    print(separator)

    for row in rows:
        exact = row["exact"]
        proxy = row["three_level_proxy"]
        ratio = compute_cost_ratio(exact["cost"], proxy["cost"])
        cells = [
            f"{row['n']:>{col_widths['n']},}",
            f"{fmt_time(exact['time_ms']):>{col_widths['exact_time']}}",
            f"{fmt_time(proxy['time_ms']):>{col_widths['proxy_time']}}",
            f"{fmt_cost(exact['cost']):>{col_widths['exact_cost']}}",
            f"{fmt_cost(proxy['cost']):>{col_widths['proxy_cost']}}",
            f"{fmt_ratio(ratio):>{col_widths['cost_ratio']}}",
        ]
        print(" | ".join(cells))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    print(f"Device: {device}  epsilon={EPSILON}  batch_size={BATCH_SIZE}")
    print(f"EMNIST split: {EMNIST_SPLIT}  data dir: {DATA_DIR}")
    print("Distance metric: L1 (three-level clustering uses Manhattan distances)")

    rows = []
    for n in N_VALUES:
        print(f"\nPreparing EMNIST N={n:,}", flush=True)
        try:
            P_red, P_blue = load_emnist_balanced(n, seed=SEED, split=EMNIST_SPLIT)
            P_red  = P_red.to(device)
            P_blue = P_blue.to(device)
        except Exception as exc:
            print(f"  Data loading failed: {exc}", flush=True)
            continue

        print("  Running Exact OT (L1)...", flush=True)
        exact_result = run_exact(P_red, P_blue)

        print("  Running Three-Level Proxy-Exact OT (L1)...", flush=True)
        proxy_result = run_three_level_proxy_exact(P_red, P_blue, device)

        rows.append({
            "n"                  : n,
            "exact"              : exact_result,
            "three_level_proxy"  : proxy_result,
        })

        del P_red, P_blue
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print_results_table(rows)


if __name__ == "__main__":
    main()
