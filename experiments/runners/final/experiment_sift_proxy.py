#!/usr/bin/env python3

import math
import sys
import time
from pathlib import Path

import numpy as np
import torch


BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    import ot
except ImportError:
    ot = None

from clustered_push_relabel.clustering.simple_copy import SimpleClustering


N_VALUES = [1_000, 5_000, 10_000]
EPSILON = 0.01
BATCH_SIZE = 512
SEED = 42
SIFT_DIM = 128

DATA_DIR = BASE_DIR / "data" / "sift"
SIFT_BASE_PATH = DATA_DIR / "sift" / "sift_base.fvecs"


def load_fvecs(path):
    """Load a .fvecs file and return a numpy float32 array of shape (N, D)."""
    with open(path, "rb") as f:
        d = np.frombuffer(f.read(4), dtype=np.int32)[0]
        f.seek(0)
        data = np.frombuffer(f.read(), dtype=np.float32)
    record_size = 1 + d
    n_vecs = len(data) // record_size
    data = data.reshape(n_vecs, record_size)
    return data[:, 1:].copy()


def load_sift_pair(n_samples, seed):
    base_vectors = load_fvecs(SIFT_BASE_PATH)
    print(f"  Loaded SIFT base descriptors: {base_vectors.shape}", flush=True)
    if base_vectors.shape[1] != SIFT_DIM:
        raise ValueError(
            f"Expected SIFT descriptors with dimension {SIFT_DIM}, "
            f"got {base_vectors.shape[1]}."
        )

    total_vectors = base_vectors.shape[0]
    if 2 * n_samples > total_vectors:
        raise ValueError(
            f"Requested 2 * n_samples = {2 * n_samples}, but only "
            f"{total_vectors} base descriptors are available."
        )

    rng = np.random.RandomState(seed)
    perm = rng.permutation(total_vectors)
    red_idx = perm[:n_samples]
    blue_idx = perm[n_samples : 2 * n_samples]

    red = base_vectors[red_idx]
    blue = base_vectors[blue_idx]

    red = red.astype(np.float32) / 255.0
    blue = blue.astype(np.float32) / 255.0

    subsample = np.vstack([red[:500], blue[:500]])
    sub_tensor = torch.from_numpy(subsample)
    diameter = float(torch.cdist(sub_tensor, sub_tensor).max().item())
    diameter = max(diameter, 1e-6)

    red = red / diameter
    blue = blue / diameter

    return torch.from_numpy(red).float(), torch.from_numpy(blue).float()


def build_proxy_cost_matrix(clustering, N, device):
    """
    Build the full N x N float proxy cost matrix using SimpleClustering output.

    For each pair (b, a):
      - If a is in adj(b): use adj_dist_float[entry]  (exact stored L2 distance)
      - Otherwise:         use d_min_b[b] + DR[nearest_s[b], a]  (triangle proxy)

    Returns a (N, N) float64 numpy array suitable for ot.emd().
    """
    DR = clustering["DR"]
    d_min_b = clustering["d_min_b"]
    nearest_s = clustering["nearest_s"]
    adj_ptr = clustering["adj_ptr"]
    adj_col = clustering["adj_col"]
    adj_dist_float = clustering["adj_dist_float"]

    C = d_min_b.unsqueeze(1) + DR[nearest_s, :]

    if adj_col.numel() > 0:
        b_indices = torch.repeat_interleave(
            torch.arange(N, device=device, dtype=torch.long),
            adj_ptr[1:] - adj_ptr[:-1],
        )
        C[b_indices, adj_col] = adj_dist_float

    return C.cpu().to(torch.float64).numpy()


def compute_cost_matrix_l2(red, blue):
    X = torch.as_tensor(red, dtype=torch.float64).contiguous()
    Y = torch.as_tensor(blue, dtype=torch.float64).contiguous()
    return torch.cdist(X, Y, p=2).numpy()


def benchmark_exact(P_red, P_blue):
    if P_red.shape[0] > 10_000:
        raise RuntimeError("Exact OT skipped: N > 10,000")
    if ot is None:
        raise RuntimeError("POT not installed")

    red_cpu = P_red.detach().cpu()
    blue_cpu = P_blue.detach().cpu()
    n = red_cpu.shape[0]
    a = np.full(n, 1.0 / n, dtype=np.float64)
    b = np.full(n, 1.0 / n, dtype=np.float64)
    C = compute_cost_matrix_l2(red_cpu, blue_cpu)

    t0 = time.perf_counter()
    plan = ot.emd(a, b, C, numItermax=10**6)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    matching = torch.from_numpy(plan.argmax(axis=0).astype(np.int64))
    matched = red_cpu[matching.to(dtype=torch.long)]
    cost = torch.norm(blue_cpu - matched, p=2, dim=1).mean().item()
    return elapsed_ms, cost


def benchmark_proxy_exact(P_red, P_blue, device):
    if P_red.shape[0] > 10_000:
        raise RuntimeError("Proxy-Exact OT skipped: N > 10,000")
    if ot is None:
        raise RuntimeError("POT not installed")

    cluster_engine = SimpleClustering(
        epsilon=EPSILON,
        tile_size=BATCH_SIZE,
    )
    clustering = cluster_engine.run(
        P_red.to(device), P_blue.to(device)
    )

    n = P_red.shape[0]
    a = np.full(n, 1.0 / n, dtype=np.float64)
    b = np.full(n, 1.0 / n, dtype=np.float64)
    C = build_proxy_cost_matrix(clustering, n, device)

    t0 = time.perf_counter()
    plan = ot.emd(a, b, C, numItermax=10**6)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    matching = torch.from_numpy(plan.argmax(axis=1).astype(np.int64))
    matched = P_red[matching.to(device=P_red.device, dtype=torch.long)]
    cost = torch.norm(P_blue - matched, p=2, dim=1).mean().item()
    return elapsed_ms, cost


def run_exact(P_red, P_blue):
    try:
        time_ms, cost = benchmark_exact(P_red, P_blue)
        return {"time_ms": time_ms, "cost": cost, "status": "ok"}
    except Exception as exc:
        print(f"  [Exact] skipped: {exc}", flush=True)
        return {"time_ms": math.nan, "cost": math.nan, "status": "skip"}


def run_proxy_exact(P_red, P_blue, device):
    try:
        time_ms, cost = benchmark_proxy_exact(P_red, P_blue, device)
        return {"time_ms": time_ms, "cost": cost, "status": "ok"}
    except Exception as exc:
        print(f"  [ProxyExact] failed: {exc}", flush=True)
        return {"time_ms": math.nan, "cost": math.nan, "status": "fail"}


def is_available(value):
    return value == value


def fmt_time(value):
    if not is_available(value):
        return "N/A"
    return f"{value:.1f} ms"


def fmt_cost(value):
    if not is_available(value):
        return "N/A"
    return f"{value:.4f}"


def fmt_ratio(value):
    if not is_available(value):
        return "N/A"
    return f"{value:.4f}"


def compute_cost_ratio(exact_cost, proxy_cost):
    if math.isnan(exact_cost) or math.isnan(proxy_cost):
        return math.nan
    return proxy_cost / exact_cost


def print_results_table(rows):
    col_widths = {
        "n": 7,
        "exact_time": 14,
        "proxy_time": 19,
        "exact_cost": 18,
        "proxy_cost": 23,
        "cost_ratio": 10,
    }
    headers = [
        ("N", col_widths["n"], ">"),
        ("Exact Time", col_widths["exact_time"], ">"),
        ("ProxyExact Time", col_widths["proxy_time"], ">"),
        ("Exact Avg L2 Cost", col_widths["exact_cost"], ">"),
        ("ProxyExact Avg L2 Cost", col_widths["proxy_cost"], ">"),
        ("Cost Ratio", col_widths["cost_ratio"], ">"),
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
        proxy = row["proxy_exact"]
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


def main():
    global SIFT_BASE_PATH

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    print(f"Device: {device}  epsilon={EPSILON}  batch_size={BATCH_SIZE}")
    print(f"SIFT base file: {SIFT_BASE_PATH}")

    if not SIFT_BASE_PATH.exists():
        fallback = DATA_DIR / "sift_base.fvecs"
        if fallback.exists():
            SIFT_BASE_PATH = fallback
            print(f"Using fallback path: {SIFT_BASE_PATH}")
        else:
            print("ERROR: sift_base.fvecs not found. Run download_sift.py first.")
            return

    rows = []
    for n in N_VALUES:
        print(f"\nPreparing SIFT N={n:,}", flush=True)
        try:
            P_red, P_blue = load_sift_pair(n, seed=SEED)
            P_red = P_red.to(device)
            P_blue = P_blue.to(device)
        except Exception as exc:
            print(f"  Data loading failed: {exc}", flush=True)
            continue

        print("  Running Exact OT...", flush=True)
        exact_result = run_exact(P_red, P_blue)

        print("  Running Proxy-Exact OT...", flush=True)
        proxy_result = run_proxy_exact(P_red, P_blue, device)

        rows.append({"n": n, "exact": exact_result, "proxy_exact": proxy_result})

        del P_red, P_blue
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print_results_table(rows)


if __name__ == "__main__":
    main()
