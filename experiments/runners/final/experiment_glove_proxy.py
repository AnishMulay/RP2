#!/usr/bin/env python3

import math
import pathlib
import statistics
import sys
import time

import numpy as np
import torch


BASE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent.parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    import ot
except ImportError:
    ot = None

from clustered_push_relabel.clustering.simple_copy import SimpleClustering


DATA_DIR = BASE_DIR / "data" / "glove"
NPY_PATH = DATA_DIR / "glove.6B.300d.npy"

N_VALUES = [1_000, 5_000, 10_000, 100_000, 200_000]
EPSILON = 0.01
BATCH_SIZE = 512
SEED = 42
WARMUP_RUNS = 0
TIMED_RUNS = 1

GLOVE_NUM_VECTORS = 400_000
GLOVE_DIM = 300
DIAMETER_SUBSAMPLE = 1_000


def load_glove_pair(n_samples, seed):
    """
    Load GloVe embeddings and return two non-overlapping sets of n_samples each.
    """
    vectors = np.load(NPY_PATH)
    print(f"  Loaded GloVe embeddings: {vectors.shape}", flush=True)
    if vectors.shape != (GLOVE_NUM_VECTORS, GLOVE_DIM):
        raise ValueError(
            f"Expected GloVe cache with shape ({GLOVE_NUM_VECTORS}, {GLOVE_DIM}), "
            f"got {vectors.shape}."
        )

    if 2 * n_samples > GLOVE_NUM_VECTORS:
        raise ValueError(
            f"Requested 2 * n_samples = {2 * n_samples}, but only "
            f"{GLOVE_NUM_VECTORS} vectors are available."
        )

    rng = np.random.RandomState(seed)
    perm = rng.permutation(GLOVE_NUM_VECTORS)
    red = vectors[perm[:n_samples]].astype(np.float32, copy=True)
    blue = vectors[perm[n_samples : 2 * n_samples]].astype(np.float32, copy=True)

    joint_size = 2 * n_samples
    subsample_size = min(DIAMETER_SUBSAMPLE, joint_size)
    subsample_positions = rng.choice(joint_size, size=subsample_size, replace=False)
    subsample = np.empty((subsample_size, GLOVE_DIM), dtype=np.float32)
    red_mask = subsample_positions < n_samples
    if red_mask.any():
        subsample[red_mask] = red[subsample_positions[red_mask]]
    if (~red_mask).any():
        subsample[~red_mask] = blue[subsample_positions[~red_mask] - n_samples]

    sub_tensor = torch.from_numpy(subsample)
    diameter = float(torch.cdist(sub_tensor, sub_tensor).max().item())
    diameter = max(diameter, 1e-6)

    red = (red / diameter).astype(np.float32, copy=False)
    blue = (blue / diameter).astype(np.float32, copy=False)

    return torch.from_numpy(red).float(), torch.from_numpy(blue).float(), diameter


def synchronize_if_cuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def empty_cache_if_cuda(device):
    if device.type == "cuda":
        torch.cuda.empty_cache()


def build_proxy_cost_matrix(clustering, N, device):
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
    return torch.cdist(X, Y, p=2).cpu().numpy()


def matching_from_plan(plan):
    return torch.from_numpy(plan.argmax(axis=0).astype(np.int64, copy=False))


def average_l2_matching_cost(P_red, P_blue, matching):
    matched_red = P_red[matching.to(device=P_red.device, dtype=torch.long)]
    return torch.norm(P_blue - matched_red, p=2, dim=1).mean().item()


def benchmark_exact_l2(P_red, P_blue):
    if P_red.shape[0] > 10_000:
        raise RuntimeError("Exact OT skipped for N > 10,000")
    if ot is None:
        raise RuntimeError("POT is not installed; exact solver unavailable.")

    red_cpu = P_red.detach().cpu()
    blue_cpu = P_blue.detach().cpu()
    n = red_cpu.shape[0]
    C = compute_cost_matrix_l2(red_cpu, blue_cpu)
    a = np.full(n, 1.0 / n, dtype=np.float64)
    b = np.full(n, 1.0 / n, dtype=np.float64)

    for _ in range(WARMUP_RUNS):
        plan = ot.emd(a, b, C, numItermax=10**6)
        del plan

    times_ms = []
    costs = []
    for _ in range(TIMED_RUNS):
        t0 = time.perf_counter()
        plan = ot.emd(a, b, C, numItermax=10**6)
        t1 = time.perf_counter()
        match_B = matching_from_plan(plan)
        costs.append(average_l2_matching_cost(red_cpu, blue_cpu, match_B))
        times_ms.append((t1 - t0) * 1000.0)
        del plan, match_B

    return statistics.median(times_ms), statistics.median(costs)


def benchmark_exact(P_red, P_blue):
    return benchmark_exact_l2(P_red, P_blue)


def benchmark_proxy_exact(P_red_gpu, P_blue_gpu, device):
    if P_red_gpu.shape[0] > 10_000:
        raise RuntimeError("Proxy-Exact OT skipped for N > 10,000")
    if ot is None:
        raise RuntimeError("POT is not installed; exact solver unavailable.")

    n = P_red_gpu.shape[0]
    a = np.full(n, 1.0 / n, dtype=np.float64)
    b = np.full(n, 1.0 / n, dtype=np.float64)
    red_cpu = P_red_gpu.detach().cpu()
    blue_cpu = P_blue_gpu.detach().cpu()

    clustering_times_ms = []
    emd_times_ms = []
    costs = []

    for _ in range(WARMUP_RUNS):
        empty_cache_if_cuda(device)
        cluster_engine = SimpleClustering(
            epsilon=EPSILON,
            tile_size=BATCH_SIZE,
        )
        clustering = cluster_engine.run(P_red_gpu, P_blue_gpu)
        C_proxy = build_proxy_cost_matrix(clustering, n, device)
        plan = ot.emd(a, b, C_proxy, numItermax=10**6)
        del cluster_engine, clustering, C_proxy, plan

    for _ in range(TIMED_RUNS):
        empty_cache_if_cuda(device)
        cluster_engine = SimpleClustering(
            epsilon=EPSILON,
            tile_size=BATCH_SIZE,
        )

        synchronize_if_cuda(device)
        t_cluster0 = time.perf_counter()
        clustering = cluster_engine.run(P_red_gpu, P_blue_gpu)
        synchronize_if_cuda(device)
        t_cluster1 = time.perf_counter()

        C_proxy = build_proxy_cost_matrix(clustering, n, device)

        t0 = time.perf_counter()
        plan = ot.emd(a, b, C_proxy, numItermax=10**6)
        t1 = time.perf_counter()

        match_B = matching_from_plan(plan)
        clustering_times_ms.append((t_cluster1 - t_cluster0) * 1000.0)
        emd_times_ms.append((t1 - t0) * 1000.0)
        costs.append(average_l2_matching_cost(red_cpu, blue_cpu, match_B))

        del cluster_engine, clustering, C_proxy, plan, match_B

    print(
        f"    Clustering time (excluded): {statistics.median(clustering_times_ms):.1f} ms",
        flush=True,
    )
    return statistics.median(emd_times_ms), statistics.median(costs)


def result_na():
    return {"time_ms": math.nan, "cost": math.nan, "status": "fail"}


def run_exact(P_red, P_blue, device):
    try:
        time_ms, cost = benchmark_exact(P_red, P_blue)
        return {"time_ms": time_ms, "cost": cost, "status": "success"}
    except Exception as exc:
        print(f"Warning: Exact failed: {exc}", flush=True)
        return result_na()


def run_proxy_exact(P_red_gpu, P_blue_gpu, device):
    try:
        time_ms, cost = benchmark_proxy_exact(P_red_gpu, P_blue_gpu, device)
        return {"time_ms": time_ms, "cost": cost, "status": "success"}
    except Exception as exc:
        print(f"Warning: ProxyExact failed: {exc}", flush=True)
        empty_cache_if_cuda(device)
        return result_na()


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
        "exact_cost": 16,
        "proxy_cost": 17,
        "ratio": 7,
    }
    headers = [
        ("N", col_widths["n"], ">"),
        ("Exact Time", col_widths["exact_time"], ">"),
        ("ProxyExact Time", col_widths["proxy_time"], ">"),
        ("Exact Cost", col_widths["exact_cost"], ">"),
        ("ProxyExact Cost", col_widths["proxy_cost"], ">"),
        ("Ratio", col_widths["ratio"], ">"),
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
            f"{fmt_ratio(ratio):>{col_widths['ratio']}}",
        ]
        print(" | ".join(cells))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    print(f"Device: {device}  epsilon={EPSILON}  batch_size={BATCH_SIZE}")
    print("Exact OT skipped for N > 10,000 (O(N^3) cost matrix)")
    print(f"GloVe cache: {NPY_PATH}")

    if not NPY_PATH.exists():
        print("ERROR: glove.6B.300d.npy not found. Run download_glove.py first.")
        return

    rows = []
    for n in N_VALUES:
        print(f"\nPreparing GloVe N={n:,}", flush=True)
        try:
            P_red_cpu, P_blue_cpu, _diameter = load_glove_pair(n, seed=SEED)
            P_red_gpu = P_red_cpu.to(device)
            P_blue_gpu = P_blue_cpu.to(device)
        except Exception as exc:
            print(f"  Data loading failed: {exc}", flush=True)
            continue

        print("  Running Exact OT...", flush=True)
        exact_result = run_exact(P_red_cpu, P_blue_cpu, device)

        print("  Running Proxy-Exact OT...", flush=True)
        proxy_result = run_proxy_exact(P_red_gpu, P_blue_gpu, device)

        rows.append({"n": n, "exact": exact_result, "proxy_exact": proxy_result})

        del P_red_cpu, P_blue_cpu, P_red_gpu, P_blue_gpu
        empty_cache_if_cuda(device)

    print_results_table(rows)


if __name__ == "__main__":
    main()
