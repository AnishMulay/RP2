#!/usr/bin/env python3

import gzip
import math
import pathlib
import pickle
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

from clustered_push_relabel.clustering.simple_precomputed import (
    SimplePrecomputedClustering,
)


DATA_DIR = BASE_DIR / "data" / "cifar_sift"
TRAIN_DESC_PATH = DATA_DIR / "cifar10_sift_train.pkl.gz"
TEST_DESC_PATH = DATA_DIR / "cifar10_sift_test.pkl.gz"

N_VALUES = [500, 1_000, 2_000, 5_000]
EPSILON = 0.01
BATCH_SIZE = 512
SEED = 42
WARMUP_RUNS = 0
TIMED_RUNS = 1
EXACT_N_LIMIT = 2_000


def synchronize_if_cuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def empty_cache_if_cuda(device):
    if device.type == "cuda":
        torch.cuda.empty_cache()


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


def load_cifar_sift_descriptors(path):
    """
    Load gzip-pickled SIFT descriptor list.
    Returns a Python list of numpy arrays, one per image, each (K_i, 128).
    """
    with gzip.open(path, "rb") as f:
        descriptors = pickle.load(f)
    print(
        f"  Loaded {len(descriptors):,} descriptor sets from {path.name}",
        flush=True,
    )
    return descriptors


def sample_image_pair(all_descriptors, n_samples, seed):
    """
    Sample two non-overlapping sets of n_samples images.
    """
    if len(all_descriptors) < 2 * n_samples:
        raise ValueError(
            f"Need at least {2 * n_samples:,} images, got "
            f"{len(all_descriptors):,}."
        )
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(all_descriptors))
    red_idx = perm[:n_samples]
    blue_idx = perm[n_samples : 2 * n_samples]
    red_descs = [all_descriptors[i] for i in red_idx]
    blue_descs = [all_descriptors[i] for i in blue_idx]
    return red_descs, blue_descs


def _move_descriptor_list_to_device(descs, device):
    return [
        torch.as_tensor(desc, dtype=torch.float32, device=device).contiguous()
        for desc in descs
    ]


def _chamfer_distance(P, Q):
    dists = torch.cdist(P, Q, compute_mode="use_mm_for_euclid_dist_if_necessary")
    fwd = dists.min(dim=1).values.mean()
    bwd = dists.min(dim=0).values.mean()
    return fwd + bwd


def compute_chamfer_matrix(descs_A, descs_B, device, tile_size=50):
    """
    Compute the full Chamfer distance matrix between two sets of images.

    Returns D [N, N] float32 tensor on `device` where D[b, a] = chamfer(B[b], A[a]).
    """
    n = len(descs_A)
    if len(descs_B) != n:
        raise ValueError("descs_A and descs_B must have the same length")

    descs_A_dev = _move_descriptor_list_to_device(descs_A, device)
    descs_B_dev = _move_descriptor_list_to_device(descs_B, device)
    D = torch.empty((n, n), dtype=torch.float32, device=device)

    for start in range(0, n, tile_size):
        print(
            f"    Computing Chamfer [blue->red]: {start}/{n} rows done...",
            flush=True,
        )
        end = min(start + tile_size, n)
        for b in range(start, end):
            P = descs_B_dev[b]
            for a in range(n):
                D[b, a] = _chamfer_distance(P, descs_A_dev[a])

    synchronize_if_cuda(device)
    print("    Computing Chamfer [blue->red]: complete.", flush=True)
    return D


def compute_chamfer_matrix_symmetric(descs, device, tile_size=50):
    """
    Compute the symmetric N x N Chamfer matrix for a single set of images.
    """
    n = len(descs)
    descs_dev = _move_descriptor_list_to_device(descs, device)
    D = torch.zeros((n, n), dtype=torch.float32, device=device)

    for start in range(0, n, tile_size):
        print(
            f"    Computing Chamfer [red->red]: {start}/{n} rows done...",
            flush=True,
        )
        end = min(start + tile_size, n)
        for i in range(start, end):
            P = descs_dev[i]
            for j in range(i + 1, n):
                value = _chamfer_distance(P, descs_dev[j])
                D[i, j] = value
                D[j, i] = value

    synchronize_if_cuda(device)
    print("    Computing Chamfer [red->red]: complete.", flush=True)
    return D


def normalize_chamfer_matrices(D_blue_to_red, D_red_to_red):
    """
    Normalize both matrices by their joint maximum value.
    """
    diameter = max(
        float(torch.maximum(D_blue_to_red.max(), D_red_to_red.max()).item()),
        1e-6,
    )
    return D_blue_to_red / diameter, D_red_to_red / diameter, diameter


def build_proxy_cost_matrix(clustering, N, device):
    """
    Build the full N x N float proxy cost matrix from
    SimplePrecomputedClustering output.

    For each pair (b, a):
      If a in adj(b): adj_dist_float[entry]
      Otherwise:      d_min_b[b] + DR[nearest_s[b], a]
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


def benchmark_exact(D_blue_to_red_norm, device):
    """
    Run ot.emd with the true normalized Chamfer distance matrix.
    """
    n = D_blue_to_red_norm.shape[0]
    if n > EXACT_N_LIMIT:
        raise RuntimeError(f"Exact OT skipped for N > {EXACT_N_LIMIT:,}")
    if ot is None:
        raise RuntimeError("POT is not installed; exact solver unavailable.")

    D_cpu = D_blue_to_red_norm.detach().cpu()
    a = np.full(n, 1.0 / n, dtype=np.float64)
    b = np.full(n, 1.0 / n, dtype=np.float64)

    # POT expects rows for the first marginal. We transpose so plan.argmax(axis=0)
    # yields the matched red index for each blue image.
    C = D_cpu.to(torch.float64).numpy().T

    for _ in range(WARMUP_RUNS):
        plan = ot.emd(a, b, C, numItermax=10**6)
        del plan

    times_ms = []
    costs = []
    blue_indices = torch.arange(n, dtype=torch.long)
    for _ in range(TIMED_RUNS):
        t0 = time.perf_counter()
        plan = ot.emd(a, b, C, numItermax=10**6)
        t1 = time.perf_counter()
        matching = torch.from_numpy(plan.argmax(axis=0).astype(np.int64, copy=False))
        cost = D_cpu[blue_indices, matching].mean().item()
        times_ms.append((t1 - t0) * 1000.0)
        costs.append(cost)
        del plan, matching

    return statistics.median(times_ms), statistics.median(costs)


def benchmark_proxy_exact(D_blue_to_red_norm, D_red_to_red_norm, device):
    """
    Run SimplePrecomputedClustering then ot.emd with proxy Chamfer distances.
    """
    n = D_blue_to_red_norm.shape[0]
    if n > EXACT_N_LIMIT:
        raise RuntimeError(f"Proxy-Exact OT skipped for N > {EXACT_N_LIMIT:,}")
    if ot is None:
        raise RuntimeError("POT is not installed; exact solver unavailable.")

    a = np.full(n, 1.0 / n, dtype=np.float64)
    b = np.full(n, 1.0 / n, dtype=np.float64)
    D_cpu = D_blue_to_red_norm.detach().cpu()
    blue_indices = torch.arange(n, dtype=torch.long)

    for _ in range(WARMUP_RUNS):
        empty_cache_if_cuda(device)
        cluster_engine = SimplePrecomputedClustering(
            epsilon=EPSILON,
            tile_size=BATCH_SIZE,
        )
        clustering = cluster_engine.run(D_red_to_red_norm, D_blue_to_red_norm)
        C_proxy = build_proxy_cost_matrix(clustering, n, device)
        plan = ot.emd(a, b, C_proxy.T, numItermax=10**6)
        del cluster_engine, clustering, C_proxy, plan

    clustering_times_ms = []
    emd_times_ms = []
    costs = []
    for _ in range(TIMED_RUNS):
        empty_cache_if_cuda(device)
        cluster_engine = SimplePrecomputedClustering(
            epsilon=EPSILON,
            tile_size=BATCH_SIZE,
        )

        synchronize_if_cuda(device)
        t_cluster0 = time.perf_counter()
        clustering = cluster_engine.run(D_red_to_red_norm, D_blue_to_red_norm)
        synchronize_if_cuda(device)
        t_cluster1 = time.perf_counter()
        clustering_times_ms.append((t_cluster1 - t_cluster0) * 1000.0)

        C_proxy = build_proxy_cost_matrix(clustering, n, device)

        t0 = time.perf_counter()
        plan = ot.emd(a, b, C_proxy.T, numItermax=10**6)
        t1 = time.perf_counter()

        matching = torch.from_numpy(plan.argmax(axis=0).astype(np.int64, copy=False))
        cost = D_cpu[blue_indices, matching].mean().item()
        emd_times_ms.append((t1 - t0) * 1000.0)
        costs.append(cost)

        del cluster_engine, clustering, C_proxy, plan, matching

    print(
        f"    Clustering time (excluded): "
        f"{statistics.median(clustering_times_ms):.1f} ms",
        flush=True,
    )
    return statistics.median(emd_times_ms), statistics.median(costs)


def run_exact(D_blue_to_red_norm, device):
    try:
        time_ms, cost = benchmark_exact(D_blue_to_red_norm, device)
        return {"time_ms": time_ms, "cost": cost, "status": "success"}
    except Exception as exc:
        print(f"Warning: Exact failed: {exc}", flush=True)
        return {"time_ms": math.nan, "cost": math.nan, "status": "fail"}


def run_proxy_exact(D_blue_to_red_norm, D_red_to_red_norm, device):
    try:
        time_ms, cost = benchmark_proxy_exact(
            D_blue_to_red_norm,
            D_red_to_red_norm,
            device,
        )
        return {"time_ms": time_ms, "cost": cost, "status": "success"}
    except Exception as exc:
        print(f"Warning: ProxyExact failed: {exc}", flush=True)
        empty_cache_if_cuda(device)
        return {"time_ms": math.nan, "cost": math.nan, "status": "fail"}


def compute_cost_ratio(exact_cost, proxy_cost):
    if math.isnan(exact_cost) or math.isnan(proxy_cost) or exact_cost == 0.0:
        return math.nan
    return proxy_cost / exact_cost


def print_results_table(rows):
    col_widths = {
        "n": 7,
        "exact_time": 14,
        "proxy_time": 19,
        "exact_cost": 12,
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

    print(header_line, flush=True)
    print(separator, flush=True)
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
        print(" | ".join(cells), flush=True)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    print("=" * 60, flush=True)
    print("Experiment: CIFAR-10 SIFT Chamfer Proxy Quality", flush=True)
    print("=" * 60, flush=True)
    print(f"Device  : {device}", flush=True)
    print(f"Epsilon : {EPSILON}  batch_size={BATCH_SIZE}", flush=True)
    print(f"Exact OT: skipped for N > {EXACT_N_LIMIT:,}", flush=True)

    for path in [TRAIN_DESC_PATH]:
        if not path.exists():
            print(f"ERROR: {path.name} not found.", flush=True)
            print("Run download_cifar_sift.py first.", flush=True)
            return

    print("\n[Data] Loading SIFT descriptors...", flush=True)
    all_descriptors = load_cifar_sift_descriptors(TRAIN_DESC_PATH)
    print(
        f"[Data] Total images available: {len(all_descriptors):,}",
        flush=True,
    )

    rows = []
    for n in N_VALUES:
        print(f"\n{'=' * 40}", flush=True)
        print(f"N = {n:,}", flush=True)
        print(f"{'=' * 40}", flush=True)

        print(f"  [1/4] Sampling {n:,} red and {n:,} blue images...", flush=True)
        try:
            red_descs, blue_descs = sample_image_pair(all_descriptors, n, SEED)
        except Exception as exc:
            print(f"  Sampling failed: {exc}", flush=True)
            continue

        print(
            f"  [2/4] Computing Chamfer distance matrices on {device}...",
            flush=True,
        )
        synchronize_if_cuda(device)
        t0 = time.perf_counter()
        D_blue_to_red = compute_chamfer_matrix(blue_descs, red_descs, device)
        D_red_to_red = compute_chamfer_matrix_symmetric(red_descs, device)
        synchronize_if_cuda(device)
        chamfer_time = (time.perf_counter() - t0) * 1000.0
        print(
            f"  Chamfer matrices computed in {chamfer_time:.1f} ms",
            flush=True,
        )

        print("  [3/4] Normalizing...", flush=True)
        D_br_norm, D_rr_norm, diameter = normalize_chamfer_matrices(
            D_blue_to_red,
            D_red_to_red,
        )
        print(f"  Diameter (max Chamfer): {diameter:.4f}", flush=True)
        del D_blue_to_red, D_red_to_red

        print("  [4/4] Running solvers...", flush=True)
        print("    Running Exact OT...", flush=True)
        exact_result = run_exact(D_br_norm, device)

        print("    Running Proxy-Exact OT...", flush=True)
        proxy_result = run_proxy_exact(D_br_norm, D_rr_norm, device)

        rows.append({"n": n, "exact": exact_result, "proxy_exact": proxy_result})

        del D_br_norm, D_rr_norm
        empty_cache_if_cuda(device)

    print(f"\n\n{'=' * 60}", flush=True)
    print("RESULTS", flush=True)
    print(f"{'=' * 60}", flush=True)
    print_results_table(rows)


if __name__ == "__main__":
    main()
