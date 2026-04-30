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

from clustered_push_relabel.clustering.simple_three_level import ThreeLevelClustering
from clustered_push_relabel.solvers.three_level_bipartite import ThreeLevelGPUSolver


N_VALUES = [1_000, 5_000, 10_000, 50_000, 100_000, 200_000, 400_000]
EPSILON = 0.01
SEED = 42
BATCH_SIZE = 512
WARMUP_RUNS = 0
TIMED_RUNS = 1
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


def synchronize_if_cuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def empty_cache_if_cuda(device):
    if device.type == "cuda":
        torch.cuda.empty_cache()


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


def benchmark_three_level_sift(P_red, P_blue, device):
    for _ in range(WARMUP_RUNS):
        empty_cache_if_cuda(device)
        solver = ThreeLevelGPUSolver(
            P_red,
            P_blue,
            EPSILON,
            batch_size=BATCH_SIZE,
            verbose=False,
            diameter=1.0,
            clustering_class=ThreeLevelClustering,
        )
        solver.debug_audit = False
        synchronize_if_cuda(device)
        solver.solve()
        synchronize_if_cuda(device)
        del solver

    times_ms = []
    costs = []
    iterations_list = []
    for _ in range(TIMED_RUNS):
        empty_cache_if_cuda(device)
        solver = ThreeLevelGPUSolver(
            P_red,
            P_blue,
            EPSILON,
            batch_size=BATCH_SIZE,
            verbose=False,
            diameter=1.0,
            clustering_class=ThreeLevelClustering,
        )
        solver.debug_audit = False
        synchronize_if_cuda(device)
        t0 = time.perf_counter()
        solver.solve()
        synchronize_if_cuda(device)
        t1 = time.perf_counter()
        costs.append(average_l2_matching_cost(P_red, P_blue, solver.match_B))
        times_ms.append((t1 - t0) * 1000.0)
        iterations_list.append(solver.iterations)
        del solver

    return (
        statistics.median(times_ms),
        statistics.median(costs),
        statistics.median(iterations_list),
    )


def result_na():
    return {"time_ms": math.nan, "cost": math.nan, "iterations": math.nan, "status": "fail"}


def run_exact(P_red, P_blue, device):
    try:
        time_ms, cost = benchmark_exact_l2(P_red, P_blue)
        return {"time_ms": time_ms, "cost": cost, "status": "success"}
    except Exception as exc:
        print(f"Warning: Exact failed: {exc}", flush=True)
        return {"time_ms": math.nan, "cost": math.nan, "status": "fail"}


def run_three_level(P_red, P_blue, device):
    try:
        time_ms, cost, iterations = benchmark_three_level_sift(P_red, P_blue, device)
        return {"time_ms": time_ms, "cost": cost, "iterations": iterations, "status": "success"}
    except Exception as exc:
        print(f"Warning: ThreeLevel failed: {exc}", flush=True)
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


def fmt_iter(value):
    if not is_available(value):
        return "N/A"
    return f"{int(value):,}"


def print_results_table(rows):
    col_widths = {
        "n": 7,
        "exact_time": 14,
        "three_level_time": 18,
        "exact_cost": 19,
        "three_level_cost": 24,
        "three_level_iters": 14,
    }
    headers = [
        ("N", col_widths["n"], ">"),
        ("Exact Time", col_widths["exact_time"], ">"),
        ("ThreeLevel Time", col_widths["three_level_time"], ">"),
        ("Exact Avg L2 Cost", col_widths["exact_cost"], ">"),
        ("ThreeLevel Avg L2 Cost", col_widths["three_level_cost"], ">"),
        ("Iters", col_widths["three_level_iters"], ">"),
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
        three_level = row["three_level"]
        cells = [
            f"{row['n']:>{col_widths['n']},}",
            f"{fmt_time(exact['time_ms']):>{col_widths['exact_time']}}",
            f"{fmt_time(three_level['time_ms']):>{col_widths['three_level_time']}}",
            f"{fmt_cost(exact['cost']):>{col_widths['exact_cost']}}",
            f"{fmt_cost(three_level['cost']):>{col_widths['three_level_cost']}}",
            f"{fmt_iter(three_level['iterations']):>{col_widths['three_level_iters']}}",
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
        exact_result = run_exact(P_red, P_blue, device)

        print("  Running ThreeLevel solver...", flush=True)
        three_level_result = run_three_level(P_red, P_blue, device)

        rows.append({"n": n, "exact": exact_result, "three_level": three_level_result})

        del P_red, P_blue
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print_results_table(rows)


if __name__ == "__main__":
    main()
