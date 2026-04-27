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

from clustered_push_relabel.clustering.simple import SimpleClustering
from clustered_push_relabel.solvers.simple_bipartite import SimpleGPUSolver


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


def benchmark_simple_glove(P_red, P_blue, device):
    for _ in range(WARMUP_RUNS):
        empty_cache_if_cuda(device)
        solver = SimpleGPUSolver(
            P_red,
            P_blue,
            epsilon=EPSILON,
            batch_size=BATCH_SIZE,
            verbose=False,
            diameter=1.0,
            clustering_class=SimpleClustering,
        )
        solver.solve()
        synchronize_if_cuda(device)
        del solver

    times_ms = []
    costs = []
    iterations_list = []
    for _ in range(TIMED_RUNS):
        empty_cache_if_cuda(device)
        synchronize_if_cuda(device)
        t0 = time.perf_counter()
        solver = SimpleGPUSolver(
            P_red,
            P_blue,
            epsilon=EPSILON,
            batch_size=BATCH_SIZE,
            verbose=False,
            diameter=1.0,
            clustering_class=SimpleClustering,
        )
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
    return {
        "time_ms": math.nan,
        "cost": math.nan,
        "iterations": math.nan,
        "status": "fail",
    }


def run_exact(P_red, P_blue, device):
    try:
        time_ms, cost = benchmark_exact_l2(P_red, P_blue)
        return {"time_ms": time_ms, "cost": cost, "status": "success"}
    except Exception as exc:
        print(f"Warning: Exact failed: {exc}", flush=True)
        return {"time_ms": math.nan, "cost": math.nan, "status": "fail"}


def run_simple(P_red, P_blue, device):
    try:
        time_ms, cost, iterations = benchmark_simple_glove(P_red, P_blue, device)
        return {
            "time_ms": time_ms,
            "cost": cost,
            "iterations": iterations,
            "status": "success",
        }
    except Exception as exc:
        print(f"Warning: Simple failed: {exc}", flush=True)
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
        "simple_time": 14,
        "exact_cost": 16,
        "simple_cost": 16,
        "simple_iters": 14,
    }
    headers = [
        ("N", col_widths["n"], ">"),
        ("Exact Time", col_widths["exact_time"], ">"),
        ("Simple Time", col_widths["simple_time"], ">"),
        ("Exact Cost", col_widths["exact_cost"], ">"),
        ("Simple Cost", col_widths["simple_cost"], ">"),
        ("Simple Iters", col_widths["simple_iters"], ">"),
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
        simple = row["simple"]
        cells = [
            f"{row['n']:>{col_widths['n']},}",
            f"{fmt_time(exact['time_ms']):>{col_widths['exact_time']}}",
            f"{fmt_time(simple['time_ms']):>{col_widths['simple_time']}}",
            f"{fmt_cost(exact['cost']):>{col_widths['exact_cost']}}",
            f"{fmt_cost(simple['cost']):>{col_widths['simple_cost']}}",
            f"{fmt_iter(simple['iterations']):>{col_widths['simple_iters']}}",
        ]
        print(" | ".join(cells))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    print(f"Device: {device}  epsilon={EPSILON}  batch_size={BATCH_SIZE}")
    print(f"GloVe cache: {NPY_PATH}")

    if not NPY_PATH.exists():
        print("ERROR: glove.6B.300d.npy not found. Run download_glove.py first.")
        return

    rows = []
    for n in N_VALUES:
        print(f"\nPreparing GloVe N={n:,}", flush=True)
        try:
            P_red, P_blue, _diameter = load_glove_pair(n, seed=SEED)
            P_red = P_red.to(device)
            P_blue = P_blue.to(device)
        except Exception as exc:
            print(f"  Data loading failed: {exc}", flush=True)
            continue

        print("  Running Exact OT...", flush=True)
        exact_result = run_exact(P_red, P_blue, device)

        print("  Running Simple solver...", flush=True)
        simple_result = run_simple(P_red, P_blue, device)

        rows.append({"n": n, "exact": exact_result, "simple": simple_result})

        del P_red, P_blue
        empty_cache_if_cuda(device)

    print_results_table(rows)


if __name__ == "__main__":
    main()
