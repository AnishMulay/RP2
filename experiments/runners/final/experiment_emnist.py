#!/usr/bin/env python3

import math
import pathlib
import statistics
import sys
import time
import warnings

import numpy as np
import torch
import torchvision


BASE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent.parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    import ot
except ImportError:
    ot = None

from clustered_push_relabel.clustering.simple_l1 import SimpleL1Clustering
from clustered_push_relabel.solvers.simple_bipartite import SimpleGPUSolver


N_VALUES = [1_000, 5_000, 10_000, 50_000, 100_000, 200_000]
EPSILON = 0.01
SEED = 42
BATCH_SIZE = 512
WARMUP_RUNS = 0
TIMED_RUNS = 1
EMNIST_SPLIT = "byclass"

DATA_DIR = BASE_DIR / "data"


def load_emnist_balanced(n_samples, seed, split):
    train_dataset = torchvision.datasets.EMNIST(
        root=str(DATA_DIR),
        split=split,
        train=True,
        download=False,
    )
    test_dataset = torchvision.datasets.EMNIST(
        root=str(DATA_DIR),
        split=split,
        train=False,
        download=False,
    )

    images = torch.cat([train_dataset.data, test_dataset.data], dim=0).cpu().numpy()
    labels = torch.cat([train_dataset.targets, test_dataset.targets], dim=0).cpu().numpy()

    images = images.reshape(-1, 28, 28).transpose(0, 2, 1).reshape(-1, 784)

    classes = np.unique(labels)
    num_classes = int(classes.size)
    samples_per_class = n_samples // num_classes
    if samples_per_class == 0:
        raise ValueError(
            f"Requested n_samples={n_samples} but EMNIST split '{split}' has "
            f"{num_classes} classes, so floor(n_samples / num_classes) == 0."
        )

    red_rng = np.random.RandomState(seed)
    blue_rng = np.random.RandomState(seed + 1)
    red_parts = []
    blue_parts = []

    for class_label in classes:
        class_indices = np.flatnonzero(labels == class_label).copy()
        needed = 2 * samples_per_class
        if class_indices.size < needed:
            warnings.warn(
                f"Skipping class {int(class_label)}: only {class_indices.size} "
                f"samples available, need {needed}.",
                stacklevel=2,
            )
            continue

        red_rng.shuffle(class_indices)
        chosen_indices = class_indices[:needed]
        red_indices = chosen_indices[:samples_per_class].copy()
        blue_indices = chosen_indices[samples_per_class:needed].copy()
        blue_rng.shuffle(blue_indices)

        red_parts.append(images[red_indices])
        blue_parts.append(images[blue_indices])

    if not red_parts or not blue_parts:
        raise ValueError(
            f"Unable to sample any EMNIST classes from split '{split}' with "
            f"{samples_per_class} samples per class."
        )

    red_images = np.concatenate(red_parts, axis=0)
    blue_images = np.concatenate(blue_parts, axis=0)

    if red_images.shape[0] != n_samples:
        warnings.warn(
            f"Collected {red_images.shape[0]} samples per set instead of the "
            f"requested {n_samples} because some classes were skipped.",
            stacklevel=2,
        )

    red_images = red_images.astype(np.float32, copy=False) / 255.0
    red_row_sums = red_images.sum(axis=1, keepdims=True)
    red_row_sums = np.maximum(red_row_sums, 1e-8)
    red_images = red_images / red_row_sums
    red_images = red_images / 2.0

    blue_images = blue_images.astype(np.float32, copy=False) / 255.0
    blue_row_sums = blue_images.sum(axis=1, keepdims=True)
    blue_row_sums = np.maximum(blue_row_sums, 1e-8)
    blue_images = blue_images / blue_row_sums
    blue_images = blue_images / 2.0

    red = torch.from_numpy(red_images).to(dtype=torch.float32)
    blue = torch.from_numpy(blue_images).to(dtype=torch.float32)
    return red, blue


def synchronize_if_cuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def empty_cache_if_cuda(device):
    if device.type == "cuda":
        torch.cuda.empty_cache()


def compute_cost_matrix_l1(red, blue):
    X = torch.as_tensor(red, dtype=torch.float64).contiguous()
    Y = torch.as_tensor(blue, dtype=torch.float64).contiguous()
    return torch.cdist(X, Y, p=1).cpu().numpy()


def matching_from_plan(plan):
    return torch.from_numpy(plan.argmax(axis=0).astype(np.int64, copy=False))


def average_l1_matching_cost(P_red, P_blue, matching):
    matched_red = P_red[matching.to(device=P_red.device, dtype=torch.long)]
    return (P_blue - matched_red).abs().sum(dim=1).mean().item()


def benchmark_exact_l1(P_red, P_blue):
    if P_red.shape[0] > 10_000:
        raise RuntimeError("Exact OT skipped for N > 10,000")
    if ot is None:
        raise RuntimeError("POT is not installed; exact solver unavailable.")

    red_cpu = P_red.detach().cpu()
    blue_cpu = P_blue.detach().cpu()
    n = red_cpu.shape[0]
    C = compute_cost_matrix_l1(red_cpu, blue_cpu)
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
        costs.append(average_l1_matching_cost(red_cpu, blue_cpu, match_B))
        times_ms.append((t1 - t0) * 1000.0)
        del plan, match_B

    return statistics.median(times_ms), statistics.median(costs)


def benchmark_simple_l1(P_red, P_blue, device):
    for _ in range(WARMUP_RUNS):
        empty_cache_if_cuda(device)
        solver = SimpleGPUSolver(
            P_red,
            P_blue,
            EPSILON,
            batch_size=BATCH_SIZE,
            verbose=False,
            clustering_class=SimpleL1Clustering,
        )
        synchronize_if_cuda(device)
        solver.solve()
        synchronize_if_cuda(device)
        del solver

    times_ms = []
    costs = []
    iterations_list = []
    for _ in range(TIMED_RUNS):
        empty_cache_if_cuda(device)
        solver = SimpleGPUSolver(
            P_red,
            P_blue,
            EPSILON,
            batch_size=BATCH_SIZE,
            verbose=False,
            clustering_class=SimpleL1Clustering,
        )
        synchronize_if_cuda(device)
        t0 = time.perf_counter()
        solver.solve()
        synchronize_if_cuda(device)
        t1 = time.perf_counter()
        costs.append(average_l1_matching_cost(P_red, P_blue, solver.match_B))
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
        time_ms, cost = benchmark_exact_l1(P_red, P_blue)
        return {"time_ms": time_ms, "cost": cost, "status": "success"}
    except Exception as exc:
        print(f"Warning: Exact failed: {exc}", flush=True)
        return {"time_ms": math.nan, "cost": math.nan, "status": "fail"}


def run_simple(P_red, P_blue, device):
    try:
        time_ms, cost, iterations = benchmark_simple_l1(P_red, P_blue, device)
        return {"time_ms": time_ms, "cost": cost, "iterations": iterations, "status": "success"}
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
        ("Exact Avg Cost", col_widths["exact_cost"], ">"),
        ("Simple Avg Cost", col_widths["simple_cost"], ">"),
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
    print(f"EMNIST split: {EMNIST_SPLIT}  data dir: {DATA_DIR}")

    rows = []
    for n in N_VALUES:
        print(f"\nPreparing EMNIST N={n:,}", flush=True)
        try:
            P_red, P_blue = load_emnist_balanced(n, seed=SEED, split=EMNIST_SPLIT)
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
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print_results_table(rows)


if __name__ == "__main__":
    main()
