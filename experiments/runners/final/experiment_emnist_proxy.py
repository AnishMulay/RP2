#!/usr/bin/env python3

import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torchvision


BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    import ot
except ImportError:
    ot = None

from clustered_push_relabel.clustering.simple_l1 import SimpleL1Clustering


N_VALUES = [1_000, 5_000, 10_000]
EPSILON = 0.01
BATCH_SIZE = 512
SEED = 42
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


def build_proxy_cost_matrix(clustering, N, device):
    """
    Build the full N x N float proxy cost matrix using SimpleL1Clustering output.

    For each pair (b, a):
      - If a is in adj(b): use adj_dist_float[entry]  (exact stored distance)
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


def compute_cost_matrix_l1(red, blue):
    X = torch.as_tensor(red, dtype=torch.float64).contiguous()
    Y = torch.as_tensor(blue, dtype=torch.float64).contiguous()
    return torch.cdist(X, Y, p=1).numpy()


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
    C = compute_cost_matrix_l1(red_cpu, blue_cpu)

    t0 = time.perf_counter()
    plan = ot.emd(a, b, C, numItermax=10**6)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    matching = torch.from_numpy(plan.argmax(axis=0).astype(np.int64))
    matched = red_cpu[matching.to(dtype=torch.long)]
    cost = (blue_cpu - matched).abs().sum(dim=1).mean().item()
    return elapsed_ms, cost


def benchmark_proxy_exact(P_red, P_blue, device):
    if P_red.shape[0] > 10_000:
        raise RuntimeError("Proxy-Exact OT skipped: N > 10,000")
    if ot is None:
        raise RuntimeError("POT not installed")

    # No simple_l1_copy variant exists; SimpleL1Clustering already preserves DR.
    cluster_engine = SimpleL1Clustering(
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
    cost = (P_blue - matched).abs().sum(dim=1).mean().item()
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
        ("Exact Avg L1 Cost", col_widths["exact_cost"], ">"),
        ("ProxyExact Avg L1 Cost", col_widths["proxy_cost"], ">"),
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
