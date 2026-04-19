#!/usr/bin/env python3

import math
import os
import pathlib
import sys
import time

import numpy as np
import torch


BASE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    import ot
except ImportError:
    ot = None

from clustered_push_relabel.clustering.simple import SimpleClustering


N_VALUES = list(range(1_000, 11_000, 1_000))
EPSILON = 0.01
SYNTHETIC_DIM = 2
SEED = 42
BATCH_SIZE = 512
WARMUP_RUNS = 0
TIMED_RUNS = 1
SAMPLE_FACTOR = 1.0


def resolve_mnist_paths(data_dir=None):
    candidate_dirs = []
    if data_dir:
        candidate_dirs.append(pathlib.Path(data_dir))

    env_data_dir = os.environ.get("MNIST_DATA_DIR")
    if env_data_dir:
        candidate_dirs.append(pathlib.Path(env_data_dir))

    candidate_dirs.extend([
        BASE_DIR / "data",
        BASE_DIR / "experiments" / "data",
        pathlib.Path.cwd() / "data",
        pathlib.Path.cwd() / "experiments" / "data",
    ])

    seen = set()
    deduped_dirs = []
    for directory in candidate_dirs:
        resolved = directory.expanduser().resolve()
        if resolved not in seen:
            deduped_dirs.append(resolved)
            seen.add(resolved)

    for directory in deduped_dirs:
        img_path = directory / "train-images-idx3-ubyte.gz"
        lbl_path = directory / "train-labels-idx1-ubyte.gz"
        if img_path.is_file() and lbl_path.is_file():
            return img_path, lbl_path

    searched = ", ".join(str(p) for p in deduped_dirs)
    raise FileNotFoundError(
        f"MNIST data files not found. Searched: {searched}. "
        "Expected train-images-idx3-ubyte.gz and train-labels-idx1-ubyte.gz."
    )


def load_mnist_flat(n_samples, seed=0, data_dir=None):
    import gzip

    img_path, lbl_path = resolve_mnist_paths(data_dir=data_dir)
    with gzip.open(img_path, "rb") as f:
        images = np.frombuffer(f.read(), dtype=np.uint8, offset=16).reshape(-1, 784)
    with gzip.open(lbl_path, "rb") as f:
        _ = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
    images = images.astype(np.float32) / 255.0
    total = images.shape[0]
    if 2 * n_samples > total:
        raise ValueError(
            f"Not enough MNIST images to sample {2*n_samples} (only {total} available)."
        )
    rng = np.random.RandomState(seed)
    perm = rng.permutation(total)
    red = torch.from_numpy(images[perm[:n_samples]]).float()
    blue = torch.from_numpy(images[perm[n_samples:2*n_samples]]).float()
    return red, blue


def generate_synthetic_2d(n, device):
    red = torch.rand((n, SYNTHETIC_DIM), dtype=torch.float32)
    blue = torch.rand((n, SYNTHETIC_DIM), dtype=torch.float32)
    return red.to(device), blue.to(device)


def load_mnist_pair(n, device):
    red, blue = load_mnist_flat(n, seed=SEED)
    return red.to(device), blue.to(device)


def normalize_points(P_red, P_blue):
    all_points = torch.cat([P_red, P_blue], dim=0)
    mean = all_points.mean(dim=0, keepdim=True)
    centered = all_points - mean
    diameter = float(2.0 * centered.norm(dim=1).max().item())
    diameter = max(diameter, 1e-8)
    return P_red / diameter, P_blue / diameter, diameter


def synchronize_if_cuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def empty_cache_if_cuda(device):
    if device.type == "cuda":
        torch.cuda.empty_cache()


def compute_cost_matrix_L2(red, blue):
    X = torch.as_tensor(red, dtype=torch.float64).contiguous()
    Y = torch.as_tensor(blue, dtype=torch.float64).contiguous()
    return torch.cdist(X, Y, p=2).cpu().numpy()


def build_proxy_matrix(clustering, N):
    """
    Proxy cost for (b, a):
      - If a is in b's adjacency list: exact stored distance
      - Otherwise: d_min_b[b] + DR[nearest_s[b], a]  (triangle through nearest sample)
    Returns a CPU float64 numpy array of shape (N_red x N_blue).
    """
    device = clustering["DR"].device
    DR = clustering["DR"]                    # (S, N_red) float
    d_min_b = clustering["d_min_b"]          # (N_blue,) float
    nearest_s = clustering["nearest_s"]      # (N_blue,) int64
    adj_ptr = clustering["adj_ptr"]          # (N_blue+1,) int64
    adj_col = clustering["adj_col"]          # (M,) int64  — red indices
    adj_dist_float = clustering["adj_dist_float"]  # (M,) float

    # (N_blue, N_red) triangle proxy, then transpose to (N_red, N_blue)
    C = (d_min_b.unsqueeze(1) + DR[nearest_s, :]).T

    if adj_col.numel() > 0:
        b_indices = torch.repeat_interleave(
            torch.arange(N, device=device, dtype=torch.long),
            adj_ptr[1:] - adj_ptr[:-1],
        )
        C[adj_col, b_indices] = adj_dist_float

    return C.cpu().to(torch.float64).numpy()


def matching_from_plan(plan):
    return torch.from_numpy(plan.argmax(axis=0).astype(np.int64, copy=False))


def matching_avg_cost(P_red, P_blue, match_B):
    match_B = match_B.to(device=P_red.device, dtype=torch.long)
    matched_red = P_red[match_B]
    return torch.norm(P_blue - matched_red, p=2, dim=1).mean().item()


def benchmark_exact(P_red, P_blue):
    if ot is None:
        raise RuntimeError("POT is not installed; exact solver unavailable.")

    red_cpu = P_red.detach().cpu()
    blue_cpu = P_blue.detach().cpu()
    n = red_cpu.shape[0]
    C = compute_cost_matrix_L2(red_cpu, blue_cpu)
    a = np.full(n, 1.0 / n, dtype=np.float64)
    b = np.full(n, 1.0 / n, dtype=np.float64)

    for _ in range(WARMUP_RUNS):
        plan = ot.emd(a, b, C, numItermax=10**6)
        del plan

    t0 = time.perf_counter()
    plan = ot.emd(a, b, C, numItermax=10**6)
    t1 = time.perf_counter()

    avg_cost = matching_avg_cost(red_cpu, blue_cpu, matching_from_plan(plan))
    del plan, C
    return (t1 - t0) * 1000.0, avg_cost


def benchmark_proxy_exact(P_red, P_blue, device):
    if ot is None:
        raise RuntimeError("POT is not installed; exact solver unavailable.")

    P_red_norm, P_blue_norm, diameter = normalize_points(P_red, P_blue)
    N = P_red_norm.shape[0]

    cluster_engine = SimpleClustering(
        epsilon=EPSILON, tile_size=BATCH_SIZE, sample_factor=SAMPLE_FACTOR
    )
    clustering = cluster_engine.run(P_red_norm, P_blue_norm)
    synchronize_if_cuda(device)

    C = build_proxy_matrix(clustering, N)
    del clustering, cluster_engine

    a = np.full(N, 1.0 / N, dtype=np.float64)
    b = np.full(N, 1.0 / N, dtype=np.float64)

    t0 = time.perf_counter()
    plan = ot.emd(a, b, C, numItermax=10**6)
    t1 = time.perf_counter()

    match_B = matching_from_plan(plan)
    avg_cost = matching_avg_cost(P_red_norm, P_blue_norm, match_B) * diameter
    del plan, match_B, C
    return (t1 - t0) * 1000.0, avg_cost


def run_exact(P_red, P_blue):
    try:
        time_ms, avg_cost = benchmark_exact(P_red, P_blue)
        return {"time_ms": time_ms, "avg_cost": avg_cost, "status": "success"}
    except Exception as exc:
        print(f"  Warning: Exact failed: {exc}", flush=True)
        return {"time_ms": math.nan, "avg_cost": math.nan, "status": "fail"}


def run_proxy_exact(P_red, P_blue, device):
    try:
        time_ms, avg_cost = benchmark_proxy_exact(P_red, P_blue, device)
        return {"time_ms": time_ms, "avg_cost": avg_cost, "status": "success"}
    except Exception as exc:
        print(f"  Warning: ProxyExact failed: {exc}", flush=True)
        empty_cache_if_cuda(device)
        return {"time_ms": math.nan, "avg_cost": math.nan, "status": "fail"}


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


def print_results_table(rows):
    col_widths = {
        "dataset": 11,
        "n": 7,
        "exact_time": 14,
        "proxy_time": 14,
        "exact_cost": 16,
        "proxy_cost": 16,
    }
    headers = [
        ("Dataset",        col_widths["dataset"],    "<"),
        ("N",              col_widths["n"],           ">"),
        ("Exact Time",     col_widths["exact_time"],  ">"),
        ("Proxy Time",     col_widths["proxy_time"],  ">"),
        ("Exact Avg Cost", col_widths["exact_cost"],  ">"),
        ("Proxy Avg Cost", col_widths["proxy_cost"],  ">"),
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
        proxy = row["proxy"]
        cells = [
            f"{row['dataset']:<{col_widths['dataset']}}",
            f"{row['n']:>{col_widths['n']},}",
            f"{fmt_time(exact['time_ms']):>{col_widths['exact_time']}}",
            f"{fmt_time(proxy['time_ms']):>{col_widths['proxy_time']}}",
            f"{fmt_cost(exact['avg_cost']):>{col_widths['exact_cost']}}",
            f"{fmt_cost(proxy['avg_cost']):>{col_widths['proxy_cost']}}",
        ]
        print(" | ".join(cells))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    datasets = [("Synthetic", generate_synthetic_2d)]
    try:
        resolve_mnist_paths()
        datasets.append(("MNIST", load_mnist_pair))
    except FileNotFoundError as exc:
        print(f"Warning: {exc} Skipping MNIST.", flush=True)

    print(f"Experiment 11: Exact vs Proxy-Exact (ot.emd with proxy distances)")
    print(f"Device: {device}  epsilon={EPSILON}  batch_size={BATCH_SIZE}")
    print(f"N values: {', '.join(f'{n:,}' for n in N_VALUES)}")
    print(f"Warmup runs: {WARMUP_RUNS}  timed runs: {TIMED_RUNS}")

    rows = []
    for dataset_name, loader in datasets:
        for n in N_VALUES:
            print(f"\nPreparing {dataset_name} N={n:,}", flush=True)
            P_red, P_blue = loader(n, device)

            print("  Running Exact...", flush=True)
            exact_result = run_exact(P_red, P_blue)

            print("  Running ProxyExact...", flush=True)
            proxy_result = run_proxy_exact(P_red, P_blue, device)

            rows.append({
                "dataset": dataset_name,
                "n": n,
                "exact": exact_result,
                "proxy": proxy_result,
            })
            del P_red, P_blue
            empty_cache_if_cuda(device)

    print_results_table(rows)


if __name__ == "__main__":
    main()
