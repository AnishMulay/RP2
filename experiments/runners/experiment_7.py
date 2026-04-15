#!/usr/bin/env python3

import math
import os
import pathlib
import statistics
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

from clustered_push_relabel.solvers.color_aware_bipartite import ColorAwareTwoLevelSolver
from clustered_push_relabel.solvers.simple_bipartite import SimpleGPUSolver


N_VALUES = [1_000, 5_000, 10_000]
EPSILON = 0.01
SYNTHETIC_DIM = 2
SEED = 42
BATCH_SIZE = 512
WARMUP_RUNS = 1
TIMED_RUNS = 3
METHODS = ["Exact", "ColorAware", "Simple"]


def resolve_mnist_paths(data_dir=None):
    """Resolve MNIST image/label file paths from common repo locations."""
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

    searched = ", ".join(str(path) for path in deduped_dirs)
    raise FileNotFoundError(
        "MNIST data files not found. Searched: "
        f"{searched}. Expected train-images-idx3-ubyte.gz and "
        "train-labels-idx1-ubyte.gz."
    )


def load_mnist_flat(n_samples, seed=0, data_dir=None):
    """Load MNIST images, normalize pixels to [0,1], sample red and blue sets."""
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
            f"Not enough MNIST images to sample {2*n_samples} "
            f"(only {total} available)."
        )
    rng = np.random.RandomState(seed)
    perm = rng.permutation(total)
    red_idx = perm[:n_samples]
    blue_idx = perm[n_samples:2*n_samples]
    red = torch.from_numpy(images[red_idx]).float()
    blue = torch.from_numpy(images[blue_idx]).float()
    return red, blue


def generate_synthetic_2d(n, device):
    red = torch.rand((n, SYNTHETIC_DIM), dtype=torch.float32)
    blue = torch.rand((n, SYNTHETIC_DIM), dtype=torch.float32)
    P_red = red.to(device)
    P_blue = blue.to(P_red.device)
    return P_red, P_blue


def load_mnist_pair(n, device):
    red, blue = load_mnist_flat(n, seed=SEED)
    return red.to(device), blue.to(device)


def synchronize_if_cuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def empty_cache_if_cuda(device):
    if device.type == "cuda":
        torch.cuda.empty_cache()


def compute_cost_matrix_L2(red, blue):
    """Compute full Euclidean distance matrix between two point sets on CPU."""
    X = torch.as_tensor(red, dtype=torch.float64).contiguous()
    Y = torch.as_tensor(blue, dtype=torch.float64).contiguous()
    return torch.cdist(X, Y, p=2).cpu().numpy()


def matching_from_plan(plan):
    """Convert a POT red-by-blue transport plan into match_B[blue] = red."""
    return torch.from_numpy(plan.argmax(axis=0).astype(np.int64, copy=False))


def average_l2_matching_cost(P_red, P_blue, matching):
    matched_red = P_red[matching.to(device=P_red.device, dtype=torch.long)]
    return torch.norm(P_blue - matched_red, p=2, dim=1).mean().item()


def solver_matching(solver):
    if hasattr(solver, "match_B"):
        return solver.match_B
    return solver.MB


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


def benchmark_solver(P_red, P_blue, device, factory):
    for _ in range(WARMUP_RUNS):
        empty_cache_if_cuda(device)
        solver = factory()
        synchronize_if_cuda(device)
        solver.solve()
        synchronize_if_cuda(device)
        del solver

    times_ms = []
    costs = []
    for _ in range(TIMED_RUNS):
        empty_cache_if_cuda(device)
        solver = factory()
        synchronize_if_cuda(device)
        t0 = time.perf_counter()
        solver.solve()
        synchronize_if_cuda(device)
        t1 = time.perf_counter()
        costs.append(average_l2_matching_cost(P_red, P_blue, solver_matching(solver)))
        times_ms.append((t1 - t0) * 1000.0)
        del solver

    return statistics.median(times_ms), statistics.median(costs)


def result_na():
    return {"time_ms": math.nan, "cost": math.nan, "status": "fail"}


def run_method(method_name, P_red, P_blue, device):
    try:
        if method_name == "Exact":
            time_ms, cost = benchmark_exact(P_red, P_blue)
        elif method_name == "ColorAware":
            time_ms, cost = benchmark_solver(
                P_red,
                P_blue,
                device,
                lambda: ColorAwareTwoLevelSolver(
                    P_red, P_blue, EPSILON, metric="L2", verbose=True
                ),
            )
        elif method_name == "Simple":
            time_ms, cost = benchmark_solver(
                P_red,
                P_blue,
                device,
                lambda: SimpleGPUSolver(
                    P_red, P_blue, EPSILON, batch_size=BATCH_SIZE, verbose=True
                ),
            )
        else:
            raise ValueError(f"Unknown method: {method_name}")
        return {"time_ms": time_ms, "cost": cost, "status": "success"}
    except Exception as exc:
        print(f"Warning: {method_name} failed: {exc}", flush=True)
        empty_cache_if_cuda(device)
        return result_na()


def is_available(value):
    return value == value


def format_time(value):
    if not is_available(value):
        return "N/A"
    return f"{value:.1f} ms"


def format_cost(value):
    if not is_available(value):
        return "N/A"
    return f"{value:.4f}"


def print_table(rows, metric_key, title, formatter):
    headers = [("Dataset", 11), ("N", 7)]
    headers.extend((method_name, 13) for method_name in METHODS)
    print()
    print(title)
    header_line = " | ".join(
        f"{label:<{width}}" if label == "Dataset" else f"{label:>{width}}"
        for label, width in headers
    )
    separator = "-+-".join("-" * width for _, width in headers)
    print(header_line)
    print(separator)
    for row in rows:
        cells = [
            f"{row['dataset']:<11}",
            f"{row['n']:>7,}",
        ]
        cells.extend(
            f"{formatter(row['results'][method_name][metric_key]):>13}"
            for method_name in METHODS
        )
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

    print(f"Device: {device}  epsilon={EPSILON}  batch_size={BATCH_SIZE}")
    print(f"Warmup runs: {WARMUP_RUNS}  timed runs: {TIMED_RUNS}")

    rows = []
    for dataset_name, loader in datasets:
        for n in N_VALUES:
            print(f"\nPreparing {dataset_name} N={n:,}", flush=True)
            P_red, P_blue = loader(n, device)
            row = {
                "dataset": dataset_name,
                "n": n,
                "results": {},
            }

            for method_name in METHODS:
                print(f"  Running {method_name}...", flush=True)
                row["results"][method_name] = run_method(
                    method_name, P_red, P_blue, device
                )

            rows.append(row)
            del P_red, P_blue
            empty_cache_if_cuda(device)

    print_table(rows, "time_ms", "Time", format_time)
    print_table(rows, "cost", "Matching Cost (Average L2 Per Pair)", format_cost)


if __name__ == "__main__":
    main()
