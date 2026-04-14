#!/usr/bin/env python3

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

from clustered_push_relabel.clustering.color_aware_two_level import ColorAwareClustering
from clustered_push_relabel.clustering.simple import SimpleClustering
from clustered_push_relabel.clustering.two_level import FastGPUClustering


N_VALUES = [50_000, 100_000, 200_000]
EPSILON = 0.01
SYNTHETIC_DIM = 2
SEED = 42
BATCH_SIZE = 512
WARMUP_RUNS = 3
TIMED_RUNS = 5


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


def sync_cuda():
    torch.cuda.synchronize()


def benchmark_run(model, P_red, P_blue):
    for _ in range(WARMUP_RUNS):
        sync_cuda()
        result = model.run(P_red, P_blue)
        sync_cuda()
        del result

    times_ms = []
    for _ in range(TIMED_RUNS):
        sync_cuda()
        t0 = time.perf_counter()
        result = model.run(P_red, P_blue)
        sync_cuda()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)
        del result

    return statistics.median(times_ms)


def method_factories():
    return [
        (
            "FastGPUClustering",
            lambda: FastGPUClustering(epsilon=EPSILON, batch_size=BATCH_SIZE),
        ),
        (
            "ColorAwareClustering",
            lambda: ColorAwareClustering(epsilon=EPSILON, batch_size=BATCH_SIZE),
        ),
        ("SimpleClustering", lambda: SimpleClustering(tile_size=BATCH_SIZE)),
    ]


def print_table(rows):
    headers = [
        ("Dataset", 11),
        ("N", 7),
        ("FastGPUClustering", 19),
        ("ColorAwareClustering", 21),
        ("SimpleClustering", 18),
    ]
    header_line = " | ".join(
        f"{name:<{width}}" if name == "Dataset" else f"{name:>{width}}"
        for name, width in headers
    )
    separator = "-+-".join("-" * width for _, width in headers)
    print()
    print(header_line)
    print(separator)
    for row in rows:
        print(
            f"{row['dataset']:<11} | "
            f"{row['n']:>7,} | "
            f"{row['FastGPUClustering']:>16.1f} ms | "
            f"{row['ColorAwareClustering']:>18.1f} ms | "
            f"{row['SimpleClustering']:>15.1f} ms"
        )


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("experiment_6.py requires CUDA.")

    device = torch.device("cuda")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)

    datasets = [
        ("Synthetic", generate_synthetic_2d),
        # ("MNIST", load_mnist_pair),
    ]
    rows = []

    print(f"Device: {device}  epsilon={EPSILON}  batch_size={BATCH_SIZE}")
    print(f"Warmup runs: {WARMUP_RUNS}  timed runs: {TIMED_RUNS}")

    for dataset_name, loader in datasets:
        for n in N_VALUES:
            print(f"\nPreparing {dataset_name} N={n:,}", flush=True)
            P_red, P_blue = loader(n, device)
            row = {"dataset": dataset_name, "n": n}

            for method_name, factory in method_factories():
                print(f"  Timing {method_name}...", flush=True)
                torch.cuda.empty_cache()
                model = factory()
                row[method_name] = benchmark_run(model, P_red, P_blue)
                del model

            rows.append(row)
            del P_red, P_blue
            torch.cuda.empty_cache()

    print_table(rows)


if __name__ == "__main__":
    main()
