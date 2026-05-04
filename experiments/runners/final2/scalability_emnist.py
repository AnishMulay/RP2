#!/usr/bin/env python3
"""
Standalone GPU scalability sweep on EMNIST image vectors.
"""

import csv
import gc
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

try:
    import torchvision
except ImportError:
    torchvision = None

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from clustered_push_relabel.solvers.simple_bipartite import SimpleGPUSolver
from clustered_push_relabel.solvers.three_level_bipartite import ThreeLevelGPUSolver

EPSILON = 0.01
BATCH_SIZE = 2048
MAX_ITERS = 999_999_999
SEED = 42
DIAMETER_TILE = 512
SPLITS = ("byclass", "letters")


def n_values():
    yield 10_000
    yield 50_000
    n = 100_000
    while True:
        yield n
        n += 50_000


def sync():
    torch.cuda.synchronize()


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()


def gpu_mem_gb():
    return (
        torch.cuda.memory_allocated() / 1024**3,
        torch.cuda.memory_reserved() / 1024**3,
    )


def candidate_roots():
    roots = [
        SCRIPT_DIR / "data",
        Path.cwd() / "data",
        Path.home() / "data",
        Path("/scratch"),
        Path("/datasets"),
    ]
    seen = set()
    for root in roots:
        key = str(root.expanduser())
        if key not in seen:
            seen.add(key)
            yield root.expanduser()


def load_emnist_dataset():
    if torchvision is None:
        raise RuntimeError("torchvision is not installed")

    errors = []
    for root in candidate_roots():
        for split in SPLITS:
            try:
                train = torchvision.datasets.EMNIST(
                    root=str(root), split=split, train=True, download=False
                )
                test = torchvision.datasets.EMNIST(
                    root=str(root), split=split, train=False, download=False
                )
                print(f"Loaded EMNIST split={split} from {root}", flush=True)
                return _flatten_emnist(train, test), split, root
            except Exception as exc:
                errors.append(f"{root} split={split}: {exc}")

    download_root = SCRIPT_DIR / "data"
    for split in SPLITS:
        try:
            train = torchvision.datasets.EMNIST(
                root=str(download_root), split=split, train=True, download=True
            )
            test = torchvision.datasets.EMNIST(
                root=str(download_root), split=split, train=False, download=True
            )
            print(f"Downloaded EMNIST split={split} to {download_root}", flush=True)
            return _flatten_emnist(train, test), split, download_root
        except Exception as exc:
            errors.append(f"{download_root} split={split} download=True: {exc}")

    raise RuntimeError("could not load EMNIST from fallback paths: " + " | ".join(errors[-6:]))


def _flatten_emnist(train, test):
    images = torch.cat([train.data, test.data], dim=0)
    images = images.reshape(-1, 28, 28).transpose(1, 2).reshape(-1, 784)
    return images.to(dtype=torch.float32).div_(255.0).contiguous()


def _diameter_with_tile(points, tile_size):
    max_dist = 0.0
    for start in range(0, points.shape[0], tile_size):
        end = min(start + tile_size, points.shape[0])
        dists = torch.cdist(
            points[start:end],
            points,
            p=2,
            compute_mode="use_mm_for_euclid_dist_if_necessary",
        )
        max_dist = max(max_dist, float(dists.max().item()))
        del dists
    return max_dist


def joint_diameter(A, B):
    points = torch.cat([A, B], dim=0)
    tile = min(DIAMETER_TILE, points.shape[0])
    while tile >= 32:
        try:
            return _diameter_with_tile(points, tile)
        except torch.cuda.OutOfMemoryError:
            cleanup()
            tile //= 2
    return _diameter_with_tile(points, 16)


def make_data(images_cpu, n, rng):
    total = images_cpu.shape[0]
    if total == 0:
        raise RuntimeError("EMNIST dataset contains no images")
    if 2 * n <= total:
        idx = rng.permutation(total)
        idx_a = idx[:n]
        idx_b = idx[n : 2 * n]
    else:
        idx_a = rng.integers(0, total, size=n)
        idx_b = rng.integers(0, total, size=n)
    A = images_cpu[torch.as_tensor(idx_a, dtype=torch.long)].to("cuda", non_blocking=True)
    B = images_cpu[torch.as_tensor(idx_b, dtype=torch.long)].to("cuda", non_blocking=True)
    diameter = joint_diameter(A, B)
    if diameter > 0.0:
        A = A / diameter
        B = B / diameter
    return A, B, diameter


def avg_matching_cost(A, B, match_B, diameter):
    match_B = match_B.to(device=A.device, dtype=torch.long)
    return torch.norm(B - A[match_B], p=2, dim=1).mean().item() * diameter


def run_solver(label, solver_cls, A, B, diameter):
    cleanup()
    solver = None
    try:
        sync()
        start = time.time()
        solver = solver_cls(
            A,
            B,
            epsilon=EPSILON,
            batch_size=BATCH_SIZE,
            verbose=False,
            diameter=1.0,
            max_iters=MAX_ITERS,
        )
        match_B = solver.solve()
        sync()
        elapsed = time.time() - start
        if match_B is None:
            match_B = solver.match_B
        cost = avg_matching_cost(A, B, match_B, diameter)
        print(f"[{label}] Time: {elapsed:.2f}s | Avg Cost: {cost:.5f}", flush=True)
        return {"status": "ok", "time": elapsed, "cost": cost}
    except torch.cuda.OutOfMemoryError:
        cleanup()
        print(f"[{label}] OOM", flush=True)
        return {"status": "oom", "time": math.nan, "cost": math.nan}
    except Exception as exc:
        cleanup()
        print(f"[{label}] ERROR: {exc}", flush=True)
        return {"status": "error", "time": math.nan, "cost": math.nan}
    finally:
        del solver
        cleanup()


def status_value(result, key):
    if result["status"] == "ok":
        return f"{result[key]:.5f}" if key == "cost" else f"{result[key]:.2f}"
    return result["status"].upper()


def print_summary(rows):
    headers = [
        "N",
        "2-Level Time (s)",
        "2-Level Avg Cost",
        "3-Level Time (s)",
        "3-Level Avg Cost",
    ]
    table = [
        [
            f"{row['n']:,}",
            status_value(row["sol2"], "time"),
            status_value(row["sol2"], "cost"),
            status_value(row["sol3"], "time"),
            status_value(row["sol3"], "cost"),
        ]
        for row in rows
    ]
    widths = [len(h) for h in headers]
    for row in table:
        widths = [max(w, len(v)) for w, v in zip(widths, row)]
    print("\nSummary", flush=True)
    print(" | ".join(f"{h:>{w}}" for h, w in zip(headers, widths)), flush=True)
    print("-+-".join("-" * w for w in widths), flush=True)
    for row in table:
        print(" | ".join(f"{v:>{w}}" for v, w in zip(row, widths)), flush=True)


def save_csv(rows):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = SCRIPT_DIR / f"scalability_emnist_results_{stamp}.csv"
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "N",
                "2-Level Time (s)",
                "2-Level Avg Cost",
                "3-Level Time (s)",
                "3-Level Avg Cost",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["n"],
                    status_value(row["sol2"], "time"),
                    status_value(row["sol2"], "cost"),
                    status_value(row["sol3"], "time"),
                    status_value(row["sol3"], "cost"),
                ]
            )
    print(f"\nSaved CSV: {path}", flush=True)


def main():
    assert torch.cuda.is_available(), "CUDA is required"
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    rng = np.random.default_rng(SEED)

    try:
        images_cpu, split, root = load_emnist_dataset()
    except Exception as exc:
        print(f"DATA ERROR: {exc}", flush=True)
        return
    print(f"EMNIST images: {images_cpu.shape[0]:,} split={split} root={root}", flush=True)

    rows = []
    sol2_active = True
    sol3_active = True

    for n in n_values():
        print(f"\n=== N = {n} ===", flush=True)
        alloc, reserved = gpu_mem_gb()
        print(f"GPU Memory: {alloc:.2f} GB allocated, {reserved:.2f} GB reserved", flush=True)

        try:
            A, B, diameter = make_data(images_cpu, n, rng)
        except torch.cuda.OutOfMemoryError:
            cleanup()
            print(f"DATA OOM while sampling or normalizing N={n:,}; stopping.", flush=True)
            break
        except Exception as exc:
            cleanup()
            print(f"DATA ERROR at N={n:,}: {exc}; stopping.", flush=True)
            break

        if sol2_active:
            sol2 = run_solver("2-Level", SimpleGPUSolver, A, B, diameter)
            if sol2["status"] == "oom":
                sol2_active = False
        else:
            sol2 = {"status": "oom", "time": math.nan, "cost": math.nan}
            print("[2-Level] OOM", flush=True)

        if sol3_active:
            sol3 = run_solver("3-Level", ThreeLevelGPUSolver, A, B, diameter)
            if sol3["status"] == "oom":
                sol3_active = False
        else:
            sol3 = {"status": "oom", "time": math.nan, "cost": math.nan}
            print("[3-Level] OOM", flush=True)

        row = {"n": n, "sol2": sol2, "sol3": sol3}
        rows.append(row)
        print(
            f"Row: {n:,} | {status_value(sol2, 'time')} | {status_value(sol2, 'cost')} | "
            f"{status_value(sol3, 'time')} | {status_value(sol3, 'cost')}",
            flush=True,
        )

        stop_for_error = sol2["status"] == "error" or sol3["status"] == "error"
        del A, B
        cleanup()
        if stop_for_error:
            print("Unrecoverable solver error recorded; stopping.", flush=True)
            break
        if not sol2_active and not sol3_active:
            print("Both solvers have OOMed; stopping.", flush=True)
            break

    print_summary(rows)
    save_csv(rows)


if __name__ == "__main__":
    main()
