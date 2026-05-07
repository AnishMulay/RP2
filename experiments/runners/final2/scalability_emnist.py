#!/usr/bin/env python3
"""
Standalone GPU scalability sweep on EMNIST image vectors.
"""

import csv
import gc
import argparse
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

from clustered_push_relabel.clustering.simple_l1 import SimpleL1Clustering
from clustered_push_relabel.clustering.simple_three_level_l1 import ThreeLevelL1Clustering
from clustered_push_relabel.solvers.simple_bipartite import SimpleGPUSolver
from clustered_push_relabel.solvers.three_level_bipartite import ThreeLevelGPUSolver

EPSILON = 0.01
MAX_ITERS = 999_999_999
SEED = 42
L1_DIAMETER = 2.0
SPLITS = ("byclass", "letters")
VALIDATION_SIZES = [1_000, 5_000, 10_000]


def scalability_n_values(start_n, step_n, max_n=None):
    n = start_n
    while max_n is None or n <= max_n:
        yield n
        n += step_n


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
    images = images.to(dtype=torch.float32).div_(255.0)
    sums = images.sum(dim=1, keepdim=True).clamp_min_(1e-8)
    images.div_(sums)
    return images.contiguous()


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
    A = A / L1_DIAMETER
    B = B / L1_DIAMETER
    return A, B, L1_DIAMETER


def avg_matching_cost(A, B, match_B, diameter):
    match_B = match_B.to(device=A.device, dtype=torch.long)
    return (B - A[match_B]).abs().sum(dim=1).mean().item() * diameter


def run_exact_solver(A, B, diameter):
    """
    Exact optimal transport cost via POT (Earth Mover's Distance).
    A, B: probability-normalized CUDA float32 tensors divided by L1 diameter.
    Returns average L1 matching cost in natural [0, 2] units.
    """
    try:
        import ot as pot
    except ImportError:
        print("[Exact] ERROR: POT library not installed (pip install POT)", flush=True)
        return {"status": "error", "cost": math.nan}

    try:
        sync()
        N = A.shape[0]
        A_cpu = A.cpu().numpy().astype(np.float64)
        B_cpu = B.cpu().numpy().astype(np.float64)

        C = np.empty((N, N), dtype=np.float64)
        dim = A_cpu.shape[1]
        block = max(1, min(N, int(32_000_000 // max(1, N * dim))))
        for start in range(0, N, block):
            end = min(start + block, N)
            diff = B_cpu[start:end, None, :] - A_cpu[None, :, :]
            C[start:end] = np.abs(diff).sum(axis=2)

        weights = np.ones(N, dtype=np.float64) / N
        avg_cost = float(pot.emd2(weights, weights, C)) * diameter

        print(f"[Exact]   Avg Cost: {avg_cost:.5f}", flush=True)
        return {"status": "ok", "cost": avg_cost}
    except Exception as exc:
        print(f"[Exact] ERROR: {exc}", flush=True)
        return {"status": "error", "cost": math.nan}


def run_solver(label, solver_cls, clustering_cls, A, B, diameter, batch_size):
    cleanup()
    solver = None
    t_cluster_start = None
    t_cluster_end = None
    t_solve_start = None
    t_solve_end = None

    try:
        sync()
        t_cluster_start = time.time()
        solver = solver_cls(
            A,
            B,
            epsilon=EPSILON,
            batch_size=batch_size,
            verbose=False,
            diameter=diameter,
            max_iters=MAX_ITERS,
            clustering_class=clustering_cls,
        )
        sync()
        t_cluster_end = time.time()
        print(
            f"[{label}] Clustering done in {t_cluster_end - t_cluster_start:.2f}s",
            flush=True,
        )
    except torch.cuda.OutOfMemoryError:
        cleanup()
        print(f"[{label}] OOM during CLUSTERING -- see last [MEM] line above", flush=True)
        return {"status": "oom", "phase": "clustering", "time": math.nan, "cost": math.nan}
    except Exception as exc:
        cleanup()
        print(f"[{label}] ERROR during CLUSTERING: {exc}", flush=True)
        return {"status": "error", "phase": "clustering", "time": math.nan, "cost": math.nan}

    match_B = None
    try:
        sync()
        t_solve_start = time.time()
        match_B = solver.solve()
        sync()
        t_solve_end = time.time()
    except torch.cuda.OutOfMemoryError:
        cleanup()
        phase = getattr(solver, "_oom_phase", "unknown")
        print(f"[{label}] OOM during SOLVE at sub-step: {phase}", flush=True)
        del solver
        solver = None
        cleanup()
        return {"status": "oom", "phase": f"solve:{phase}", "time": math.nan, "cost": math.nan}
    except Exception as exc:
        cleanup()
        phase = getattr(solver, "_oom_phase", "unknown")
        print(f"[{label}] ERROR during SOLVE at sub-step {phase}: {exc}", flush=True)
        del solver
        solver = None
        cleanup()
        return {"status": "error", "phase": f"solve:{phase}", "time": math.nan, "cost": math.nan}

    try:
        if match_B is None:
            match_B = solver.match_B
        cost = avg_matching_cost(A, B, match_B, diameter)
        t_total = (t_cluster_end - t_cluster_start) + (t_solve_end - t_solve_start)
        print(
            f"[{label}] Cluster: {t_cluster_end - t_cluster_start:.2f}s  "
            f"Solve: {t_solve_end - t_solve_start:.2f}s  "
            f"Total: {t_total:.2f}s  |  Avg Cost: {cost:.5f}",
            flush=True,
        )
        return {
            "status": "ok",
            "phase": "done",
            "time": t_total,
            "time_cluster": t_cluster_end - t_cluster_start,
            "time_solve": t_solve_end - t_solve_start,
            "cost": cost,
        }
    except Exception as exc:
        cleanup()
        print(f"[{label}] ERROR during cost computation: {exc}", flush=True)
        return {"status": "error", "phase": "cost", "time": math.nan, "cost": math.nan}
    finally:
        del solver
        cleanup()


def status_value(result, key):
    if result["status"] == "ok":
        value = result.get(key, math.nan)
        return f"{value:.5f}" if key == "cost" else f"{value:.2f}"
    return result["status"].upper()


def exact_value(result):
    if result["status"] == "ok":
        return f"{result['cost']:.5f}"
    return result["status"].upper()


def print_validation_summary(rows):
    headers = [
        "N",
        "Exact Avg Cost",
        "2-Level Time (s)",
        "2-Level Cluster Time (s)",
        "2-Level Solve Time (s)",
        "2-Level Avg Cost",
        "3-Level Time (s)",
        "3-Level Cluster Time (s)",
        "3-Level Solve Time (s)",
        "3-Level Avg Cost",
    ]
    table = [
        [
            f"{row['n']:,}",
            exact_value(row["exact"]),
            status_value(row["sol2"], "time"),
            status_value(row["sol2"], "time_cluster"),
            status_value(row["sol2"], "time_solve"),
            status_value(row["sol2"], "cost"),
            status_value(row["sol3"], "time"),
            status_value(row["sol3"], "time_cluster"),
            status_value(row["sol3"], "time_solve"),
            status_value(row["sol3"], "cost"),
        ]
        for row in rows
    ]
    print_table("Validation Summary", headers, table)


def print_scalability_summary(rows):
    headers = [
        "N",
        "2-Level Time (s)",
        "2-Level Cluster Time (s)",
        "2-Level Solve Time (s)",
        "2-Level Avg Cost",
        "3-Level Time (s)",
        "3-Level Cluster Time (s)",
        "3-Level Solve Time (s)",
        "3-Level Avg Cost",
    ]
    table = [
        [
            f"{row['n']:,}",
            status_value(row["sol2"], "time"),
            status_value(row["sol2"], "time_cluster"),
            status_value(row["sol2"], "time_solve"),
            status_value(row["sol2"], "cost"),
            status_value(row["sol3"], "time"),
            status_value(row["sol3"], "time_cluster"),
            status_value(row["sol3"], "time_solve"),
            status_value(row["sol3"], "cost"),
        ]
        for row in rows
    ]
    print_table("Scalability Summary", headers, table)


def print_table(title, headers, table):
    widths = [len(h) for h in headers]
    for row in table:
        widths = [max(w, len(v)) for w, v in zip(widths, row)]
    print(f"\n{title}", flush=True)
    print(" | ".join(f"{h:>{w}}" for h, w in zip(headers, widths)), flush=True)
    print("-+-".join("-" * w for w in widths), flush=True)
    for row in table:
        print(" | ".join(f"{v:>{w}}" for v, w in zip(row, widths)), flush=True)


def save_validation_csv(rows):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = SCRIPT_DIR / f"scalability_emnist_validation_{stamp}.csv"
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "N",
                "Exact Avg Cost",
                "2-Level Time (s)",
                "2-Level Cluster Time (s)",
                "2-Level Solve Time (s)",
                "2-Level Avg Cost",
                "3-Level Time (s)",
                "3-Level Cluster Time (s)",
                "3-Level Solve Time (s)",
                "3-Level Avg Cost",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["n"],
                    exact_value(row["exact"]),
                    status_value(row["sol2"], "time"),
                    status_value(row["sol2"], "time_cluster"),
                    status_value(row["sol2"], "time_solve"),
                    status_value(row["sol2"], "cost"),
                    status_value(row["sol3"], "time"),
                    status_value(row["sol3"], "time_cluster"),
                    status_value(row["sol3"], "time_solve"),
                    status_value(row["sol3"], "cost"),
                ]
            )
    print(f"\nSaved validation CSV: {path}", flush=True)


def save_scalability_csv(rows):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = SCRIPT_DIR / f"scalability_emnist_scalability_{stamp}.csv"
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "N",
                "2-Level Time (s)",
                "2-Level Cluster Time (s)",
                "2-Level Solve Time (s)",
                "2-Level Avg Cost",
                "3-Level Time (s)",
                "3-Level Cluster Time (s)",
                "3-Level Solve Time (s)",
                "3-Level Avg Cost",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["n"],
                    status_value(row["sol2"], "time"),
                    status_value(row["sol2"], "time_cluster"),
                    status_value(row["sol2"], "time_solve"),
                    status_value(row["sol2"], "cost"),
                    status_value(row["sol3"], "time"),
                    status_value(row["sol3"], "time_cluster"),
                    status_value(row["sol3"], "time_solve"),
                    status_value(row["sol3"], "cost"),
                ]
            )
    print(f"\nSaved scalability CSV: {path}", flush=True)


def run_validation_phase(images_cpu, rng, batch_size):
    print("\n=== Phase 1: Validation ===", flush=True)
    val_rows = []
    for n in VALIDATION_SIZES:
        print(f"\n=== Validation N = {n} ===", flush=True)
        alloc, reserved = gpu_mem_gb()
        print(f"GPU Memory: {alloc:.2f} GB allocated, {reserved:.2f} GB reserved", flush=True)
        try:
            A, B, diameter = make_data(images_cpu, n, rng)
        except torch.cuda.OutOfMemoryError:
            cleanup()
            print(f"DATA OOM while sampling or normalizing validation N={n:,}; skipping.", flush=True)
            continue
        except Exception as exc:
            cleanup()
            print(f"DATA ERROR at validation N={n:,}: {exc}; skipping.", flush=True)
            continue

        exact = run_exact_solver(A, B, diameter)
        sol2 = run_solver("2-Level", SimpleGPUSolver, SimpleL1Clustering, A, B, diameter, batch_size)
        sol3 = run_solver("3-Level", ThreeLevelGPUSolver, ThreeLevelL1Clustering, A, B, diameter, batch_size)
        val_rows.append({"n": n, "exact": exact, "sol2": sol2, "sol3": sol3})

        del A, B
        cleanup()

    print_validation_summary(val_rows)
    save_validation_csv(val_rows)
    return val_rows


def run_scalability_phase(images_cpu, rng, start_n, step_n, max_n, batch_size):
    print("\n=== Phase 2: Scalability ===", flush=True)
    rows = []
    sol2_active = True
    sol3_active = True

    for n in scalability_n_values(start_n, step_n, max_n):
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
            sol2 = run_solver("2-Level", SimpleGPUSolver, SimpleL1Clustering, A, B, diameter, batch_size)
            if sol2["status"] == "oom":
                sol2_active = False
        else:
            sol2 = {"status": "oom", "time": math.nan, "cost": math.nan}
            print("[2-Level] OOM", flush=True)

        if sol3_active:
            sol3 = run_solver("3-Level", ThreeLevelGPUSolver, ThreeLevelL1Clustering, A, B, diameter, batch_size)
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

    print_scalability_summary(rows)
    save_scalability_csv(rows)
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Standalone L1/Manhattan EMNIST push-relabel scalability sweep."
    )
    parser.add_argument("--skip-validation", action="store_true", help="Skip exact/POT validation and run only scalability.")
    parser.add_argument("--start-n", type=int, default=50_000)
    parser.add_argument("--step-n", type=int, default=50_000)
    parser.add_argument("--max-n", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=2048)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA is required"
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print("Metric: L1 / Manhattan on probability-normalized EMNIST histograms", flush=True)
    print(f"Analytic L1 diameter: {L1_DIAMETER}", flush=True)
    print(
        f"Scalability N: start={args.start_n:,}, step={args.step_n:,}, "
        f"max={'unbounded' if args.max_n is None else f'{args.max_n:,}'}",
        flush=True,
    )
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    rng = np.random.default_rng(SEED)

    try:
        images_cpu, split, root = load_emnist_dataset()
    except Exception as exc:
        print(f"DATA ERROR: {exc}", flush=True)
        return
    print(f"EMNIST images: {images_cpu.shape[0]:,} split={split} root={root}", flush=True)

    if not args.skip_validation:
        run_validation_phase(images_cpu, rng, args.batch_size)
    run_scalability_phase(images_cpu, rng, args.start_n, args.step_n, args.max_n, args.batch_size)


if __name__ == "__main__":
    main()
