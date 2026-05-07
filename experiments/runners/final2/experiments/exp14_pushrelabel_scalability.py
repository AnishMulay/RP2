#!/usr/bin/env python3
"""
Experiment 14 — EMNIST push-relabel scalability.

Runs only the 2-level and 3-level GPU push-relabel solvers on EMNIST byclass
L1 image histograms. There is no dense exact OT baseline and no synthetic/NYC
sub-sweep in this experiment.
"""

import argparse
import gc
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchvision

FINAL2_DIR = Path(__file__).resolve().parent.parent
BASE_DIR = FINAL2_DIR.parent.parent.parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(FINAL2_DIR) not in sys.path:
    sys.path.insert(0, str(FINAL2_DIR))

from clustered_push_relabel.clustering.simple_l1 import SimpleL1Clustering
from clustered_push_relabel.clustering.simple_three_level_l1 import ThreeLevelL1Clustering
from clustered_push_relabel.solvers.simple_bipartite import SimpleGPUSolver
from clustered_push_relabel.solvers.three_level_bipartite import ThreeLevelGPUSolver
from shared import fmt_time, fmt_iters

EXP_ID = 14
EXP_NAME = "EMNIST Push-Relabel Scalability — 2L vs 3L"
DATASET = "EMNIST"

N_VALUES = list(range(50_000, 1_000_001, 50_000))

EPSILON = 0.01
SEED = 42
BATCH_SIZE = 512
MAX_ITERS = 500_000
MEMORY_HEADROOM = 0.85
ESTIMATE_SAFETY_FACTOR = 2.5

DATA_DIR = BASE_DIR / "data"
EMNIST_SPLIT = "byclass"


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _clear(device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _is_oom(exc):
    return isinstance(exc, torch.cuda.OutOfMemoryError) or (
        isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()
    )


def _cuda_free_gb(device):
    if device.type != "cuda":
        return math.nan
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    free_bytes, _total_bytes = torch.cuda.mem_get_info(device_index)
    return free_bytes / 1e9


def _estimate_solver_gb(n, dim, tile_size=BATCH_SIZE):
    input_bytes = 2 * n * dim * 4
    tile_bytes = n * tile_size * (4 + 1)

    s2 = max(1, int(math.ceil(math.sqrt(n))))
    two_level_bytes = input_bytes + (12 * n * s2) + tile_bytes

    s1 = max(1, int(math.ceil(float(n) ** (2.0 / 3.0))))
    s3 = max(1, int(math.ceil(float(n) ** (1.0 / 3.0))))
    three_level_bytes = input_bytes + (8 * n * s3) + (8 * s1 * s3) + tile_bytes

    return ESTIMATE_SAFETY_FACTOR * max(two_level_bytes, three_level_bytes) / 1e9


def _has_enough_memory_for_next_n(n, dim, device):
    if device.type != "cuda":
        return True, math.nan, math.nan
    free_gb = _cuda_free_gb(device)
    needed_gb = _estimate_solver_gb(n, dim)
    return needed_gb <= free_gb * MEMORY_HEADROOM, needed_gb, free_gb


def load_emnist_equal(n_samples, seed, split=EMNIST_SPLIT):
    train = torchvision.datasets.EMNIST(root=str(DATA_DIR), split=split, train=True, download=False)
    test = torchvision.datasets.EMNIST(root=str(DATA_DIR), split=split, train=False, download=False)
    images = torch.cat([train.data, test.data], dim=0).numpy()
    images = images.reshape(-1, 28, 28).transpose(0, 2, 1).reshape(-1, 784)

    needed = 2 * n_samples
    if images.shape[0] < needed:
        raise ValueError(
            f"EMNIST has {images.shape[0]:,} total samples, need {needed:,} for N={n_samples:,}"
        )

    rng = np.random.RandomState(seed)
    chosen = rng.permutation(images.shape[0])[:needed]
    red = images[chosen[:n_samples]].astype(np.float32) / 255.0
    blue = images[chosen[n_samples:needed]].astype(np.float32) / 255.0
    for arr in (red, blue):
        s = arr.sum(axis=1, keepdims=True)
        np.maximum(s, 1e-8, out=s)
        arr /= s
        # The GPU solvers expect costs in [0, 1]. Probability histograms have
        # L1 distances in [0, 2], so scale coordinates by 1/2 for this solver run.
        arr *= 0.5

    if red.shape[0] != n_samples or blue.shape[0] != n_samples:
        raise RuntimeError(f"internal sampling error: got red={red.shape[0]}, blue={blue.shape[0]}")
    return torch.from_numpy(red), torch.from_numpy(blue)


def _run_solver2(red, blue, device):
    engine = solver = clustering = None
    try:
        _clear(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        engine = SimpleL1Clustering(epsilon=EPSILON, tile_size=BATCH_SIZE)
        _sync(device)
        t0 = time.perf_counter()
        clustering = engine.run(red, blue)
        _sync(device)
        cluster_ms = (time.perf_counter() - t0) * 1000.0

        solver = SimpleGPUSolver(
            None,
            None,
            epsilon=EPSILON,
            batch_size=BATCH_SIZE,
            verbose=False,
            max_iters=MAX_ITERS,
            diameter=1.0,
            precomputed_clustering=clustering,
        )
        solver.debug_audit = False
        solver.debug_stop_on_first_violation = False

        _sync(device)
        t0 = time.perf_counter()
        solver.solve()
        _sync(device)
        solve_ms = (time.perf_counter() - t0) * 1000.0

        peak_gb = torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else math.nan
        result = {
            "cluster_ms": cluster_ms,
            "solve_ms": solve_ms,
            "peak_gb": peak_gb,
            "iters": solver.iterations,
            "cost": math.nan,
            "status": "ok",
        }
        return result
    except Exception as exc:
        if _is_oom(exc):
            return _solver_error_row("oom")
        print(f"    2L error: {exc}", flush=True)
        return _solver_error_row("error")
    finally:
        del solver, clustering, engine
        _clear(device)


def _run_solver3(red, blue, device):
    engine = solver = clustering = None
    try:
        _clear(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        engine = ThreeLevelL1Clustering(epsilon=EPSILON, tile_size=BATCH_SIZE)
        _sync(device)
        t0 = time.perf_counter()
        clustering = engine.run(red, blue)
        _sync(device)
        cluster_ms = (time.perf_counter() - t0) * 1000.0

        solver = ThreeLevelGPUSolver(
            None,
            None,
            epsilon=EPSILON,
            batch_size=BATCH_SIZE,
            verbose=False,
            max_iters=MAX_ITERS,
            diameter=1.0,
            precomputed_clustering=clustering,
        )
        solver.debug_audit = False
        solver.debug_stop_on_first_violation = False

        _sync(device)
        t0 = time.perf_counter()
        solver.solve()
        _sync(device)
        solve_ms = (time.perf_counter() - t0) * 1000.0

        peak_gb = torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else math.nan
        result = {
            "cluster_ms": cluster_ms,
            "solve_ms": solve_ms,
            "peak_gb": peak_gb,
            "iters": solver.iterations,
            "cost": math.nan,
            "status": "ok",
        }
        return result
    except Exception as exc:
        if _is_oom(exc):
            return _solver_error_row("oom")
        print(f"    3L error: {exc}", flush=True)
        return _solver_error_row("error")
    finally:
        del solver, clustering, engine
        _clear(device)


def _solver_error_row(status):
    return {
        "cluster_ms": math.nan,
        "solve_ms": math.nan,
        "peak_gb": math.nan,
        "iters": math.nan,
        "cost": math.nan,
        "status": status,
    }


def _make_row(n, sol2, sol3, status="ok"):
    return {
        "dataset": DATASET,
        "n": n,
        "status": status,
        "cluster2_ms": sol2["cluster_ms"],
        "solve2_ms": sol2["solve_ms"],
        "mem2_gb": sol2["peak_gb"],
        "iters2": sol2["iters"],
        "cost2": sol2["cost"],
        "status2": sol2["status"],
        "cluster3_ms": sol3["cluster_ms"],
        "solve3_ms": sol3["solve_ms"],
        "mem3_gb": sol3["peak_gb"],
        "iters3": sol3["iters"],
        "cost3": sol3["cost"],
        "status3": sol3["status"],
    }


def run(device, **kwargs):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    print(f"\n{'='*65}", flush=True)
    print(f"  Exp {EXP_ID}: {EXP_NAME}", flush=True)
    print(f"  Device: {device}  epsilon={EPSILON}  batch={BATCH_SIZE}", flush=True)
    print(f"  N sweep: {N_VALUES[0]:,}, {N_VALUES[1]:,}, ...", flush=True)
    print(f"{'='*65}", flush=True)

    if device.type != "cuda":
        print("  CUDA is required for these GPU solvers; skipping experiment.", flush=True)
        return []

    rows = []
    solver2_stopped = False
    solver3_stopped = False

    for n in N_VALUES:
        print(f"\n  N = {n:,}", flush=True)

        ok_memory, needed_gb, free_gb = _has_enough_memory_for_next_n(n, 784, device)
        if not ok_memory:
            print(
                f"    Memory guard stopped before allocation: estimate={needed_gb:.2f} GB, "
                f"free={free_gb:.2f} GB",
                flush=True,
            )
            break

        try:
            red_cpu, blue_cpu = load_emnist_equal(n, SEED)
        except FileNotFoundError as exc:
            print(f"    EMNIST data not found: {exc}", flush=True)
            break
        except (ValueError, RuntimeError) as exc:
            print(f"    EMNIST exhausted at N={n:,}: {exc}", flush=True)
            break

        try:
            red = red_cpu.to(device=device, dtype=torch.float32, non_blocking=True)
            blue = blue_cpu.to(device=device, dtype=torch.float32, non_blocking=True)
            del red_cpu, blue_cpu
            _sync(device)
        except Exception as exc:
            del red_cpu, blue_cpu
            if _is_oom(exc):
                _clear(device)
                print("    OOM while moving EMNIST batch to GPU; stopping sweep.", flush=True)
                break
            raise

        sol2 = _solver_error_row("skip") if solver2_stopped else _run_solver2(red, blue, device)
        if sol2["status"] == "oom":
            solver2_stopped = True
        print(
            f"    2L: {sol2['status']}  cluster={fmt_time(sol2['cluster_ms'])}  "
            f"solve={fmt_time(sol2['solve_ms'])}  mem={_fmt_mem(sol2['peak_gb'], sol2['status'])}  "
            f"iters={fmt_iters(sol2['iters'])}",
            flush=True,
        )

        sol3 = _solver_error_row("skip") if solver3_stopped else _run_solver3(red, blue, device)
        if sol3["status"] == "oom":
            solver3_stopped = True
        print(
            f"    3L: {sol3['status']}  cluster={fmt_time(sol3['cluster_ms'])}  "
            f"solve={fmt_time(sol3['solve_ms'])}  mem={_fmt_mem(sol3['peak_gb'], sol3['status'])}  "
            f"iters={fmt_iters(sol3['iters'])}",
            flush=True,
        )

        rows.append(_make_row(n, sol2, sol3))

        del red, blue
        _clear(device)

        if solver2_stopped and solver3_stopped:
            print("  Both solvers have stopped due to OOM; ending sweep.", flush=True)
            break

    return rows


def _fmt_status(status):
    if status in {"ok", "oom", "skip", "error"}:
        return status
    return str(status)


def _fmt_mem(value, status):
    status = _fmt_status(status)
    if status == "oom":
        return "OOM"
    if status != "ok" or math.isnan(float(value)):
        return "N/A"
    return f"{float(value):.2f} GB"


COL_SPECS = [
    ("Dataset", 8),
    ("N", 9),
    ("2L Clust", 10),
    ("2L Solve", 10),
    ("2L Mem", 10),
    ("2L Iters", 9),
    ("2L", 6),
    ("3L Clust", 10),
    ("3L Solve", 10),
    ("3L Mem", 10),
    ("3L Iters", 9),
    ("3L", 6),
]

FMT_FNS = {
    "Dataset": lambda r: r["dataset"],
    "N": lambda r: f"{r['n']:,}",
    "2L Clust": lambda r: fmt_time(r["cluster2_ms"]),
    "2L Solve": lambda r: fmt_time(r["solve2_ms"]),
    "2L Mem": lambda r: _fmt_mem(r["mem2_gb"], r["status2"]),
    "2L Iters": lambda r: fmt_iters(r["iters2"]),
    "2L": lambda r: _fmt_status(r["status2"]),
    "3L Clust": lambda r: fmt_time(r["cluster3_ms"]),
    "3L Solve": lambda r: fmt_time(r["solve3_ms"]),
    "3L Mem": lambda r: _fmt_mem(r["mem3_gb"], r["status3"]),
    "3L Iters": lambda r: fmt_iters(r["iters3"]),
    "3L": lambda r: _fmt_status(r["status3"]),
}


def print_table(rows):
    headers = [h for h, _ in COL_SPECS]
    widths = [max(len(h), w) for h, w in COL_SPECS]
    print("\n" + " | ".join(f"{h:>{w}}" for h, w in zip(headers, widths)))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        cells = [FMT_FNS[h](row) for h in headers]
        print(" | ".join(f"{c:>{w}}" for c, w in zip(cells, widths)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run Experiment 14 EMNIST scalability.")
    ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = run(dev)
    print_table(results)
