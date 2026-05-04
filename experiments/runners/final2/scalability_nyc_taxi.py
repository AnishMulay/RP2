#!/usr/bin/env python3
"""
Standalone GPU scalability sweep on NYC Taxi coordinate data.
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
    import pandas as pd
except ImportError:
    pd = None

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
DIAMETER_TILE = 1024
VALIDATION_SIZES = [1_000, 5_000, 10_000]

NYC_LAT_MIN, NYC_LAT_MAX = 40.4, 41.0
NYC_LON_MIN, NYC_LON_MAX = -74.4, -73.6

PICKUP_LAT = ("pickup_latitude", "pickup_lat", "pu_lat", "pulatitude")
PICKUP_LON = ("pickup_longitude", "pickup_lon", "pickup_long", "pu_lon", "pulongitude")
DROPOFF_LAT = ("dropoff_latitude", "dropoff_lat", "do_lat", "dolatitude")
DROPOFF_LON = ("dropoff_longitude", "dropoff_lon", "dropoff_long", "do_lon", "dolongitude")


def scalability_n_values():
    n = 50_000
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


def search_dirs():
    roots = [
        SCRIPT_DIR / "data",
        Path.cwd() / "data",
        Path("/scratch"),
        Path.home() / "data",
    ]
    seen = set()
    for root in roots:
        root = root.expanduser()
        key = str(root)
        if key not in seen:
            seen.add(key)
            yield root


def taxi_candidates():
    for root in search_dirs():
        if not root.exists():
            continue
        files = []
        for pattern in ("*.csv", "*.parquet"):
            files.extend(root.rglob(pattern))
        for path in sorted(files):
            name = path.name.lower()
            if "taxi" in name or "nyc" in name:
                yield path


def _find_col(columns, variants):
    lower = {str(c).lower(): c for c in columns}
    for variant in variants:
        if variant.lower() in lower:
            return lower[variant.lower()]
    return None


def _valid_coord_frame(df, lat_col, lon_col):
    out = df[[lat_col, lon_col]].copy()
    out.columns = ["lat", "lon"]
    out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
    out["lon"] = pd.to_numeric(out["lon"], errors="coerce")
    out = out.dropna()
    out = out[
        out["lat"].between(NYC_LAT_MIN, NYC_LAT_MAX)
        & out["lon"].between(NYC_LON_MIN, NYC_LON_MAX)
    ]
    return out.reset_index(drop=True)


def load_taxi_file(path):
    if pd is None:
        raise RuntimeError("pandas is not installed")
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise RuntimeError(f"unsupported taxi file type: {path}")

    plat = _find_col(df.columns, PICKUP_LAT)
    plon = _find_col(df.columns, PICKUP_LON)
    if plat is None or plon is None:
        raise RuntimeError("no usable pickup latitude/longitude columns")

    pickup = _valid_coord_frame(df, plat, plon)
    dlat = _find_col(df.columns, DROPOFF_LAT)
    dlon = _find_col(df.columns, DROPOFF_LON)
    dropoff = None
    if dlat is not None and dlon is not None:
        dropoff = _valid_coord_frame(df, dlat, dlon)

    if len(pickup) < 10_000:
        raise RuntimeError(f"fewer than 10,000 usable pickup rows: {len(pickup):,}")
    if dropoff is not None and len(dropoff) < 10_000:
        dropoff = None

    pickup_np = pickup[["lon", "lat"]].to_numpy(dtype=np.float32)
    dropoff_np = None if dropoff is None else dropoff[["lon", "lat"]].to_numpy(dtype=np.float32)
    return pickup_np, dropoff_np


def load_taxi_dataset():
    errors = []
    for path in taxi_candidates():
        try:
            pickup, dropoff = load_taxi_file(path)
            side = "pickup/dropoff" if dropoff is not None else "pickup-only"
            print(f"Loaded taxi data from {path} ({side}, {len(pickup):,} pickup rows)", flush=True)
            return pickup, dropoff, path
        except Exception as exc:
            errors.append(f"{path}: {exc}")
            print(f"Skipping taxi candidate {path}: {exc}", flush=True)
    if errors:
        raise RuntimeError("no usable taxi file found: " + " | ".join(errors[-5:]))
    raise RuntimeError("no .csv or .parquet taxi/nyc file found in fallback directories")


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
    while tile >= 64:
        try:
            return _diameter_with_tile(points, tile)
        except torch.cuda.OutOfMemoryError:
            cleanup()
            tile //= 2
    return _diameter_with_tile(points, 32)


def sample_indices(total, n, rng):
    if 2 * n <= total:
        idx = rng.permutation(total)
        return idx[:n], idx[n : 2 * n]
    return (
        rng.integers(0, total, size=n),
        rng.integers(0, total, size=n),
    )


def make_data(pickup_np, dropoff_np, n, rng):
    if len(pickup_np) < 10_000:
        raise RuntimeError(f"total usable taxi rows < 10,000: {len(pickup_np):,}")
    idx_a, idx_b = sample_indices(len(pickup_np), n, rng)
    if dropoff_np is not None and len(dropoff_np) >= n:
        idx_drop = idx_a if len(dropoff_np) == len(pickup_np) else rng.integers(0, len(dropoff_np), size=n)
        A_np = dropoff_np[idx_drop]
    else:
        A_np = pickup_np[idx_a]
    B_np = pickup_np[idx_b]

    A = torch.as_tensor(A_np, dtype=torch.float32, device="cuda")
    B = torch.as_tensor(B_np, dtype=torch.float32, device="cuda")
    mins = torch.cat([A, B], dim=0).min(dim=0).values
    A = A - mins
    B = B - mins
    diameter = joint_diameter(A, B)
    if diameter > 0.0:
        A = A / diameter
        B = B / diameter
    return A, B, diameter


def avg_matching_cost(A, B, match_B, diameter):
    match_B = match_B.to(device=A.device, dtype=torch.long)
    return torch.norm(B - A[match_B], p=2, dim=1).mean().item() * diameter


def run_exact_solver(A, B, diameter):
    """
    Exact optimal transport cost via POT (Earth Mover's Distance).
    A, B: normalized CUDA float32 tensors of shape (N, d).
    Returns average matching cost in original coordinates.
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
            C[start:end] = np.sqrt((diff ** 2).sum(axis=2))

        weights = np.ones(N, dtype=np.float64) / N
        avg_cost_normalized = pot.emd2(weights, weights, C)
        avg_cost = float(avg_cost_normalized) * diameter

        print(f"[Exact]   Avg Cost: {avg_cost:.5f}", flush=True)
        return {"status": "ok", "cost": avg_cost}
    except Exception as exc:
        print(f"[Exact] ERROR: {exc}", flush=True)
        return {"status": "error", "cost": math.nan}


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


def exact_value(result):
    if result["status"] == "ok":
        return f"{result['cost']:.5f}"
    return result["status"].upper()


def print_validation_summary(rows):
    headers = [
        "N",
        "Exact Avg Cost",
        "2-Level Time (s)",
        "2-Level Avg Cost",
        "3-Level Time (s)",
        "3-Level Avg Cost",
    ]
    table = [
        [
            f"{row['n']:,}",
            exact_value(row["exact"]),
            status_value(row["sol2"], "time"),
            status_value(row["sol2"], "cost"),
            status_value(row["sol3"], "time"),
            status_value(row["sol3"], "cost"),
        ]
        for row in rows
    ]
    print_table("Validation Summary", headers, table)


def print_scalability_summary(rows):
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
    path = SCRIPT_DIR / f"scalability_nyc_taxi_validation_{stamp}.csv"
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "N",
                "Exact Avg Cost",
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
                    exact_value(row["exact"]),
                    status_value(row["sol2"], "time"),
                    status_value(row["sol2"], "cost"),
                    status_value(row["sol3"], "time"),
                    status_value(row["sol3"], "cost"),
                ]
            )
    print(f"\nSaved validation CSV: {path}", flush=True)


def save_scalability_csv(rows):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = SCRIPT_DIR / f"scalability_nyc_taxi_scalability_{stamp}.csv"
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
    print(f"\nSaved scalability CSV: {path}", flush=True)


def run_validation_phase(pickup_np, dropoff_np, rng):
    print("\n=== Phase 1: Validation ===", flush=True)
    val_rows = []
    for n in VALIDATION_SIZES:
        print(f"\n=== Validation N = {n} ===", flush=True)
        alloc, reserved = gpu_mem_gb()
        print(f"GPU Memory: {alloc:.2f} GB allocated, {reserved:.2f} GB reserved", flush=True)
        try:
            A, B, diameter = make_data(pickup_np, dropoff_np, n, rng)
        except torch.cuda.OutOfMemoryError:
            cleanup()
            print(f"DATA OOM while sampling or normalizing validation N={n:,}; skipping.", flush=True)
            continue
        except Exception as exc:
            cleanup()
            print(f"DATA ERROR at validation N={n:,}: {exc}; skipping.", flush=True)
            continue

        exact = run_exact_solver(A, B, diameter)
        sol2 = run_solver("2-Level", SimpleGPUSolver, A, B, diameter)
        sol3 = run_solver("3-Level", ThreeLevelGPUSolver, A, B, diameter)
        val_rows.append({"n": n, "exact": exact, "sol2": sol2, "sol3": sol3})

        del A, B
        cleanup()

    print_validation_summary(val_rows)
    save_validation_csv(val_rows)
    return val_rows


def run_scalability_phase(pickup_np, dropoff_np, rng):
    print("\n=== Phase 2: Scalability ===", flush=True)
    rows = []
    sol2_active = True
    sol3_active = True

    for n in scalability_n_values():
        print(f"\n=== N = {n} ===", flush=True)
        alloc, reserved = gpu_mem_gb()
        print(f"GPU Memory: {alloc:.2f} GB allocated, {reserved:.2f} GB reserved", flush=True)

        try:
            A, B, diameter = make_data(pickup_np, dropoff_np, n, rng)
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

    print_scalability_summary(rows)
    save_scalability_csv(rows)
    return rows


def main():
    assert torch.cuda.is_available(), "CUDA is required"
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    rng = np.random.default_rng(SEED)

    try:
        pickup_np, dropoff_np, path = load_taxi_dataset()
    except Exception as exc:
        print(f"DATA ERROR: {exc}", flush=True)
        return
    print(f"Taxi source: {path}", flush=True)

    run_validation_phase(pickup_np, dropoff_np, rng)
    run_scalability_phase(pickup_np, dropoff_np, rng)


if __name__ == "__main__":
    main()
