#!/usr/bin/env python3

import argparse
import math
import pathlib
import statistics
import sys
import time

import numpy as np
import pandas as pd
import torch


BASE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent.parent  # → RP2/
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    import ot
except ImportError:
    ot = None

from clustered_push_relabel.solvers.simple_bipartite import SimpleGPUSolver


DEFAULT_DATA_PATH = pathlib.Path("./nyc_data/yellow_tripdata_2014-01.parquet")
DEFAULT_DAY = "2014-01-01"

N_VALUES = [1_000, 5_000, 10_000, 50_000, 100_000, 200_000]
EPSILON = 0.01
BATCH_SIZE = 2048
SEED = 42
WARMUP_RUNS = 0
TIMED_RUNS = 1

LON0 = -74.0
LAT0 = 40.7
R_EARTH = 6_371_000.0  # meters

NYC_LAT_MIN, NYC_LAT_MAX = 40.0, 41.0
NYC_LON_MIN, NYC_LON_MAX = -75.0, -73.0

_PICKUP_LAT_VARIANTS  = ["pickup_latitude",  "pickup_lat"]
_PICKUP_LON_VARIANTS  = ["pickup_longitude", "pickup_lon", "pickup_long"]
_DROPOFF_LAT_VARIANTS = ["dropoff_latitude", "dropoff_lat"]
_DROPOFF_LON_VARIANTS = ["dropoff_longitude", "dropoff_lon", "dropoff_long"]
_PICKUP_TIME_VARIANTS = ["tpep_pickup_datetime", "lpep_pickup_datetime", "pickup_datetime"]


def _find_col(df, variants, label):
    for v in variants:
        if v in df.columns:
            return v
    raise KeyError(f"Cannot find {label} column; tried: {variants}")


def load_taxi_data(path, day):
    """Load and filter NYC yellow taxi parquet for a single day."""
    df = pd.read_parquet(path)

    time_col = _find_col(df, _PICKUP_TIME_VARIANTS, "pickup datetime")
    if df[time_col].dt.tz is None:
        df[time_col] = pd.to_datetime(df[time_col]).dt.tz_localize("America/New_York")
    else:
        df[time_col] = df[time_col].dt.tz_convert("America/New_York")
    if day is None:
        print("No date filter applied — using all rows.")
    else:
        mask_day = df[time_col].dt.date == pd.Timestamp(day).date()
        df = df[mask_day]

    plat_col = _find_col(df, _PICKUP_LAT_VARIANTS,  "pickup latitude")
    plon_col = _find_col(df, _PICKUP_LON_VARIANTS,  "pickup longitude")
    dlat_col = _find_col(df, _DROPOFF_LAT_VARIANTS, "dropoff latitude")
    dlon_col = _find_col(df, _DROPOFF_LON_VARIANTS, "dropoff longitude")

    df = df.dropna(subset=[plat_col, plon_col, dlat_col, dlon_col])
    df = df[
        df[plat_col].between(NYC_LAT_MIN, NYC_LAT_MAX) &
        df[plon_col].between(NYC_LON_MIN, NYC_LON_MAX) &
        df[dlat_col].between(NYC_LAT_MIN, NYC_LAT_MAX) &
        df[dlon_col].between(NYC_LON_MIN, NYC_LON_MAX)
    ]
    df = df.rename(columns={
        plat_col: "pickup_latitude",
        plon_col: "pickup_longitude",
        dlat_col: "dropoff_latitude",
        dlon_col: "dropoff_longitude",
    })
    df = df.reset_index(drop=True)
    print(f"Loaded {len(df):,} rows after filtering.")
    return df


def make_point_sets(df, n, rng):
    """
    Sample two independent non-overlapping batches of n rows.
    Batch 1 → pickup  coords → B (blue, proposes outward).
    Batch 2 → dropoff coords → A (red,  matched into).
    Returns A_raw [N,2], B_raw [N,2] in raw lon/lat degrees, or (None, None).
    """
    if len(df) < 2 * n:
        return None, None
    shuffled = rng.permutation(len(df))
    idx_B = shuffled[:n]
    idx_A = shuffled[n: 2 * n]
    B_raw = df.iloc[idx_B][["pickup_longitude",  "pickup_latitude"]].values.astype(np.float32)
    A_raw = df.iloc[idx_A][["dropoff_longitude", "dropoff_latitude"]].values.astype(np.float32)
    return A_raw, B_raw


def project_to_meters(A_raw, B_raw):
    """Equirectangular projection: lon/lat degrees → local XY meters."""
    lat0_rad = math.radians(LAT0)
    cos_lat0 = math.cos(lat0_rad)

    def _proj(coords):
        x = R_EARTH * np.radians(coords[:, 0] - LON0) * cos_lat0
        y = R_EARTH * np.radians(coords[:, 1] - LAT0)
        return np.stack([x, y], axis=1).astype(np.float32)

    return _proj(A_raw), _proj(B_raw)


def normalize_points(A_m, B_m):
    """Normalize both clouds by the joint bounding-box diameter."""
    all_pts = np.vstack([A_m, B_m])
    diameter = float((all_pts.max(axis=0) - all_pts.min(axis=0)).max())
    diameter = max(diameter, 1e-6)
    return (A_m / diameter).astype(np.float32), (B_m / diameter).astype(np.float32), diameter


def synchronize_if_cuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def empty_cache_if_cuda(device):
    if device.type == "cuda":
        torch.cuda.empty_cache()


def compute_cost_matrix_l2(red, blue):
    X = torch.as_tensor(red,  dtype=torch.float64).contiguous()
    Y = torch.as_tensor(blue, dtype=torch.float64).contiguous()
    return torch.cdist(X, Y, p=2).cpu().numpy()


def matching_from_plan(plan):
    return torch.from_numpy(plan.argmax(axis=0).astype(np.int64, copy=False))


def average_l2_matching_cost(P_red, P_blue, matching):
    matched_red = P_red[matching.to(device=P_red.device, dtype=torch.long)]
    return torch.norm(P_blue - matched_red, p=2, dim=1).mean().item()


def benchmark_exact(P_red, P_blue):
    if P_red.shape[0] > 10_000:
        raise RuntimeError("Exact OT skipped for N > 10,000")
    if ot is None:
        raise RuntimeError("POT is not installed; exact solver unavailable.")

    red_cpu  = P_red.detach().cpu()
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


def benchmark_simple(P_red, P_blue, device, diameter):
    """
    Time the full wall time (clustering + solve) for SimpleGPUSolver.
    Points are normalized so diameter=1.0 is passed to the solver;
    avg_cost_meters = avg_cost_normalized * diameter.
    """
    for _ in range(WARMUP_RUNS):
        empty_cache_if_cuda(device)
        solver = SimpleGPUSolver(
            P_red, P_blue,
            EPSILON,
            batch_size=BATCH_SIZE,
            verbose=False,
            diameter=1.0,
        )
        synchronize_if_cuda(device)
        solver.solve()
        synchronize_if_cuda(device)
        del solver

    times_ms = []
    costs_m = []
    iterations_list = []
    for _ in range(TIMED_RUNS):
        empty_cache_if_cuda(device)
        solver = SimpleGPUSolver(
            P_red, P_blue,
            EPSILON,
            batch_size=BATCH_SIZE,
            verbose=False,
            diameter=1.0,
        )
        synchronize_if_cuda(device)
        t0 = time.perf_counter()
        solver.solve()
        synchronize_if_cuda(device)
        t1 = time.perf_counter()
        cost_norm = average_l2_matching_cost(P_red, P_blue, solver.match_B)
        costs_m.append(cost_norm * diameter)
        times_ms.append((t1 - t0) * 1000.0)
        iterations_list.append(solver.iterations)
        del solver

    return (
        statistics.median(times_ms),
        statistics.median(costs_m),
        statistics.median(iterations_list),
    )


def result_na():
    return {"time_ms": math.nan, "cost": math.nan, "iterations": math.nan, "status": "fail"}


def run_exact(P_red, P_blue, diameter):
    """P_red and P_blue must be CPU tensors (normalized)."""
    try:
        time_ms, cost_norm = benchmark_exact(P_red, P_blue)
        return {"time_ms": time_ms, "cost": cost_norm * diameter, "status": "success"}
    except Exception as exc:
        print(f"Warning: Exact failed: {exc}", flush=True)
        return {"time_ms": math.nan, "cost": math.nan, "status": "fail"}


def run_simple(P_red, P_blue, device, diameter):
    try:
        time_ms, cost_m, iterations = benchmark_simple(P_red, P_blue, device, diameter)
        return {"time_ms": time_ms, "cost": cost_m, "iterations": iterations, "status": "success"}
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


def fmt_cost_m(value):
    if not is_available(value):
        return "N/A"
    return f"{value:.1f} m"


def fmt_iter(value):
    if not is_available(value):
        return "N/A"
    return f"{int(value):,}"


def print_results_table(rows):
    col_widths = {
        "n":            7,
        "exact_time":  14,
        "simple_time": 14,
        "exact_cost":  16,
        "simple_cost": 16,
        "simple_iters":14,
    }
    headers = [
        ("N",               col_widths["n"],            ">"),
        ("Exact Time",      col_widths["exact_time"],   ">"),
        ("Simple Time",     col_widths["simple_time"],  ">"),
        ("Exact Avg Cost",  col_widths["exact_cost"],   ">"),
        ("Simple Avg Cost", col_widths["simple_cost"],  ">"),
        ("Simple Iters",    col_widths["simple_iters"], ">"),
    ]

    header_line = " | ".join(
        f"{label:{align}{width}}" for label, width, align in headers
    )
    separator = "-+-".join("-" * width for _, width, _ in headers)

    print()
    print(header_line)
    print(separator)
    for row in rows:
        exact  = row["exact"]
        simple = row["simple"]
        cells = [
            f"{row['n']:>{col_widths['n']},}",
            f"{fmt_time(exact['time_ms']):>{col_widths['exact_time']}}",
            f"{fmt_time(simple['time_ms']):>{col_widths['simple_time']}}",
            f"{fmt_cost_m(exact['cost']):>{col_widths['exact_cost']}}",
            f"{fmt_cost_m(simple['cost']):>{col_widths['simple_cost']}}",
            f"{fmt_iter(simple['iterations']):>{col_widths['simple_iters']}}",
        ]
        print(" | ".join(cells))


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark SimpleGPUSolver vs exact OT on NYC yellow taxi data."
    )
    parser.add_argument(
        "--data", type=pathlib.Path, default=DEFAULT_DATA_PATH,
        help="Path to yellow taxi parquet file",
    )
    parser.add_argument(
        "--day", type=str, default=None,
        help="Date to filter (YYYY-MM-DD)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    print(f"Device  : {device}")
    print(f"Data    : {args.data}")
    print(f"Day     : {args.day}")
    print(f"Epsilon : {EPSILON}  batch_size={BATCH_SIZE}")

    df = load_taxi_data(args.data, args.day)

    rng = np.random.default_rng(SEED)
    rows = []

    for n in N_VALUES:
        print(f"\nPreparing NYC N={n:,}", flush=True)

        A_raw, B_raw = make_point_sets(df, n, rng)
        if A_raw is None:
            print(f"  Not enough rows for 2×{n:,} (have {len(df):,}). Skipping.")
            continue

        A_m, B_m = project_to_meters(A_raw, B_raw)
        A_norm, B_norm, diameter = normalize_points(A_m, B_m)
        print(f"  Diameter: {diameter / 1000:.2f} km", flush=True)

        P_red_cpu  = torch.from_numpy(A_norm).float()
        P_blue_cpu = torch.from_numpy(B_norm).float()
        P_red  = P_red_cpu.to(device)
        P_blue = P_blue_cpu.to(device)

        print("  Running Exact OT...", flush=True)
        exact_result = run_exact(P_red_cpu, P_blue_cpu, diameter)

        print("  Running Simple solver...", flush=True)
        simple_result = run_simple(P_red, P_blue, device, diameter)

        rows.append({"n": n, "exact": exact_result, "simple": simple_result})

        del P_red, P_blue, P_red_cpu, P_blue_cpu
        empty_cache_if_cuda(device)

    print("\n\n=== RESULTS ===")
    print_results_table(rows)


if __name__ == "__main__":
    main()
