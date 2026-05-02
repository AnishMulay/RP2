#!/usr/bin/env python3
"""
Experiment 5 — NYC Taxi: Exact OT vs 2-Level Solver vs 3-Level Solver (Scalability)
Distance: L2 in projected metres.
"""

import argparse
import math
import pathlib
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

FINAL2_DIR = Path(__file__).resolve().parent.parent
BASE_DIR = FINAL2_DIR.parent.parent.parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(FINAL2_DIR) not in sys.path:
    sys.path.insert(0, str(FINAL2_DIR))

try:
    import ot
except ImportError:
    ot = None

try:
    import pandas as pd
except ImportError:
    pd = None

from clustered_push_relabel.solvers.simple_bipartite import SimpleGPUSolver
from clustered_push_relabel.solvers.three_level_bipartite import ThreeLevelGPUSolver
from shared import compute_ratio, fmt_time, fmt_cost, fmt_ratio, fmt_iters, fmt_bytes

EXP_ID = 5
EXP_NAME = "NYC Taxi — Exact vs 2L-Solver vs 3L-Solver (Scalability)"
DATASET = "NYC Taxi"

DEFAULT_DATA_PATH = pathlib.Path("./nyc_data/yellow_tripdata_2014-01.parquet")

N_VALUES = [1_000, 5_000, 10_000, 50_000, 100_000, 200_000]
EPSILON = 0.01
BATCH_SIZE = 2048
SEED = 42
EXACT_N_LIMIT = 10_000

LON0, LAT0, R_EARTH = -74.0, 40.7, 6_371_000.0
NYC_LAT_MIN, NYC_LAT_MAX = 40.0, 41.0
NYC_LON_MIN, NYC_LON_MAX = -75.0, -73.0

_PICKUP_LAT  = ["pickup_latitude",  "pickup_lat"]
_PICKUP_LON  = ["pickup_longitude", "pickup_lon", "pickup_long"]
_DROPOFF_LAT = ["dropoff_latitude", "dropoff_lat"]
_DROPOFF_LON = ["dropoff_longitude","dropoff_lon","dropoff_long"]
_PICKUP_TIME = ["tpep_pickup_datetime","lpep_pickup_datetime","pickup_datetime"]

COL_SPECS = [
    ("N",           7),
    ("Exact Time", 14),
    ("2L Time",    12),
    ("3L Time",    12),
    ("Exact Cost", 14),
    ("2L Cost",    12),
    ("3L Cost",    12),
    ("2L Ratio",    9),
    ("3L Ratio",    9),
    ("2L Iters",   10),
    ("3L Iters",   10),
    ("2L Peak Mem", 12),
    ("3L Peak Mem", 12),
]

FMT_FNS = {
    "N":          lambda r: f"{r['n']:,}",
    "Exact Time": lambda r: fmt_time(r["exact"]["time_ms"]),
    "2L Time":    lambda r: fmt_time(r["sol2"]["time_ms"]),
    "3L Time":    lambda r: fmt_time(r["sol3"]["time_ms"]),
    "Exact Cost": lambda r: fmt_cost(r["exact"]["cost"]),
    "2L Cost":    lambda r: fmt_cost(r["sol2"]["cost"]),
    "3L Cost":    lambda r: fmt_cost(r["sol3"]["cost"]),
    "2L Ratio":   lambda r: fmt_ratio(compute_ratio(r["exact"]["cost"], r["sol2"]["cost"])),
    "3L Ratio":   lambda r: fmt_ratio(compute_ratio(r["exact"]["cost"], r["sol3"]["cost"])),
    "2L Iters":   lambda r: fmt_iters(r["sol2"].get("iters", math.nan)),
    "3L Iters":   lambda r: fmt_iters(r["sol3"].get("iters", math.nan)),
    "2L Peak Mem": lambda r: fmt_bytes(r["sol2"]["peak_mem_bytes"]),
    "3L Peak Mem": lambda r: fmt_bytes(r["sol3"]["peak_mem_bytes"]),
}


def _find_col(df, variants, label):
    for v in variants:
        if v in df.columns:
            return v
    raise KeyError(f"Cannot find {label} column; tried: {variants}")


def load_taxi(path, day=None):
    if pd is None:
        raise RuntimeError("pandas not installed")
    path = pathlib.Path(path)
    if str(path).endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_parquet(path)

    if day is not None:
        tc = _find_col(df, _PICKUP_TIME, "pickup datetime")
        df[tc] = pd.to_datetime(df[tc], utc=True, errors="coerce")
        if df[tc].dt.tz is None:
            df[tc] = df[tc].dt.tz_localize("America/New_York")
        else:
            df[tc] = df[tc].dt.tz_convert("America/New_York")
        df = df[df[tc].dt.date == pd.Timestamp(day).date()]

    plat = _find_col(df, _PICKUP_LAT, "pickup lat")
    plon = _find_col(df, _PICKUP_LON, "pickup lon")
    dlat = _find_col(df, _DROPOFF_LAT, "dropoff lat")
    dlon = _find_col(df, _DROPOFF_LON, "dropoff lon")

    df = df.dropna(subset=[plat, plon, dlat, dlon])
    df = df[df[plat].between(NYC_LAT_MIN, NYC_LAT_MAX) &
            df[plon].between(NYC_LON_MIN, NYC_LON_MAX) &
            df[dlat].between(NYC_LAT_MIN, NYC_LAT_MAX) &
            df[dlon].between(NYC_LON_MIN, NYC_LON_MAX)]
    df = df.rename(columns={plat: "pickup_latitude", plon: "pickup_longitude",
                             dlat: "dropoff_latitude", dlon: "dropoff_longitude"})
    return df.reset_index(drop=True)


def _project(coords):
    import math as _math
    cos_lat = _math.cos(_math.radians(LAT0))
    x = R_EARTH * np.radians(coords[:, 0] - LON0) * cos_lat
    y = R_EARTH * np.radians(coords[:, 1] - LAT0)
    return np.stack([x, y], axis=1).astype(np.float32)


def make_points(df, n, rng):
    if len(df) < 2 * n:
        return None, None
    idx = rng.permutation(len(df))
    B_raw = df.iloc[idx[:n]][["pickup_longitude", "pickup_latitude"]].values.astype(np.float32)
    A_raw = df.iloc[idx[n:2*n]][["dropoff_longitude","dropoff_latitude"]].values.astype(np.float32)
    A_m = _project(A_raw)
    B_m = _project(B_raw)
    all_pts = np.vstack([A_m, B_m])
    diam = float((all_pts.max(0) - all_pts.min(0)).max())
    diam = max(diam, 1e-6)
    return (A_m / diam).astype(np.float32), (B_m / diam).astype(np.float32), diam


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def _clear(device):
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _avg_l2(red, blue, match):
    matched = red[match.to(device=red.device, dtype=torch.long)]
    return torch.norm(blue - matched, p=2, dim=1).mean().item()


def _run_exact(red_cpu, blue_cpu, diameter):
    if red_cpu.shape[0] > EXACT_N_LIMIT:
        raise RuntimeError(f"Exact skipped: N > {EXACT_N_LIMIT:,}")
    if ot is None:
        raise RuntimeError("POT not installed")
    n = red_cpu.shape[0]
    C = torch.cdist(red_cpu.double(), blue_cpu.double(), p=2).numpy()
    a = np.full(n, 1.0 / n, np.float64)
    b = np.full(n, 1.0 / n, np.float64)
    t0 = time.perf_counter()
    plan = ot.emd(a, b, C, numItermax=10**6)
    elapsed = (time.perf_counter() - t0) * 1000.0
    match = torch.from_numpy(plan.argmax(axis=0).astype(np.int64))
    cost = _avg_l2(red_cpu, blue_cpu, match) * diameter
    return elapsed, cost


def _run_solver2(red, blue, device, diameter):
    _clear(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    solver = SimpleGPUSolver(red, blue, EPSILON, batch_size=BATCH_SIZE, verbose=False, diameter=1.0)
    _sync(device)
    t0 = time.perf_counter()
    solver.solve()
    _sync(device)
    peak_mem = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else math.nan
    elapsed = (time.perf_counter() - t0) * 1000.0
    cost = _avg_l2(red, blue, solver.match_B) * diameter
    iters = solver.iterations
    del solver
    return elapsed, cost, iters, peak_mem


def _run_solver3(red, blue, device, diameter):
    _clear(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    solver = ThreeLevelGPUSolver(red, blue, EPSILON, batch_size=BATCH_SIZE, verbose=False, diameter=1.0)
    _sync(device)
    t0 = time.perf_counter()
    solver.solve()
    _sync(device)
    peak_mem = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else math.nan
    elapsed = (time.perf_counter() - t0) * 1000.0
    cost = _avg_l2(red, blue, solver.match_B) * diameter
    iters = solver.iterations
    del solver
    return elapsed, cost, iters, peak_mem


def _safe_exact(fn, label):
    try:
        t, c = fn()
        return {"time_ms": t, "cost": c, "status": "ok"}
    except Exception as exc:
        print(f"    [{label}] skipped: {exc}", flush=True)
        return {"time_ms": math.nan, "cost": math.nan, "status": "skip"}


def _safe_solver(fn, label):
    try:
        t, c, it, peak_mem = fn()
        return {"time_ms": t, "cost": c, "iters": it, "peak_mem_bytes": peak_mem, "status": "ok"}
    except Exception as exc:
        print(f"    [{label}] failed: {exc}", flush=True)
        return {"time_ms": math.nan, "cost": math.nan, "iters": math.nan, "peak_mem_bytes": math.nan, "status": "fail"}


def run(device, nyc_data_path=None, nyc_day=None, **kwargs):
    data_path = pathlib.Path(nyc_data_path) if nyc_data_path else DEFAULT_DATA_PATH

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    print(f"\n{'─'*60}", flush=True)
    print(f"  Exp {EXP_ID}: {EXP_NAME}", flush=True)
    print(f"  Device: {device}  ε={EPSILON}  data: {data_path}", flush=True)
    print(f"{'─'*60}", flush=True)

    if not data_path.exists():
        print(f"  ERROR: {data_path} not found. Skipping experiment.", flush=True)
        return []

    df = load_taxi(data_path, nyc_day)
    print(f"  Loaded {len(df):,} taxi rows.", flush=True)

    rng = np.random.default_rng(SEED)
    rows = []

    for n in N_VALUES:
        print(f"\n  N = {n:,}", flush=True)
        result = make_points(df, n, rng)
        if result[0] is None:
            print(f"    Not enough rows. Skipping.", flush=True)
            continue
        A_norm, B_norm, diameter = result
        print(f"    Diameter: {diameter/1000:.2f} km", flush=True)

        red_cpu  = torch.from_numpy(A_norm)
        blue_cpu = torch.from_numpy(B_norm)
        red  = red_cpu.to(device)
        blue = blue_cpu.to(device)

        print(f"    [1/3] Exact OT ...", flush=True)
        exact = _safe_exact(lambda: _run_exact(red_cpu, blue_cpu, diameter), "Exact")

        print(f"    [2/3] 2-Level Solver ...", flush=True)
        sol2 = _safe_solver(lambda: _run_solver2(red, blue, device, diameter), "2L-Solver")

        print(f"    [3/3] 3-Level Solver ...", flush=True)
        sol3 = _safe_solver(lambda: _run_solver3(red, blue, device, diameter), "3L-Solver")

        rows.append({"n": n, "exact": exact, "sol2": sol2, "sol3": sol3})

        del red, blue, red_cpu, blue_cpu
        _clear(device)

    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA_PATH))
    ap.add_argument("--day", default=None)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = run(dev, nyc_data_path=args.data, nyc_day=args.day)
    for r in results:
        e, s2, s3 = r["exact"], r["sol2"], r["sol3"]
        print(f"N={r['n']:>8,} | exact={fmt_time(e['time_ms'])} "
              f"| 2L={fmt_time(s2['time_ms'])} iters={fmt_iters(s2.get('iters',math.nan))} "
              f"| 3L={fmt_time(s3['time_ms'])} iters={fmt_iters(s3.get('iters',math.nan))}")
