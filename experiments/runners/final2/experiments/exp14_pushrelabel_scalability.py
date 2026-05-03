#!/usr/bin/env python3
"""
Experiment 14 — Push-relabel scalability limits across synthetic, EMNIST,
and NYC Taxi vector datasets.

For each dataset and N, this sweep measures whether a dense N x N float32 CUDA
matrix fits, then runs the 2-level and 3-level push-relabel GPU solvers using
precomputed L1 clusterings. A method is not retried at larger N after its first
OOM for a dataset.
"""

import argparse
import gc
import math
import pathlib
import sys
import time
import warnings
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

try:
    import pandas as pd
except ImportError:
    pd = None

from clustered_push_relabel.clustering.simple_l1 import SimpleL1Clustering
from clustered_push_relabel.clustering.simple_three_level_l1 import ThreeLevelL1Clustering
from clustered_push_relabel.solvers.simple_bipartite import SimpleGPUSolver
from clustered_push_relabel.solvers.three_level_bipartite import ThreeLevelGPUSolver
from shared import fmt_time, fmt_cost, fmt_iters

EXP_ID = 14
EXP_NAME = "Push-Relabel Scalability Limits — Synthetic / EMNIST / NYC Taxi"
DATASET = "Multi"

N_VALUES = [10_000, 25_000, 50_000, 100_000, 200_000, 500_000, 1_000_000]

EPSILON = 0.01
SEED = 42
TILE_SIZE = 512

DATA_DIR = BASE_DIR / "data"
EMNIST_SPLIT = "byclass"

DEFAULT_DATA_PATH = pathlib.Path("./nyc_data/yellow_tripdata_2014-01.parquet")

LON0, LAT0, R_EARTH = -74.0, 40.7, 6_371_000.0
NYC_LAT_MIN, NYC_LAT_MAX = 40.0, 41.0
NYC_LON_MIN, NYC_LON_MAX = -75.0, -73.0

_PICKUP_LAT = ["pickup_latitude", "pickup_lat"]
_PICKUP_LON = ["pickup_longitude", "pickup_lon", "pickup_long"]
_DROPOFF_LAT = ["dropoff_latitude", "dropoff_lat"]
_DROPOFF_LON = ["dropoff_longitude", "dropoff_lon", "dropoff_long"]
_PICKUP_TIME = ["tpep_pickup_datetime", "lpep_pickup_datetime", "pickup_datetime"]

_NYC_DATA_PATH = DEFAULT_DATA_PATH
_NYC_DAY = None
_TAXI_CACHE = {}


def generate_synthetic(n, device):
    torch.manual_seed(SEED)
    A = torch.rand(n, 2, device=device, dtype=torch.float32)
    B = torch.rand(n, 2, device=device, dtype=torch.float32)
    return A, B


def load_emnist(n, seed, split=EMNIST_SPLIT):
    n_samples = n
    train = torchvision.datasets.EMNIST(root=str(DATA_DIR), split=split, train=True,  download=False)
    test  = torchvision.datasets.EMNIST(root=str(DATA_DIR), split=split, train=False, download=False)
    images = torch.cat([train.data, test.data], dim=0).numpy()
    labels = torch.cat([train.targets, test.targets], dim=0).numpy()
    images = images.reshape(-1, 28, 28).transpose(0, 2, 1).reshape(-1, 784)

    classes = np.unique(labels)
    spc = n_samples // len(classes)
    if spc == 0:
        raise ValueError(f"n_samples={n_samples} too small for {len(classes)} EMNIST classes")

    rng_r = np.random.RandomState(seed)
    rng_b = np.random.RandomState(seed + 1)
    red_parts, blue_parts = [], []
    for cls in classes:
        idx = np.flatnonzero(labels == cls).copy()
        needed = 2 * spc
        if idx.size < needed:
            warnings.warn(f"Class {cls}: {idx.size} available, need {needed}. Skipping.")
            continue
        rng_r.shuffle(idx)
        chosen = idx[:needed]
        red_parts.append(images[chosen[:spc]])
        blue_parts.append(images[chosen[spc:needed]])

    red  = np.concatenate(red_parts).astype(np.float32)  / 255.0
    blue = np.concatenate(blue_parts).astype(np.float32) / 255.0
    for arr in (red, blue):
        s = arr.sum(axis=1, keepdims=True)
        np.maximum(s, 1e-8, out=s)
        # Treat each EMNIST image as a probability measure over the pixel grid.
        # This is the standard OT image-histogram scaling; do not divide by 2,
        # because these proxy experiments report costs in natural [0, 2] L1 units.
        arr /= s

    return torch.from_numpy(red), torch.from_numpy(blue)


def _find_col(df, variants, label):
    for v in variants:
        if v in df.columns:
            return v
    raise KeyError(f"Cannot find {label} column; tried: {variants}")


def load_taxi(path_or_n, day=None):
    if isinstance(path_or_n, (int, np.integer)):
        n = int(path_or_n)
        seed = SEED if day is None else int(day)
        data_path = pathlib.Path(_NYC_DATA_PATH)
        if not data_path.exists():
            raise FileNotFoundError(data_path)

        cache_key = (str(data_path), _NYC_DAY)
        if cache_key not in _TAXI_CACHE:
            _TAXI_CACHE[cache_key] = load_taxi(data_path, _NYC_DAY)
        df = _TAXI_CACHE[cache_key]

        rng = np.random.default_rng(seed)
        result = make_points(df, n, rng)
        if result[0] is None:
            raise ValueError(f"not enough taxi rows: need {2*n:,}, have {len(df):,}")
        A_m, B_m, _diameter = result
        return torch.from_numpy(A_m), torch.from_numpy(B_m)

    if pd is None:
        raise RuntimeError("pandas not installed")
    path = pathlib.Path(path_or_n)
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


DATASET_CONFIGS = [
    {"name": "Synthetic-2D", "type": "synthetic"},
    {"name": "EMNIST",       "type": "real", "loader": load_emnist},
    {"name": "NYC-Taxi",     "type": "real", "loader": load_taxi},
]


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _clear(device):
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _is_oom(exc):
    return isinstance(exc, torch.cuda.OutOfMemoryError) or (
        isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()
    )


def _avg_l1(red, blue, match):
    matched = red[match.to(device=red.device, dtype=torch.long)]
    return (blue - matched).abs().sum(dim=1).mean().item()


def _dense_alloc_test(n, device):
    if device.type != "cuda":
        return {"status": "skip", "mem_gb": float("nan")}
    try:
        torch.cuda.reset_peak_memory_stats(device)
        M = torch.empty(n, n, dtype=torch.float32, device=device)
        torch.cuda.synchronize(device)
        peak = torch.cuda.max_memory_allocated(device) / 1e9
        del M
        torch.cuda.empty_cache()
        return {"status": "ok", "mem_gb": peak}
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        torch.cuda.empty_cache()
        return {"status": "oom", "mem_gb": float("nan")}


def _run_solver2(A, B, device):
    try:
        _clear(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        engine = SimpleL1Clustering(epsilon=EPSILON, tile_size=TILE_SIZE)
        _sync(device)
        t0 = time.perf_counter()
        c = engine.run(A, B)
        _sync(device)
        cluster_time_ms = (time.perf_counter() - t0) * 1000.0

        solver = SimpleGPUSolver(None, None, epsilon=EPSILON, batch_size=TILE_SIZE,
                                 verbose=False, diameter=1.0, precomputed_clustering=c)
        _sync(device)
        t0 = time.perf_counter()
        solver.solve()
        _sync(device)
        solve_time_ms = (time.perf_counter() - t0) * 1000.0

        peak_mem_gb = (
            torch.cuda.max_memory_allocated(device) / 1e9
            if device.type == "cuda" else float("nan")
        )
        cost = _avg_l1(A, B, solver.match_B)
        iters = solver.iterations
        del solver, c, engine
        return cluster_time_ms, solve_time_ms, peak_mem_gb, iters, cost, "ok"
    except (torch.cuda.OutOfMemoryError, RuntimeError, ValueError) as exc:
        _clear(device)
        if _is_oom(exc):
            return (float("nan"), float("nan"), float("nan"),
                    float("nan"), float("nan"), "oom")
        print(f"    2L error: {exc}", flush=True)
        return (float("nan"), float("nan"), float("nan"),
                float("nan"), float("nan"), "error")


def _run_solver3(A, B, device):
    try:
        _clear(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        engine = ThreeLevelL1Clustering(epsilon=EPSILON, tile_size=TILE_SIZE)
        _sync(device)
        t0 = time.perf_counter()
        c = engine.run(A, B)
        _sync(device)
        cluster_time_ms = (time.perf_counter() - t0) * 1000.0

        solver = ThreeLevelGPUSolver(None, None, epsilon=EPSILON, batch_size=TILE_SIZE,
                                     verbose=False, diameter=1.0, precomputed_clustering=c)
        _sync(device)
        t0 = time.perf_counter()
        solver.solve()
        _sync(device)
        solve_time_ms = (time.perf_counter() - t0) * 1000.0

        peak_mem_gb = (
            torch.cuda.max_memory_allocated(device) / 1e9
            if device.type == "cuda" else float("nan")
        )
        cost = _avg_l1(A, B, solver.match_B)
        iters = solver.iterations
        del solver, c, engine
        return cluster_time_ms, solve_time_ms, peak_mem_gb, iters, cost, "ok"
    except (torch.cuda.OutOfMemoryError, RuntimeError, ValueError) as exc:
        _clear(device)
        if _is_oom(exc):
            return (float("nan"), float("nan"), float("nan"),
                    float("nan"), float("nan"), "oom")
        print(f"    3L error: {exc}", flush=True)
        return (float("nan"), float("nan"), float("nan"),
                float("nan"), float("nan"), "error")


def _na_row(ds_name, n, status):
    nan = float("nan")
    return {
        "dataset": ds_name, "n": n,
        "dense_mem_gb": nan, "dense_status": status,
        "cluster2_ms": nan, "solve2_ms": nan,
        "mem2_gb": nan, "iters2": nan, "cost2": nan, "status2": status,
        "cluster3_ms": nan, "solve3_ms": nan,
        "mem3_gb": nan, "iters3": nan, "cost3": nan, "status3": status,
    }


def _label_row(row, dataset_has_row):
    row["dataset_label"] = "" if dataset_has_row else row["dataset"]
    return row


def run(device, **kwargs):
    global _NYC_DATA_PATH, _NYC_DAY

    if kwargs.get("nyc_data_path"):
        _NYC_DATA_PATH = pathlib.Path(kwargs["nyc_data_path"])
    else:
        _NYC_DATA_PATH = DEFAULT_DATA_PATH
    _NYC_DAY = kwargs.get("nyc_day")

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    print(f"\n{'='*65}", flush=True)
    print(f"  Exp {EXP_ID}: {EXP_NAME}", flush=True)
    print(f"  Device: {device}  ε={EPSILON}", flush=True)
    print(f"{'='*65}", flush=True)

    rows = []

    for ds_cfg in DATASET_CONFIGS:
        ds_name = ds_cfg["name"]
        print(f"\n{'─'*65}", flush=True)
        print(f"  Dataset: {ds_name}", flush=True)
        print(f"{'─'*65}", flush=True)

        dense_oom = False
        solver2_oom = False
        solver3_oom = False
        dataset_has_row = False

        for n in N_VALUES:
            print(f"\n  N = {n:,}", flush=True)

            if ds_cfg["type"] == "synthetic":
                A, B = generate_synthetic(n, device)
            else:
                try:
                    A, B = ds_cfg["loader"](n, SEED)
                    A, B = A.to(device), B.to(device)
                except FileNotFoundError as exc:
                    print(f"    Data file not found — skipping {ds_name}: {exc}", flush=True)
                    break
                except (ValueError, RuntimeError) as exc:
                    print(f"    Dataset exhausted at N={n:,} — stopping: {exc}", flush=True)
                    break
                except Exception as exc:
                    print(f"    Unexpected load error: {exc}", flush=True)
                    rows.append(_label_row(_na_row(ds_name, n, "load_error"), dataset_has_row))
                    dataset_has_row = True
                    continue

            if not dense_oom:
                d = _dense_alloc_test(n, device)
                if d["status"] == "oom":
                    dense_oom = True
                    print(f"    Dense: OOM  (N×N float32 = {n**2 * 4 / 1e9:.1f} GB theoretical)", flush=True)
                elif d["status"] == "skip":
                    print("    Dense: skip  peak=N/A", flush=True)
                else:
                    print(f"    Dense: ok  peak={d['mem_gb']:.2f} GB", flush=True)
            else:
                d = {"status": "oom", "mem_gb": float("nan")}

            if not solver2_oom:
                ct2, st2, m2, it2, c2, s2 = _run_solver2(A, B, device)
                if s2 == "oom":
                    solver2_oom = True
                print(f"    2L: {s2}  cluster={ct2:.0f}ms  solve={st2:.0f}ms  "
                      f"mem={m2:.2f}GB  iters={it2}", flush=True)
            else:
                ct2, st2, m2, it2, c2, s2 = [float("nan")] * 5 + ["oom"]

            if not solver3_oom:
                ct3, st3, m3, it3, c3, s3 = _run_solver3(A, B, device)
                if s3 == "oom":
                    solver3_oom = True
                print(f"    3L: {s3}  cluster={ct3:.0f}ms  solve={st3:.0f}ms  "
                      f"mem={m3:.2f}GB  iters={it3}", flush=True)
            else:
                ct3, st3, m3, it3, c3, s3 = [float("nan")] * 5 + ["oom"]

            row = {
                "dataset": ds_name, "n": n,
                "dense_mem_gb": d["mem_gb"], "dense_status": d["status"],
                "cluster2_ms": ct2, "solve2_ms": st2,
                "mem2_gb": m2, "iters2": it2, "cost2": c2, "status2": s2,
                "cluster3_ms": ct3, "solve3_ms": st3,
                "mem3_gb": m3, "iters3": it3, "cost3": c3, "status3": s3,
            }
            rows.append(_label_row(row, dataset_has_row))
            dataset_has_row = True

            del A, B
            gc.collect()
            _clear(device)

            if dense_oom and solver2_oom and solver3_oom:
                print(f"  All methods OOM — stopping sweep for {ds_name}.", flush=True)
                break

    return rows


def _fmt_mem(value, status):
    status = _fmt_status(status)
    if status == "oom":
        return "OOM"
    if status != "ok" or math.isnan(float(value)):
        return "N/A"
    return f"{float(value):.2f} GB"


def _fmt_status(status):
    if status in {"ok", "oom", "skip", "error"}:
        return status
    if status in {"load_error", "fail"}:
        return "error"
    if status in {"exhausted"}:
        return "skip"
    return str(status)


COL_SPECS = [
    ("Dataset", 14),
    ("N", 9),
    ("Dense", 9),
    ("Dense Mem", 11),
    ("2L Clust", 10),
    ("2L Solve", 10),
    ("2L Mem", 10),
    ("3L Clust", 10),
    ("3L Solve", 10),
    ("3L Mem", 10),
]

FMT_FNS = {
    "Dataset": lambda r: r.get("dataset_label", r["dataset"]),
    "N": lambda r: f"{r['n']:,}",
    "Dense": lambda r: _fmt_status(r["dense_status"]),
    "Dense Mem": lambda r: _fmt_mem(r["dense_mem_gb"], r["dense_status"]),
    "2L Clust": lambda r: fmt_time(r["cluster2_ms"]),
    "2L Solve": lambda r: fmt_time(r["solve2_ms"]),
    "2L Mem": lambda r: _fmt_mem(r["mem2_gb"], r["status2"]),
    "3L Clust": lambda r: fmt_time(r["cluster3_ms"]),
    "3L Solve": lambda r: fmt_time(r["solve3_ms"]),
    "3L Mem": lambda r: _fmt_mem(r["mem3_gb"], r["status3"]),
}

DIAG_COL_SPECS = [
    ("Dataset", 14),
    ("N", 9),
    ("Dense", 9),
    ("2L Cost", 12),
    ("2L Iters", 9),
    ("2L", 6),
    ("3L Cost", 12),
    ("3L Iters", 9),
    ("3L", 6),
]

DIAG_FMT_FNS = {
    "Dataset": lambda r: r.get("dataset_label", r["dataset"]),
    "N": lambda r: f"{r['n']:,}",
    "Dense": lambda r: _fmt_status(r["dense_status"]),
    "2L Cost": lambda r: fmt_cost(r["cost2"]),
    "2L Iters": lambda r: fmt_iters(r["iters2"]),
    "2L": lambda r: _fmt_status(r["status2"]),
    "3L Cost": lambda r: fmt_cost(r["cost3"]),
    "3L Iters": lambda r: fmt_iters(r["iters3"]),
    "3L": lambda r: _fmt_status(r["status3"]),
}


def _print_split_table(title, col_specs, fmt_fns, rows):
    headers = [h for h, _ in col_specs]
    widths = [max(len(h), w) for h, w in col_specs]
    print(f"\n{title}")
    print(" | ".join(f"{h:>{w}}" for h, w in zip(headers, widths)))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        cells = [fmt_fns[h](row) for h in headers]
        print(" | ".join(f"{c:>{w}}" for c, w in zip(cells, widths)))


def print_table(rows):
    _print_split_table("Main Table", COL_SPECS, FMT_FNS, rows)
    _print_split_table("Status Table", DIAG_COL_SPECS, DIAG_FMT_FNS, rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nyc-data", default=None)
    ap.add_argument("--nyc-day", default=None)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = run(dev, nyc_data_path=args.nyc_data, nyc_day=args.nyc_day)
    print_table(results)
