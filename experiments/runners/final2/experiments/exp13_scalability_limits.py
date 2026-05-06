#!/usr/bin/env python3
"""
Experiment 13 — Scalability limits across synthetic and real vector datasets.

For each dataset and N, this sweep measures dense GPU allocation pressure and
the 2-level / 3-level L1 proxy solvers. A method is not retried at larger N
after its first OOM for a dataset.
"""

import argparse
import gzip
import math
import pathlib
import pickle
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
    import ot
except ImportError:
    ot = None

from clustered_push_relabel.clustering.simple_l1 import SimpleL1Clustering
from clustered_push_relabel.clustering.simple_three_level_l1 import ThreeLevelL1Clustering
from shared import fmt_time, build_two_level_proxy_matrix, build_three_level_proxy_matrix

EXP_ID = 13
EXP_NAME = "Scalability Limits — All Datasets"
DATASET = "Multi"

N_VALUES = [
    1_000,
    5_000,
    10_000,
    25_000,
    50_000,
    100_000,
    200_000,
    500_000,
    1_000_000,
]

EPSILON = 0.01
SEED = 42
TILE_SIZE = 512

DATA_DIR = BASE_DIR / "data"
EMNIST_SPLIT = "byclass"
CIFAR_DATA_DIR = BASE_DIR / "data" / "cifar_sift"
TRAIN_DESC_PATH = CIFAR_DATA_DIR / "cifar10_sift_train.pkl.gz"
NEWSGROUPS_DATA_DIR = BASE_DIR / "data" / "newsgroups_glove"
EMBEDDINGS_PATH = NEWSGROUPS_DATA_DIR / "newsgroups_embeddings.pkl.gz"
MAX_WORDS = 300


def load_emnist(n, seed, split=EMNIST_SPLIT):
    train = torchvision.datasets.EMNIST(root=str(DATA_DIR), split=split, train=True, download=False)
    test = torchvision.datasets.EMNIST(root=str(DATA_DIR), split=split, train=False, download=False)
    images = torch.cat([train.data, test.data], dim=0).numpy()
    labels = torch.cat([train.targets, test.targets], dim=0).numpy()
    images = images.reshape(-1, 28, 28).transpose(0, 2, 1).reshape(-1, 784)

    classes = np.unique(labels)
    spc = n // len(classes)
    if spc == 0:
        raise ValueError(f"n_samples={n} too small for {len(classes)} EMNIST classes")

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

    red = np.concatenate(red_parts).astype(np.float32) / 255.0
    blue = np.concatenate(blue_parts).astype(np.float32) / 255.0
    for arr in (red, blue):
        s = arr.sum(axis=1, keepdims=True)
        np.maximum(s, 1e-8, out=s)
        # Treat each EMNIST image as a probability measure over the pixel grid.
        # This is the standard OT image-histogram scaling; do not divide by 2,
        # because these proxy experiments report costs in natural [0, 2] L1 units.
        arr /= s

    return torch.from_numpy(red), torch.from_numpy(blue)


def load_descriptors(path):
    with gzip.open(path, "rb") as f:
        descs = pickle.load(f)
    print(f"  Loaded {len(descs):,} descriptor sets from {pathlib.Path(path).name}", flush=True)
    return descs


def sample_pair(all_descs, n, seed):
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(all_descs))
    return ([all_descs[i] for i in perm[:n]], [all_descs[i] for i in perm[n:2 * n]])


def _descriptor_means(descs):
    rows = []
    for d in descs:
        arr = np.asarray(d, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] == 0:
            raise ValueError("descriptor set must be a non-empty 2D array")
        rows.append(arr.mean(axis=0))
    return torch.from_numpy(np.stack(rows).astype(np.float32))


def load_cifar_sift(n, seed):
    if not TRAIN_DESC_PATH.exists():
        raise FileNotFoundError(
            f"{TRAIN_DESC_PATH} not found. Run download_cifar_sift.py first."
        )
    all_descs = load_descriptors(TRAIN_DESC_PATH)
    if 2 * n > len(all_descs):
        raise ValueError(
            f"not enough CIFAR SIFT descriptor sets: need {2*n:,}, have {len(all_descs):,}"
        )
    red_descs, blue_descs = sample_pair(all_descs, n, seed)
    return _descriptor_means(red_descs), _descriptor_means(blue_descs)


def load_embeddings(path):
    with gzip.open(path, "rb") as f:
        embs = pickle.load(f)
    print(f"  Loaded {len(embs):,} document embeddings from {Path(path).name}", flush=True)
    return embs


def _embedding_means(descs):
    rows = []
    for d in descs:
        arr = np.asarray(d[:MAX_WORDS] if len(d) > MAX_WORDS else d, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] == 0:
            raise ValueError("embedding set must be a non-empty 2D array")
        rows.append(arr.mean(axis=0))
    return torch.from_numpy(np.stack(rows).astype(np.float32))


def load_newsgroups(n, seed):
    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(
            f"{EMBEDDINGS_PATH} not found. Run download_newsgroups_glove.py first."
        )
    all_embs = load_embeddings(EMBEDDINGS_PATH)
    if 2 * n > len(all_embs):
        raise ValueError(
            f"not enough Newsgroups documents: need {2*n:,}, have {len(all_embs):,}"
        )
    red_descs, blue_descs = sample_pair(all_embs, n, seed)
    return _embedding_means(red_descs), _embedding_means(blue_descs)


DATASET_CONFIGS = [
    {"name": "Synthetic-2D", "type": "synthetic"},
    {"name": "EMNIST", "type": "real", "loader": load_emnist, "max_n": None},
    {"name": "CIFAR-SIFT", "type": "real", "loader": load_cifar_sift},
    {"name": "Newsgroups", "type": "real", "loader": load_newsgroups},
    {"name": "NYC Taxi", "type": "skip"},
]


def _dataset_key(name):
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _select_dataset_configs(dataset_filter):
    if dataset_filter is None:
        return DATASET_CONFIGS

    if isinstance(dataset_filter, str):
        requested = [part.strip() for part in dataset_filter.split(",") if part.strip()]
    else:
        requested = [str(part).strip() for part in dataset_filter if str(part).strip()]

    if not requested:
        return DATASET_CONFIGS

    configs_by_key = {_dataset_key(cfg["name"]): cfg for cfg in DATASET_CONFIGS}
    selected = []
    unknown = []
    seen = set()
    for name in requested:
        key = _dataset_key(name)
        cfg = configs_by_key.get(key)
        if cfg is None:
            unknown.append(name)
            continue
        if key not in seen:
            selected.append(cfg)
            seen.add(key)

    if unknown:
        valid = ", ".join(cfg["name"] for cfg in DATASET_CONFIGS)
        raise ValueError(f"Unknown dataset filter(s): {', '.join(unknown)}. Valid datasets: {valid}")

    return selected


def _is_cuda_oom(exc):
    return isinstance(exc, torch.cuda.OutOfMemoryError) or (
        isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()
    )


def _dense_alloc_test(n, device):
    if device.type != "cuda":
        return {"status": "skip", "mem_gb": float("nan"), "time_ms": float("nan")}
    try:
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        M = torch.empty(n, n, dtype=torch.float32, device=device)
        torch.cuda.synchronize(device)
        elapsed = (time.perf_counter() - t0) * 1000.0
        peak = torch.cuda.max_memory_allocated(device)
        del M
        torch.cuda.empty_cache()
        return {"status": "ok", "mem_gb": peak / 1e9, "time_ms": elapsed}
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {"status": "oom", "mem_gb": float("nan"), "time_ms": float("nan")}


def _run_solver2(red, blue, device):
    try:
        if ot is None:
            raise RuntimeError("POT not installed")
        n = red.shape[0]
        a = np.full(n, 1.0 / n, np.float64)
        b = np.full(n, 1.0 / n, np.float64)
        torch.cuda.reset_peak_memory_stats(device)
        t0 = time.perf_counter()
        engine = SimpleL1Clustering(epsilon=EPSILON, tile_size=TILE_SIZE)
        c = engine.run(red, blue)
        C = build_two_level_proxy_matrix(c, n, device)
        plan = ot.emd(a, b, C, numItermax=10**6)
        elapsed = (time.perf_counter() - t0) * 1000.0
        match = torch.from_numpy(plan.argmax(axis=1).astype(np.int64))
        cost = (blue - red[match.to(device)]).abs().sum(dim=1).mean().item()
        torch.cuda.synchronize(device)
        peak_mem_gb = torch.cuda.max_memory_allocated(device) / 1e9
        return elapsed, cost, peak_mem_gb, "ok"
    except Exception as exc:
        if _is_cuda_oom(exc):
            torch.cuda.empty_cache()
            return float("nan"), float("nan"), float("nan"), "oom"
        torch.cuda.empty_cache()
        print(f"    2L error: {exc}", flush=True)
        return float("nan"), float("nan"), float("nan"), "error"


def _run_solver3(red, blue, device):
    try:
        if ot is None:
            raise RuntimeError("POT not installed")
        n = red.shape[0]
        a = np.full(n, 1.0 / n, np.float64)
        b = np.full(n, 1.0 / n, np.float64)
        torch.cuda.reset_peak_memory_stats(device)
        t0 = time.perf_counter()
        engine = ThreeLevelL1Clustering(epsilon=EPSILON, tile_size=TILE_SIZE)
        c = engine.run(red, blue)
        C = build_three_level_proxy_matrix(c, n, device)
        plan = ot.emd(a, b, C, numItermax=10**6)
        elapsed = (time.perf_counter() - t0) * 1000.0
        match = torch.from_numpy(plan.argmax(axis=1).astype(np.int64))
        cost = (blue - red[match.to(device)]).abs().sum(dim=1).mean().item()
        torch.cuda.synchronize(device)
        peak_mem_gb = torch.cuda.max_memory_allocated(device) / 1e9
        return elapsed, cost, peak_mem_gb, "ok"
    except Exception as exc:
        if _is_cuda_oom(exc):
            torch.cuda.empty_cache()
            return float("nan"), float("nan"), float("nan"), "oom"
        torch.cuda.empty_cache()
        print(f"    3L error: {exc}", flush=True)
        return float("nan"), float("nan"), float("nan"), "error"


def _make_row(ds_name, n, status):
    nan = float("nan")
    return {
        "dataset": ds_name,
        "dataset_label": ds_name,
        "n": n,
        "dense_mem_gb": nan,
        "dense_status": status,
        "sol2_mem_gb": nan,
        "sol2_time_ms": nan,
        "sol2_cost": nan,
        "sol2_status": status,
        "sol3_mem_gb": nan,
        "sol3_time_ms": nan,
        "sol3_cost": nan,
        "sol3_status": status,
    }


def _label_row(row, seen_dataset):
    row["dataset_label"] = row["dataset"] if not seen_dataset else ""
    return row


def run(device, **kwargs):
    rows = []
    dataset_filter = kwargs.get("dataset", kwargs.get("datasets"))
    dataset_configs = _select_dataset_configs(dataset_filter)

    if dataset_filter is not None:
        selected_names = ", ".join(cfg["name"] for cfg in dataset_configs)
        print(f"Dataset filter: {selected_names}", flush=True)

    for ds_cfg in dataset_configs:
        ds_name = ds_cfg["name"]
        if ds_cfg["type"] == "skip":
            print(f"\nSkipping {ds_name}: data file is not present.", flush=True)
            continue

        print(f"\n{'='*60}", flush=True)
        print(f"  Dataset: {ds_name}", flush=True)
        print(f"{'='*60}", flush=True)

        dense_oom = False
        solver2_oom = False
        solver3_oom = False
        dataset_has_row = False

        for n in N_VALUES:
            print(f"\n  N = {n:,}", flush=True)

            if ds_cfg["type"] == "synthetic":
                torch.manual_seed(SEED)
                red = torch.rand(n, 2, device=device)
                blue = torch.rand(n, 2, device=device)
                load_status = "ok"
            else:
                try:
                    red, blue = ds_cfg["loader"](n, SEED)
                    red, blue = red.to(device), blue.to(device)
                    load_status = "ok"
                except FileNotFoundError as exc:
                    print(f"    Data not found: {exc} — skipping dataset", flush=True)
                    break
                except ValueError as exc:
                    print(f"    Dataset exhausted at N={n:,}: {exc}", flush=True)
                    rows.append(_label_row(_make_row(ds_name, n, "exhausted"), dataset_has_row))
                    dataset_has_row = True
                    break
                except Exception as exc:
                    print(f"    Load error: {exc}", flush=True)
                    rows.append(_label_row(_make_row(ds_name, n, "error"), dataset_has_row))
                    dataset_has_row = True
                    continue

            if load_status != "ok":
                continue

            if not dense_oom:
                d_res = _dense_alloc_test(n, device)
                print(f"    Dense: {d_res['status']}  mem={d_res['mem_gb']:.2f}GB", flush=True)
                if d_res["status"] == "oom":
                    dense_oom = True
            else:
                d_res = {"status": "oom", "mem_gb": float("nan"), "time_ms": float("nan")}

            if not solver2_oom:
                t2, c2, m2, s2 = _run_solver2(red, blue, device)
                print(f"    2L: {s2}  mem={m2:.2f}GB  time={t2:.0f}ms", flush=True)
                if s2 == "oom":
                    solver2_oom = True
            else:
                t2, c2, m2, s2 = float("nan"), float("nan"), float("nan"), "oom"

            if not solver3_oom:
                t3, c3, m3, s3 = _run_solver3(red, blue, device)
                print(f"    3L: {s3}  mem={m3:.2f}GB  time={t3:.0f}ms", flush=True)
                if s3 == "oom":
                    solver3_oom = True
            else:
                t3, c3, m3, s3 = float("nan"), float("nan"), float("nan"), "oom"

            row = {
                "dataset": ds_name,
                "n": n,
                "dense_mem_gb": d_res["mem_gb"],
                "dense_status": d_res["status"],
                "sol2_mem_gb": m2,
                "sol2_time_ms": t2,
                "sol2_cost": c2,
                "sol2_status": s2,
                "sol3_mem_gb": m3,
                "sol3_time_ms": t3,
                "sol3_cost": c3,
                "sol3_status": s3,
            }
            rows.append(_label_row(row, dataset_has_row))
            dataset_has_row = True

            torch.cuda.empty_cache()

            if dense_oom and solver2_oom and solver3_oom:
                print(f"    All methods OOM — stopping sweep for {ds_name}", flush=True)
                break

    return rows


def _fmt_mem(row, value_key, status_key):
    status = row[status_key]
    value = row[value_key]
    if status == "oom":
        return "OOM"
    if status != "ok" or math.isnan(float(value)):
        return "N/A"
    return f"{float(value):.2f} GB"


COL_SPECS = [
    ("Dataset", 14),
    ("N", 9),
    ("Dense Mem", 11),
    ("Dense", 7),
    ("2L Mem", 10),
    ("2L Time", 11),
    ("2L", 7),
    ("3L Mem", 10),
    ("3L Time", 11),
    ("3L", 7),
]

FMT_FNS = {
    "Dataset": lambda r: f"{r.get('dataset_label', r['dataset']):<14}",
    "N": lambda r: f"{r['n']:,}",
    "Dense Mem": lambda r: _fmt_mem(r, "dense_mem_gb", "dense_status"),
    "Dense": lambda r: r["dense_status"],
    "2L Mem": lambda r: _fmt_mem(r, "sol2_mem_gb", "sol2_status"),
    "2L Time": lambda r: fmt_time(r["sol2_time_ms"]),
    "2L": lambda r: r["sol2_status"],
    "3L Mem": lambda r: _fmt_mem(r, "sol3_mem_gb", "sol3_status"),
    "3L Time": lambda r: fmt_time(r["sol3_time_ms"]),
    "3L": lambda r: r["sol3_status"],
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Experiment 13 scalability limits.")
    parser.add_argument(
        "--dataset",
        default=None,
        help="Optional dataset filter, e.g. EMNIST. Comma-separated values are allowed.",
    )
    args = parser.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = run(dev, dataset=args.dataset)
    for r in results:
        print(
            f"{r.get('dataset_label', r['dataset']):<14} N={r['n']:>9,} "
            f"dense={r['dense_status']} 2L={r['sol2_status']} 3L={r['sol3_status']}"
        )
