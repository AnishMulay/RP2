#!/usr/bin/env python3
"""
Experiment 16 — Cleanup Matching Slack.

The 2-level push-relabel solver stops once at most epsilon * N blues remain
free, then matches those remaining free blues to free reds arbitrarily.  Those
cleanup edges are not required to be admissible.  This experiment measures the
slack introduced by that final arbitrary matching step.
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch

FINAL2_DIR = Path(__file__).resolve().parent.parent
BASE_DIR = FINAL2_DIR.parent.parent.parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(FINAL2_DIR) not in sys.path:
    sys.path.insert(0, str(FINAL2_DIR))

from clustered_push_relabel.clustering.simple_l1 import SimpleL1Clustering
from clustered_push_relabel.clustering.simple_precomputed import SimplePrecomputedClustering
from clustered_push_relabel.solvers.simple_bipartite import SimpleGPUSolver
from shared import compute_ratio, fmt_cost, fmt_iters, fmt_ratio, fmt_time

from experiments import exp05_nyc_scalability as exp05
from experiments import exp10_mnist_proxy_dissimilar as exp10
from experiments import exp15_old_vs_new_proxy as exp15


EXP_ID = 16
EXP_NAME = "Cleanup Matching Slack — Arbitrary Final Matches"
DATASET = "Multi"

EPSILON = 0.01
BATCH_SIZE = 512
NYC_BATCH_SIZE = 2048
SEED = 42

DEFAULT_N_BY_DATASET = {
    "synthetic2d": [500, 1_000, 1_500, 2_000, 2_500],
    "mnist_equal": [5_000, 10_000, 15_000, 20_000, 25_000],
    "mnist_biased": [5_000, 10_000, 15_000, 20_000, 25_000],
    "mnist_dissimilar": [500, 1_000, 2_000, 3_000, 5_000],
    "emnist_equal": [5_000, 10_000, 15_000, 20_000, 25_000],
    "emnist_biased": [5_000, 10_000, 15_000, 20_000, 25_000],
    "cifar_sift": [1_000, 2_000, 3_000, 5_000, 7_000, 10_000],
    "newsgroups": [1_000, 2_000, 3_000, 5_000, 7_000, 10_000],
    "nyc_taxi": [1_000, 5_000, 10_000, 50_000, 100_000, 200_000],
}
DEFAULT_DATASETS = list(DEFAULT_N_BY_DATASET.keys())


@dataclass(frozen=True)
class CleanupCase:
    red: torch.Tensor | None
    blue: torch.Tensor | None
    metric: str
    distance_note: str
    scale: float = 1.0
    red_descs: list | None = None
    blue_descs: list | None = None


DATASET_NAMES = {
    "synthetic2d": "Synthetic2D",
    "mnist_equal": "MNIST-Equal",
    "mnist_biased": "MNIST-Biased",
    "mnist_dissimilar": "MNIST-Dissim",
    "emnist_equal": "EMNIST-Equal",
    "emnist_biased": "EMNIST-Biased",
    "cifar_sift": "CIFAR-SIFT",
    "newsgroups": "Newsgroups",
    "nyc_taxi": "NYC-Taxi",
}


COL_SPECS = [
    ("Dataset", 13),
    ("N", 8),
    ("Cleanup", 8),
    ("Exact OT", 11),
    ("Proxy Slack", 12),
    ("True Slack", 12),
    ("Proxy Frac", 10),
    ("True Frac", 10),
    ("Solve", 10),
    ("Iters", 9),
]

FMT_FNS = {
    "Dataset": lambda r: r.get("dataset_label", r["dataset"]),
    "N": lambda r: f"{r['n']:,}",
    "Cleanup": lambda r: fmt_iters(r["cleanup_count"]),
    "Exact OT": lambda r: fmt_cost(r["exact_cost"]),
    "Proxy Slack": lambda r: fmt_cost(r["proxy_slack_cost_sum"]),
    "True Slack": lambda r: fmt_cost(r["true_slack_cost_sum"]),
    "Proxy Frac": lambda r: fmt_ratio(r["proxy_slack_fraction"]),
    "True Frac": lambda r: fmt_ratio(r["true_slack_fraction"]),
    "Solve": lambda r: fmt_time(r["solve_time_ms"]),
    "Iters": lambda r: fmt_iters(r["iterations"]),
}

DIAG_COL_SPECS = [
    ("Dataset", 13),
    ("N", 8),
    ("Status", 8),
    ("Exact Time", 11),
    ("Direct", 8),
    ("Proxy Avg", 10),
    ("True Avg", 10),
    ("Proxy Min", 10),
    ("True Min", 10),
]

DIAG_FMT_FNS = {
    "Dataset": lambda r: r.get("dataset_label", r["dataset"]),
    "N": lambda r: f"{r['n']:,}",
    "Status": lambda r: r["status"],
    "Exact Time": lambda r: fmt_time(r["exact_time_ms"]),
    "Direct": lambda r: fmt_iters(r["cleanup_direct_count"]),
    "Proxy Avg": lambda r: fmt_cost(r["proxy_slack_cost_avg"]),
    "True Avg": lambda r: fmt_cost(r["true_slack_cost_avg"]),
    "Proxy Min": lambda r: fmt_cost(r["proxy_slack_cost_min"]),
    "True Min": lambda r: fmt_cost(r["true_slack_cost_min"]),
}


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _clear(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _time_ms(device: torch.device, fn: Callable):
    _sync(device)
    t0 = time.perf_counter()
    out = fn()
    _sync(device)
    return out, (time.perf_counter() - t0) * 1000.0


def _cdist(x: torch.Tensor, y: torch.Tensor, metric: str) -> torch.Tensor:
    if metric == "l1":
        return torch.cdist(x, y, p=1)
    if metric == "l2":
        return torch.cdist(x, y, p=2, compute_mode="use_mm_for_euclid_dist_if_necessary")
    raise ValueError(f"unsupported vector metric: {metric}")


def _as_cleanup_case(case: exp15.LoadedCase) -> CleanupCase:
    return CleanupCase(
        red=case.red,
        blue=case.blue,
        metric=case.metric,
        distance_note=case.distance_note,
        red_descs=case.red_descs,
        blue_descs=case.blue_descs,
    )


def load_mnist_dissimilar(n: int, seed: int, _device: torch.device) -> CleanupCase:
    red, blue = exp10.load_mnist_dissimilar(n, seed)
    return CleanupCase(
        red=red,
        blue=blue,
        metric="l1",
        distance_note="L1 on probability-normalized MNIST histograms; dissimilar digit split",
    )


_TAXI_CACHE = {}


def load_nyc_taxi(n: int, seed: int, _device: torch.device) -> CleanupCase:
    data_path = pathlib.Path(_NYC_DATA_PATH)
    if not data_path.exists():
        raise FileNotFoundError(data_path)
    cache_key = (str(data_path), _NYC_DAY)
    if cache_key not in _TAXI_CACHE:
        _TAXI_CACHE[cache_key] = exp05.load_taxi(data_path, _NYC_DAY)
    df = _TAXI_CACHE[cache_key]
    rng = np.random.default_rng(seed)
    result = exp05.make_points(df, n, rng)
    if result[0] is None:
        raise ValueError(f"not enough taxi rows: need {2*n:,}, have {len(df):,}")
    red_np, blue_np, diameter = result
    return CleanupCase(
        red=torch.from_numpy(red_np),
        blue=torch.from_numpy(blue_np),
        metric="l2",
        distance_note="L2 on projected NYC taxi points, normalized for solving and reported in metres",
        scale=float(diameter),
    )


def _load_from_exp15(key: str, n: int, seed: int, device: torch.device) -> CleanupCase:
    loader = exp15.DATASET_LOADERS[key][1]
    return _as_cleanup_case(loader(n, seed, device))


DATASET_LOADERS = {
    "synthetic2d": lambda n, seed, device: _load_from_exp15("synthetic2d", n, seed, device),
    "mnist_equal": lambda n, seed, device: _load_from_exp15("mnist_equal", n, seed, device),
    "mnist_biased": lambda n, seed, device: _load_from_exp15("mnist_biased", n, seed, device),
    "mnist_dissimilar": load_mnist_dissimilar,
    "emnist_equal": lambda n, seed, device: _load_from_exp15("emnist_equal", n, seed, device),
    "emnist_biased": lambda n, seed, device: _load_from_exp15("emnist_biased", n, seed, device),
    "cifar_sift": lambda n, seed, device: _load_from_exp15("cifar_sift", n, seed, device),
    "newsgroups": lambda n, seed, device: _load_from_exp15("newsgroups", n, seed, device),
    "nyc_taxi": load_nyc_taxi,
}

_NYC_DATA_PATH = exp05.DEFAULT_DATA_PATH
_NYC_DAY = None


def _run_exact_vector(case: CleanupCase, device: torch.device) -> tuple[float, float]:
    assert case.red is not None and case.blue is not None
    red = case.red.to(device)
    blue = case.blue.to(device)
    C = _cdist(blue, red, case.metric).to(torch.float32) * float(case.scale)
    try:
        return exp15.run_emd2(C)
    finally:
        del C, red, blue
        _clear(device)


def _run_solver_vector(case: CleanupCase, device: torch.device):
    assert case.red is not None and case.blue is not None
    red = case.red.to(device)
    blue = case.blue.to(device)
    clustering_class = SimpleL1Clustering if case.metric == "l1" else None
    tile_size = NYC_BATCH_SIZE if case.metric == "l2" and red.shape[1] == 2 else BATCH_SIZE
    solver = SimpleGPUSolver(
        red,
        blue,
        EPSILON,
        batch_size=tile_size,
        verbose=False,
        diameter=1.0,
        clustering_class=clustering_class,
    )
    solver.solve()
    return solver, red, blue


def _run_precomputed(case: CleanupCase, device: torch.device):
    loaded = exp15.LoadedCase(
        red=None,
        blue=None,
        metric=case.metric,
        distance_note=case.distance_note,
        red_descs=case.red_descs,
        blue_descs=case.blue_descs,
    )
    bundle = exp15.build_distance_bundle(loaded, device)
    C_eval = bundle.C_true * float(bundle.scale)
    exact = exp15.run_emd2(C_eval)
    del C_eval
    clustering = SimplePrecomputedClustering(epsilon=EPSILON, tile_size=BATCH_SIZE).run(
        bundle.D_rr, bundle.C_true
    )
    solver = SimpleGPUSolver(
        None,
        None,
        EPSILON,
        batch_size=BATCH_SIZE,
        verbose=False,
        diameter=1.0,
        precomputed_clustering=clustering,
    )
    _sync(device)
    t0 = time.perf_counter()
    solver.solve()
    _sync(device)
    solve_time_ms = (time.perf_counter() - t0) * 1000.0
    return exact, solver, bundle, clustering, solve_time_ms


def _cleanup_proxy_ints(solver: SimpleGPUSolver) -> tuple[torch.Tensor, int]:
    cleanup_b = solver.cleanup_blues.to(dtype=torch.long)
    if cleanup_b.numel() == 0:
        return torch.empty(0, device=solver.device, dtype=torch.long), 0

    cleanup_a = solver.match_B[cleanup_b].to(dtype=torch.long)
    proxy = (
        solver.d_min_b_int[cleanup_b].to(torch.long)
        + solver.y_A[cleanup_a].to(torch.long)
        - solver.V[solver.nearest_s[cleanup_b], cleanup_a].to(torch.long)
    )

    direct_count = 0
    for i, b_value in enumerate(cleanup_b.detach().cpu().tolist()):
        a_value = cleanup_a[i]
        start = int(solver.adj_ptr[b_value].item())
        end = int(solver.adj_ptr[b_value + 1].item())
        if start == end:
            continue
        adj = solver.adj_col[start:end]
        hit = torch.nonzero(adj == a_value, as_tuple=True)[0]
        if hit.numel() > 0:
            proxy[i] = solver.adj_dist_int[start + int(hit[0].item())].to(torch.long)
            direct_count += 1

    return proxy, direct_count


def _vector_true_cleanup_costs(
    red: torch.Tensor,
    blue: torch.Tensor,
    metric: str,
    cleanup_b: torch.Tensor,
    cleanup_a: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    if cleanup_b.numel() == 0:
        return torch.empty(0, device=blue.device, dtype=torch.float32)
    diffs = blue[cleanup_b] - red[cleanup_a]
    if metric == "l1":
        costs = diffs.abs().sum(dim=1)
    elif metric == "l2":
        costs = torch.norm(diffs, p=2, dim=1)
    else:
        raise ValueError(f"unsupported vector metric: {metric}")
    return costs.to(torch.float32) * float(scale)


def _summarize_slack(
    solver: SimpleGPUSolver,
    true_costs: torch.Tensor,
    scale: float,
    exact_cost: float,
    exact_time_ms: float,
    solve_time_ms: float,
    display_name: str,
    n: int,
    metric: str,
    distance_note: str,
) -> dict:
    cleanup_b = solver.cleanup_blues.to(dtype=torch.long)
    cleanup_a = solver.match_B[cleanup_b].to(dtype=torch.long) if cleanup_b.numel() else cleanup_b
    proxy_int, direct_count = _cleanup_proxy_ints(solver)

    dual_int = (
        solver.y_B[cleanup_b].to(torch.float32)
        + solver.y_A[cleanup_a].to(torch.float32)
        if cleanup_b.numel()
        else torch.empty(0, device=solver.device, dtype=torch.float32)
    )
    unit = float(EPSILON) * float(scale)
    proxy_slack_total_scale = (proxy_int.to(torch.float32) - dual_int) * unit
    true_slack_total_scale = true_costs.to(solver.device, dtype=torch.float32) - dual_int * unit
    proxy_slack = proxy_slack_total_scale / float(n)
    true_slack = true_slack_total_scale / float(n)

    proxy_total = float(proxy_slack_total_scale.sum().item()) if proxy_slack.numel() else 0.0
    true_total = float(true_slack_total_scale.sum().item()) if true_slack.numel() else 0.0
    proxy_sum = float(proxy_slack.sum().item()) if proxy_slack.numel() else 0.0
    true_sum = float(true_slack.sum().item()) if true_slack.numel() else 0.0

    return {
        "dataset": display_name,
        "n": n,
        "status": "ok",
        "error": "",
        "metric": metric,
        "distance_note": distance_note,
        "epsilon": EPSILON,
        "scale": float(scale),
        "exact_cost": float(exact_cost),
        "exact_total_cost": float(exact_cost) * float(n),
        "exact_time_ms": float(exact_time_ms),
        "solve_time_ms": float(solve_time_ms),
        "iterations": int(solver.iterations),
        "cleanup_count": int(cleanup_b.numel()),
        "cleanup_direct_count": int(direct_count),
        "proxy_slack_cost_total": proxy_total,
        "true_slack_cost_total": true_total,
        "proxy_slack_cost_sum": proxy_sum,
        "true_slack_cost_sum": true_sum,
        "proxy_slack_cost_avg": proxy_sum / max(int(cleanup_b.numel()), 1),
        "true_slack_cost_avg": true_sum / max(int(cleanup_b.numel()), 1),
        "proxy_slack_cost_min": float(proxy_slack.min().item()) if proxy_slack.numel() else 0.0,
        "true_slack_cost_min": float(true_slack.min().item()) if true_slack.numel() else 0.0,
        "proxy_slack_cost_max": float(proxy_slack.max().item()) if proxy_slack.numel() else 0.0,
        "true_slack_cost_max": float(true_slack.max().item()) if true_slack.numel() else 0.0,
        "proxy_slack_fraction": compute_ratio(exact_cost, proxy_sum),
        "true_slack_fraction": compute_ratio(exact_cost, true_sum),
    }


def _skip_row(display_name: str, n: int, exc: Exception | str) -> dict:
    nan = math.nan
    return {
        "dataset": display_name,
        "n": n,
        "status": "skip",
        "error": str(exc),
        "metric": "",
        "distance_note": "",
        "epsilon": EPSILON,
        "scale": nan,
        "exact_cost": nan,
        "exact_total_cost": nan,
        "exact_time_ms": nan,
        "solve_time_ms": nan,
        "iterations": nan,
        "cleanup_count": nan,
        "cleanup_direct_count": nan,
        "proxy_slack_cost_total": nan,
        "true_slack_cost_total": nan,
        "proxy_slack_cost_sum": nan,
        "true_slack_cost_sum": nan,
        "proxy_slack_cost_avg": nan,
        "true_slack_cost_avg": nan,
        "proxy_slack_cost_min": nan,
        "true_slack_cost_min": nan,
        "proxy_slack_cost_max": nan,
        "true_slack_cost_max": nan,
        "proxy_slack_fraction": nan,
        "true_slack_fraction": nan,
    }


def run_one_case(dataset_key: str, n: int, device: torch.device, seed: int) -> dict:
    display_name = DATASET_NAMES[dataset_key]
    print(f"\n  Dataset={display_name}  N={n:,}", flush=True)

    if device.type != "cuda":
        raise RuntimeError("cleanup slack experiment requires CUDA push-relabel solver")

    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    case = DATASET_LOADERS[dataset_key](n, seed, device)
    print(f"    Distance: {case.distance_note}", flush=True)

    if case.metric == "precomputed_chamfer":
        (exact_cost, exact_time), solver, bundle, clustering, solve_time = _run_precomputed(case, device)
        cleanup_b = solver.cleanup_blues.to(dtype=torch.long)
        cleanup_a = solver.match_B[cleanup_b].to(dtype=torch.long) if cleanup_b.numel() else cleanup_b
        true_costs = (
            bundle.C_true[cleanup_b, cleanup_a] * float(bundle.scale)
            if cleanup_b.numel()
            else torch.empty(0, device=device, dtype=torch.float32)
        )
        scale = float(bundle.scale)
        row = _summarize_slack(
            solver, true_costs, scale, exact_cost, exact_time, solve_time,
            display_name, n, case.metric, case.distance_note
        )
        del solver, bundle, clustering, true_costs
        _clear(device)
        return row

    exact, exact_wall_time = _time_ms(device, lambda: _run_exact_vector(case, device))
    exact_cost, exact_ot_time = exact
    print(f"    Exact OT: cost={exact_cost:.6f} time={exact_ot_time:.0f} ms", flush=True)

    (solver, red, blue), solve_time = _time_ms(device, lambda: _run_solver_vector(case, device))
    cleanup_b = solver.cleanup_blues.to(dtype=torch.long)
    cleanup_a = solver.match_B[cleanup_b].to(dtype=torch.long) if cleanup_b.numel() else cleanup_b
    true_costs = _vector_true_cleanup_costs(
        red, blue, case.metric, cleanup_b, cleanup_a, float(case.scale)
    )
    row = _summarize_slack(
        solver, true_costs, float(case.scale), exact_cost, exact_wall_time,
        solve_time, display_name, n, case.metric, case.distance_note
    )
    print(
        f"    Cleanup={row['cleanup_count']:,}  "
        f"proxy_slack={row['proxy_slack_cost_sum']:.6f} "
        f"true_slack={row['true_slack_cost_sum']:.6f}",
        flush=True,
    )
    del solver, red, blue, true_costs
    _clear(device)
    return row


def _label_row(row: dict, dataset_has_row: bool) -> dict:
    row["dataset_label"] = "" if dataset_has_row else row["dataset"]
    return row


def run(device: torch.device, **kwargs) -> list[dict]:
    global _NYC_DATA_PATH, _NYC_DAY

    if kwargs.get("nyc_data_path"):
        _NYC_DATA_PATH = pathlib.Path(kwargs["nyc_data_path"])
    else:
        _NYC_DATA_PATH = exp05.DEFAULT_DATA_PATH
    _NYC_DAY = kwargs.get("nyc_day")

    dataset_keys = list(kwargs.get("dataset_keys", DEFAULT_DATASETS))
    n_values_by_dataset = kwargs.get("n_values_by_dataset", DEFAULT_N_BY_DATASET)
    seed = int(kwargs.get("seed", SEED))

    print(f"\n{'=' * 65}", flush=True)
    print(f"  Exp {EXP_ID}: {EXP_NAME}", flush=True)
    print(f"  Device: {device}  epsilon={EPSILON}", flush=True)
    print(f"{'=' * 65}", flush=True)
    print(
        "  Slack costs are reported as sum(cleanup slack) / N; fractions divide by exact OT cost.",
        flush=True,
    )

    rows = []
    for key in dataset_keys:
        dataset_has_row = False
        for n in n_values_by_dataset[key]:
            display_name = DATASET_NAMES[key]
            try:
                row = run_one_case(key, int(n), device, seed)
            except Exception as exc:
                print(f"    Skipping {display_name} N={int(n):,}: {exc}", flush=True)
                row = _skip_row(display_name, int(n), exc)
                _clear(device)
            rows.append(_label_row(row, dataset_has_row))
            dataset_has_row = True
    return rows


def print_table(rows: list[dict]) -> None:
    headers = [h for h, _ in COL_SPECS]
    widths = [max(w, len(h)) for h, w in COL_SPECS]
    print()
    print(" | ".join(f"{h:>{w}}" for h, w in zip(headers, widths)))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(" | ".join(f"{FMT_FNS[h](row):>{w}}" for h, w in zip(headers, widths)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nyc-data", default=None)
    ap.add_argument("--nyc-day", default=None)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = run(dev, nyc_data_path=args.nyc_data, nyc_day=args.nyc_day)
    print_table(results)
