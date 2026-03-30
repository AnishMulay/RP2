#!/usr/bin/env python3
"""GloVe-100 recall experiment: weak HNSW seeds + cover expansion vs strong HNSW alone.

Downloads glove-100-angular.hdf5 from ann-benchmarks automatically if not present.
The key comparison is:
  - HNSW(M=16) alone              [cheap index, baseline]
  - HNSW(M=64) alone              [expensive index, strong baseline]
  - HNSW(M=16) + cluster_search   [cheap index + cover expansion, proposed method]

Usage:
    python experiments/runners/glove_recall_experiment.py
    python experiments/runners/glove_recall_experiment.py --n 10000
    python experiments/runners/glove_recall_experiment.py --n 100000
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys
import time
import urllib.request

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import h5py
except ImportError as exc:
    raise ImportError("h5py is required: pip install h5py") from exc

try:
    import hnswlib
except ImportError as exc:
    raise ImportError("hnswlib is required: pip install hnswlib") from exc

BASE_DIR = pathlib.Path(__file__).resolve().parents[2]
SRC_DIR  = BASE_DIR / "src"
DATA_DIR = BASE_DIR / "data"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cluster_search import CoverIndex, build_cover, cluster_search
from clustered_push_relabel.utils.distance import TiledEuclideanKernel

GLOVE_URL      = "http://ann-benchmarks.com/glove-100-angular.hdf5"
GLOVE_FILENAME = "glove-100-angular.hdf5"

K_VALUES       = [10, 20, 50, 100]
EF_CONSTRUCTION = 200

# ---------------------------------------------------------------------------
# Dataset download + load
# ---------------------------------------------------------------------------

def _download_with_progress(url: str, dest: pathlib.Path) -> None:
    print(f"  Downloading {url}")
    print(f"  -> {dest}")

    def _progress(block_count, block_size, total_size):
        if total_size <= 0:
            return
        downloaded = block_count * block_size
        pct = min(downloaded / total_size * 100, 100)
        bar = int(pct // 2)
        print(f"\r  [{'#' * bar:<50}] {pct:5.1f}%", end="", flush=True)

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    try:
        urllib.request.urlretrieve(url, tmp, reporthook=_progress)
        tmp.rename(dest)
        print()
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def ensure_glove(data_dir: pathlib.Path) -> pathlib.Path:
    path = data_dir / GLOVE_FILENAME
    if path.exists():
        print(f"  Found cached dataset at {path}")
        return path
    _download_with_progress(GLOVE_URL, path)
    return path


def load_glove(
    path: pathlib.Path,
    n: int | None,
    n_queries: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (base, queries, ground_truth_neighbors).

    ground_truth_neighbors shape: (n_queries, 100) — top-100 neighbor indices
    into the returned base array.  When n < full dataset size, neighbors are
    filtered to only those that fall within [0, n).
    """
    with h5py.File(path, "r") as f:
        full_base    = f["train"][:]          # (1183514, 100) float32
        full_queries = f["test"][:]           # (10000,   100) float32
        full_gt      = f["neighbors"][:, :100]  # (10000, 100) int32

    if n is not None:
        full_base = full_base[:n]
        # filter ground truth neighbors to those within [0, n)
        filtered_gt = []
        for row in full_gt[:n_queries]:
            valid = [idx for idx in row if idx < n]
            filtered_gt.append(valid)
        # pad / truncate to 100 columns
        gt = np.full((n_queries, 100), -1, dtype=np.int32)
        for i, row in enumerate(filtered_gt):
            gt[i, :len(row)] = row
        full_gt = gt

    queries = full_queries[:n_queries]
    gt      = full_gt[:n_queries]
    return full_base.astype(np.float32), queries.astype(np.float32), gt


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GloVe-100 recall experiment")
    p.add_argument("--n", type=int, default=10_000,
                   help="Number of base vectors to use (default 10000; max ~1.18M)")
    p.add_argument("--n_queries", type=int, default=100,
                   help="Number of query vectors to evaluate")
    p.add_argument("--epsilon", type=float, default=0.01)
    p.add_argument("--k_cluster", type=int, default=4)
    p.add_argument("--M_weak", type=int, default=16,
                   help="HNSW M for the weak (cheap) index used as seeds")
    p.add_argument("--M_strong", type=int, default=64,
                   help="HNSW M for the strong (expensive) index baseline")
    p.add_argument("--k_search_multiplier", type=int, default=4,
                   help="k_prime passed to cluster_search = max(K_VALUES) * multiplier")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Index helpers
# ---------------------------------------------------------------------------

def build_hnsw(dataset_cpu: np.ndarray, m: int, label: str) -> hnswlib.Index:
    n, dim = dataset_cpu.shape
    print(f"  Building HNSW M={m} ({label})...")
    t0 = time.perf_counter()
    idx = hnswlib.Index(space="l2", dim=dim)
    idx.init_index(max_elements=n, ef_construction=EF_CONSTRUCTION, M=m)
    idx.add_items(dataset_cpu, np.arange(n, dtype=np.int32))
    print(f"    done in {time.perf_counter() - t0:.2f}s")
    return idx


def query_hnsw(idx: hnswlib.Index, qvec: np.ndarray, k: int) -> list[int]:
    idx.set_ef(max(k * 2, 50))
    labels, _ = idx.knn_query(qvec, k=k)
    return [int(x) for x in labels[0]]


def topk_from_distances(
    distances_sq: torch.Tensor,
    k: int,
    candidate_ids: list[int] | None = None,
) -> list[int]:
    if candidate_ids is None:
        k_eff = min(k, distances_sq.numel())
        return torch.topk(distances_sq, k=k_eff, largest=False).indices.tolist()
    if not candidate_ids:
        return []
    ct = torch.tensor(candidate_ids, device=distances_sq.device, dtype=torch.long)
    cd = distances_sq.index_select(0, ct)
    k_eff = min(k, ct.numel())
    return ct[torch.topk(cd, k=k_eff, largest=False).indices].tolist()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    dataset: torch.Tensor,
    queries: torch.Tensor,
    queries_cpu: np.ndarray,
    ground_truth: np.ndarray,
    cover_index: CoverIndex,
    hnsw_weak: hnswlib.Index,
    hnsw_strong: hnswlib.Index,
    epsilon: float,
    m_weak: int,
    k_search_multiplier: int,
) -> tuple[dict, dict, dict, dict, dict, list[dict]]:
    """
    Returns:
        weak_recall    : {k: mean recall}  — HNSW(M_weak) alone
        strong_recall  : {k: mean recall}  — HNSW(M_strong) alone
        cluster_recall : {k: mean recall}  — HNSW(M_weak) seeds + cover expansion
        ratio_mean     : {k: list[float]}  — mean dist ratio per rank
        ratio_max      : {k: list[float]}  — max  dist ratio per rank
        rows           : per-query CSV data
    """
    kernel    = TiledEuclideanKernel(chunk_size=4096)
    workspace = kernel.prepare_workspace(dataset)
    n_queries = queries.shape[0]
    max_k_prime = max(K_VALUES) * k_search_multiplier

    weak_hits    = {k: 0.0 for k in K_VALUES}
    strong_hits  = {k: 0.0 for k in K_VALUES}
    cluster_hits = {k: 0.0 for k in K_VALUES}
    ratio_sum    = {k: np.zeros(k) for k in K_VALUES}
    ratio_max_acc= {k: np.zeros(k) for k in K_VALUES}
    rows: list[dict] = []

    for q_idx in range(n_queries):
        if q_idx % 10 == 0:
            print(f"  query {q_idx+1}/{n_queries}")

        query     = queries[q_idx : q_idx + 1]
        query_cpu = queries_cpu[q_idx : q_idx + 1]

        # squared L2 distances to all dataset points
        distances_sq = kernel.compute_dist_tile(query, workspace).squeeze(1)  # (N,)

        # ground truth from HDF5 (pre-computed on full dataset)
        # fall back to brute-force when gt entry is -1 (filtered out)
        gt_row = ground_truth[q_idx]
        gt_valid = [int(x) for x in gt_row if x >= 0]

        # cluster_search — called ONCE per query at max budget
        seeds        = query_hnsw(hnsw_weak, query_cpu, k_search=m_weak)
        all_candidates = cluster_search(
            seeds, cover_index, dataset, kernel,
            k_prime=max_k_prime, epsilon=epsilon
        )

        for k in K_VALUES:
            # ground truth set for this k
            # prefer hdf5 gt; fall back to brute-force from distances_sq
            if len(gt_valid) >= k:
                true_set = set(gt_valid[:k])
            else:
                true_set = set(topk_from_distances(distances_sq, k))

            true_ids_ordered = (
                gt_valid[:k] if len(gt_valid) >= k
                else topk_from_distances(distances_sq, k)
            )
            true_dists = torch.sqrt(
                distances_sq[torch.tensor(true_ids_ordered, device=distances_sq.device)]
            ).cpu().numpy()

            # HNSW weak alone
            weak_result   = query_hnsw(hnsw_weak,   query_cpu, k)
            weak_hits[k] += len(true_set.intersection(weak_result)) / k

            # HNSW strong alone
            strong_result   = query_hnsw(hnsw_strong, query_cpu, k)
            strong_hits[k] += len(true_set.intersection(strong_result)) / k

            # clustering-aided (reuse candidates)
            candidates      = all_candidates[:k * k_search_multiplier]
            cluster_result  = topk_from_distances(distances_sq, k, candidates)
            cluster_hits[k]+= len(true_set.intersection(cluster_result)) / k

            # distance ratio per rank
            for rank, pid in enumerate(cluster_result[:k]):
                ret_dist  = float(torch.sqrt(distances_sq[pid]).cpu())
                true_dist = float(true_dists[rank]) if rank < len(true_dists) else 1e-12
                ratio = ret_dist / true_dist if true_dist > 1e-12 else 1.0
                ratio_sum[k][rank]     += ratio
                ratio_max_acc[k][rank]  = max(ratio_max_acc[k][rank], ratio)

            rows.append({
                "query_idx":      q_idx,
                "k":              k,
                "weak_recall":    len(true_set.intersection(weak_result))   / k,
                "strong_recall":  len(true_set.intersection(strong_result)) / k,
                "cluster_recall": len(true_set.intersection(cluster_result))/ k,
                "n_candidates":   len(all_candidates),
            })

    nq = float(n_queries)
    return (
        {k: weak_hits[k]    / nq for k in K_VALUES},
        {k: strong_hits[k]  / nq for k in K_VALUES},
        {k: cluster_hits[k] / nq for k in K_VALUES},
        {k: (ratio_sum[k] / nq).tolist() for k in K_VALUES},
        {k: ratio_max_acc[k].tolist()    for k in K_VALUES},
        rows,
    )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_summary(weak, strong, cluster, ratio_mean, ratio_max) -> None:
    print()
    print(f"{'k':>4} | {'HNSW-weak':>9} | {'HNSW-strong':>11} | {'Cluster':>9} | {'mean ratio':>10} | {'max ratio':>9}")
    print("-" * 68)
    for k in K_VALUES:
        mr = float(np.mean(ratio_mean[k]))
        xr = float(np.max(ratio_max[k]))
        print(f"{k:>4} | {weak[k]:>9.4f} | {strong[k]:>11.4f} | {cluster[k]:>9.4f} | {mr:>10.4f} | {xr:>9.4f}")
    print()


def save_recall_plot(weak, strong, cluster, out_dir, m_weak, m_strong, n) -> pathlib.Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(K_VALUES, [weak[k]    for k in K_VALUES], marker="o", linestyle="--",
            label=f"HNSW only (M={m_weak}, cheap)")
    ax.plot(K_VALUES, [strong[k]  for k in K_VALUES], marker="^", linestyle="--",
            label=f"HNSW only (M={m_strong}, strong)")
    ax.plot(K_VALUES, [cluster[k] for k in K_VALUES], marker="s",
            label=f"HNSW(M={m_weak}) + cover expansion")
    ax.set_xlabel("k")
    ax.set_ylabel("Recall@k")
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(K_VALUES)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title(f"GloVe-100 (n={n:,}): Recall@k")
    fig.tight_layout()
    path = out_dir / f"glove_recall_n{n}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def save_ratio_plot(ratio_mean, ratio_max, out_dir, n) -> pathlib.Path:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=False)
    for ax, k in zip(axes.flatten(), K_VALUES):
        ranks = list(range(1, k + 1))
        ax.plot(ranks, ratio_mean[k], label="mean ratio")
        ax.plot(ranks, ratio_max[k],  label="max ratio", linestyle="--")
        ax.axhline(1.0, color="gray", linewidth=0.8, linestyle=":")
        ax.set_xlabel("rank")
        ax.set_ylabel("dist ratio")
        ax.set_title(f"k={k}")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"GloVe-100 (n={n:,}): returned dist / true dist per rank")
    fig.tight_layout()
    path = out_dir / f"glove_ratio_n{n}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def save_csv(rows, out_dir, n) -> pathlib.Path:
    path = out_dir / f"glove_results_n{n}.csv"
    fields = ["query_idx", "k", "weak_recall", "strong_recall", "cluster_recall", "n_candidates"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # --- dataset ---
    print("Preparing GloVe-100 dataset...")
    glove_path = ensure_glove(DATA_DIR)
    base_np, query_np, ground_truth = load_glove(glove_path, n=args.n, n_queries=args.n_queries)
    print(f"  base: {base_np.shape}  queries: {query_np.shape}")

    # GloVe vectors from ann-benchmarks are already L2-normalized
    dataset     = torch.from_numpy(base_np).to(device)
    queries     = torch.from_numpy(query_np).to(device)
    dataset_cpu = base_np
    queries_cpu = query_np

    # --- cover ---
    print("Building clustering cover...")
    t0 = time.perf_counter()
    cover = build_cover(dataset, epsilon=args.epsilon, k=args.k_cluster)
    torch.cuda.synchronize()
    cover_time = time.perf_counter() - t0
    cover_index = CoverIndex(cover)
    print(f"  done in {cover_time:.2f}s  |  {cover_index}")

    # --- HNSW indexes ---
    hnsw_weak   = build_hnsw(dataset_cpu, args.M_weak,   "weak")
    hnsw_strong = build_hnsw(dataset_cpu, args.M_strong, "strong")

    # --- evaluate ---
    print("Evaluating...")
    t0 = time.perf_counter()
    weak_r, strong_r, cluster_r, ratio_mean, ratio_max, rows = evaluate(
        dataset=dataset,
        queries=queries,
        queries_cpu=queries_cpu,
        ground_truth=ground_truth,
        cover_index=cover_index,
        hnsw_weak=hnsw_weak,
        hnsw_strong=hnsw_strong,
        epsilon=args.epsilon,
        m_weak=args.M_weak,
        k_search_multiplier=args.k_search_multiplier,
    )
    eval_time = time.perf_counter() - t0
    print(f"  done in {eval_time:.2f}s")

    print_summary(weak_r, strong_r, cluster_r, ratio_mean, ratio_max)

    out_dir = BASE_DIR / "experiments" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    p1 = save_recall_plot(weak_r, strong_r, cluster_r, out_dir, args.M_weak, args.M_strong, args.n)
    p2 = save_ratio_plot(ratio_mean, ratio_max, out_dir, args.n)
    p3 = save_csv(rows, out_dir, args.n)

    print(f"Saved recall plot  -> {p1}")
    print(f"Saved ratio plot   -> {p2}")
    print(f"Saved CSV          -> {p3}")
    print(f"Timing: cover={cover_time:.2f}s  eval={eval_time:.2f}s")


if __name__ == "__main__":
    main()
