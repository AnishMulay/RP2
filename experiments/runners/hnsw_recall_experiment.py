#!/usr/bin/env python3

from __future__ import annotations

import argparse
import pathlib
import sys
from collections import defaultdict

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import hnswlib
except ImportError as exc:
    raise ImportError("hnswlib is required for this experiment.") from exc

BASE_DIR = pathlib.Path(__file__).resolve().parents[2]
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from clustered_push_relabel.clustering.k_level import k_level_cluster
from clustered_push_relabel.utils.distance import TiledEuclideanKernel


K_SEARCH_VALUES = [1, 2, 4, 8, 16, 24, 32, 48, 64]
NOISE_STD = 0.5
EF_CONSTRUCTION = 200


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HNSW recall experiment with clustering-aided reranking.")
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--n_clusters", type=int, default=50)
    parser.add_argument("--epsilon", type=float, default=0.5)
    parser.add_argument("--k_cluster", type=int, default=4)
    parser.add_argument("--M", type=int, default=16)
    parser.add_argument("--n_queries", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires CUDA because the dataset and clustering are specified to run on GPU.")
    return torch.device("cuda")


def sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def generate_dataset(n: int, dim: int, n_clusters: int, device: torch.device) -> torch.Tensor:
    centroids = torch.randn(n_clusters, dim, device=device)
    assignments = torch.randint(0, n_clusters, (n,), device=device)
    points = centroids[assignments] + NOISE_STD * torch.randn(n, dim, device=device)
    return torch.nn.functional.normalize(points, p=2, dim=1)


def build_cover_dicts(blue_cover: tuple[torch.Tensor, torch.Tensor, torch.Tensor], n_points: int) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    center_ids, level_ids, point_ids = blue_cover
    if not (center_ids.numel() == level_ids.numel() == point_ids.numel()):
        raise ValueError("blue_cover tensors must be parallel 1D tensors of equal length.")

    point_to_centers_sets: dict[int, set[int]] = defaultdict(set)
    center_to_members_sets: dict[int, set[int]] = defaultdict(set)

    # k_level_cluster builds cover edges against the concatenated red/blue workspace.
    # Here x == y == dataset, so collapse both copies back to the underlying dataset ID.
    for center_id, point_id in zip(center_ids.tolist(), point_ids.tolist()):
        point_idx = int(point_id) % n_points
        center_idx = int(center_id)
        point_to_centers_sets[point_idx].add(center_idx)
        center_to_members_sets[center_idx].add(point_idx)

    point_to_centers = {
        point_idx: sorted(center_ids_for_point)
        for point_idx, center_ids_for_point in point_to_centers_sets.items()
    }
    center_to_members = {
        center_idx: sorted(member_ids)
        for center_idx, member_ids in center_to_members_sets.items()
    }
    return point_to_centers, center_to_members


def build_hnsw_index(dataset_cpu: np.ndarray, m: int) -> hnswlib.Index:
    n, dim = dataset_cpu.shape
    index = hnswlib.Index(space="l2", dim=dim)
    index.init_index(max_elements=n, ef_construction=EF_CONSTRUCTION, M=m)
    index.add_items(dataset_cpu, np.arange(n, dtype=np.int32))
    return index


def query_hnsw(index: hnswlib.Index, query_vector: np.ndarray, query_idx: int, k_search: int) -> list[int]:
    index.set_ef(max(k_search, 50))
    labels, _ = index.knn_query(query_vector, k=k_search)
    return [int(label) for label in labels[0] if int(label) != query_idx][:k_search]


def get_anchor_point(index: hnswlib.Index, query_vector: np.ndarray) -> int:
    index.set_ef(50)
    labels, _ = index.knn_query(query_vector, k=1)
    return int(labels[0, 0])


def topk_from_distances(
    distances_sq: torch.Tensor,
    k: int,
    exclude_idx: int,
    candidate_ids: list[int] | None = None,
) -> list[int]:
    if candidate_ids is None:
        working = distances_sq.clone()
        working[exclude_idx] = float("inf")
        k_eff = min(k, working.numel() - 1)
        if k_eff <= 0:
            return []
        return torch.topk(working, k=k_eff, largest=False).indices.tolist()

    if not candidate_ids:
        return []

    candidate_tensor = torch.tensor(candidate_ids, device=distances_sq.device, dtype=torch.long)
    keep_mask = candidate_tensor != exclude_idx
    candidate_tensor = candidate_tensor[keep_mask]
    if candidate_tensor.numel() == 0:
        return []

    candidate_distances = distances_sq.index_select(0, candidate_tensor)
    k_eff = min(k, candidate_tensor.numel())
    topk_local = torch.topk(candidate_distances, k=k_eff, largest=False).indices
    return candidate_tensor[topk_local].tolist()


def evaluate(
    dataset: torch.Tensor,
    dataset_cpu: np.ndarray,
    point_to_centers: dict[int, list[int]],
    center_to_members: dict[int, list[int]],
    index: hnswlib.Index,
    query_indices: list[int],
) -> tuple[dict[int, float], dict[int, float]]:
    max_k = max(K_SEARCH_VALUES)
    kernel = TiledEuclideanKernel(chunk_size=4096)
    workspace = kernel.prepare_workspace(dataset)

    hnsw_hits = {k: 0.0 for k in K_SEARCH_VALUES}
    clustering_hits = {k: 0.0 for k in K_SEARCH_VALUES}

    for query_idx in query_indices:
        query = dataset[query_idx : query_idx + 1]
        distances_sq = kernel.compute_dist_tile(query, workspace).squeeze(1)
        brute_force_top = topk_from_distances(distances_sq, max_k, exclude_idx=query_idx)

        query_vector = dataset_cpu[query_idx : query_idx + 1]
        anchor_point = get_anchor_point(index, query_vector)
        candidate_ids_set = {anchor_point}
        for center_idx in point_to_centers.get(anchor_point, []):
            candidate_ids_set.update(center_to_members.get(center_idx, []))
        candidate_ids = sorted(candidate_ids_set)
        clustering_top = topk_from_distances(
            distances_sq,
            max_k,
            exclude_idx=query_idx,
            candidate_ids=candidate_ids,
        )

        brute_force_sets = {
            k_search: set(brute_force_top[:k_search])
            for k_search in K_SEARCH_VALUES
        }

        for k_search in K_SEARCH_VALUES:
            hnsw_result = query_hnsw(index, query_vector, query_idx, k_search)
            clustering_result = clustering_top[:k_search]

            hnsw_hits[k_search] += len(brute_force_sets[k_search].intersection(hnsw_result)) / k_search
            clustering_hits[k_search] += len(brute_force_sets[k_search].intersection(clustering_result)) / k_search

    n_queries = float(len(query_indices))
    hnsw_recall = {k: hnsw_hits[k] / n_queries for k in K_SEARCH_VALUES}
    clustering_recall = {k: clustering_hits[k] / n_queries for k in K_SEARCH_VALUES}
    return hnsw_recall, clustering_recall


def save_plot(hnsw_recall: dict[int, float], clustering_recall: dict[int, float], m: int) -> pathlib.Path:
    output_dir = BASE_DIR / "experiments" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "recall_vs_k.png"

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(K_SEARCH_VALUES, [hnsw_recall[k] for k in K_SEARCH_VALUES], marker="o", label=f"HNSW (M={m})")
    ax.plot(K_SEARCH_VALUES, [clustering_recall[k] for k in K_SEARCH_VALUES], marker="s", label="Clustering-aided")
    ax.axvline(m, color="gray", linestyle="--", linewidth=1.5)
    ax.text(m + 0.5, 0.05, f"M={m}", color="gray", rotation=90, va="bottom")
    ax.set_xlabel("k_search")
    ax.set_ylabel("Recall")
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(K_SEARCH_VALUES)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def print_results_table(hnsw_recall: dict[int, float], clustering_recall: dict[int, float]) -> None:
    print()
    print(f"{'k_search':>8} | {'HNSW recall':>11} | {'Clustering-aided recall':>24}")
    print("-" * 50)
    for k_search in K_SEARCH_VALUES:
        print(f"{k_search:>8d} | {hnsw_recall[k_search]:>11.4f} | {clustering_recall[k_search]:>24.4f}")


def main() -> None:
    args = parse_args()
    device = require_cuda()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    if args.n_queries > args.n:
        raise ValueError("--n_queries must be less than or equal to --n.")

    print(f"Generating dataset on {device}...")
    dataset = generate_dataset(args.n, args.dim, args.n_clusters, device)
    sync_if_needed(device)

    print("Building k-level cover...")
    cover = k_level_cluster(dataset, dataset, epsilon=args.epsilon, k=args.k_cluster)
    point_to_centers, center_to_members = build_cover_dicts(cover["blue_cover"], args.n)

    print("Preparing HNSW index...")
    dataset_cpu = dataset.detach().cpu().numpy().astype(np.float32, copy=False)
    index = build_hnsw_index(dataset_cpu, args.M)

    query_indices = torch.randperm(args.n, device=device)[: args.n_queries].cpu().tolist()

    print("Evaluating recall...")
    hnsw_recall, clustering_recall = evaluate(
        dataset=dataset,
        dataset_cpu=dataset_cpu,
        point_to_centers=point_to_centers,
        center_to_members=center_to_members,
        index=index,
        query_indices=query_indices,
    )

    print_results_table(hnsw_recall, clustering_recall)
    output_path = save_plot(hnsw_recall, clustering_recall, args.M)
    print()
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
