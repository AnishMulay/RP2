"""Cluster-search reranking over shell covers."""

from __future__ import annotations

import heapq

import torch


def cluster_search(
    seeds: list[int],
    cover_index,
    dataset: torch.Tensor,
    kernel,
    k_prime: int,
    epsilon: float,
) -> list[int]:
    """Expands HNSW seed points through a shell cover.

    Args:
        seeds: Exactly M point IDs returned by HNSW (where M is the HNSW base-layer
            connection count), covering the regime where HNSW is reliable.
        cover_index: A ``CoverIndex`` built from ``build_cover``.
        dataset: Full dataset tensor of shape ``(N, D)``.
        kernel: A ``TiledEuclideanKernel`` instance. It is accepted for API
            compatibility but is not used by this algorithm.
        k_prime: Target number of neighbors to return.
        epsilon: Cover discretization parameter used for shell expansion.

    Returns:
        A list of unique point IDs of length at most ``k_prime``.
    """

    _ = kernel

    heap: list[tuple[float, int, int, float]] = []
    results: set[int] = set()
    result_list: list[int] = []

    for seed in seeds:
        for center_id, v_level in cover_index.get_shells(seed):
            delta_c = epsilon * v_level
            heapq.heappush(heap, (delta_c, center_id, 0, delta_c))

    while len(results) < k_prime:
        if not heap:
            break
        _proxy_distance, center_id, level_id, delta_c = heapq.heappop(heap)

        if level_id == 0:
            if center_id not in results:
                results.add(center_id)
                result_list.append(center_id)
        else:
            for point_id in cover_index.get_shell_members(center_id, level_id):
                if point_id not in results:
                    results.add(point_id)
                    result_list.append(point_id)

        next_level = level_id + 1
        if next_level <= cover_index.get_max_level(center_id):
            next_proxy = delta_c + epsilon * next_level
            heapq.heappush(heap, (next_proxy, center_id, next_level, delta_c))

    return result_list[:k_prime]
