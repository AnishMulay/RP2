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

    shell_cache: dict[int, dict[int, int]] = {}
    for seed in seeds:
        shell_cache.setdefault(seed, dict(cover_index.get_shells(seed)))

    def min_proxy_distance_to_seeds(point_id: int) -> float:
        center_to_level_p = shell_cache.setdefault(point_id, dict(cover_index.get_shells(point_id)))
        best_proxy = float("inf")

        for seed in seeds:
            center_to_level_seed = shell_cache[seed]
            shared_centers = center_to_level_seed.keys() & center_to_level_p.keys()
            if not shared_centers:
                continue

            best_center = min(
                shared_centers,
                key=lambda center_id: max(center_to_level_seed[center_id], center_to_level_p[center_id]),
            )
            proxy_distance = float(center_to_level_seed[best_center] + center_to_level_p[best_center]) * epsilon
            if proxy_distance < best_proxy:
                best_proxy = proxy_distance

        return best_proxy

    result_list.sort(key=lambda point_id: (min_proxy_distance_to_seeds(point_id), point_id))
    return result_list[:k_prime]
