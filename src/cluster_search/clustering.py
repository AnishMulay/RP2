"""Single-set cover construction utilities for cluster_search."""

from __future__ import annotations

import torch

from clustered_push_relabel.utils.distance import TiledEuclideanKernel


def _sample_hierarchy_levels(n: int, k: int, device: torch.device) -> torch.Tensor:
    """Samples point hierarchy levels using the current k-level clustering logic."""

    prob = n ** (-1.0 / k)
    levels = torch.zeros(n, dtype=torch.long, device=device)
    for _ in range(k - 1):
        mask = torch.rand(n, device=device) <= prob
        levels += mask.long()
    if (levels == k - 1).sum() == 0:
        levels[torch.randint(0, n, (1,), device=device)] = k - 1
    return levels


def _process_level(
    centers: torch.Tensor,
    center_indices: torch.Tensor,
    bounds: torch.Tensor,
    epsilon: float,
    kernel: TiledEuclideanKernel,
    workspace: dict[str, torch.Tensor],
    batch_size: int,
) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    """Builds shell assignments for one sampled center level."""

    n_centers = centers.shape[0]
    chunk_center_ids: list[torch.Tensor] = []
    chunk_level_ids: list[torch.Tensor] = []
    chunk_point_ids: list[torch.Tensor] = []

    new_bounds = bounds.clone()

    for start in range(0, n_centers, batch_size):
        end = min(start + batch_size, n_centers)
        batch_centers = centers[start:end]
        batch_indices = center_indices[start:end]

        dists = kernel.compute_dist_tile(batch_centers, workspace).t()
        batch_min, _ = dists.min(dim=0)
        new_bounds = torch.minimum(new_bounds, batch_min)

        mask = dists < bounds.unsqueeze(0)
        if not mask.any():
            continue

        valid_indices = torch.nonzero(mask)
        rows = valid_indices[:, 0]
        cols = valid_indices[:, 1]

        valid_dists = torch.sqrt(dists[rows, cols])
        levels = torch.ceil(valid_dists / epsilon).to(torch.long)
        global_center_ids = batch_indices[rows]

        chunk_center_ids.append(global_center_ids.cpu())
        chunk_level_ids.append(levels.cpu())
        chunk_point_ids.append(cols.cpu())

    if chunk_center_ids:
        edges = (
            torch.cat(chunk_center_ids),
            torch.cat(chunk_level_ids),
            torch.cat(chunk_point_ids),
        )
    else:
        empty = torch.empty(0, dtype=torch.long)
        edges = (empty, empty.clone(), empty.clone())

    return edges, new_bounds


def build_cover(
    P: torch.Tensor,
    epsilon: float,
    k: int,
    metric: str = "L2",
    batch_size: int = 2048,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Builds a single clustering cover over one point set.

    Args:
        P: A 2D tensor of shape ``(N, D)`` containing the point set.
        epsilon: Base shell radius used to quantize distances into level IDs.
        k: Number of hierarchy levels to sample.
        metric: Distance metric label. Must be ``"L2"`` or ``"L1"``.
        batch_size: Number of centers to process per tiled distance batch.

    Returns:
        A tuple ``(center_ids, level_ids, point_ids)`` of parallel 1D CPU
        ``torch.long`` tensors. All IDs are direct indices into ``P`` in the
        range ``0..N-1``.

    Raises:
        ValueError: If ``P`` is not 2D, if ``k`` is less than 1, or if
            ``metric`` is not ``"L2"`` or ``"L1"``.
    """

    if P.dim() != 2:
        raise ValueError("P must be a 2D tensor of shape (N, D).")
    if k < 1:
        raise ValueError("k must be at least 1.")
    if metric not in {"L2", "L1"}:
        raise ValueError('metric must be either "L2" or "L1".')

    n = P.shape[0]
    if n == 0:
        empty = torch.empty(0, dtype=torch.long)
        return empty, empty.clone(), empty.clone()

    kernel = TiledEuclideanKernel(chunk_size=batch_size)
    workspace = kernel.prepare_workspace(P)
    sampled_levels = _sample_hierarchy_levels(n, k, P.device)

    all_center_ids: list[torch.Tensor] = []
    all_level_ids: list[torch.Tensor] = []
    all_point_ids: list[torch.Tensor] = []
    bounds = torch.full((n,), float("inf"), device=P.device)

    for level in range(k - 1, -1, -1):
        level_mask = sampled_levels == level
        if not level_mask.any():
            continue

        center_indices = torch.nonzero(level_mask).squeeze(1)
        centers = P[center_indices]
        (center_ids, level_ids, point_ids), bounds = _process_level(
            centers=centers,
            center_indices=center_indices,
            bounds=bounds,
            epsilon=epsilon,
            kernel=kernel,
            workspace=workspace,
            batch_size=batch_size,
        )
        all_center_ids.append(center_ids)
        all_level_ids.append(level_ids)
        all_point_ids.append(point_ids)

    if not all_center_ids:
        empty = torch.empty(0, dtype=torch.long)
        return empty, empty.clone(), empty.clone()

    return (
        torch.cat(all_center_ids),
        torch.cat(all_level_ids),
        torch.cat(all_point_ids),
    )
