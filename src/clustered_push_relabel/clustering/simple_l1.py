import math
from typing import Dict

import torch


class SimpleL1Clustering:
    """
    Memory-efficient L1 clustering representation for the push-relabel solver.

    Red points  = A  (right side, N points)  matched into
    Blue points = B  (left side,  N points)  propose outwards

    The output structure matches SimpleClustering exactly. Distances are
    Manhattan distances, and integer costs use ceil(distance / epsilon).
    """

    def __init__(
        self,
        epsilon: float,
        tile_size: int = 2048,
        sample_factor: float = 1.0,
    ):
        if tile_size <= 0:
            raise ValueError("tile_size must be positive")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if sample_factor <= 0:
            raise ValueError("sample_factor must be positive")
        self.epsilon = float(epsilon)
        self.tile_size = int(tile_size)
        self.sample_factor = float(sample_factor)

    def run(self, A: torch.Tensor, B: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        A : (N, d)  red points, CUDA, floating-point
        B : (N, d)  blue points, CUDA, floating-point, same device as A

        Returns
        -------
        dict with the same keys as SimpleClustering.run.
        """
        _validate(A, B)
        device = A.device
        N = A.shape[0]
        T = self.tile_size
        eps = self.epsilon

        sample_mask = torch.rand(N, device=device) < (
            self.sample_factor / math.sqrt(N)
        )
        if not sample_mask.any():
            sample_mask[torch.randint(N, (1,), device=device)] = True
        sampled_idx = sample_mask.nonzero(as_tuple=True)[0]
        A_s = A[sampled_idx]

        DR = torch.cdist(A_s, A, p=1)
        DB = torch.cdist(B, A_s, p=1)
        d_min_b, nearest_s = DB.min(dim=1)
        DR_int = (DR / eps).ceil_().to(torch.int32)
        d_min_b_int = (d_min_b / eps).ceil_().to(torch.int32)

        counts = torch.zeros(N, dtype=torch.long, device=device)
        for start in range(0, N, T):
            end = min(start + T, N)
            dist_tile = torch.cdist(B, A[start:end], p=1)
            mask = dist_tile < d_min_b.unsqueeze(1)
            counts.add_(mask.sum(dim=1))

        adj_ptr = torch.zeros(N + 1, dtype=torch.long, device=device)
        adj_ptr[1:] = counts.cumsum(0)
        M = int(adj_ptr[-1].item())
        adj_col = torch.empty(M, dtype=torch.long, device=device)
        adj_dist_int = torch.empty(M, dtype=torch.int32, device=device)
        adj_dist_float = torch.empty(M, dtype=A.dtype, device=device)
        del counts

        if M == 0:
            return _pack(
                sampled_idx,
                A_s,
                DR,
                DR_int,
                DB,
                d_min_b,
                d_min_b_int,
                nearest_s,
                adj_ptr,
                adj_col,
                adj_dist_int,
                adj_dist_float,
            )

        cursor = adj_ptr[:-1].clone()

        for start in range(0, N, T):
            end = min(start + T, N)
            dist_tile = torch.cdist(B, A[start:end], p=1)
            mask = dist_tile < d_min_b.unsqueeze(1)

            b_idx, t_idx = mask.nonzero(as_tuple=True)
            if b_idx.numel() == 0:
                continue

            write_pos = cursor[b_idx] + _group_offsets(b_idx)
            adj_col[write_pos] = (t_idx + start).long()
            actual_dists = dist_tile[b_idx, t_idx]
            adj_dist_float[write_pos] = actual_dists
            adj_dist_int[write_pos] = (actual_dists / eps).ceil_().to(torch.int32)

            cursor.scatter_add_(0, b_idx, torch.ones_like(b_idx))

        del cursor
        return _pack(
            sampled_idx,
            A_s,
            DR,
            DR_int,
            DB,
            d_min_b,
            d_min_b_int,
            nearest_s,
            adj_ptr,
            adj_col,
            adj_dist_int,
            adj_dist_float,
        )

    def get_adj(self, b: int, result: Dict) -> torch.Tensor:
        """Zero-copy adjacency slice for blue point b."""
        return get_adj(b, result)


def get_adj(b: int, result: Dict) -> torch.Tensor:
    """Return the adjacency list of blue point b as a 1-D tensor of red indices."""
    ptr = result["adj_ptr"]
    return result["adj_col"][ptr[b] : ptr[b + 1]]


def _validate(A: torch.Tensor, B: torch.Tensor) -> None:
    if A.device != B.device:
        raise ValueError("A and B must be on the same device")
    if A.device.type != "cuda":
        raise ValueError("SimpleClustering requires CUDA tensors")
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("A and B must be rank-2 tensors")
    if A.shape != B.shape:
        raise ValueError("A and B must have the same shape (N, d)")
    if not A.is_floating_point() or not B.is_floating_point():
        raise TypeError("A and B must be floating-point tensors")
    if A.shape[0] == 0:
        raise ValueError("A and B must be non-empty")


def _group_offsets(b_idx: torch.Tensor) -> torch.Tensor:
    """
    For a sorted 1-D integer tensor, return the 0-based intra-group offset of
    each element.
    """
    M = b_idx.numel()
    if M <= 1:
        return torch.zeros(M, dtype=torch.long, device=b_idx.device)

    same = torch.zeros(M, dtype=torch.long, device=b_idx.device)
    same[1:] = (b_idx[1:] == b_idx[:-1]).long()

    cumsum = same.cumsum(0)
    start_vals = cumsum.masked_fill(same.bool(), 0)
    baseline = torch.cummax(start_vals, dim=0).values
    return cumsum - baseline


def _pack(
    sampled_idx,
    A_s,
    DR,
    DR_int,
    DB,
    d_min_b,
    d_min_b_int,
    nearest_s,
    adj_ptr,
    adj_col,
    adj_dist_int,
    adj_dist_float,
) -> Dict[str, torch.Tensor]:
    return {
        "sampled_idx": sampled_idx,
        "A_sampled": A_s,
        "DR": DR,
        "DR_int": DR_int,
        "DB": DB,
        "d_min_b": d_min_b,
        "d_min_b_int": d_min_b_int,
        "nearest_s": nearest_s,
        "adj_ptr": adj_ptr,
        "adj_col": adj_col,
        "adj_dist_int": adj_dist_int,
        "adj_dist_float": adj_dist_float,
    }
