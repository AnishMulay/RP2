import math
from typing import Dict

import torch


class SimpleClustering:
    """
    Single-sample-level clustering representation for balanced red/blue sets.

    Red points are A/right-side points and blue points are B/left-side points.
    Blue points propose to red points. The representation consists of sampled
    red centers, dense distances to those centers, and CSR adjacency lists from
    each blue point to red points that are closer than its nearest sampled red.
    """

    def __init__(self, tile_size=2048):
        if tile_size <= 0:
            raise ValueError("tile_size must be positive")
        self.tile_size = int(tile_size)

    def run(self, A, B) -> Dict[str, torch.Tensor]:
        if A.device != B.device:
            raise ValueError("A and B must be on the same device")
        if A.device.type != "cuda":
            raise ValueError("SimpleClustering requires CUDA tensors")
        if A.ndim != 2 or B.ndim != 2:
            raise ValueError("A and B must be rank-2 tensors")
        if A.shape != B.shape:
            raise ValueError("A and B must have matching shape (N, d)")
        if not A.is_floating_point() or not B.is_floating_point():
            raise TypeError("A and B must be floating-point tensors")

        device = A.device
        n, _ = A.shape
        if n <= 0:
            raise ValueError("A and B must contain at least one point")

        p = 1.0 / math.sqrt(n)
        sample_mask = torch.rand(n, device=device) < p
        if not sample_mask.any():
            sample_mask[torch.randint(n, (1,), device=device)] = True

        sampled_idx = sample_mask.nonzero(as_tuple=True)[0]
        A_sampled = A[sampled_idx]

        DR = torch.cdist(
            A_sampled,
            A,
            compute_mode="use_mm_for_euclid_dist_if_necessary",
        )
        DB = torch.cdist(
            B,
            A_sampled,
            compute_mode="use_mm_for_euclid_dist_if_necessary",
        )
        d_min_b, nearest_s = DB.min(dim=1)

        d_min_b_sq = d_min_b.pow(2)
        A_sq = A.pow(2).sum(dim=1)
        B_sq = B.pow(2).sum(dim=1)

        pair_b_parts = []
        pair_a_parts = []

        for start in range(0, n, self.tile_size):
            end = min(start + self.tile_size, n)
            A_tile = A[start:end]
            A_tile_sq = A_sq[start:end]

            cross = torch.mm(B, A_tile.t())
            d_sq_tile = B_sq[:, None] + A_tile_sq[None, :] - 2.0 * cross
            mask = d_sq_tile < d_min_b_sq[:, None]

            b_idx, t_idx = mask.nonzero(as_tuple=True)
            pair_b_parts.append(b_idx)
            pair_a_parts.append(t_idx + start)

        pairs_b = torch.cat(pair_b_parts)
        pairs_a = torch.cat(pair_a_parts)

        order = torch.argsort(pairs_b, stable=True)
        pairs_b = pairs_b[order]
        pairs_a = pairs_a[order]

        counts = torch.bincount(pairs_b, minlength=n)
        adj_ptr = torch.zeros(n + 1, dtype=torch.long, device=device)
        adj_ptr[1:] = counts.cumsum(0)

        return {
            "sampled_idx": sampled_idx,
            "A_sampled": A_sampled,
            "DR": DR,
            "DB": DB,
            "d_min_b": d_min_b,
            "nearest_s": nearest_s,
            "adj_ptr": adj_ptr,
            "adj_col": pairs_a,
        }

    def get_adj(self, b, result):
        return get_adj(b, result)


def get_adj(b, result):
    """Return the CSR adjacency slice for blue point b without copying."""
    ptr = result["adj_ptr"]
    return result["adj_col"][ptr[b] : ptr[b + 1]]
