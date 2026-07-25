import math
from typing import Dict

import torch


def _gpu_mem(label):
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        res = torch.cuda.memory_reserved() / 1024**3
        print(
            f"  [MEM-cluster] {label}: alloc={alloc:.2f}GB res={res:.2f}GB",
            flush=True,
        )


class SimplePrecomputedClustering:
    """
    Clustering representation for precomputed pairwise distances.

    Inputs are:
      - D_red_to_red[i, j]  = distance(red_i, red_j)
      - D_blue_to_red[b, a] = distance(blue_b, red_a)
    """

    def __init__(
        self,
        epsilon: float,
        tile_size: int = 512,
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

    def run(
        self,
        D_red_to_red: torch.Tensor,
        D_blue_to_red: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        _validate(D_red_to_red, D_blue_to_red)
        device = D_red_to_red.device
        n = D_red_to_red.shape[0]
        tile_size = self.tile_size
        eps = self.epsilon

        sample_mask = torch.rand(n, device=device) < (
            1.0 / (self.sample_factor * math.sqrt(n))
        )
        if not sample_mask.any():
            sample_mask[torch.randint(n, (1,), device=device)] = True
        sampled_idx = sample_mask.nonzero(as_tuple=True)[0]
        A_sampled = sampled_idx.to(torch.float32).unsqueeze(1)

        _gpu_mem("before DR/DB slice")
        DR = D_red_to_red[sampled_idx, :]
        DB = D_blue_to_red[:, sampled_idx]
        d_min_b, nearest_s = DB.min(dim=1)
        del DB
        _gpu_mem("after DB deleted")

        DR_int = (DR / eps).ceil_().to(torch.int32)
        d_min_b_int = (d_min_b / eps).ceil_().to(torch.int32)

        _gpu_mem("before pass 1")
        counts = torch.zeros(n, dtype=torch.long, device=device)
        d_min_b_col = d_min_b.unsqueeze(1)
        for start in range(0, n, tile_size):
            end = min(start + tile_size, n)
            dist_tile = D_blue_to_red[:, start:end]
            mask = dist_tile < d_min_b_col
            counts.add_(mask.sum(dim=1))
        _gpu_mem("after pass 1")

        adj_ptr = torch.zeros(n + 1, dtype=torch.long, device=device)
        adj_ptr[1:] = counts.cumsum(0)
        m = int(adj_ptr[-1].item())
        adj_col = torch.empty(m, dtype=torch.long, device=device)
        adj_dist_int = torch.empty(m, dtype=torch.int32, device=device)
        adj_dist_float = torch.empty(m, dtype=torch.float32, device=device)
        del counts
        _gpu_mem("after adj arrays allocated")

        if m == 0:
            _gpu_mem("before return")
            return _pack(
                sampled_idx,
                A_sampled,
                DR,
                DR_int,
                d_min_b,
                d_min_b_int,
                nearest_s,
                adj_ptr,
                adj_col,
                adj_dist_int,
                adj_dist_float,
            )

        cursor = adj_ptr[:-1].clone()
        for start in range(0, n, tile_size):
            end = min(start + tile_size, n)
            dist_tile = D_blue_to_red[:, start:end]
            mask = dist_tile < d_min_b_col
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
        _gpu_mem("after pass 2")
        _gpu_mem("before return")
        return _pack(
            sampled_idx,
            A_sampled,
            DR,
            DR_int,
            d_min_b,
            d_min_b_int,
            nearest_s,
            adj_ptr,
            adj_col,
            adj_dist_int,
            adj_dist_float,
        )

    def get_adj(self, b: int, result: Dict) -> torch.Tensor:
        return get_adj(b, result)


def get_adj(b: int, result: Dict) -> torch.Tensor:
    ptr = result["adj_ptr"]
    return result["adj_col"][ptr[b] : ptr[b + 1]]


def _validate(D_red_to_red: torch.Tensor, D_blue_to_red: torch.Tensor) -> None:
    if D_red_to_red.device != D_blue_to_red.device:
        raise ValueError("distance tensors must be on the same device")
    if D_red_to_red.device.type not in ("cuda", "cpu"):
        raise ValueError("SimplePrecomputedClustering requires CUDA or CPU tensors")
    if D_red_to_red.dtype != torch.float32 or D_blue_to_red.dtype != torch.float32:
        raise TypeError("distance tensors must be float32")
    if D_red_to_red.ndim != 2 or D_blue_to_red.ndim != 2:
        raise ValueError("distance tensors must be rank-2")
    n = D_red_to_red.shape[0]
    if n == 0:
        raise ValueError("distance tensors must be non-empty")
    if D_red_to_red.shape != (n, n):
        raise ValueError("D_red_to_red must have shape (N, N)")
    if D_blue_to_red.shape != (n, n):
        raise ValueError("D_blue_to_red must have shape (N, N) with the same N")


def _group_offsets(b_idx: torch.Tensor) -> torch.Tensor:
    """
    For a sorted 1-D integer tensor, return the 0-based intra-group offset
    of each element.
    """
    m = b_idx.numel()
    if m <= 1:
        return torch.zeros(m, dtype=torch.long, device=b_idx.device)

    same = torch.zeros(m, dtype=torch.long, device=b_idx.device)
    same[1:] = (b_idx[1:] == b_idx[:-1]).long()

    cumsum = same.cumsum(0)
    start_vals = cumsum.masked_fill(same.bool(), 0)
    baseline = torch.cummax(start_vals, dim=0).values
    return cumsum - baseline


def _pack(
    sampled_idx,
    A_sampled,
    DR,
    DR_int,
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
        "A_sampled": A_sampled,
        "DR": DR,
        "DR_int": DR_int,
        "d_min_b": d_min_b,
        "d_min_b_int": d_min_b_int,
        "nearest_s": nearest_s,
        "adj_ptr": adj_ptr,
        "adj_col": adj_col,
        "adj_dist_int": adj_dist_int,
        "adj_dist_float": adj_dist_float,
    }
