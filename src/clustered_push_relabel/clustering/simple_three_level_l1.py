from typing import Dict

import torch


def _gpu_mem(label: str) -> None:
    return


class ThreeLevelL1Clustering:
    """
    Three-level memory-efficient clustering for the push-relabel solver.

    Distances are Manhattan/L1 throughout, but the hierarchical sampling and
    proxy structure match the L2 three-level implementation exactly:

      A1 ⊆ A sampled at rate N^(-1/3)  →  |A1| ≈ N^(2/3)
      A2 ⊆ A1 sampled at rate N^(-1/3) →  |A2| ≈ N^(1/3)

      Adj_B(b)   = {a ∈ A : d(b, a)  < d(b, nearest A1)}
      Adj_A1(a1) = {a ∈ A : d(a1, a) < d(a1, nearest A2)}
      DR         = distances from A2 to all of A

    Proxy cost for pair (b, a)
    ──────────────────────────
    Level 0 (direct)     a ∈ Adj_B(b)                →  d(b, a)
    Level 1 (via A1)     a ∈ Adj_A1(nearest_s1[b])   →  d(b, s1_b) + d(s1_b, a)
    Level 2 (via A2)     fallback, always covers rest  →  d(b, s2_b) + DR[s2_b, a]

    Unlike the L2 variant, there is no squared-distance GEMM shortcut for exact
    L1 distances, so tiled torch.cdist(..., p=1) calls are used in the A1-nearest
    and CSR passes while preserving the same asymptotic structure and outputs.
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
        ──────────
        A : (N, d)  red  points — CUDA, floating-point
        B : (N, d)  blue points — CUDA, floating-point, same device as A

        Returns
        ───────
        dict with keys:
            sampled_idx_A1    (S1,)     indices into A for A1 centers
            sampled_idx_A2    (S2,)     indices into A for A2 centers
            A1_sampled        (S1, d)   coordinates of A1 centers
            A2_sampled        (S2, d)   coordinates of A2 centers
            DR                (S2, N)   d(A2[i], A[j])  — float32, kept for proxy/slack
            DR_int            (S2, N)   ceil(DR / eps)  — int32
            nearest_s1        (N,)      index into A1 for each blue b
            d_min_b_A1        (N,)      d(b, nearest A1 center)
            d_min_b_A1_int    (N,)      ceil(d_min_b_A1 / eps) — int32
            nearest_s2        (N,)      index into A2 for each blue b
            d_min_b_A2        (N,)      d(b, nearest A2 center)
            d_min_b_A2_int    (N,)      ceil(d_min_b_A2 / eps) — int32
            nearest_s2_A1     (S1,)     index into A2 for each a1 ∈ A1
            d_min_A1_A2       (S1,)     d(a1, nearest A2 center)
            d_min_A1_A2_int   (S1,)     ceil(d_min_A1_A2 / eps) — int32
            adj_B_ptr         (N+1,)    CSR row pointers for Adj_B  (int64)
            adj_B_col         (MB,)     red indices in Adj_B  (int64)
            adj_B_dist_int    (MB,)     ceil(d(b,a)/eps) for Adj_B entries  (int32)
            adj_B_dist_float  (MB,)     d(b,a) for Adj_B entries  (float32)
            adj_A1_ptr        (S1+1,)   CSR row pointers for Adj_A1  (int64)
            adj_A1_col        (MA1,)    red indices in Adj_A1  (int64)
            adj_A1_dist_int   (MA1,)    ceil(d(a1,a)/eps) for Adj_A1 entries  (int32)
            adj_A1_dist_float (MA1,)    d(a1,a) for Adj_A1 entries  (float32)
        """
        _validate(A, B)
        device = A.device
        N = A.shape[0]
        T = self.tile_size
        eps = self.epsilon

        rate1 = self.sample_factor / (float(N) ** (1.0 / 3.0))
        mask_A1 = torch.rand(N, device=device) < rate1
        if not mask_A1.any():
            mask_A1[torch.randint(N, (1,), device=device)] = True
        sampled_idx_A1 = mask_A1.nonzero(as_tuple=True)[0]
        A1 = A[sampled_idx_A1]
        S1 = sampled_idx_A1.shape[0]

        rate2 = self.sample_factor / (float(N) ** (1.0 / 3.0))
        mask_A2 = torch.rand(S1, device=device) < rate2
        if not mask_A2.any():
            mask_A2[torch.randint(S1, (1,), device=device)] = True
        local_idx_A2 = mask_A2.nonzero(as_tuple=True)[0]
        sampled_idx_A2 = sampled_idx_A1[local_idx_A2]
        A2 = A[sampled_idx_A2]

        _gpu_mem(f"sampled A1={S1}, A2={A2.shape[0]}")

        DR = torch.cdist(A2, A, p=1)
        DR_int = (DR / eps).ceil_().to(torch.int32)
        _gpu_mem("after DR")

        DB_A2 = torch.cdist(B, A2, p=1)
        d_min_b_A2, nearest_s2 = DB_A2.min(dim=1)
        del DB_A2
        d_min_b_A2_int = (d_min_b_A2 / eps).ceil_().to(torch.int32)
        _gpu_mem("after DB_A2")

        DA1_A2 = torch.cdist(A1, A2, p=1)
        d_min_A1_A2, nearest_s2_A1 = DA1_A2.min(dim=1)
        del DA1_A2
        d_min_A1_A2_int = (d_min_A1_A2 / eps).ceil_().to(torch.int32)
        _gpu_mem("after DA1_A2")

        d_min_b_A1 = torch.full((N,), float("inf"), dtype=A.dtype, device=device)
        nearest_s1 = torch.zeros(N, dtype=torch.long, device=device)

        for start in range(0, S1, T):
            end = min(start + T, S1)
            dist_tile = torch.cdist(B, A1[start:end], p=1)
            tile_min, tile_argmin = dist_tile.min(dim=1)
            update = tile_min < d_min_b_A1
            d_min_b_A1 = torch.where(update, tile_min, d_min_b_A1)
            nearest_s1 = torch.where(update, tile_argmin + start, nearest_s1)
            del dist_tile

        d_min_b_A1_int = (d_min_b_A1 / eps).ceil_().to(torch.int32)
        _gpu_mem("after DB_A1 tiled min")

        bool_buf = torch.empty(N, T, dtype=torch.bool, device=device)

        counts_B = torch.zeros(N, dtype=torch.long, device=device)
        for start in range(0, N, T):
            end = min(start + T, N)
            t = end - start
            dist_tile = torch.cdist(B, A[start:end], p=1)
            torch.lt(dist_tile, d_min_b_A1.unsqueeze(1), out=bool_buf[:, :t])
            counts_B.add_(bool_buf[:, :t].sum(dim=1))
            del dist_tile
        _gpu_mem("Adj_B pass 1 done")

        adj_B_ptr = torch.zeros(N + 1, dtype=torch.long, device=device)
        adj_B_ptr[1:] = counts_B.cumsum(0)
        MB = int(adj_B_ptr[-1].item())
        adj_B_col = torch.empty(MB, dtype=torch.long, device=device)
        adj_B_dist_int = torch.empty(MB, dtype=torch.int32, device=device)
        adj_B_dist_float = torch.empty(MB, dtype=A.dtype, device=device)
        del counts_B
        _gpu_mem(f"Adj_B allocated MB={MB}")

        cursor_B = adj_B_ptr[:-1].clone()
        for start in range(0, N, T):
            end = min(start + T, N)
            t = end - start
            dist_tile = torch.cdist(B, A[start:end], p=1)
            torch.lt(dist_tile, d_min_b_A1.unsqueeze(1), out=bool_buf[:, :t])
            b_idx, t_idx = bool_buf[:, :t].nonzero(as_tuple=True)
            if b_idx.numel() == 0:
                del dist_tile
                continue
            write_pos = cursor_B[b_idx] + _group_offsets(b_idx)
            adj_B_col[write_pos] = (t_idx + start).long()
            actual_dists = dist_tile[b_idx, t_idx]
            adj_B_dist_float[write_pos] = actual_dists
            adj_B_dist_int[write_pos] = (actual_dists / eps).ceil_().to(torch.int32)
            cursor_B.scatter_add_(0, b_idx, torch.ones_like(b_idx))
            del dist_tile
        del cursor_B, bool_buf
        _gpu_mem("Adj_B pass 2 done")

        bool_buf_A1 = torch.empty(S1, T, dtype=torch.bool, device=device)

        counts_A1 = torch.zeros(S1, dtype=torch.long, device=device)
        for start in range(0, N, T):
            end = min(start + T, N)
            t = end - start
            dist_tile = torch.cdist(A1, A[start:end], p=1)
            torch.lt(dist_tile, d_min_A1_A2.unsqueeze(1), out=bool_buf_A1[:, :t])
            counts_A1.add_(bool_buf_A1[:, :t].sum(dim=1))
            del dist_tile
        _gpu_mem("Adj_A1 pass 1 done")

        adj_A1_ptr = torch.zeros(S1 + 1, dtype=torch.long, device=device)
        adj_A1_ptr[1:] = counts_A1.cumsum(0)
        MA1 = int(adj_A1_ptr[-1].item())
        adj_A1_col = torch.empty(MA1, dtype=torch.long, device=device)
        adj_A1_dist_int = torch.empty(MA1, dtype=torch.int32, device=device)
        adj_A1_dist_float = torch.empty(MA1, dtype=A.dtype, device=device)
        del counts_A1
        _gpu_mem(f"Adj_A1 allocated MA1={MA1}")

        cursor_A1 = adj_A1_ptr[:-1].clone()
        for start in range(0, N, T):
            end = min(start + T, N)
            t = end - start
            dist_tile = torch.cdist(A1, A[start:end], p=1)
            torch.lt(dist_tile, d_min_A1_A2.unsqueeze(1), out=bool_buf_A1[:, :t])
            a1_idx, t_idx = bool_buf_A1[:, :t].nonzero(as_tuple=True)
            if a1_idx.numel() == 0:
                del dist_tile
                continue
            write_pos = cursor_A1[a1_idx] + _group_offsets(a1_idx)
            adj_A1_col[write_pos] = (t_idx + start).long()
            actual_dists = dist_tile[a1_idx, t_idx]
            adj_A1_dist_float[write_pos] = actual_dists
            adj_A1_dist_int[write_pos] = (actual_dists / eps).ceil_().to(torch.int32)
            cursor_A1.scatter_add_(0, a1_idx, torch.ones_like(a1_idx))
            del dist_tile
        del cursor_A1, bool_buf_A1
        _gpu_mem("Adj_A1 pass 2 done")

        _gpu_mem("before return")
        return _pack(
            sampled_idx_A1,
            sampled_idx_A2,
            A1,
            A2,
            DR,
            DR_int,
            nearest_s1,
            d_min_b_A1,
            d_min_b_A1_int,
            nearest_s2,
            d_min_b_A2,
            d_min_b_A2_int,
            nearest_s2_A1,
            d_min_A1_A2,
            d_min_A1_A2_int,
            adj_B_ptr,
            adj_B_col,
            adj_B_dist_int,
            adj_B_dist_float,
            adj_A1_ptr,
            adj_A1_col,
            adj_A1_dist_int,
            adj_A1_dist_float,
        )

    def get_adj_B(self, b: int, result: Dict) -> torch.Tensor:
        """Zero-copy adjacency slice for blue point b."""
        return get_adj_B(b, result)

    def get_adj_A1(self, a1: int, result: Dict) -> torch.Tensor:
        """Zero-copy adjacency slice for A1 center a1."""
        return get_adj_A1(a1, result)


def get_adj_B(b: int, result: Dict) -> torch.Tensor:
    """Return the Adj_B list for blue point b as a 1-D tensor of red indices."""
    ptr = result["adj_B_ptr"]
    return result["adj_B_col"][ptr[b] : ptr[b + 1]]


def get_adj_A1(a1: int, result: Dict) -> torch.Tensor:
    """Return the Adj_A1 list for A1 center a1 as a 1-D tensor of red indices."""
    ptr = result["adj_A1_ptr"]
    return result["adj_A1_col"][ptr[a1] : ptr[a1 + 1]]


def _validate(A: torch.Tensor, B: torch.Tensor) -> None:
    if A.device != B.device:
        raise ValueError("A and B must be on the same device")
    if A.device.type != "cuda":
        raise ValueError("ThreeLevelL1Clustering requires CUDA tensors")
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("A and B must be rank-2 tensors")
    if A.shape != B.shape:
        raise ValueError("A and B must have the same shape (N, d)")
    if not A.is_floating_point() or not B.is_floating_point():
        raise TypeError("A and B must be floating-point tensors")
    if A.shape[0] == 0:
        raise ValueError("A and B must be non-empty")


def _group_offsets(idx: torch.Tensor) -> torch.Tensor:
    """
    For a SORTED 1-D integer tensor, return the 0-based intra-group offset
    of each element.

    Example: [0, 0, 0, 1, 1, 2]  →  [0, 1, 2, 0, 1, 0]
    """
    M = idx.numel()
    if M <= 1:
        return torch.zeros(M, dtype=torch.long, device=idx.device)
    same = torch.zeros(M, dtype=torch.long, device=idx.device)
    same[1:] = (idx[1:] == idx[:-1]).long()
    cumsum = same.cumsum(0)
    start_vals = cumsum.masked_fill(same.bool(), 0)
    baseline = torch.cummax(start_vals, dim=0).values
    return cumsum - baseline


def _pack(
    sampled_idx_A1,
    sampled_idx_A2,
    A1,
    A2,
    DR,
    DR_int,
    nearest_s1,
    d_min_b_A1,
    d_min_b_A1_int,
    nearest_s2,
    d_min_b_A2,
    d_min_b_A2_int,
    nearest_s2_A1,
    d_min_A1_A2,
    d_min_A1_A2_int,
    adj_B_ptr,
    adj_B_col,
    adj_B_dist_int,
    adj_B_dist_float,
    adj_A1_ptr,
    adj_A1_col,
    adj_A1_dist_int,
    adj_A1_dist_float,
) -> Dict[str, torch.Tensor]:
    return {
        "sampled_idx_A1": sampled_idx_A1,
        "sampled_idx_A2": sampled_idx_A2,
        "A1_sampled": A1,
        "A2_sampled": A2,
        "DR": DR,
        "DR_int": DR_int,
        "nearest_s1": nearest_s1,
        "d_min_b_A1": d_min_b_A1,
        "d_min_b_A1_int": d_min_b_A1_int,
        "nearest_s2": nearest_s2,
        "d_min_b_A2": d_min_b_A2,
        "d_min_b_A2_int": d_min_b_A2_int,
        "nearest_s2_A1": nearest_s2_A1,
        "d_min_A1_A2": d_min_A1_A2,
        "d_min_A1_A2_int": d_min_A1_A2_int,
        "adj_B_ptr": adj_B_ptr,
        "adj_B_col": adj_B_col,
        "adj_B_dist_int": adj_B_dist_int,
        "adj_B_dist_float": adj_B_dist_float,
        "adj_A1_ptr": adj_A1_ptr,
        "adj_A1_col": adj_A1_col,
        "adj_A1_dist_int": adj_A1_dist_int,
        "adj_A1_dist_float": adj_A1_dist_float,
    }
