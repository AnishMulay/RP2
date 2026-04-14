import math
from typing import Dict

import torch


class SimpleClustering:
    """
    Memory-efficient clustering representation for the push-relabel solver.

    Red points  = A  (right side, N points)  — matched into
    Blue points = B  (left side,  N points)  — propose outwards

    Two output structures
    ─────────────────────
    DR  (S × N)             distance from each sampled red center to every red point
    CSR (adj_ptr, adj_col)  for each blue b, red points a with d(b,a) < d_min_b
                            where d_min_b = distance to b's nearest sampled red center
                            condition is STRICTLY less-than

    Space budget (N=200 K, d=2, tile=512)
    ──────────────────────────────────────
    DR + DB          ~680 MB   (kept; push-relabel uses them for slack)
    float_buf        ~400 MB   (pre-allocated once, reused every tile)
    bool_buf         ~100 MB   (pre-allocated once, reused every tile)
    adj_col          ~560 MB   (final output, allocated after pass 1 gives exact M)
    cursor           ~1.6 MB   (per-blue write cursor in pass 2)
    ─────────────────────────────────────
    Peak             ~1.74 GB  (vs ~4+ GB in the accumulate-and-sort approach)

    The original approach accumulated all (b,a) pairs across tiles, then called
    argsort — which briefly created 4-5 copies of the pairs buffer (~3.4 GB for
    N=200 K).  This version avoids that entirely with a two-pass CSR build:
      Pass 1: count admissible pairs per blue → adj_ptr via cumsum
      Pass 2: write adj_col directly using pre-computed write cursors

    This is safe because torch.nonzero() on a 2D tensor scans row-major, so
    b_idx from each tile is always sorted — intra-group write offsets can be
    computed in O(M) with a vectorised cummax, no sort needed.
    """

    def __init__(self, tile_size: int = 2048):
        if tile_size <= 0:
            raise ValueError("tile_size must be positive")
        self.tile_size = int(tile_size)

    # ── Public interface ──────────────────────────────────────────────────────

    def run(self, A: torch.Tensor, B: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ──────────
        A : (N, d)  red  points — CUDA, floating-point
        B : (N, d)  blue points — CUDA, floating-point, same device as A

        Returns
        ───────
        dict with keys:
            sampled_idx  (S,)    indices into A of sampled red centers
            A_sampled    (S, d)  coordinates of sampled red centers
            DR           (S, N)  DR[i,j] = d(A_sampled[i], A[j])
            DB           (N, S)  DB[b,s] = d(B[b], A_sampled[s])
            d_min_b      (N,)    d(B[b], nearest sampled red)
            nearest_s    (N,)    index into sampled_idx for each blue
            adj_ptr      (N+1,)  CSR row pointers  (int64)
            adj_col      (M,)    red indices per blue in CSR order  (int64)
        """
        _validate(A, B)
        device = A.device
        N      = A.shape[0]
        T      = self.tile_size

        # ── 1. Sample red centers  w.p. 1/√N ─────────────────────────────────
        sample_mask = torch.rand(N, device=device) < (1.0 / math.sqrt(N))
        if not sample_mask.any():
            sample_mask[torch.randint(N, (1,), device=device)] = True
        sampled_idx = sample_mask.nonzero(as_tuple=True)[0]    # (S,)
        A_s         = A[sampled_idx]                            # (S, d)

        # ── 2. DR and DB  (real L2, kept for slack computations) ──────────────
        DR = torch.cdist(A_s, A,
                         compute_mode="use_mm_for_euclid_dist_if_necessary")  # (S, N)
        DB = torch.cdist(B,  A_s,
                         compute_mode="use_mm_for_euclid_dist_if_necessary")  # (N, S)
        d_min_b, nearest_s = DB.min(dim=1)                     # (N,)  each

        # ── 3. Pre-computed norms and threshold ───────────────────────────────
        d_min_b_sq = d_min_b.pow(2)            # (N,)  threshold per blue
        A_sq       = A.pow(2).sum(dim=1)       # (N,)  ||a||²
        B_sq       = B.pow(2).sum(dim=1)       # (N,)  ||b||²

        # ── 4. Pre-allocate tile buffers  (allocated ONCE, reused every tile) ─
        #  float_buf : squared distances for one tile of red points
        #  bool_buf  : admissibility mask for one tile
        # Keeping these alive outside the loop means zero GPU allocations in
        # the hot path.
        float_buf = torch.empty(N, T, dtype=A.dtype,    device=device)
        bool_buf  = torch.empty(N, T, dtype=torch.bool, device=device)

        # ── 5. Pass 1: count admissible pairs per blue ────────────────────────
        counts = torch.zeros(N, dtype=torch.long, device=device)

        for start in range(0, N, T):
            end = min(start + T, N)
            t   = end - start
            _sq_dist_inplace(B, A[start:end], B_sq, A_sq[start:end],
                             float_buf[:, :t])
            torch.lt(float_buf[:, :t], d_min_b_sq.unsqueeze(1),
                     out=bool_buf[:, :t])
            counts.add_(bool_buf[:, :t].sum(dim=1))

        # ── 6. Build CSR pointers; allocate column buffer ─────────────────────
        adj_ptr     = torch.zeros(N + 1, dtype=torch.long, device=device)
        adj_ptr[1:] = counts.cumsum(0)
        M           = int(adj_ptr[-1].item())
        adj_col     = torch.empty(M, dtype=torch.long, device=device)
        del counts

        if M == 0:
            del float_buf, bool_buf
            return _pack(sampled_idx, A_s, DR, DB, d_min_b, nearest_s,
                         adj_ptr, adj_col)

        # ── 7. Pass 2: fill adj_col in CSR order ──────────────────────────────
        # cursor[b] = next write position for blue point b.
        # Starts at adj_ptr[b] and advances as pairs are placed.
        # At the end of pass 2, cursor[b] == adj_ptr[b+1] for every b.
        cursor = adj_ptr[:-1].clone()   # (N,) int64

        for start in range(0, N, T):
            end = min(start + T, N)
            t   = end - start
            _sq_dist_inplace(B, A[start:end], B_sq, A_sq[start:end],
                             float_buf[:, :t])
            torch.lt(float_buf[:, :t], d_min_b_sq.unsqueeze(1),
                     out=bool_buf[:, :t])

            # b_idx is sorted because nonzero() scans row-major on the 2D mask
            b_idx, t_idx = bool_buf[:, :t].nonzero(as_tuple=True)
            if b_idx.numel() == 0:
                continue

            # Write positions: cursor[b] + intra-group offset of this pair
            # (no sort needed — b_idx is already sorted)
            write_pos       = cursor[b_idx] + _group_offsets(b_idx)
            adj_col[write_pos] = (t_idx + start).long()

            # Advance cursor for every blue point that received pairs this tile
            cursor.scatter_add_(0, b_idx, torch.ones_like(b_idx))

        del float_buf, bool_buf, cursor
        return _pack(sampled_idx, A_s, DR, DB, d_min_b, nearest_s, adj_ptr, adj_col)

    def get_adj(self, b: int, result: Dict) -> torch.Tensor:
        """Zero-copy adjacency slice for blue point b (no data movement)."""
        return get_adj(b, result)


# ── Module-level helpers ──────────────────────────────────────────────────────

def get_adj(b: int, result: Dict) -> torch.Tensor:
    """Return the adjacency list of blue point b as a 1-D tensor of red indices."""
    ptr = result["adj_ptr"]
    return result["adj_col"][ptr[b] : ptr[b + 1]]


# ── Internal helpers ──────────────────────────────────────────────────────────

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


def _sq_dist_inplace(
    B: torch.Tensor,
    A_tile: torch.Tensor,
    B_sq: torch.Tensor,
    A_tile_sq: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """
    Compute squared L2 distances ||B[i] - A_tile[j]||² into `out` (N × t),
    fully in-place — zero extra allocations.

    Identity used:
        ||b - a||² = ||b||² + ||a||² - 2(b · a)

    The matmul `B @ A_tile.T` maps to a single cuBLAS sgemm.  In-place
    add/mul avoid the temporary tensor that broadcasting would otherwise
    create.

    clamp_(min=0) eliminates small negative values from floating-point
    cancellation when b ≈ a, which would produce NaN or false non-admissibility.
    """
    torch.mm(B, A_tile.t(), out=out)        # out  ← b · a
    out.mul_(-2.0)                          # out  ← -2(b · a)
    out.add_(B_sq.unsqueeze(1))             # out  ← ||b||² - 2(b·a)
    out.add_(A_tile_sq.unsqueeze(0))        # out  ← ||b||² + ||a||² - 2(b·a)
    out.clamp_(min=0.0)                     # guard against fp cancellation


def _group_offsets(b_idx: torch.Tensor) -> torch.Tensor:
    """
    For a SORTED 1-D integer tensor, return the 0-based intra-group offset of
    each element — i.e. how many times this value has appeared before this
    position.

    Example:  b_idx = [0, 0, 0, 1, 1, 2]  →  [0, 1, 2, 0, 1, 0]

    Algorithm (O(M), fully vectorised, no Python loop over elements):
      1. same[i]     = 1 if b_idx[i] == b_idx[i-1] (i.e. not a group start)
      2. cumsum      = prefix sum of same  (monotone; resets at each group start
                       but without the right baseline)
      3. start_vals  = cumsum where same == 0 (group starts), else 0
      4. baseline    = cummax(start_vals)  — forward-fills each group-start
                       cumsum value across the whole group
      5. offset      = cumsum - baseline

    Proof of correctness:
      At a group start position p: cumsum[p] = k (some value), baseline[p] = k,
      so offset[p] = 0. ✓
      For position p+j within the same group: cumsum[p+j] = k+j,
      baseline[p+j] = k (forward-filled by cummax), so offset[p+j] = j. ✓
    """
    M = b_idx.numel()
    if M <= 1:
        return torch.zeros(M, dtype=torch.long, device=b_idx.device)

    same       = torch.zeros(M, dtype=torch.long, device=b_idx.device)
    same[1:]   = (b_idx[1:] == b_idx[:-1]).long()   # 0 at group starts, 1 inside

    cumsum     = same.cumsum(0)
    # At group starts (same == 0), keep cumsum value; zero out non-starts.
    start_vals = cumsum.masked_fill(same.bool(), 0)
    # Forward-fill: cummax propagates the last group-start's cumsum value.
    baseline   = torch.cummax(start_vals, dim=0).values
    return cumsum - baseline


def _pack(sampled_idx, A_s, DR, DB, d_min_b, nearest_s,
          adj_ptr, adj_col) -> Dict[str, torch.Tensor]:
    return {
        "sampled_idx" : sampled_idx,
        "A_sampled"   : A_s,
        "DR"          : DR,
        "DB"          : DB,
        "d_min_b"     : d_min_b,
        "nearest_s"   : nearest_s,
        "adj_ptr"     : adj_ptr,
        "adj_col"     : adj_col,
    }