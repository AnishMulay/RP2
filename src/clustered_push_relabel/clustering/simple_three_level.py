import math
from typing import Dict

import torch


def _gpu_mem(label: str) -> None:
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024 ** 3
        res   = torch.cuda.memory_reserved()  / 1024 ** 3
        print(f"  [MEM-3lvl] {label}: alloc={alloc:.2f}GB res={res:.2f}GB",
              flush=True)


class ThreeLevelClustering:
    """
    Three-level memory-efficient clustering for the push-relabel solver.

    Two-level recap (simple.py)
    ───────────────────────────
    Sample P ⊆ A at rate 1/√N  →  |P| ≈ N^(1/2)
    Adj_B(b)  = {a ∈ A : d(b,a) < d(b, nearest P)}    total ≈ N × N^(1/2) = N^(3/2)
    DR matrix = |P| × N                                 size  ≈ N^(3/2)
    Both structures balance at N^(3/2).

    Three-level improvement
    ───────────────────────
    Sample A1 ⊆ A  at rate N^(-1/3)  →  |A1| ≈ N^(2/3)   (first  level centers)
    Sample A2 ⊆ A1 at rate N^(-1/3)  →  |A2| ≈ N^(1/3)   (second level centers)

    Adj_B(b)    = {a ∈ A  : d(b,  a)  < d(b,  nearest A1)}   total ≈ N·N^(1/3)  = N^(4/3)
    Adj_A1(a1)  = {a ∈ A  : d(a1, a)  < d(a1, nearest A2)}   total ≈ N^(2/3)·N^(2/3) = N^(4/3)
    DR matrix   = |A2| × N                                     size  ≈ N^(1/3)·N = N^(4/3)

    All three balance at N^(4/3)  <  N^(3/2).

    Proxy cost for pair (b, a)
    ──────────────────────────
    Level 0 (direct)     a ∈ Adj_B(b)                →  d(b, a)
    Level 1 (via A1)     a ∈ Adj_A1(nearest_s1[b])   →  d(b, s1_b) + d(s1_b, a)
    Level 2 (via A2)     fallback, always covers rest  →  d(b, s2_b) + DR[s2_b, a]

    Memory notes
    ────────────
    DR  kept as float32  (needed by proxy cost builder and push-relabel slack).
    B→A1 distances are never materialised in full; a tiled running-min is used
    instead, matching the two-pass CSR pattern from simple.py exactly.
    float_buf / bool_buf for Adj_B are N × T  (same as simple.py).
    float_buf / bool_buf for Adj_A1 are S1 × T  (reuse same T, free B buffers first).
    """

    def __init__(
        self,
        epsilon: float,
        tile_size: int = 2048,
        sample_factor: float = 1.0,
        profile_memory: bool = False,
    ):
        if tile_size <= 0:
            raise ValueError("tile_size must be positive")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if sample_factor <= 0:
            raise ValueError("sample_factor must be positive")
        self.epsilon       = float(epsilon)
        self.tile_size     = int(tile_size)
        self.sample_factor = float(sample_factor)
        self.profile_memory = bool(profile_memory)
        self.memory_profile = {}

    # ── Opt-in stage-level peak-memory profiling (default OFF, zero overhead
    # when disabled; see NOTES_FOR_REBUTTAL.md). No-op on CPU. ────────────────
    def _prof_reset(self):
        if self.profile_memory and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def _prof_record(self, label):
        if self.profile_memory and torch.cuda.is_available():
            torch.cuda.synchronize()
            self.memory_profile[label] = torch.cuda.max_memory_allocated() / 1024 ** 3

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
        N      = A.shape[0]
        T      = self.tile_size
        eps    = self.epsilon
        if self.profile_memory:
            self.memory_profile = {}

        # ── 1. Sample A1 ⊆ A at rate N^(-1/3)  →  E[|A1|] = N^(2/3) ─────────
        self._prof_reset()
        rate1   = 1.0 / (self.sample_factor * (float(N) ** (1.0 / 3.0)))
        mask_A1 = torch.rand(N, device=device) < rate1
        if not mask_A1.any():
            mask_A1[torch.randint(N, (1,), device=device)] = True
        sampled_idx_A1 = mask_A1.nonzero(as_tuple=True)[0]   # (S1,)
        A1             = A[sampled_idx_A1]                    # (S1, d)
        S1             = sampled_idx_A1.shape[0]

        # ── 2. Sample A2 ⊆ A1 at rate N^(-1/3)  →  E[|A2|] = N^(1/3) ─────────
        #     Effective rate from A is N^(-2/3).
        rate2   = 1.0 / (self.sample_factor * (float(N) ** (1.0 / 3.0)))
        mask_A2 = torch.rand(S1, device=device) < rate2
        if not mask_A2.any():
            mask_A2[torch.randint(S1, (1,), device=device)] = True
        local_idx_A2   = mask_A2.nonzero(as_tuple=True)[0]   # indices into A1
        sampled_idx_A2 = sampled_idx_A1[local_idx_A2]        # indices into A
        A2             = A[sampled_idx_A2]                    # (S2, d)
        S2             = sampled_idx_A2.shape[0]
        self._prof_record("landmark_sampling_A1_A2")

        _gpu_mem(f"sampled A1={S1}, A2={S2}")

        # ── 3. Pre-compute squared norms (reused across all tiled operations) ──
        B_sq  = B.pow(2).sum(dim=1)    # (N,)
        A_sq  = A.pow(2).sum(dim=1)    # (N,)
        A1_sq = A1.pow(2).sum(dim=1)   # (S1,)

        # ── 4. DR: A2 × A  (S2 ≈ N^(1/3) is tiny; one cdist call is fine) ─────
        #     Kept as float32; needed by proxy cost builder and push-relabel slack.
        self._prof_reset()
        DR     = torch.cdist(A2, A,
                             compute_mode="use_mm_for_euclid_dist_if_necessary")  # (S2, N)
        DR_int = (DR / eps).ceil_().to(torch.int32)                               # (S2, N)
        self._prof_record("DR_construction")
        _gpu_mem("after DR")

        # ── 5. DB_A2: nearest A2 center for each b  (N × S2 ≈ N^(4/3), tiny) ──
        # ── 6. DA1_A2: nearest A2 center for each a1  (S1 × S2, tiny) ──────────
        # Profiled together as the "landmark-assignment" stage (DB / d_min_b
        # equivalent for the 3-level pipeline).
        self._prof_reset()
        DB_A2              = torch.cdist(B, A2,
                                         compute_mode="use_mm_for_euclid_dist_if_necessary")
        d_min_b_A2, nearest_s2 = DB_A2.min(dim=1)    # (N,) each
        del DB_A2
        d_min_b_A2_int = (d_min_b_A2 / eps).ceil_().to(torch.int32)
        _gpu_mem("after DB_A2")

        DA1_A2                  = torch.cdist(A1, A2,
                                              compute_mode="use_mm_for_euclid_dist_if_necessary")
        d_min_A1_A2, nearest_s2_A1 = DA1_A2.min(dim=1)   # (S1,) each
        del DA1_A2
        d_min_A1_A2_int = (d_min_A1_A2 / eps).ceil_().to(torch.int32)
        self._prof_record("landmark_assignment_A2")
        _gpu_mem("after DA1_A2")

        # ── 7. DB_A1: nearest A1 center for each b  (tiled over A1) ────────────
        #     The full N × S1 matrix (≈ N^(5/3) entries) is never materialised.
        #     We tile over A1 using the same float_buf + _sq_dist_inplace pattern
        #     as simple.py, tracking a running squared-minimum so that sqrt is
        #     taken only once after the loop.
        self._prof_reset()
        float_buf     = torch.empty(N, T, dtype=A.dtype, device=device)
        d_min_b_A1_sq = torch.full((N,), float("inf"), dtype=A.dtype, device=device)
        nearest_s1    = torch.zeros(N, dtype=torch.long, device=device)

        for start in range(0, S1, T):
            end = min(start + T, S1)
            t   = end - start
            _sq_dist_inplace(B, A1[start:end], B_sq, A1_sq[start:end],
                             float_buf[:, :t])
            tile_min_sq, tile_argmin = float_buf[:, :t].min(dim=1)   # (N,) each
            update        = tile_min_sq < d_min_b_A1_sq
            d_min_b_A1_sq = torch.where(update, tile_min_sq, d_min_b_A1_sq)
            nearest_s1    = torch.where(update, tile_argmin + start, nearest_s1)

        d_min_b_A1     = d_min_b_A1_sq.sqrt()
        d_min_b_A1_int = (d_min_b_A1 / eps).ceil_().to(torch.int32)
        del d_min_b_A1_sq
        self._prof_record("landmark_assignment_A1_tiled")
        _gpu_mem("after DB_A1 tiled min")

        # ── 8. Adj_B: two-pass CSR, threshold = d_min_b_A1 ─────────────────────
        #     Condition: d(b, a)² < d_min_b_A1[b]²  (comparison in squared space)
        #     Pattern mirrors simple.py exactly; float_buf (N × T) is already live.
        bool_buf           = torch.empty(N, T, dtype=torch.bool, device=device)
        d_min_b_A1_sq_thr  = d_min_b_A1.pow(2)   # (N,) — threshold in squared space

        # Pass 1 — count admissible (b, a) pairs per blue
        self._prof_reset()
        counts_B = torch.zeros(N, dtype=torch.long, device=device)
        for start in range(0, N, T):
            end = min(start + T, N)
            t   = end - start
            _sq_dist_inplace(B, A[start:end], B_sq, A_sq[start:end],
                             float_buf[:, :t])
            torch.lt(float_buf[:, :t], d_min_b_A1_sq_thr.unsqueeze(1),
                     out=bool_buf[:, :t])
            counts_B.add_(bool_buf[:, :t].sum(dim=1))
        self._prof_record("adj_B_csr_pass1_counting")
        _gpu_mem("Adj_B pass 1 done")

        adj_B_ptr = torch.zeros(N + 1, dtype=torch.long, device=device)
        adj_B_ptr[1:] = counts_B.cumsum(0)
        MB               = int(adj_B_ptr[-1].item())
        adj_B_col        = torch.empty(MB, dtype=torch.long,  device=device)
        adj_B_dist_int   = torch.empty(MB, dtype=torch.int32, device=device)
        adj_B_dist_float = torch.empty(MB, dtype=A.dtype,     device=device)
        del counts_B
        _gpu_mem(f"Adj_B allocated MB={MB}")

        # Pass 2 — fill adj_B_col and adj_B_dist_*
        self._prof_reset()
        cursor_B = adj_B_ptr[:-1].clone()   # (N,) write cursors, start at row offset
        for start in range(0, N, T):
            end = min(start + T, N)
            t   = end - start
            _sq_dist_inplace(B, A[start:end], B_sq, A_sq[start:end],
                             float_buf[:, :t])
            torch.lt(float_buf[:, :t], d_min_b_A1_sq_thr.unsqueeze(1),
                     out=bool_buf[:, :t])
            b_idx, t_idx = bool_buf[:, :t].nonzero(as_tuple=True)   # row-major → b_idx sorted
            if b_idx.numel() == 0:
                continue
            write_pos                   = cursor_B[b_idx] + _group_offsets(b_idx)
            adj_B_col[write_pos]        = (t_idx + start).long()
            actual_dists                = torch.sqrt(float_buf[:, :t][b_idx, t_idx])
            adj_B_dist_float[write_pos] = actual_dists
            adj_B_dist_int[write_pos]   = (actual_dists / eps).ceil_().to(torch.int32)
            cursor_B.scatter_add_(0, b_idx, torch.ones_like(b_idx))
        del cursor_B, d_min_b_A1_sq_thr
        self._prof_record("adj_B_csr_pass2_fill")
        _gpu_mem("Adj_B pass 2 done")

        # ── 9. Adj_A1: two-pass CSR, threshold = d_min_A1_A2 ───────────────────
        #     Structurally identical to Adj_B but rows are A1 points, not blues.
        #     _sq_dist_inplace works unchanged: A1 plays the role of B here.
        #     Free the N-row buffers first, then allocate S1-row replacements so
        #     peak memory stays bounded.
        del float_buf, bool_buf

        float_buf_A1       = torch.empty(S1, T, dtype=A.dtype,    device=device)
        bool_buf_A1        = torch.empty(S1, T, dtype=torch.bool, device=device)
        d_min_A1_A2_sq_thr = d_min_A1_A2.pow(2)   # (S1,)

        # Pass 1 — count admissible (a1, a) pairs per A1 center
        self._prof_reset()
        counts_A1 = torch.zeros(S1, dtype=torch.long, device=device)
        for start in range(0, N, T):
            end = min(start + T, N)
            t   = end - start
            _sq_dist_inplace(A1, A[start:end], A1_sq, A_sq[start:end],
                             float_buf_A1[:, :t])
            torch.lt(float_buf_A1[:, :t], d_min_A1_A2_sq_thr.unsqueeze(1),
                     out=bool_buf_A1[:, :t])
            counts_A1.add_(bool_buf_A1[:, :t].sum(dim=1))
        self._prof_record("adj_A1_csr_pass1_counting")
        _gpu_mem("Adj_A1 pass 1 done")

        adj_A1_ptr = torch.zeros(S1 + 1, dtype=torch.long, device=device)
        adj_A1_ptr[1:] = counts_A1.cumsum(0)
        MA1               = int(adj_A1_ptr[-1].item())
        adj_A1_col        = torch.empty(MA1, dtype=torch.long,  device=device)
        adj_A1_dist_int   = torch.empty(MA1, dtype=torch.int32, device=device)
        adj_A1_dist_float = torch.empty(MA1, dtype=A.dtype,     device=device)
        del counts_A1
        _gpu_mem(f"Adj_A1 allocated MA1={MA1}")

        # Pass 2 — fill adj_A1_col and adj_A1_dist_*
        self._prof_reset()
        cursor_A1 = adj_A1_ptr[:-1].clone()   # (S1,) write cursors
        for start in range(0, N, T):
            end = min(start + T, N)
            t   = end - start
            _sq_dist_inplace(A1, A[start:end], A1_sq, A_sq[start:end],
                             float_buf_A1[:, :t])
            torch.lt(float_buf_A1[:, :t], d_min_A1_A2_sq_thr.unsqueeze(1),
                     out=bool_buf_A1[:, :t])
            a1_idx, t_idx = bool_buf_A1[:, :t].nonzero(as_tuple=True)  # a1_idx sorted
            if a1_idx.numel() == 0:
                continue
            write_pos                    = cursor_A1[a1_idx] + _group_offsets(a1_idx)
            adj_A1_col[write_pos]        = (t_idx + start).long()
            actual_dists                 = torch.sqrt(float_buf_A1[:, :t][a1_idx, t_idx])
            adj_A1_dist_float[write_pos] = actual_dists
            adj_A1_dist_int[write_pos]   = (actual_dists / eps).ceil_().to(torch.int32)
            cursor_A1.scatter_add_(0, a1_idx, torch.ones_like(a1_idx))
        del cursor_A1, float_buf_A1, bool_buf_A1, d_min_A1_A2_sq_thr
        self._prof_record("adj_A1_csr_pass2_fill")
        _gpu_mem("Adj_A1 pass 2 done")

        _gpu_mem("before return")
        return _pack(
            sampled_idx_A1, sampled_idx_A2, A1, A2,
            DR, DR_int,
            nearest_s1,  d_min_b_A1,  d_min_b_A1_int,
            nearest_s2,  d_min_b_A2,  d_min_b_A2_int,
            nearest_s2_A1, d_min_A1_A2, d_min_A1_A2_int,
            adj_B_ptr,  adj_B_col,  adj_B_dist_int,  adj_B_dist_float,
            adj_A1_ptr, adj_A1_col, adj_A1_dist_int, adj_A1_dist_float,
        )

    def get_adj_B(self, b: int, result: Dict) -> torch.Tensor:
        """Zero-copy adjacency slice for blue point b."""
        return get_adj_B(b, result)

    def get_adj_A1(self, a1: int, result: Dict) -> torch.Tensor:
        """Zero-copy adjacency slice for A1 center a1."""
        return get_adj_A1(a1, result)


# ── Module-level helpers ──────────────────────────────────────────────────────

def get_adj_B(b: int, result: Dict) -> torch.Tensor:
    """Return the Adj_B list for blue point b as a 1-D tensor of red indices."""
    ptr = result["adj_B_ptr"]
    return result["adj_B_col"][ptr[b]: ptr[b + 1]]


def get_adj_A1(a1: int, result: Dict) -> torch.Tensor:
    """Return the Adj_A1 list for A1 center a1 as a 1-D tensor of red indices."""
    ptr = result["adj_A1_ptr"]
    return result["adj_A1_col"][ptr[a1]: ptr[a1 + 1]]


# ── Internal helpers ──────────────────────────────────────────────────────────

def _validate(A: torch.Tensor, B: torch.Tensor) -> None:
    if A.device != B.device:
        raise ValueError("A and B must be on the same device")
    if A.device.type not in ("cuda", "cpu"):
        raise ValueError("ThreeLevelClustering requires CUDA or CPU tensors")
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
    Compute squared L2 distances ||B[i] - A_tile[j]||² into `out` (rows × t),
    fully in-place — zero extra allocations.  Mirrors simple.py exactly.

    Works for any (rows, d) tensor in place of B:
        - Pass B       (N,  d) for Adj_B and DB_A1 loops
        - Pass A1      (S1, d) for Adj_A1 loop
    """
    torch.mm(B, A_tile.t(), out=out)    # out ← b · a
    out.mul_(-2.0)                      # out ← -2(b·a)
    out.add_(B_sq.unsqueeze(1))         # out ← ||b||² - 2(b·a)
    out.add_(A_tile_sq.unsqueeze(0))    # out ← ||b||² + ||a||² - 2(b·a)
    out.clamp_(min=0.0)                 # guard fp cancellation near b ≈ a


def _group_offsets(idx: torch.Tensor) -> torch.Tensor:
    """
    For a SORTED 1-D integer tensor, return the 0-based intra-group offset
    of each element.  Mirrors simple.py exactly (variable renamed idx → generic).

    Example: [0, 0, 0, 1, 1, 2]  →  [0, 1, 2, 0, 1, 0]
    """
    M = idx.numel()
    if M <= 1:
        return torch.zeros(M, dtype=torch.long, device=idx.device)
    same       = torch.zeros(M, dtype=torch.long, device=idx.device)
    same[1:]   = (idx[1:] == idx[:-1]).long()
    cumsum     = same.cumsum(0)
    start_vals = cumsum.masked_fill(same.bool(), 0)
    baseline   = torch.cummax(start_vals, dim=0).values
    return cumsum - baseline


def _pack(
    sampled_idx_A1, sampled_idx_A2, A1, A2,
    DR, DR_int,
    nearest_s1,  d_min_b_A1,  d_min_b_A1_int,
    nearest_s2,  d_min_b_A2,  d_min_b_A2_int,
    nearest_s2_A1, d_min_A1_A2, d_min_A1_A2_int,
    adj_B_ptr,  adj_B_col,  adj_B_dist_int,  adj_B_dist_float,
    adj_A1_ptr, adj_A1_col, adj_A1_dist_int, adj_A1_dist_float,
) -> Dict[str, torch.Tensor]:
    return {
        "sampled_idx_A1"   : sampled_idx_A1,
        "sampled_idx_A2"   : sampled_idx_A2,
        "A1_sampled"       : A1,
        "A2_sampled"       : A2,
        "DR"               : DR,
        "DR_int"           : DR_int,
        "nearest_s1"       : nearest_s1,
        "d_min_b_A1"       : d_min_b_A1,
        "d_min_b_A1_int"   : d_min_b_A1_int,
        "nearest_s2"       : nearest_s2,
        "d_min_b_A2"       : d_min_b_A2,
        "d_min_b_A2_int"   : d_min_b_A2_int,
        "nearest_s2_A1"    : nearest_s2_A1,
        "d_min_A1_A2"      : d_min_A1_A2,
        "d_min_A1_A2_int"  : d_min_A1_A2_int,
        "adj_B_ptr"        : adj_B_ptr,
        "adj_B_col"        : adj_B_col,
        "adj_B_dist_int"   : adj_B_dist_int,
        "adj_B_dist_float" : adj_B_dist_float,
        "adj_A1_ptr"       : adj_A1_ptr,
        "adj_A1_col"       : adj_A1_col,
        "adj_A1_dist_int"  : adj_A1_dist_int,
        "adj_A1_dist_float": adj_A1_dist_float,
    }