# Color-Aware Clustered Push-Relabel — Implementation Specification

This document is the single source of truth for implementing the color-aware two-level
clustered push-relabel solver. The VS Code agent must follow this spec exactly. Do not
infer, generalise, or substitute any part of it. If anything is unclear, stop and re-read
this document before writing any code.

---

## 0. Repository Context

- Package root: `src/clustered_push_relabel/`
- Clustering modules live in: `src/clustered_push_relabel/clustering/`
- Solver modules live in: `src/clustered_push_relabel/solvers/`
- Utilities live in: `src/clustered_push_relabel/utils/`
- Experiments live in: `experiments/runners/`
- GPU framework: PyTorch only. No CuPy, no custom CUDA.
- The existing files `bipartite.py`, `two_level.py`, `k_level.py`, and all existing
  experiments must not be modified.

**Three new files to create — in this order:**

1. `src/clustered_push_relabel/clustering/color_aware_two_level.py`
2. `src/clustered_push_relabel/solvers/color_aware_bipartite.py`
3. `experiments/runners/e2_color_aware_vs_exact.py`

---

## 1. Problem Setup

- Input: `P_red` — n red (A) points; `P_blue` — n blue (B) points. Both shape `(N, D)`.
- Goal: produce an ε-approximate min-cost bipartite matching M between A and B.
- All distances must be normalised to `[0, 1]` by dividing by the diameter Δ before
  clustering. Normalisation is done inside the solver's `__init__`, not in the experiment.
- ε is the shared error parameter used in both clustering and push-relabel.

---

## 2. File 1 — Clustering: `color_aware_two_level.py`

### 2.1 Purpose

Implements color-aware two-level clustering. Produces shell-level COO triplets consumed
by the solver's CSR builder.

### 2.2 Imports

```python
import math
import torch
from ..utils.distance import TiledEuclideanKernel, TiledManhattanKernel
```

### 2.3 Class: `ColorAwareClustering`

#### `__init__(self, epsilon, batch_size=256, metric='L2')`

Store `self.epsilon`, `self.batch_size`, `self.metric`. Nothing else.

#### `run(self, P_red_norm, P_blue_norm) -> (blue_coo, red_coo)`

Both inputs are pre-normalised float tensors of shape `(N, D)` on the same device.
Returns two COO tuples. See §2.4 for return convention.

### 2.4 Return Convention

```
blue_coo = (b_c, b_l, b_p)   # all torch.long, same device as input
red_coo  = (r_c, r_l, r_p)   # all torch.long, same device as input
```

- Center IDs (`b_c`, `r_c`): per-color, in `[0, N)`.
  `b_c` indexes into `P_blue_norm`; `r_c` indexes into `P_red_norm`.
- Point IDs (`b_p`, `r_p`): global convention `[0, 2N)`.
  Red points = `[0, N)`; blue points = `[N, 2N)`.
  This matches the convention used by `FastGPUClustering` in `two_level.py`.
- Level IDs (`b_l`, `r_l`): `floor(d(point, center) / epsilon)`, dtype `torch.long`.
  These are **shell** levels — each point stored exactly once at its own level.

### 2.5 Distance Helper

Build once, reuse for all calls:

```python
if self.metric == 'L1':
    kernel = TiledManhattanKernel(chunk_size=self.batch_size)
else:
    kernel = TiledEuclideanKernel(chunk_size=self.batch_size)
ws_all = kernel.prepare_workspace(P_all)   # workspace over all 2N points
```

`get_dists(query, ws_all)` returns shape `(2N, |query|)` of actual distances:
- For L1: `kernel.compute_dist_tile(query, ws_all)` directly (already actual distances).
- For L2: `torch.sqrt(kernel.compute_dist_tile(query, ws_all))` (kernel returns squared).

### 2.6 Algorithm Steps Inside `run()`

**Step 1 — Setup:**
```python
device   = P_red_norm.device
N        = P_red_norm.shape[0]
n_total  = 2 * N
P_all    = torch.cat([P_red_norm, P_blue_norm], dim=0)  # global IDs [0, 2N)
ws_all   = kernel.prepare_workspace(P_all)
```

**Step 2 — Sampling:**
```python
p_sample     = 1.0 / math.sqrt(n_total)
sampled_mask = torch.bernoulli(torch.full((n_total,), p_sample, device=device)).bool()
P1_red_mask  = sampled_mask[:N]    # (N,) bool — True = this red center is sampled
P1_blue_mask = sampled_mask[N:]    # (N,) bool — True = this blue center is sampled
# Guarantee at least one sampled center per color
if P1_red_mask.sum() == 0:
    P1_red_mask[0] = True
if P1_blue_mask.sum() == 0:
    P1_blue_mask[0] = True
```

**Step 3 — Nearest sampled distance per point:**
```python
sampled_red_pts  = P_red_norm[P1_red_mask]    # (k_r, D)
sampled_blue_pts = P_blue_norm[P1_blue_mask]  # (k_b, D)
d_min_red  = get_dists(sampled_red_pts,  ws_all).min(dim=1).values   # (2N,)
d_min_blue = get_dists(sampled_blue_pts, ws_all).min(dim=1).values   # (2N,)
```

**Step 4 — Build COO for red centers (batched):**
```python
red_c_list, red_l_list, red_p_list = [], [], []
for batch_start in range(0, N, self.batch_size):
    batch_end = min(batch_start + self.batch_size, N)
    c_ids = torch.arange(batch_start, batch_end, device=device, dtype=torch.long)
    c_pts = P_red_norm[c_ids]                           # (b, D)
    dists = get_dists(c_pts, ws_all)                    # (2N, b)

    is_sampled         = P1_red_mask[c_ids]             # (b,) bool
    non_sampled_member = dists < d_min_red.unsqueeze(1) # (2N, b)
    sampled_member     = is_sampled.unsqueeze(0).expand(n_total, -1)  # (2N, b)
    member             = sampled_member | non_sampled_member

    levels = (dists / self.epsilon).long()              # (2N, b)

    p_idx, c_local = torch.where(member)
    if p_idx.numel() > 0:
        red_c_list.append(c_ids[c_local])
        red_l_list.append(levels[p_idx, c_local])
        red_p_list.append(p_idx)                        # global IDs [0, 2N)
```

**Step 5 — Build COO for blue centers (same pattern):**

Identical to Step 4, replacing:
- `P_red_norm` → `P_blue_norm`
- `P1_red_mask` → `P1_blue_mask`
- `d_min_red` → `d_min_blue`
- appending to `blue_c_list`, `blue_l_list`, `blue_p_list`

**Step 6 — Concatenate and return:**
```python
def _cat_or_empty(lst, device):
    if not lst:
        return torch.empty(0, dtype=torch.long, device=device)
    return torch.cat(lst)

r_c = _cat_or_empty(red_c_list,  device)
r_l = _cat_or_empty(red_l_list,  device)
r_p = _cat_or_empty(red_p_list,  device)
b_c = _cat_or_empty(blue_c_list, device)
b_l = _cat_or_empty(blue_l_list, device)
b_p = _cat_or_empty(blue_p_list, device)
return (b_c, b_l, b_p), (r_c, r_l, r_p)
```

---

## 3. File 2 — Solver: `color_aware_bipartite.py`

### 3.1 Module-Level Helpers

Copy these three functions verbatim from `bipartite.py` (they are module-level there).
Do not import from `bipartite.py` — copy them into this file:

```python
def _ensure_long_arange(owner, attr_name, size, device):
    buf = getattr(owner, attr_name, None)
    if buf is None or buf.numel() < size:
        buf = torch.arange(size, device=device, dtype=torch.long)
        setattr(owner, attr_name, buf)
    return buf[:size]

def _ensure_zero_long_buffer(owner, attr_name, size, device):
    buf = getattr(owner, attr_name, None)
    if buf is None or buf.numel() < size:
        buf = torch.empty(size, device=device, dtype=torch.long)
        setattr(owner, attr_name, buf)
    buf = buf[:size]
    buf.zero_()
    return buf

def _ensure_bool_buffer(owner, attr_name, size, device):
    buf = getattr(owner, attr_name, None)
    if buf is None or buf.numel() < size:
        buf = torch.empty(size, device=device, dtype=torch.bool)
        setattr(owner, attr_name, buf)
    return buf[:size]
```

### 3.2 Imports

```python
import torch
import math
import gc
from ..clustering.color_aware_two_level import ColorAwareClustering
```

### 3.3 Class: `ColorAwareTwoLevelSolver`

#### `__init__(self, P_red, P_blue, epsilon, batch_size=None, metric='L2', verbose=False)`

**Step 1 — Store args:**
```python
self.device     = P_red.device
self.N          = P_red.shape[0]
self.epsilon    = epsilon
self.metric     = metric
self.verbose    = verbose
self.batch_size = 256 if batch_size is None else batch_size
```

**Step 2 — Diameter normalisation:**
```python
P_all = torch.cat([P_red, P_blue], dim=0)
if metric == 'L1':
    delta = (P_all.max(dim=0).values - P_all.min(dim=0).values).sum()
else:
    delta = ((P_all.max(dim=0).values - P_all.min(dim=0).values).pow(2).sum()).sqrt()
delta       = delta.clamp(min=1e-8)
self.delta  = delta
P_red_norm  = P_red.float()  / delta
P_blue_norm = P_blue.float() / delta
```

Do NOT store `P_red` or `P_blue` as instance attributes.

**Step 3 — Clustering:**
```python
cluster_engine       = ColorAwareClustering(epsilon, batch_size=self.batch_size, metric=metric)
blue_coo, red_coo    = cluster_engine.run(P_red_norm, P_blue_norm)
```

**Step 4 — Build index:**
```python
self._build_csr_and_inv(blue_coo, red_coo)
```

**Step 5 — Cleanup and state init:**
```python
del blue_coo, red_coo, cluster_engine, P_red_norm, P_blue_norm, P_all
gc.collect()
self.yA = torch.zeros(self.N, device=self.device, dtype=torch.int32)
self.yB = torch.full((self.N,), 1,  device=self.device, dtype=torch.int32)
self.MA = torch.full((self.N,), -1, device=self.device, dtype=torch.int32)
self.MB = torch.full((self.N,), -1, device=self.device, dtype=torch.int32)
```

---

### 3.4 Method: `_build_csr_and_inv(self, blue_coo, red_coo)`

This method builds **seven** index structures. Read carefully — several are new and do not
exist in any existing solver.

#### 3.4.1 Inputs

```
blue_coo = (b_c, b_l, b_p)  — center [0,N), level, point [0,2N)  — blue centers
red_coo  = (r_c, r_l, r_p)  — center [0,N), level, point [0,2N)  — red centers
```

#### 3.4.2 Unification and Filtering (same as existing solver)

```python
b_c, b_l, b_p = blue_coo
r_c, r_l, r_p = red_coo
N = self.N

b_c_shifted  = b_c + N                              # blue centers → [N, 2N)
all_centers  = torch.cat([b_c_shifted, r_c])
all_levels   = torch.cat([b_l, r_l])
all_points   = torch.cat([b_p, r_p])
is_red_point = all_points < N

centers_with_red  = torch.unique(all_centers[is_red_point])
centers_with_blue = torch.unique(all_centers[~is_red_point])
if centers_with_red.numel() == 0 or centers_with_blue.numel() == 0:
    raise ValueError("No valid clusters (empty intersection).")
valid_centers = centers_with_red[torch.isin(centers_with_red, centers_with_blue)]
if valid_centers.numel() == 0:
    raise ValueError("No valid clusters (empty intersection).")

mask_valid   = torch.isin(all_centers, valid_centers)
all_centers  = all_centers[mask_valid]
all_levels   = all_levels[mask_valid]
all_points   = all_points[mask_valid]
is_red_point = is_red_point[mask_valid]

center_map           = torch.searchsorted(valid_centers, all_centers)
self.num_active_centers = int(valid_centers.numel())
```

#### 3.4.3 Global Level Bound

```python
max_level_global = int(all_levels.max().item())
self.max_level_global = max_level_global
L = max_level_global + 1   # number of level slots (0..max_level_global)
K = self.num_active_centers
```

All bucket IDs are computed as: `bucket_id = center_dense * L + level`
Total number of buckets = `K * L`.

#### 3.4.4 Structure 1 — Shell CSR for A-points

Groups A-points by `(center_dense, level)` bucket. Each A-point stored exactly once
at its own shell level k_a.

```python
red_mask    = is_red_point
red_centers = center_map[red_mask]
red_points  = all_points[red_mask]        # already [0, N)
red_levels  = all_levels[red_mask].long()

red_bucket_ids = red_centers * L + red_levels   # shape (num_red_entries,)

perm_r = torch.argsort(red_bucket_ids)
self.shell_red_indices = red_points[perm_r].to(device=self.device, dtype=torch.long)
self.shell_red_levels  = red_levels[perm_r].to(device=self.device, dtype=torch.long)
sorted_red_buckets     = red_bucket_ids[perm_r]

r_counts = torch.bincount(sorted_red_buckets, minlength=K * L)
r_counts_long = r_counts.to(device=self.device, dtype=torch.long)
self.shell_red_offsets = torch.cat([
    torch.zeros(1, device=self.device, dtype=torch.long),
    torch.cumsum(r_counts_long, 0)
])
# shell_red_offsets has shape (K*L + 1,)
# shell_red_offsets[bucket_id] .. shell_red_offsets[bucket_id+1] gives A-points
# in the SHELL (center_dense, level).
```

Also build expand array for scatter ops:
```python
self.shell_red_expand_bucket_ids = torch.repeat_interleave(
    torch.arange(K * L, device=self.device, dtype=torch.long), r_counts_long
)
```

#### 3.4.5 Structure 2 — max_level per center

```python
# For each center, the highest shell level that has at least one A-point.
center_has_red = red_centers   # dense center IDs for all A-point entries
level_per_entry = red_levels
max_level_per_center = torch.zeros(K, device=self.device, dtype=torch.long)
max_level_per_center.scatter_reduce_(
    0, center_has_red.to(self.device), level_per_entry.to(self.device),
    reduce="amax", include_self=True
)
self.max_level_per_center = max_level_per_center   # shape (K,)
```

#### 3.4.6 Structure 3 — INV for B-points (inverted index: blue point → buckets)

Same as existing `blue_offsets` / `blue_center_indices` in `bipartite.py`, but stores
`(center_dense, level)` bucket IDs instead of just center IDs.

```python
blue_mask    = ~is_red_point
blue_centers = center_map[blue_mask]
blue_points  = all_points[blue_mask] - N    # rebase to [0, N)
blue_levels  = all_levels[blue_mask].long()

blue_bucket_ids = blue_centers * L + blue_levels   # (center, level) bucket per entry

perm_b = torch.argsort(blue_points)
self.inv_b_bucket_ids = blue_bucket_ids[perm_b].to(device=self.device, dtype=torch.long)
self.inv_b_levels     = blue_levels[perm_b].to(device=self.device, dtype=torch.long)
sorted_b_pts          = blue_points[perm_b]

b_counts = torch.bincount(sorted_b_pts, minlength=N)
b_counts_long = b_counts.to(device=self.device, dtype=torch.long)
self.inv_b_offsets = torch.cat([
    torch.zeros(1, device=self.device, dtype=torch.long),
    torch.cumsum(b_counts_long, 0)
])
# inv_b_offsets[b] .. inv_b_offsets[b+1] gives the list of bucket_ids for blue point b.
# inv_b_levels[i] gives the shell level k_b for the i-th entry — used as proxy 2*k_b.
```

#### 3.4.7 Structure 4 — INV for A-points (inverted index: red point → buckets)

Same structure as INV_B, but for A-points. Used in Step 7 (incremental pre-processing
update). Each A-point stored once at its own shell level k_a.

```python
perm_ra = torch.argsort(red_points)
self.inv_a_bucket_ids = red_bucket_ids[perm_r][  # already perm_r-sorted above
    torch.argsort(torch.argsort(perm_r))          # un-sort to get per-point order
]
```

Actually, build it directly to avoid confusion:
```python
perm_a = torch.argsort(red_points)
self.inv_a_bucket_ids = red_bucket_ids[perm_a].to(device=self.device, dtype=torch.long)
self.inv_a_levels     = red_levels[perm_a].to(device=self.device, dtype=torch.long)
sorted_a_pts          = red_points[perm_a]

a_counts = torch.bincount(sorted_a_pts, minlength=N)
a_counts_long = a_counts.to(device=self.device, dtype=torch.long)
self.inv_a_offsets = torch.cat([
    torch.zeros(1, device=self.device, dtype=torch.long),
    torch.cumsum(a_counts_long, 0)
])
# inv_a_offsets[a] .. inv_a_offsets[a+1] gives the list of shell bucket_ids for A-point a.
# Each entry is the bucket at a's OWN shell level k_a.
```

#### 3.4.8 Structures 5+6 — ball_sizes, d_max, max_list (vectorized GPU init)

All three are built with vectorized GPU ops. No Python loops.

ball_sizes[q*L+k] = number of A-points in shells 0..k of center q.
Computed as prefix sum of shell sizes per center via reshape + cumsum.

d_max[q*L+k] = 0 if ball non-empty, -1 if empty. Initially all y(a)=0
so every non-empty ball has d_max=0 and max_list = entire ball.

max_list_offsets/values/count form a second CSR preallocated to ball_sizes.
Initially populated vectorized: each shell entry (a, center q, level k_a)
writes a into balls (q, k_a), (q, k_a+1), ..., (q, max_level[q]).
Position in max_list for ball (q,k) = max_list_offsets[q*L+k] + local_pos,
where local_pos = position of a within center q's shell CSR entries.

ZERO Python loops in init. Only .item() calls allowed in __init__ are for
determining allocation sizes (torch.empty requires Python int).

---

### 3.5 Method: `solve(self)`

This implements the 7-step push-relabel phase. All dual weights are integers (units of ε).

```python
def solve(self):
    N          = self.N
    K          = self.num_active_centers
    L          = self.max_level_global + 1
    device     = self.device
    B_free     = torch.arange(N, device=device, dtype=torch.long)
    iteration  = 0

    while True:
        num_free = B_free.numel()
        if num_free <= self.epsilon * N:
            break
        if iteration > 50000:
            break
        iteration += 1

        # ── STEP 1: Find zero-slack candidate buckets ─────────────────────
        # For each b in B_free, scan INV_B(b).
        # For each bucket entry (bucket_id, k_b):
        #   target = 2 * k_b - y(b)
        #   if d_max[bucket_id] == target  →  this bucket is a candidate
        #
        # Implementation: expand INV_B for all free B-points, compute target,
        # compare against d_max, collect candidates.

        starts_b  = self.inv_b_offsets[B_free]
        ends_b    = self.inv_b_offsets[B_free + 1]
        lengths_b = ends_b - starts_b
        total_inv = int(lengths_b.sum().item())

        if total_inv == 0:
            self.yB[B_free] += 1
            continue

        cum_b   = torch.cumsum(lengths_b, 0)
        seg_b   = cum_b - lengths_b
        g_range = _ensure_long_arange(self, "_inv_arange", total_inv, device)
        rep_starts_b  = torch.repeat_interleave(starts_b, lengths_b)
        offsets_b     = g_range - torch.repeat_interleave(seg_b, lengths_b)
        inv_edge_idx  = rep_starts_b + offsets_b

        active_b_ids    = torch.repeat_interleave(B_free, lengths_b)   # (E,)
        active_bkt_ids  = self.inv_b_bucket_ids[inv_edge_idx]           # (E,)
        active_kb       = self.inv_b_levels[inv_edge_idx]               # (E,)

        target          = 2 * active_kb - self.yB[active_b_ids].long()  # (E,)
        dmax_vals       = self.d_max[active_bkt_ids].long()              # (E,)
        is_candidate    = (dmax_vals == target) & (dmax_vals >= 0)      # (E,) bool
        # dmax >= 0 filters out empty balls (d_max == -1 means empty)

        if not is_candidate.any():
            self.yB[B_free] += 1
            continue

        cand_b   = active_b_ids[is_candidate]    # free B-point for each candidate
        cand_bkt = active_bkt_ids[is_candidate]  # ball bucket_id
        cand_kb  = active_kb[is_candidate]        # k_b (proxy = 2*k_b)

        Step 2 uses the Gumbel-max trick for vectorized weighted sampling.
        No Python loop over b. One scatter_reduce amax over all candidates.

        # ── STEP 3: Proposal — each b draws one a from max_list ──────────
        # For each b_with_cand[i], pick uniformly from
        # max_list_values[max_list_offsets[chosen_bkt[i]] : ... + max_list_count[chosen_bkt[i]]]

        ml_starts  = self.max_list_offsets[chosen_bkt]           # (num_b_cand,)
        ml_lens    = self.max_list_count[chosen_bkt]             # (num_b_cand,)
        rand_idx   = (torch.rand(num_b_cand, device=device) * ml_lens.float()).long()
        rand_idx   = rand_idx.clamp(max=ml_lens - 1)
        proposal_a = self.max_list_values[ml_starts + rand_idx]  # (num_b_cand,) proposed A-points

        # ── STEP 4: Conflict resolution — each a accepts one proposal ─────
        # Each a that received ≥1 proposal accepts exactly one uniformly at random.
        # Use scatter with random priority: assign random keys, keep argmin per a.

        num_props = num_b_cand
        rand_prio = torch.rand(num_props, device=device)

        # For each a that was proposed to, find the b with minimum random priority
        # (effectively uniform random selection among proposals).
        proposal_b = b_with_cand   # b_with_cand[i] proposed to proposal_a[i]

        # scatter_reduce to find min priority per a
        min_prio_per_a = torch.full((N,), float('inf'), device=device)
        min_prio_per_a.scatter_reduce_(
            0, proposal_a, rand_prio, reduce="amin", include_self=True
        )

        # An (a, b) pair is accepted if its priority equals the minimum for that a
        accepted_mask = rand_prio == min_prio_per_a[proposal_a]   # (num_props,)

        r_new = proposal_a[accepted_mask]    # newly matched A-points
        b_new = proposal_b[accepted_mask]    # newly matched B-points
        # Each a in r_new appears exactly once (by construction of argmin).

        # ── STEP 5: Matching update + F_B update ─────────────────────────
        if r_new.numel() > 0:
            # Find evictions: A-points in r_new that were previously matched
            was_matched = self.MA[r_new] != -1
            evicted_b   = self.MA[r_new[was_matched]].to(torch.long).clone()

            # Clear evicted B-points' matching
            if evicted_b.numel() > 0:
                self.MB[evicted_b] = -1

            # Write new matches
            self.MA[r_new] = b_new.to(self.MA.dtype)
            self.MB[b_new] = r_new.to(self.MB.dtype)

            # Update F_B: remove matched b_new, add evicted_b
            keep_mask = _ensure_bool_buffer(self, "_keep_free_mask", num_free, device)
            keep_mask.fill_(True)
            keep_mask[torch.searchsorted(B_free, b_new)] = False
            still_free = B_free[keep_mask]
            if evicted_b.numel() > 0:
                F_B_new, _ = torch.sort(torch.cat([still_free, evicted_b]))
            else:
                F_B_new = still_free
        else:
            F_B_new = B_free

        # ── STEP 6: Dual update ───────────────────────────────────────────
        # y(b) += 1 for all b still in F_B (after Step 5 update)
        # y(a) += 1 for all a in new_matches (r_new)
        self.yB[F_B_new] += 1
        if r_new.numel() > 0:
            self.yA[r_new] += 1   # INCREMENT, not decrement

        Step 7 is fully vectorized on GPU:
        1. Expand INV_A for r_new → all shell entries for changed A-points.
        2. Expand each shell entry to all balls k_a..max_level[q] → affected ball IDs.
        3. Deduplicate affected ball IDs with .unique().
        4. For each affected ball, expand its shell CSR range to get all A-points in the ball.
        5. scatter_reduce amax over y(a) → new d_max per affected ball.
        6. Filter for A-points with y(a)==new_d_max → new max_list members.
        7. Compute write positions via searchsorted + rank, scatter write.

        ZERO Python loops in solve(). Only .item() calls allowed are three per phase
        for _ensure_long_arange buffer sizing (inv expansion, ball expansion, ball entries).

        B_free = F_B_new

    self.cleanup_remaining_points()
```

---

### 3.6 Method: `cleanup_remaining_points(self)`

```python
def cleanup_remaining_points(self):
    free_b = torch.nonzero(self.MB == -1).squeeze(1)
    free_r = torch.nonzero(self.MA == -1).squeeze(1)
    count  = min(free_b.numel(), free_r.numel())
    if count > 0:
        self.MB[free_b[:count]] = free_r[:count].to(self.MB.dtype)
        self.MA[free_r[:count]] = free_b[:count].to(self.MA.dtype)
```

---

## 4. File 3 — Experiment: `e2_color_aware_vs_exact.py`

### 4.1 Purpose

Compares `ColorAwareTwoLevelSolver` against POT-Exact (network simplex) on MNIST,
logging results to CSV. Mirrors the structure of `e1_mnist_vs_exact.py` exactly.

### 4.2 What to copy verbatim from `e1_mnist_vs_exact.py`

- `resolve_mnist_paths()`
- `load_mnist_flat()`
- `compute_cost_matrix_L1()`
- `average_l1_matching_cost()`
- `reset_peak_memory_stats_if_cuda()`
- `peak_memory_allocated_mb()`
- `synchronize_if_cuda()`
- The POT-Exact block (unchanged)
- All CSV writing patterns and error handling (try/except/finally, `del solver`)
- The `BASE_DIR` path setup at the top

### 4.3 Changes from `e1_mnist_vs_exact.py`

**Import — replace solver imports with:**
```python
from clustered_push_relabel.solvers.color_aware_bipartite import ColorAwareTwoLevelSolver
```

**Solver block — replace the 2-Level and k-Level blocks with one block:**
```python
try:
    reset_peak_memory_stats_if_cuda(device)
    t_start = time.time()
    solver = ColorAwareTwoLevelSolver(P_red, P_blue, args.epsilon, metric="L1")
    synchronize_if_cuda(device)
    t_mid = time.time()
    solver.solve()
    synchronize_if_cuda(device)
    t_end = time.time()
    total_time = t_end - t_start
    clust_time = t_mid - t_start   # __init__ time (includes clustering)
    solve_time = t_end - t_mid     # solve() time only
    cost_val   = average_l1_matching_cost(P_red, P_blue, solver.MB)
    peak_mem   = peak_memory_allocated_mb(device)
    status     = "success"
except Exception as e:
    print(f"[ColorAware-2L] Exception: {e}")
    import traceback; traceback.print_exc()
    status     = "fail"
    total_time = clust_time = solve_time = float('nan')
    cost_val   = float('nan')
    peak_mem   = float('nan')
finally:
    try: del solver
    except: pass
```

**No normalisation in the experiment.** Pass `P_red` and `P_blue` raw (MNIST pixels
already in `[0, 1]`). The solver's `__init__` handles diameter normalisation internally.
Cost is measured in the original unnormalised space via `average_l1_matching_cost`.

**CSV algo label:** `"ColorAware-2L"`

**CSV k column:** write `2` (fixed).

**Default output file:** `results_e2_color_aware.csv`

**Remove:** `--k` argument, `--with_sinkhorn` argument, all Sinkhorn code.

**Default n values:** `[500, 1000, 2000]`

**Default epsilon:** `0.05`

**CSV headers:** same 15 columns as e1:
`dataset, n, dim, epsilon, k, trial, algo, status, total_time_s, cluster_time_s,
solver_time_s, peak_gpu_mem_mb, cost, abs_error, rel_error`

---

## 5. Critical Correctness Notes

These must be followed precisely. They encode decisions made by the algorithm designer
and professor and cannot be inferred from the existing codebase.

1. **Balls not shells in CSR lookup.** `d_max[bucket_id]` covers the entire ball
   (shells 0..k), not just shell k. `max_list` likewise covers the entire ball.

2. **Proxy is 2·k_b, not 2·max(k_a, k_b).** The proxy is always twice the B-point's
   level in the cluster. It never depends on the specific A-point's level.

3. **d_max = -1 means empty ball.** Always check `dmax >= 0` before treating a
   bucket as a candidate.

4. **y(a) increments, never decrements.** Newly matched A-points get `y(a) += 1`.
   This is different from some push-relabel variants. Do not negate or subtract.

5. **F_B update before dual update.** Evicted B-points must be in F_B before
   `y(b) += 1` is applied (Step 6 uses the post-Step-5 F_B).

6. **Evicted B-points only re-propose next phase.** They are added to F_B for the
   next iteration, not for any re-proposal in the current phase.

7. **max_list preallocated to ball size.** Never reallocate. Use `max_list_count` to
   track the live count. The preallocated capacity is `ball_sizes[bid]`.

8. **Step 7 Case 1 overwrites max_list entirely.** When ya_old == d_max, the new
   max_list = {a} only. Write a at offset 0 of the ball's max_list slice and set count=1.

9. **Step 7 Case 2 appends.** When ya_new == d_max (a rises to join), append a
   at position `max_list_count[bid]` and increment the count.

10. **Shell CSR stores A-points only.** Blue points never appear in `shell_red_indices`.

11. **Diameter normalisation uses the combined point set.** Δ is computed from
    `cat([P_red, P_blue])`, not from P_red or P_blue separately.
