#!/usr/bin/env python3
"""
Shared utilities for final2 experiments.

Provides:
  - Formatting helpers
  - Three-level precomputed clustering (from D_rr / D_br matrices)
  - Three-level proxy cost-matrix builder
  - Terminal table printing
  - Markdown result writing
"""

import datetime
import math
from pathlib import Path

import numpy as np
import torch


# ── Formatting ────────────────────────────────────────────────────────────────

def is_nan(v):
    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return True


def fmt_time(v):
    return "N/A" if is_nan(v) else f"{float(v):.1f} ms"


def fmt_cost(v):
    return "N/A" if is_nan(v) else f"{float(v):.4f}"


def fmt_ratio(v):
    return "N/A" if is_nan(v) else f"{float(v):.4f}"


def fmt_iters(v):
    return "N/A" if is_nan(v) else f"{int(v):,}"


def fmt_bytes(v):
    if is_nan(v):
        return "N/A"
    v = float(v)
    if v >= 1024 ** 3:
        return f"{v / 1024**3:.2f} GiB"
    if v >= 1024 ** 2:
        return f"{v / 1024**2:.1f} MiB"
    if v >= 1024:
        return f"{v / 1024:.1f} KiB"
    return f"{v:.0f} B"


def compute_ratio(exact, approx):
    if is_nan(exact) or is_nan(approx) or float(exact) == 0.0:
        return math.nan
    return float(approx) / float(exact)


def agg_mean(values):
    vals = [float(v) for v in values if not math.isnan(float(v))]
    return math.nan if not vals else float(sum(vals) / len(vals))


def agg_median(values):
    vals = [float(v) for v in values if not math.isnan(float(v))]
    return math.nan if not vals else float(np.median(vals))


def agg_std(values):
    vals = [float(v) for v in values if not math.isnan(float(v))]
    return math.nan if len(vals) < 2 else float(np.std(vals))


def agg_p90(values):
    vals = [float(v) for v in values if not math.isnan(float(v))]
    return math.nan if not vals else float(np.percentile(vals, 90))


def agg_max(values):
    vals = [float(v) for v in values if not math.isnan(float(v))]
    return math.nan if not vals else float(max(vals))


def fmt_stat(v):
    return "N/A" if is_nan(v) else f"{float(v):.4f}"


def compute_gamma(match, red, blue, adj_ptr, adj_col):
    if adj_col.numel() == 0:
        return math.nan

    pair_costs = (red[match] - blue).abs().sum(dim=1)
    is_local = torch.zeros(match.shape[0], dtype=torch.bool)
    for b in range(match.shape[0]):
        start = adj_ptr[b].item()
        end = adj_ptr[b + 1].item()
        is_local[b] = (adj_col[start:end] == match[b]).any()

    total = pair_costs.sum().item()
    if total <= 0.0:
        return 0.0
    return pair_costs[~is_local].sum().item() / total


# ── Three-level precomputed clustering ───────────────────────────────────────
# Builds a dict compatible with ThreeLevelGPUSolver(precomputed_clustering=...)
# from precomputed (N x N) distance matrices D_rr and D_br.


def _group_offsets(b_idx: torch.Tensor) -> torch.Tensor:
    """0-based intra-group offset for each element in a sorted integer tensor."""
    m = b_idx.numel()
    if m <= 1:
        return torch.zeros(m, dtype=torch.long, device=b_idx.device)
    same = torch.zeros(m, dtype=torch.long, device=b_idx.device)
    same[1:] = (b_idx[1:] == b_idx[:-1]).long()
    cumsum = same.cumsum(0)
    start_vals = cumsum.masked_fill(same.bool(), 0)
    baseline = torch.cummax(start_vals, dim=0).values
    return cumsum - baseline


def _build_csr_adjacency(dist_mat, threshold_per_row, N_rows, N_cols,
                          epsilon, tile_size, device, distance_scale=1.0):
    """
    Build CSR adjacency where entry (i, j) is included iff
    dist_mat[i, j] < threshold_per_row[i].

    dist_mat   : (N_rows, N_cols) float32 on device
    threshold  : (N_rows,) float32 on device, in scaled distance units

    Returns (ptr, col, dist_int, dist_float).
    """
    thr = threshold_per_row.unsqueeze(1)  # (N_rows, 1)
    counts = torch.zeros(N_rows, dtype=torch.long, device=device)
    for start in range(0, N_cols, tile_size):
        end = min(start + tile_size, N_cols)
        counts.add_((dist_mat[:, start:end] / distance_scale < thr).sum(dim=1))

    ptr = torch.zeros(N_rows + 1, dtype=torch.long, device=device)
    ptr[1:] = counts.cumsum(0)
    M = int(ptr[-1].item())
    col = torch.empty(M, dtype=torch.long, device=device)
    dist_int = torch.empty(M, dtype=torch.int32, device=device)
    dist_float = torch.empty(M, dtype=torch.float32, device=device)
    del counts

    if M > 0:
        cursor = ptr[:-1].clone()
        for start in range(0, N_cols, tile_size):
            end = min(start + tile_size, N_cols)
            tile = dist_mat[:, start:end]
            scaled_tile = tile / distance_scale
            mask = scaled_tile < thr
            ri, ti = mask.nonzero(as_tuple=True)
            if ri.numel() == 0:
                continue
            wp = cursor[ri] + _group_offsets(ri)
            col[wp] = (ti + start).long()
            d = scaled_tile[ri, ti]
            dist_float[wp] = d
            dist_int[wp] = (d / epsilon).ceil_().to(torch.int32)
            cursor.scatter_add_(0, ri, torch.ones_like(ri))
        del cursor

    return ptr, col, dist_int, dist_float


def run_three_level_precomputed(D_rr, D_br, epsilon, tile_size=512):
    """
    Build a ThreeLevelGPUSolver-compatible clustering dict from precomputed
    pairwise distance matrices.

    Parameters
    ----------
    D_rr : (N, N) float32 CUDA tensor — red-to-red distances
    D_br : (N, N) float32 CUDA tensor — blue-to-red distances
    epsilon : float
    tile_size : int

    Returns
    -------
    dict with the same field names as ThreeLevelClustering.run() output.
    """
    device = D_rr.device
    N = D_rr.shape[0]
    eps = epsilon

    # 1. Sample A1 ⊆ A at rate N^(-1/3)  →  E[|A1|] ≈ N^(2/3)
    rate1 = 1.0 / (float(N) ** (1.0 / 3.0))
    mask_A1 = torch.rand(N, device=device) < rate1
    if not mask_A1.any():
        mask_A1[torch.randint(N, (1,), device=device)] = True
    sampled_idx_A1 = mask_A1.nonzero(as_tuple=True)[0]
    S1 = sampled_idx_A1.shape[0]

    # 2. Sample A2 ⊆ A1 at rate N^(-1/3)  →  E[|A2|] ≈ N^(1/3)
    rate2 = 1.0 / (float(N) ** (1.0 / 3.0))
    mask_A2 = torch.rand(S1, device=device) < rate2
    if not mask_A2.any():
        mask_A2[torch.randint(S1, (1,), device=device)] = True
    local_idx_A2 = mask_A2.nonzero(as_tuple=True)[0]
    sampled_idx_A2 = sampled_idx_A1[local_idx_A2]
    S2 = sampled_idx_A2.shape[0]

    # 3. DR: (S2, N) distances from A2 centers to all reds
    DR = D_rr[sampled_idx_A2, :]

    # 4. Nearest A2 center for each blue
    DB_to_A2 = D_br[:, sampled_idx_A2]          # (N, S2)
    d_min_b_A2, nearest_s2 = DB_to_A2.min(dim=1)
    del DB_to_A2

    # 5. Nearest A2 for each A1 center
    D_A1_to_A2 = D_rr[sampled_idx_A1][:, sampled_idx_A2]  # (S1, S2)
    d_min_A1_A2, nearest_s2_A1 = D_A1_to_A2.min(dim=1)
    del D_A1_to_A2

    # 6. Nearest A1 center for each blue
    DB_to_A1 = D_br[:, sampled_idx_A1]          # (N, S1)
    d_min_b_A1, nearest_s1 = DB_to_A1.min(dim=1)
    del DB_to_A1

    D_A1_to_reds = D_rr[sampled_idx_A1, :]      # (S1, N)

    # The push-relabel solver (Lahn et al. 2022) requires max cost = 1.
    # The 3-level Thorup-Zwick proxy has worst-case distortion 2k-1 = 5x.
    # We normalize all proxy distances by proxy_max so the solver sees
    # costs in [0,1], restoring the O(1/eps^2) phase bound.
    # The caller multiplies matching costs by proxy_max to recover
    # costs in the original normalized scale.
    proxy_max = max(
        float(d_min_b_A2.max()) + float(DR.max()),
        float(d_min_b_A1.max()) + float(D_A1_to_reds.max()),
        1.0,
        1e-6,
    )

    DR = DR / proxy_max
    d_min_b_A2 = d_min_b_A2 / proxy_max
    d_min_b_A1 = d_min_b_A1 / proxy_max
    d_min_A1_A2 = d_min_A1_A2 / proxy_max

    DR_int = (DR / eps).ceil_().to(torch.int32)
    d_min_b_A2_int = (d_min_b_A2 / eps).ceil_().to(torch.int32)
    d_min_A1_A2_int = (d_min_A1_A2 / eps).ceil_().to(torch.int32)
    d_min_b_A1_int = (d_min_b_A1 / eps).ceil_().to(torch.int32)

    # 7. Adj_B: {a : d(b, a) < d_min_b_A1[b]}
    adj_B_ptr, adj_B_col, adj_B_dist_int, adj_B_dist_float = _build_csr_adjacency(
        D_br, d_min_b_A1, N, N, eps, tile_size, device, distance_scale=proxy_max
    )

    # 8. Adj_A1: {a : d(a1, a) < d_min_A1_A2[a1]}
    adj_A1_ptr, adj_A1_col, adj_A1_dist_int, adj_A1_dist_float = _build_csr_adjacency(
        D_A1_to_reds, d_min_A1_A2, S1, N, eps, tile_size, device,
        distance_scale=proxy_max
    )
    del D_A1_to_reds

    return {
        "proxy_max": proxy_max,
        "sampled_idx_A1": sampled_idx_A1,
        "sampled_idx_A2": sampled_idx_A2,
        "A1_sampled": None,
        "A2_sampled": None,
        "DR": DR,
        "DR_int": DR_int,
        "d_min_b_A1": d_min_b_A1,
        "d_min_b_A1_int": d_min_b_A1_int,
        "nearest_s1": nearest_s1,
        "d_min_b_A2": d_min_b_A2,
        "d_min_b_A2_int": d_min_b_A2_int,
        "nearest_s2": nearest_s2,
        "d_min_A1_A2": d_min_A1_A2,
        "d_min_A1_A2_int": d_min_A1_A2_int,
        "nearest_s2_A1": nearest_s2_A1,
        "adj_B_ptr": adj_B_ptr,
        "adj_B_col": adj_B_col,
        "adj_B_dist_int": adj_B_dist_int,
        "adj_B_dist_float": adj_B_dist_float,
        "adj_A1_ptr": adj_A1_ptr,
        "adj_A1_col": adj_A1_col,
        "adj_A1_dist_int": adj_A1_dist_int,
        "adj_A1_dist_float": adj_A1_dist_float,
    }


def build_three_level_proxy_matrix(clustering, N, device):
    """
    Build the full N×N proxy cost matrix from a three-level clustering dict.

    Proxy priority (each level overwrites the coarser estimate):
      Level 2 (base):  C[b,a] = d_min_b_A2[b] + DR[nearest_s2[b], a]
      Level 1 (refine):C[b,a] = d_min_b_A1[b] + d(s1_b, a)  for a ∈ Adj_A1(s1_b)
      Level 0 (direct):C[b,a] = d(b, a)                      for a ∈ Adj_B(b)

    Returns (N, N) float64 numpy array for ot.emd().
    """
    DR = clustering["DR"]
    d_min_b_A2 = clustering["d_min_b_A2"]
    nearest_s2 = clustering["nearest_s2"]
    d_min_b_A1 = clustering["d_min_b_A1"]
    nearest_s1 = clustering["nearest_s1"]
    adj_B_ptr = clustering["adj_B_ptr"]
    adj_B_col = clustering["adj_B_col"]
    adj_B_dist_float = clustering["adj_B_dist_float"]
    adj_A1_ptr = clustering["adj_A1_ptr"]
    adj_A1_col = clustering["adj_A1_col"]
    adj_A1_dist_float = clustering["adj_A1_dist_float"]
    S1 = int(adj_A1_ptr.shape[0]) - 1

    # Level 2 base
    C = d_min_b_A2.unsqueeze(1) + DR[nearest_s2, :]   # (N, N) float32

    # Level 1 — group blues by nearest A1 center, scatter-overwrite
    sorted_order = torch.argsort(nearest_s1)
    group_counts = torch.bincount(nearest_s1, minlength=S1)
    group_ptr = torch.zeros(S1 + 1, dtype=torch.long, device=device)
    group_ptr[1:] = group_counts.cumsum(0)

    for a1_i in range(S1):
        a1_s = int(adj_A1_ptr[a1_i].item())
        a1_e = int(adj_A1_ptr[a1_i + 1].item())
        if a1_s == a1_e:
            continue
        a_cols = adj_A1_col[a1_s:a1_e]
        a_dists = adj_A1_dist_float[a1_s:a1_e]
        g_s = int(group_ptr[a1_i].item())
        g_e = int(group_ptr[a1_i + 1].item())
        if g_s == g_e:
            continue
        blues = sorted_order[g_s:g_e]
        C[blues.unsqueeze(1), a_cols.unsqueeze(0)] = (
            d_min_b_A1[blues].unsqueeze(1) + a_dists.unsqueeze(0)
        )

    # Level 0 — direct
    if adj_B_col.numel() > 0:
        b_idx = torch.repeat_interleave(
            torch.arange(N, device=device, dtype=torch.long),
            adj_B_ptr[1:] - adj_B_ptr[:-1],
        )
        C[b_idx, adj_B_col] = adj_B_dist_float

    return C.cpu().to(torch.float64).numpy()


# ── Two-level proxy cost-matrix builder (from SimplePrecomputedClustering) ───

def build_two_level_proxy_matrix(clustering, N, device):
    """
    Build N×N proxy cost matrix from SimplePrecomputedClustering output.
    For each (b, a): if a ∈ adj(b) use exact dist; else d_min_b[b] + DR[nearest_s[b], a].
    Returns float64 numpy array.
    """
    DR = clustering["DR"]
    d_min_b = clustering["d_min_b"]
    nearest_s = clustering["nearest_s"]
    adj_ptr = clustering["adj_ptr"]
    adj_col = clustering["adj_col"]
    adj_dist_float = clustering["adj_dist_float"]

    C = d_min_b.unsqueeze(1) + DR[nearest_s, :]
    if adj_col.numel() > 0:
        b_idx = torch.repeat_interleave(
            torch.arange(N, device=device, dtype=torch.long),
            adj_ptr[1:] - adj_ptr[:-1],
        )
        C[b_idx, adj_col] = adj_dist_float
    return C.cpu().to(torch.float64).numpy()


# ── Terminal table printer ────────────────────────────────────────────────────

def print_results_table(title, col_specs, rows, fmt_fns):
    """
    Print a bordered results table to stdout.

    col_specs : list of (header_str, width_int)
    rows      : list of dicts
    fmt_fns   : dict mapping column header → callable(row) → str
    """
    widths = [max(len(h), w) for h, w in col_specs]
    headers = [h for h, _ in col_specs]

    top = "┌" + "┬".join("─" * (w + 2) for w in widths) + "┐"
    hdr = "│" + "│".join(f" {h:^{w}} " for h, w in zip(headers, widths)) + "│"
    mid = "├" + "┼".join("─" * (w + 2) for w in widths) + "┤"
    bot = "└" + "┴".join("─" * (w + 2) for w in widths) + "┘"

    if title:
        print(f"\n{title}")
    print(top)
    print(hdr)
    print(mid)
    for row in rows:
        cells = [fmt_fns[h](row) for h in headers]
        line = "│" + "│".join(f" {c:>{w}} " for c, w in zip(cells, widths)) + "│"
        print(line)
    print(bot)


# ── Markdown writer ───────────────────────────────────────────────────────────

def get_results_path(results_dir: Path) -> Path:
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return results_dir / f"results_{ts}.md"


def _md_table(col_specs, rows, fmt_fns):
    headers = [h for h, _ in col_specs]
    header_row = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---:" for _ in headers) + " |"
    lines = [header_row, separator]
    for row in rows:
        cells = [fmt_fns[h](row) for h in headers]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_markdown(filepath: Path, sections: list):
    """
    Write a markdown results file.

    sections : list of dicts with keys:
        title     : str
        subtitle  : str (optional)
        col_specs : list of (header, width)
        rows      : list of row dicts
        fmt_fns   : dict header → callable(row) → str
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"# Experiment Results",
        f"",
        f"Generated: {ts}",
        f"",
    ]
    for sec in sections:
        lines.append(f"## {sec['title']}")
        if sec.get("subtitle"):
            lines.append(f"")
            lines.append(f"_{sec['subtitle']}_")
        lines.append(f"")
        if sec.get("rows"):
            lines.append(_md_table(sec["col_specs"], sec["rows"], sec["fmt_fns"]))
        else:
            lines.append("_No results collected._")
        lines.append(f"")
    filepath.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nResults written to: {filepath}", flush=True)
