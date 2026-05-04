#!/usr/bin/env python3
"""
Validate 2-Level and 3-Level feasibility/admissibility invariants.

This is a correctness check on small synthetic 2D point clouds. The audits
allocate dense N x N matrices, so N is intentionally capped at 10,000.
"""

import gc
import math
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
SRC_DIR = REPO_ROOT / "src"
if not SRC_DIR.exists():
    REPO_ROOT = SCRIPT_DIR.parent.parent
    SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from clustered_push_relabel.solvers.simple_bipartite import SimpleGPUSolver
from clustered_push_relabel.solvers.three_level_bipartite import ThreeLevelGPUSolver

EPSILON = 0.01
BATCH_SIZE = 2048
MAX_ITERS = 999_999_999
SEED = 42
DIAMETER_TILE = 1024
SIZES = [500, 1000, 2000, 5000, 10000]


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _diameter_with_tile(points, tile_size):
    max_dist = 0.0
    for start in range(0, points.shape[0], tile_size):
        end = min(start + tile_size, points.shape[0])
        dists = torch.cdist(
            points[start:end],
            points,
            p=2,
            compute_mode="use_mm_for_euclid_dist_if_necessary",
        )
        max_dist = max(max_dist, float(dists.max().item()))
        del dists
    return max_dist


def joint_diameter(A, B):
    points = torch.cat([A, B], dim=0)
    tile = min(DIAMETER_TILE, points.shape[0])
    while tile >= 64:
        try:
            return _diameter_with_tile(points, tile)
        except torch.cuda.OutOfMemoryError:
            cleanup()
            tile //= 2
    return _diameter_with_tile(points, 32)


def make_data(n):
    A = torch.rand(n, 2, device="cuda", dtype=torch.float32)
    B = torch.rand(n, 2, device="cuda", dtype=torch.float32)
    diameter = joint_diameter(A, B)
    if diameter > 0.0:
        A = A / diameter
        B = B / diameter
    return A, B, diameter


def avg_matching_cost(A, B, match_B, diameter):
    match_B = match_B.to(device=A.device, dtype=torch.long)
    return torch.norm(B - A[match_B], p=2, dim=1).mean().item() * diameter


def audit_two_level(solver):
    """
    Full N x N feasibility and admissibility check for SimpleGPUSolver.

    Feasibility:  y_B[b] + y_A[a] <= proxy_cost(b, a) + 1   for ALL (b, a)
    Admissibility: y_B[b] + y_A[a] == proxy_cost(b, a)       for MATCHED (b, a)

    Proxy cost:
      if a in adj(b):  adj_dist_int[entry]
      else:            d_min_b_int[b] + DR_int[nearest_s[b]][a]
                       = d_min_b_int[b] + (y_A[a] - V[nearest_s[b]][a])
                         where V[s][a] = y_A[a] - DR_int[s][a]
    """
    device = solver.device
    N = solver.N

    y_A = solver.y_A.to(torch.long)
    y_B = solver.y_B.to(torch.long)

    # Build proxy cost matrix (N x N) in int64.
    # Start from the A2/Set-1 triangle proxy for all pairs.
    V_rows = solver.V[solver.nearest_s].to(torch.long)
    proxy = (
        solver.d_min_b_int.to(torch.long).unsqueeze(1)
        + y_A.unsqueeze(0)
        - V_rows
    )

    # Override with direct adj cost where a is in adj(b) (strictly lower proxy).
    all_b = torch.arange(N, device=device, dtype=torch.long)
    lengths = solver.adj_ptr[1:] - solver.adj_ptr[:-1]
    total_edges = int(lengths.sum().item())
    if total_edges > 0:
        edge_arange = torch.arange(total_edges, device=device, dtype=torch.long)
        row_pos = torch.repeat_interleave(all_b, lengths)
        cum_len = lengths.cumsum(0)
        packed_st = cum_len - lengths
        active_edge_idx = (
            torch.repeat_interleave(solver.adj_ptr[:-1], lengths)
            + edge_arange
            - torch.repeat_interleave(packed_st, lengths)
        )
        active_b = row_pos
        active_a = solver.adj_col[active_edge_idx]
        proxy[active_b, active_a] = solver.adj_dist_int[active_edge_idx].to(torch.long)

    # Feasibility check.
    lhs = y_B.unsqueeze(1) + y_A.unsqueeze(0)
    excess = lhs - proxy - 1
    n_feas_violations = int((excess > 0).sum().item())
    worst_excess = int(excess.max().item())

    # Admissibility check for phase-matched pairs (exclude cleanup blues).
    is_cleanup = torch.zeros(N, dtype=torch.bool, device=device)
    if solver.cleanup_blues.numel() > 0:
        is_cleanup[solver.cleanup_blues] = True
    matched_b = (solver.match_B != -1).nonzero(as_tuple=True)[0]
    phase_matched_b = matched_b[~is_cleanup[matched_b]]
    n_adm_violations = 0
    worst_adm_diff = 0
    if phase_matched_b.numel() > 0:
        matched_a = solver.match_B[phase_matched_b]
        diff = lhs[phase_matched_b, matched_a] - proxy[phase_matched_b, matched_a]
        n_adm_violations = int((diff != 0).sum().item())
        worst_adm_diff = int(diff.abs().max().item()) if diff.numel() > 0 else 0

    print(
        f"  [2-Level Audit] Feasibility violations: {n_feas_violations}  "
        f"(worst excess={worst_excess})",
        flush=True,
    )
    print(
        f"  [2-Level Audit] Admissibility violations: {n_adm_violations}  "
        f"(worst |diff|={worst_adm_diff})",
        flush=True,
    )

    return {
        "feasibility_violations": n_feas_violations,
        "feasibility_worst_excess": worst_excess,
        "admissibility_violations": n_adm_violations,
        "admissibility_worst_diff": worst_adm_diff,
    }


def empty_row(n):
    return {
        "N": n,
        "2l_feas_ok": False,
        "2l_adm_ok": False,
        "3l_feas_ok": False,
        "3l_adm_ok": False,
        "2l_cost": math.nan,
        "3l_cost": math.nan,
    }


def run_two_level(A, B, diameter, row):
    solver2 = None
    try:
        solver2 = SimpleGPUSolver(
            A,
            B,
            epsilon=EPSILON,
            batch_size=BATCH_SIZE,
            verbose=False,
            diameter=1.0,
            max_iters=MAX_ITERS,
        )
        match_B2 = solver2.solve()
        result2 = audit_two_level(solver2)
        cost2 = avg_matching_cost(A, B, match_B2, diameter)
        row["2l_feas_ok"] = result2["feasibility_violations"] == 0
        row["2l_adm_ok"] = result2["admissibility_violations"] == 0
        row["2l_cost"] = cost2
        print(
            f"  [2-Level] Avg cost: {cost2:.5f}  "
            f"Feasibility OK: {row['2l_feas_ok']}  "
            f"Admissibility OK: {row['2l_adm_ok']}",
            flush=True,
        )
    except torch.cuda.OutOfMemoryError:
        print("  [2-Level] OOM", flush=True)
    except Exception as exc:
        print(f"  [2-Level] ERROR: {exc}", flush=True)
    finally:
        if solver2 is not None:
            del solver2
        cleanup()


def run_three_level(A, B, diameter, row):
    solver3 = None
    try:
        solver3 = ThreeLevelGPUSolver(
            A,
            B,
            epsilon=EPSILON,
            batch_size=BATCH_SIZE,
            verbose=False,
            diameter=1.0,
            max_iters=MAX_ITERS,
        )
        solver3.debug_audit = True
        solver3.debug_stop_on_first_violation = True
        match_B3 = solver3.solve()
        verify3 = solver3.verify_solution()
        cost3 = avg_matching_cost(A, B, match_B3, diameter)
        row["3l_feas_ok"] = verify3["feasibility_violations"] == 0
        row["3l_adm_ok"] = verify3["admissibility_violations"] == 0
        row["3l_cost"] = cost3
        print(
            f"  [3-Level] Avg cost: {cost3:.5f}  "
            f"Feasibility OK: {row['3l_feas_ok']}  "
            f"Admissibility OK: {row['3l_adm_ok']}",
            flush=True,
        )
    except torch.cuda.OutOfMemoryError:
        print("  [3-Level] OOM", flush=True)
    except RuntimeError as exc:
        print(f"  [3-Level] VIOLATION DETECTED: {exc}", flush=True)
    except Exception as exc:
        print(f"  [3-Level] ERROR: {exc}", flush=True)
    finally:
        if solver3 is not None:
            del solver3
        cleanup()


def format_bool(value):
    return "YES" if value else "NO"


def format_cost(value):
    if math.isnan(value):
        return "   --  "
    return f"{value:0.5f}"


def print_summary(rows):
    print("\nN     | 2L Feas OK | 2L Adm OK | 3L Feas OK | 3L Adm OK | 2L Cost | 3L Cost")
    print("------+------------+-----------+------------+-----------+---------+--------")
    for row in rows:
        print(
            f"{row['N']:<5} |"
            f" {format_bool(row['2l_feas_ok']):^10} |"
            f" {format_bool(row['2l_adm_ok']):^9} |"
            f" {format_bool(row['3l_feas_ok']):^10} |"
            f" {format_bool(row['3l_adm_ok']):^9} |"
            f" {format_cost(row['2l_cost']):>7} |"
            f" {format_cost(row['3l_cost']):>7}",
            flush=True,
        )

    all_passed = all(
        row["2l_feas_ok"]
        and row["2l_adm_ok"]
        and row["3l_feas_ok"]
        and row["3l_adm_ok"]
        for row in rows
    )
    if all_passed:
        print("\nALL FEASIBILITY CHECKS PASSED", flush=True)
    else:
        print("\nVIOLATIONS DETECTED — see details above.", flush=True)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this validation script")

    torch.manual_seed(SEED)
    rows = []

    for n in SIZES:
        print(f"\n=== N = {n} ===", flush=True)
        A = None
        B = None
        row = empty_row(n)
        try:
            A, B, diameter = make_data(n)
            run_two_level(A, B, diameter, row)
            run_three_level(A, B, diameter, row)
        except torch.cuda.OutOfMemoryError:
            print(f"  [Data] OOM while preparing N={n}", flush=True)
        except Exception as exc:
            print(f"  [Data] ERROR while preparing N={n}: {exc}", flush=True)
        finally:
            if A is not None:
                del A
            if B is not None:
                del B
            cleanup()
            rows.append(row)

    print_summary(rows)


if __name__ == "__main__":
    main()
