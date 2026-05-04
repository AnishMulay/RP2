#!/usr/bin/env python3
"""
Standalone comparison sweep on synthetic 2D point clouds.

Table 1 compares exact EMD, POT Sinkhorn, and push-relabel on small point
clouds. Table 2 compares scalable methods on larger point clouds.
"""

import csv
import gc
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# GeomLoss/PyKeOps excluded: PyKeOps requires runtime CUDA kernel compilation
# incompatible with CUDA 13.x (system CUDA 13.2, PyKeOps 2.3).
# POT bregman log-domain Sinkhorn is the Sinkhorn baseline.

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from clustered_push_relabel.solvers.simple_bipartite import SimpleGPUSolver
from clustered_push_relabel.solvers.three_level_bipartite import ThreeLevelGPUSolver

EPSILON = 0.01
BATCH_SIZE = 2048
MAX_ITERS = 999_999_999

DIAMETER_TILE = 1024
ACCURACY_SIZES = [1_000, 5_000, 10_000, 15_000]
SEED = 42
SINKHORN_REGS = [0.1, 0.01, 0.001]
MAX_SINKHORN_N = 15_000
MAX_EXACT_EMD_N = 2_000  # exact EMD is O(N^3); impractical above this


def scalability_n_values():
    n = 50_000
    while True:
        yield n
        n += 50_000


def sync():
    torch.cuda.synchronize()


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()


def gpu_mem_gb():
    return (
        torch.cuda.memory_allocated() / 1024**3,
        torch.cuda.memory_reserved() / 1024**3,
    )


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


def run_exact_emd(A_cuda, B_cuda, diameter):
    """
    Exact W1 transport cost using POT ot.emd2. Runs on CPU in float64.
    Only call for N <= MAX_EXACT_EMD_N.
    Returns {"status": "ok", "cost": float, "time": float}
         or {"status": "error"/"oom", "cost": nan, "time": nan}
    """
    if A_cuda.shape[0] > MAX_EXACT_EMD_N:
        print(
            f"  [Exact EMD] SKIP (N={A_cuda.shape[0]:,} > MAX_EXACT_EMD_N={MAX_EXACT_EMD_N:,}; "
            f"O(N^3) solver impractical at this size)",
            flush=True,
        )
        return {"status": "skip", "cost": math.nan, "time": math.nan, "peak_gb": 0.0}

    try:
        import ot as pot

        sync()
        t0 = time.time()
        N = A_cuda.shape[0]
        A_np = A_cuda.cpu().numpy().astype(np.float64)
        B_np = B_cuda.cpu().numpy().astype(np.float64)

        diff = B_np[:, None, :] - A_np[None, :, :]
        C = np.sqrt((diff**2).sum(axis=2))

        a = np.ones(N, dtype=np.float64) / N
        b = np.ones(N, dtype=np.float64) / N

        cost_normalized = pot.emd2(a, b, C)
        sync()
        elapsed = time.time() - t0
        cost = float(cost_normalized) * diameter

        print(f"  [Exact EMD] Time: {elapsed:.2f}s | Avg Cost: {cost:.5f}", flush=True)
        return {"status": "ok", "cost": cost, "time": elapsed, "peak_gb": 0.0}

    except MemoryError:
        print("  [Exact EMD] OOM", flush=True)
        return {"status": "oom", "cost": math.nan, "time": math.nan, "peak_gb": 0.0}
    except Exception as exc:
        print(f"  [Exact EMD] ERROR: {exc}", flush=True)
        return {"status": "error", "cost": math.nan, "time": math.nan, "peak_gb": 0.0}


def sinkhorn_has_memory(n):
    props = torch.cuda.get_device_properties(0)
    available = props.total_memory - torch.cuda.memory_allocated()
    required = n * n * 4 * 2
    return available >= required, available, required


def run_sinkhorn_pot(A_cuda, B_cuda, diameter, reg):
    """
    Sinkhorn via POT bregman log-domain routine (stabilized, GPU).
    Reports soft transport plan cost = sum_ij T_ij * C_ij in original scale.
    This is a biased upper bound on the true OT cost due to entropic smoothing.
    Convergence is verified by checking marginal constraint violations on T.
    Only runs for N <= MAX_SINKHORN_N (N x N cost matrix memory limit).
    """
    N = A_cuda.shape[0]

    if N > MAX_SINKHORN_N:
        return {"status": "skip", "cost": math.nan, "time": math.nan, "peak_gb": math.nan}

    has_mem, available, required = sinkhorn_has_memory(N)
    if not has_mem:
        print(
            f"  [Sinkhorn eps_reg={reg}] OOM precheck "
            f"(need {required/1024**3:.2f} GB, have {available/1024**3:.2f} GB)",
            flush=True,
        )
        return {"status": "oom", "cost": math.nan, "time": math.nan, "peak_gb": math.nan}

    try:
        import ot as pot
        a = torch.ones(N, device=A_cuda.device, dtype=torch.float32) / N
        b = torch.ones(N, device=A_cuda.device, dtype=torch.float32) / N

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        sync()
        t0 = time.time()

        # Build N x N L2 cost matrix, timed because it is part of Sinkhorn's cost.
        C = torch.cdist(
            B_cuda, A_cuda, p=2,
            compute_mode="use_mm_for_euclid_dist_if_necessary",
        )  # (N, N) float32, GPU

        # Log-domain stabilized Sinkhorn, numerically robust at small reg.
        # This direct bregman call is the correct POT API.
        T, _log = pot.bregman.sinkhorn_log(
            a, b, C,
            reg=reg,
            numItermax=5000,
            stopThr=1e-6,
            log=True,
            warn=False,
        )

        sync()
        elapsed = time.time() - t0
        peak_gb = torch.cuda.max_memory_allocated() / 1024**3

        # Convergence check: verify marginal constraints directly on T.
        # T must satisfy T.sum(dim=1) == a and T.sum(dim=0) == b.
        # Threshold 1e-4 is conservative for float32 GPU arithmetic.
        row_err = float((T.sum(dim=1) - a).abs().max().item())
        col_err = float((T.sum(dim=0) - b).abs().max().item())
        converged = (row_err < 1e-4) and (col_err < 1e-4)

        if not converged:
            del T, C
            cleanup()
            print(
                f"  [Sinkhorn eps_reg={reg}] DNC "
                f"(row_err={row_err:.2e}, col_err={col_err:.2e})",
                flush=True,
            )
            return {"status": "dnc", "cost": math.nan, "time": elapsed, "peak_gb": math.nan}

        # Soft plan cost: sum_ij T_ij * C_ij = average transport cost under T.
        # Biased upward vs true OT cost by O(reg * entropy(T)).
        soft_cost_normalized = float((T * C).sum().item())
        soft_cost = soft_cost_normalized * diameter

        del T, C
        cleanup()

        print(
            f"  [Sinkhorn eps_reg={reg}] Time: {elapsed:.2f}s | "
            f"Soft Plan Cost: {soft_cost:.5f} "
            f"(row_err={row_err:.2e}, col_err={col_err:.2e})",
            flush=True,
        )
        return {"status": "ok", "cost": soft_cost, "time": elapsed, "peak_gb": peak_gb}

    except torch.cuda.OutOfMemoryError:
        cleanup()
        print(f"  [Sinkhorn eps_reg={reg}] OOM", flush=True)
        return {"status": "oom", "cost": math.nan, "time": math.nan, "peak_gb": math.nan}
    except Exception as exc:
        cleanup()
        print(f"  [Sinkhorn eps_reg={reg}] ERROR: {exc}", flush=True)
        return {"status": "error", "cost": math.nan, "time": math.nan, "peak_gb": math.nan}


def run_solver(label, solver_cls, A, B, diameter):
    cleanup()
    solver = None
    t_cluster_start = None
    t_cluster_end = None
    t_solve_start = None
    t_solve_end = None

    try:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        sync()
        t_cluster_start = time.time()
        solver = solver_cls(
            A, B,
            epsilon=EPSILON,
            batch_size=BATCH_SIZE,
            verbose=False,
            diameter=1.0,
            max_iters=MAX_ITERS,
        )
        sync()
        t_cluster_end = time.time()
        print(
            f"  [{label}] Clustering done in {t_cluster_end - t_cluster_start:.2f}s",
            flush=True,
        )
    except torch.cuda.OutOfMemoryError:
        peak_oom = torch.cuda.max_memory_allocated() / 1024**3
        gc.collect()
        torch.cuda.empty_cache()
        print(
            f"  [{label}] OOM during CLUSTERING "
            f"(peak before OOM: {peak_oom:.3f} GB) -- see last [MEM] line above",
            flush=True,
        )
        return {
            "status": "oom",
            "phase": "clustering",
            "time": math.nan,
            "cost": math.nan,
            "peak_gb": peak_oom,
        }
    except Exception as exc:
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  [{label}] ERROR during CLUSTERING: {exc}", flush=True)
        return {
            "status": "error",
            "phase": "clustering",
            "time": math.nan,
            "cost": math.nan,
            "peak_gb": math.nan,
        }

    match_B = None
    try:
        sync()
        t_solve_start = time.time()
        match_B = solver.solve()
        sync()
        t_solve_end = time.time()
        peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    except torch.cuda.OutOfMemoryError:
        peak_oom = torch.cuda.max_memory_allocated() / 1024**3
        cleanup()
        phase = getattr(solver, "_oom_phase", "unknown")
        print(f"  [{label}] OOM during SOLVE at sub-step: {phase}", flush=True)
        del solver
        solver = None
        cleanup()
        return {
            "status": "oom",
            "phase": f"solve:{phase}",
            "time": math.nan,
            "cost": math.nan,
            "peak_gb": peak_oom,
        }
    except Exception as exc:
        cleanup()
        phase = getattr(solver, "_oom_phase", "unknown")
        print(f"  [{label}] ERROR during SOLVE at sub-step {phase}: {exc}", flush=True)
        del solver
        solver = None
        cleanup()
        return {
            "status": "error",
            "phase": f"solve:{phase}",
            "time": math.nan,
            "cost": math.nan,
            "peak_gb": math.nan,
        }

    try:
        if match_B is None:
            match_B = solver.match_B
        cost = avg_matching_cost(A, B, match_B, diameter)
        t_total = (t_cluster_end - t_cluster_start) + (t_solve_end - t_solve_start)
        print(
            f"  [{label}] Cluster: {t_cluster_end - t_cluster_start:.2f}s  "
            f"Solve: {t_solve_end - t_solve_start:.2f}s  "
            f"Total: {t_total:.2f}s  |  Avg Cost: {cost:.5f}",
            flush=True,
        )
        return {
            "status": "ok",
            "phase": "done",
            "time": t_total,
            "time_cluster": t_cluster_end - t_cluster_start,
            "time_solve": t_solve_end - t_solve_start,
            "cost": cost,
            "peak_gb": peak_gb,
        }
    except Exception as exc:
        cleanup()
        print(f"  [{label}] ERROR during cost computation: {exc}", flush=True)
        return {
            "status": "error",
            "phase": "cost",
            "time": math.nan,
            "cost": math.nan,
            "peak_gb": math.nan,
        }
    finally:
        del solver
        cleanup()


def result_value(result, key):
    status = result.get("status", "error")
    if status == "ok":
        value = result.get(key, math.nan)
        if not math.isfinite(value):
            return "N/A"
        return f"{value:.5f}" if key == "cost" else f"{value:.2f}"
    if status == "dnc":
        return "DNC"
    if status == "skip":
        return "SKIP"
    return status.upper()


def delta_from_exact(result, exact_cost):
    if result["status"] != "ok" or exact_cost is None or not math.isfinite(exact_cost):
        return math.nan
    return (result["cost"] - exact_cost) / exact_cost * 100.0


def delta_value(result, exact_cost):
    if result.get("status") != "ok":
        return "N/A"
    if exact_cost is None or not math.isfinite(exact_cost):
        return "N/A"
    cost = result.get("cost", math.nan)
    if not math.isfinite(cost):
        return "N/A"
    delta = (cost - exact_cost) / exact_cost * 100.0
    return f"{delta:+.2f}%"


def csv_float(value):
    if value is None or not math.isfinite(value):
        return ""
    return f"{value:.10g}"


def csv_result_value(result, key):
    if result["status"] == "dnc":
        return "DNC"
    return csv_float(result.get(key, math.nan))


def progress_pair(result):
    if result["status"] == "ok":
        return f"{result_value(result, 'time')}s / {result_value(result, 'cost')}"
    return f"{result['status'].upper()} / {result['status'].upper()}"


def peak_value(result, is_cpu=False):
    if is_cpu:
        return "CPU"
    peak_gb = result.get("peak_gb", math.nan)
    if math.isfinite(peak_gb):
        return f"{peak_gb:.3f}"
    return result_value(result, "time")


def print_table(title, headers, table):
    widths = [len(h) for h in headers]
    for row in table:
        widths = [max(w, len(str(v))) for w, v in zip(widths, row)]
    print(f"\n{title}", flush=True)
    print(" | ".join(f"{h:>{w}}" for h, w in zip(headers, widths)), flush=True)
    print("-+-".join("-" * w for w in widths), flush=True)
    for row in table:
        print(" | ".join(f"{str(v):>{w}}" for v, w in zip(row, widths)), flush=True)


def print_accuracy_table(rows):
    headers = [
        "N",
        "Method",
        "eps/blur",
        "Time (s)",
        "Avg Cost or Divergence",
        "Delta from Exact (%)",
        "Peak GPU (GB)",
    ]
    table = []
    for row in rows:
        exact_cost = row["exact_cost"]
        is_cpu = row["method"] == "Exact EMD"
        table.append(
            [
                f"{row['n']:,}",
                row["method"],
                row["epsilon_or_blur"],
                result_value(row["result"], "time"),
                result_value(row["result"], "cost"),
                delta_value(row["result"], exact_cost),
                peak_value(row["result"], is_cpu=is_cpu),
            ]
        )
    print_table("Table 1 - Accuracy", headers, table)
    print(
        "\nNotes: Sinkhorn costs are soft plan costs and biased upper estimates of OT. "
        "Push-relabel rows report hard matching average L2 cost with additive epsilon guarantee.",
        flush=True,
    )


def print_scalability_table(rows):
    headers = [
        "N",
        "2-Level Time (s)",
        "2-Level Cost",
        "2-Level Peak (GB)",
        "3-Level Time (s)",
        "3-Level Cost",
        "3-Level Peak (GB)",
    ]
    table = []
    for row in rows:
        table.append(
            [
                f"{row['n']:,}",
                result_value(row["sol2"], "time"),
                result_value(row["sol2"], "cost"),
                peak_value(row["sol2"]),
                result_value(row["sol3"], "time"),
                result_value(row["sol3"], "cost"),
                peak_value(row["sol3"]),
            ]
        )
    print_table("Table 2 - Scalability", headers, table)
    print("\nSinkhorn (POT): N/A (N x N matrix) for scalability rows.", flush=True)


def save_accuracy_csv(rows, stamp):
    path = SCRIPT_DIR / f"comparison_synthetic_accuracy_{stamp}.csv"
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "N",
                "Method",
                "epsilon_or_blur",
                "time_s",
                "cost",
                "delta_from_exact_pct",
                "peak_gpu_gb",
                "cost_type",
                "hardware",
            ]
        )
        for row in rows:
            result = row["result"]
            delta = delta_from_exact(result, row["exact_cost"])
            writer.writerow(
                [
                    row["n"],
                    row["method"],
                    row["epsilon_or_blur"],
                    csv_result_value(result, "time"),
                    csv_result_value(result, "cost"),
                    csv_float(delta),
                    csv_float(result.get("peak_gb", math.nan)),
                    row["cost_type"],
                    row["hardware"],
                ]
            )
    print(f"\nSaved accuracy CSV: {path}", flush=True)
    return path


def save_scalability_csv(rows, stamp):
    path = SCRIPT_DIR / f"comparison_synthetic_scalability_{stamp}.csv"
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "N",
                "two_level_time_s",
                "two_level_cost",
                "two_level_peak_gb",
                "two_level_status",
                "three_level_time_s",
                "three_level_cost",
                "three_level_peak_gb",
                "three_level_status",
                "sinkhorn_pot",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["n"],
                    csv_result_value(row["sol2"], "time"),
                    csv_result_value(row["sol2"], "cost"),
                    csv_float(row["sol2"].get("peak_gb", math.nan)),
                    row["sol2"]["status"],
                    csv_result_value(row["sol3"], "time"),
                    csv_result_value(row["sol3"], "cost"),
                    csv_float(row["sol3"].get("peak_gb", math.nan)),
                    row["sol3"]["status"],
                    "N/A (N x N matrix)",
                ]
            )
    print(f"\nSaved scalability CSV: {path}", flush=True)
    return path


def append_accuracy_row(
    rows,
    n,
    method,
    epsilon_or_blur,
    result,
    exact_cost,
    cost_type,
    hardware,
):
    rows.append(
        {
            "n": n,
            "method": method,
            "epsilon_or_blur": epsilon_or_blur,
            "result": result,
            "exact_cost": exact_cost,
            "cost_type": cost_type,
            "hardware": hardware,
        }
    )


def run_accuracy_phase():
    print(
        "\n=== Table 1: Accuracy ===\n"
        f"Sizes: {ACCURACY_SIZES}\n"
        f"Exact EMD: N <= {MAX_EXACT_EMD_N} only (O(N^3) CPU solver).\n"
        f"Sinkhorn:  N <= {MAX_SINKHORN_N} only (N x N GPU matrix).\n"
        "Peak GPU memory measured per solver via torch.cuda.reset_peak_memory_stats()\n"
        "+ torch.cuda.max_memory_allocated(). GPU cleared between every solver call.\n"
        "Sinkhorn soft-plan costs are biased upward by entropic smoothing.\n"
        "Delta from Exact includes this bias. Our solver eps=0.01 is an additive\n"
        "per-pair error bound -- a different quantity from Sinkhorn eps_reg.\n",
        flush=True,
    )
    rows = []

    for n in ACCURACY_SIZES:
        print(f"\n=== Accuracy N = {n:,} ===", flush=True)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

        try:
            A, B, diameter = make_data(n)
        except torch.cuda.OutOfMemoryError:
            cleanup()
            print(f"  DATA OOM at N={n}; skipping.", flush=True)
            continue
        except Exception as exc:
            cleanup()
            print(f"  DATA ERROR at N={n}: {exc}; skipping.", flush=True)
            continue

        alloc, reserved = gpu_mem_gb()
        print(
            f"GPU Memory: {alloc:.2f} GB allocated, {reserved:.2f} GB reserved",
            flush=True,
        )

        gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
        exact = run_exact_emd(A, B, diameter)
        exact_cost = exact["cost"] if exact["status"] == "ok" else math.nan
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
        append_accuracy_row(
            rows,
            n,
            "Exact EMD",
            "",
            exact,
            exact_cost,
            "hard_matching_cost",
            "CPU (single-thread C++)",
        )

        for reg in SINKHORN_REGS:
            sinkhorn = run_sinkhorn_pot(A, B, diameter, reg)
            gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
            append_accuracy_row(
                rows,
                n,
                "Sinkhorn (POT)",
                reg,
                sinkhorn,
                exact_cost,
                "soft_plan_cost",
                "GPU (RTX 2060 Super, 8GB VRAM)",
            )

        sol2 = run_solver("2-Level", SimpleGPUSolver, A, B, diameter)
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
        append_accuracy_row(
            rows,
            n,
            "2-Level Push-Relabel",
            EPSILON,
            sol2,
            exact_cost,
            "hard_matching_cost",
            "GPU (RTX 2060 Super, 8GB VRAM)",
        )

        sol3 = run_solver("3-Level", ThreeLevelGPUSolver, A, B, diameter)
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
        append_accuracy_row(
            rows,
            n,
            "3-Level Push-Relabel",
            EPSILON,
            sol3,
            exact_cost,
            "hard_matching_cost",
            "GPU (RTX 2060 Super, 8GB VRAM)",
        )

        del A, B
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()

    print_accuracy_table(rows)
    return rows


def run_scalability_phase():
    print("\n=== Table 2: Scalability ===", flush=True)
    print("Sinkhorn (POT): N/A (N x N matrix) for this phase.", flush=True)
    rows = []
    sol2_active = True
    sol3_active = True
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    for n in scalability_n_values():
        print(f"\n=== Scalability N = {n:,} ===", flush=True)
        alloc, reserved = gpu_mem_gb()
        print(
            f"GPU Memory: {alloc:.2f} GB allocated, {reserved:.2f} GB reserved",
            flush=True,
        )

        try:
            A, B, diameter = make_data(n)
        except torch.cuda.OutOfMemoryError:
            cleanup()
            print(f"DATA OOM while generating or normalizing N={n:,}; stopping.", flush=True)
            break
        except Exception as exc:
            cleanup()
            print(f"DATA ERROR at N={n:,}: {exc}; stopping.", flush=True)
            break

        if sol2_active:
            gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
            sol2 = run_solver("2-Level", SimpleGPUSolver, A, B, diameter)
            if sol2["status"] == "oom":
                sol2_active = False
        else:
            sol2 = {"status": "oom", "time": math.nan, "cost": math.nan, "peak_gb": math.nan}
            print("  [2-Level] OOM", flush=True)

        gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()

        if sol3_active:
            sol3 = run_solver("3-Level", ThreeLevelGPUSolver, A, B, diameter)
            if sol3["status"] == "oom":
                sol3_active = False
        else:
            sol3 = {"status": "oom", "time": math.nan, "cost": math.nan, "peak_gb": math.nan}
            print("  [3-Level] OOM", flush=True)

        gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()

        rows.append({"n": n, "sol2": sol2, "sol3": sol3})
        print(
            f"Row: {n:,} | 2-Level {progress_pair(sol2)} | 3-Level {progress_pair(sol3)}",
            flush=True,
        )

        stop_for_error = sol2["status"] == "error" or sol3["status"] == "error"
        del A, B
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
        if stop_for_error:
            print("Unrecoverable solver error recorded; stopping.", flush=True)
            break
        if not sol2_active and not sol3_active:
            print("Both push-relabel solvers have OOMed; stopping.", flush=True)
            break

    print_scalability_table(rows)
    return rows


def main():
    assert torch.cuda.is_available(), "CUDA is required"
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    accuracy_rows = run_accuracy_phase()
    scalability_rows = run_scalability_phase()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_accuracy_csv(accuracy_rows, stamp)
    save_scalability_csv(scalability_rows, stamp)


if __name__ == "__main__":
    main()
