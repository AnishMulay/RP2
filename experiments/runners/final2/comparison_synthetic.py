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
# which is incompatible with CUDA 13.x (system: CUDA 13.2, PyKeOps 2.3).
# POT sinkhorn (log-domain stabilized) is used as the Sinkhorn baseline.

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
ACCURACY_SIZES = [1_000, 5_000, 10_000]
ACCURACY_SEEDS = [42, 123, 777, 999, 1337]
SCALABILITY_SEEDS = [42, 123, 777]
SINKHORN_REGS = [0.1, 0.01, 0.001]


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
    Only call for N <= 10000.
    Returns {"status": "ok", "cost": float, "time": float}
         or {"status": "error"/"oom", "cost": nan, "time": nan}
    """
    if A_cuda.shape[0] > 10_000:
        return {"status": "skip", "cost": math.nan, "time": math.nan}

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
        return {"status": "ok", "cost": cost, "time": elapsed}

    except MemoryError:
        print("  [Exact EMD] OOM", flush=True)
        return {"status": "oom", "cost": math.nan, "time": math.nan}
    except Exception as exc:
        print(f"  [Exact EMD] ERROR: {exc}", flush=True)
        return {"status": "error", "cost": math.nan, "time": math.nan}


def sinkhorn_has_memory(n):
    props = torch.cuda.get_device_properties(0)
    available = props.total_memory - torch.cuda.memory_allocated()
    required = n * n * 4 * 2
    return available >= required, available, required


def run_sinkhorn_pot(A_cuda, B_cuda, diameter, reg):
    """
    Sinkhorn via POT with PyTorch GPU backend.
    Returns soft transport plan cost = sum_ij T_ij * C_ij in original scale.
    This is a biased upper bound on the true OT cost.
    """
    if A_cuda.shape[0] > 10_000:
        return {"status": "skip", "cost": math.nan, "time": math.nan}

    try:
        import ot as pot

        N = A_cuda.shape[0]
        has_memory, available, required = sinkhorn_has_memory(N)
        if not has_memory:
            print(
                f"  [Sinkhorn eps_reg={reg}] OOM precheck "
                f"(available={available / 1024**3:.2f} GiB, "
                f"required={required / 1024**3:.2f} GiB)",
                flush=True,
            )
            return {"status": "oom", "cost": math.nan, "time": math.nan}

        a = torch.ones(N, device=A_cuda.device, dtype=torch.float32) / N
        b = torch.ones(N, device=A_cuda.device, dtype=torch.float32) / N

        sync()
        t0 = time.time()

        C = torch.cdist(
            B_cuda,
            A_cuda,
            p=2,
            compute_mode="use_mm_for_euclid_dist_if_necessary",
        )

        T, sinkhorn_log_dict = pot.sinkhorn(
            a,
            b,
            C,
            reg=reg,
            method="sinkhorn_log",
            numItermax=10000,
            stopThr=1e-9,
            log=True,
        )

        sync()
        elapsed = time.time() - t0

        err_list = sinkhorn_log_dict.get("err", [])
        converged = (len(err_list) == 0) or (float(err_list[-1]) <= 1e-9)

        if not converged:
            print(
                f"  [Sinkhorn eps_reg={reg}] DNC "
                f"(did not converge, last_err={err_list[-1]:.2e})",
                flush=True,
            )
            del T, C
            cleanup()
            return {"status": "dnc", "cost": math.nan, "time": elapsed}

        soft_cost_normalized = (T * C).sum().item()
        soft_cost = soft_cost_normalized * diameter

        del T, C
        cleanup()

        print(
            f"  [Sinkhorn eps_reg={reg}] Time: {elapsed:.2f}s | "
            f"Soft Plan Cost: {soft_cost:.5f}",
            flush=True,
        )
        return {"status": "ok", "cost": soft_cost, "time": elapsed}

    except torch.cuda.OutOfMemoryError:
        cleanup()
        print(f"  [Sinkhorn eps_reg={reg}] OOM", flush=True)
        return {"status": "oom", "cost": math.nan, "time": math.nan}
    except Exception as exc:
        cleanup()
        print(f"  [Sinkhorn eps_reg={reg}] ERROR: {exc}", flush=True)
        return {"status": "error", "cost": math.nan, "time": math.nan}


def run_solver(label, solver_cls, A, B, diameter):
    cleanup()
    solver = None
    t_cluster_start = None
    t_cluster_end = None
    t_solve_start = None
    t_solve_end = None

    try:
        sync()
        t_cluster_start = time.time()
        solver = solver_cls(
            A,
            B,
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
        cleanup()
        print(f"  [{label}] OOM during CLUSTERING -- see last [MEM] line above", flush=True)
        return {"status": "oom", "phase": "clustering", "time": math.nan, "cost": math.nan}
    except Exception as exc:
        cleanup()
        print(f"  [{label}] ERROR during CLUSTERING: {exc}", flush=True)
        return {"status": "error", "phase": "clustering", "time": math.nan, "cost": math.nan}

    match_B = None
    try:
        sync()
        t_solve_start = time.time()
        match_B = solver.solve()
        sync()
        t_solve_end = time.time()
    except torch.cuda.OutOfMemoryError:
        cleanup()
        phase = getattr(solver, "_oom_phase", "unknown")
        print(f"  [{label}] OOM during SOLVE at sub-step: {phase}", flush=True)
        del solver
        solver = None
        cleanup()
        return {"status": "oom", "phase": f"solve:{phase}", "time": math.nan, "cost": math.nan}
    except Exception as exc:
        cleanup()
        phase = getattr(solver, "_oom_phase", "unknown")
        print(f"  [{label}] ERROR during SOLVE at sub-step {phase}: {exc}", flush=True)
        del solver
        solver = None
        cleanup()
        return {"status": "error", "phase": f"solve:{phase}", "time": math.nan, "cost": math.nan}

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
        }
    except Exception as exc:
        cleanup()
        print(f"  [{label}] ERROR during cost computation: {exc}", flush=True)
        return {"status": "error", "phase": "cost", "time": math.nan, "cost": math.nan}
    finally:
        del solver
        cleanup()


def result_value(result, key):
    if result["status"] == "ok":
        value = result.get(key, math.nan)
        return f"{value:.5f}" if key == "cost" else f"{value:.2f}"
    if result["status"] == "dnc":
        return "DNC"
    return result["status"].upper()


def delta_from_exact(result, exact_cost):
    if result["status"] != "ok" or exact_cost is None or not math.isfinite(exact_cost):
        return math.nan
    return (result["cost"] - exact_cost) / exact_cost * 100.0


def delta_value(result, exact_cost):
    if result["status"] == "dnc":
        return "N/A"
    delta = delta_from_exact(result, exact_cost)
    if math.isfinite(delta):
        return f"{delta:.2f}"
    return "N/A"


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


import statistics as _statistics


def _aggregate(rows, n, method, eps_or_blur, key):
    vals = [
        r["result"].get(key, math.nan)
        for r in rows
        if r["n"] == n
        and r["method"] == method
        and str(r["epsilon_or_blur"]) == str(eps_or_blur)
        and r["result"]["status"] == "ok"
        and math.isfinite(r["result"].get(key, math.nan))
    ]
    if not vals:
        return math.nan, math.nan
    mean = sum(vals) / len(vals)
    std = _statistics.stdev(vals) if len(vals) > 1 else 0.0
    return mean, std


def _fmt_mean_std(mean, std, decimals):
    if not math.isfinite(mean):
        return "N/A"
    fmt = f".{decimals}f"
    return f"{mean:{fmt}}±{std:{fmt}}"


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
        "Time (s) mean±std",
        "Cost mean±std",
        "Δ mean (%)",
    ]
    table = []
    seen = set()
    groups = []
    for row in rows:
        key = (row["n"], row["method"], row["epsilon_or_blur"])
        if key not in seen:
            seen.add(key)
            groups.append(key)
    for n, method, epsilon_or_blur in groups:
        time_mean, time_std = _aggregate(rows, n, method, epsilon_or_blur, "time")
        cost_mean, cost_std = _aggregate(rows, n, method, epsilon_or_blur, "cost")
        exact_mean, _ = _aggregate(rows, n, "Exact EMD", "", "cost")
        delta = math.nan
        if math.isfinite(cost_mean) and math.isfinite(exact_mean):
            delta = (cost_mean - exact_mean) / exact_mean * 100.0
        table.append(
            [
                f"{n:,}",
                method,
                epsilon_or_blur,
                _fmt_mean_std(time_mean, time_std, 2),
                _fmt_mean_std(cost_mean, cost_std, 5),
                f"{delta:.2f}" if math.isfinite(delta) else "N/A",
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
        "3-Level Time (s)",
        "3-Level Cost",
    ]
    table = []
    for row in rows:
        n = row["n"]
        flat_rows = []
        for result in row["sol2"]:
            flat_rows.append({"n": n, "method": "sol2", "epsilon_or_blur": "", "result": result})
        for result in row["sol3"]:
            flat_rows.append({"n": n, "method": "sol3", "epsilon_or_blur": "", "result": result})
        sol2_time_mean, sol2_time_std = _aggregate(flat_rows, n, "sol2", "", "time")
        sol2_cost_mean, sol2_cost_std = _aggregate(flat_rows, n, "sol2", "", "cost")
        sol3_time_mean, sol3_time_std = _aggregate(flat_rows, n, "sol3", "", "time")
        sol3_cost_mean, sol3_cost_std = _aggregate(flat_rows, n, "sol3", "", "cost")
        table.append(
            [
                f"{n:,}",
                _fmt_mean_std(sol2_time_mean, sol2_time_std, 2),
                _fmt_mean_std(sol2_cost_mean, sol2_cost_std, 5),
                _fmt_mean_std(sol3_time_mean, sol3_time_std, 2),
                _fmt_mean_std(sol3_cost_mean, sol3_cost_std, 5),
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
                "seed",
                "Method",
                "epsilon_or_blur",
                "time_s",
                "cost",
                "delta_from_exact_pct",
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
                    row["seed"],
                    row["method"],
                    row["epsilon_or_blur"],
                    csv_result_value(result, "time"),
                    csv_result_value(result, "cost"),
                    csv_float(delta),
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
                "two_level_status",
                "three_level_time_s",
                "three_level_cost",
                "three_level_status",
                "sinkhorn_pot",
            ]
        )
        for row in rows:
            n = row["n"]
            flat_rows = []
            for result in row["sol2"]:
                flat_rows.append({"n": n, "method": "sol2", "epsilon_or_blur": "", "result": result})
            for result in row["sol3"]:
                flat_rows.append({"n": n, "method": "sol3", "epsilon_or_blur": "", "result": result})
            sol2_time_mean, sol2_time_std = _aggregate(flat_rows, n, "sol2", "", "time")
            sol2_cost_mean, sol2_cost_std = _aggregate(flat_rows, n, "sol2", "", "cost")
            sol3_time_mean, sol3_time_std = _aggregate(flat_rows, n, "sol3", "", "time")
            sol3_cost_mean, sol3_cost_std = _aggregate(flat_rows, n, "sol3", "", "cost")
            writer.writerow(
                [
                    n,
                    _fmt_mean_std(sol2_time_mean, sol2_time_std, 2),
                    _fmt_mean_std(sol2_cost_mean, sol2_cost_std, 5),
                    "ok" if math.isfinite(sol2_time_mean) else "N/A",
                    _fmt_mean_std(sol3_time_mean, sol3_time_std, 2),
                    _fmt_mean_std(sol3_cost_mean, sol3_cost_std, 5),
                    "ok" if math.isfinite(sol3_time_mean) else "N/A",
                    "N/A (N x N matrix)",
                ]
            )
    print(f"\nSaved scalability CSV: {path}", flush=True)
    return path


def append_accuracy_row(
    rows,
    n,
    seed,
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
            "seed": seed,
            "method": method,
            "epsilon_or_blur": epsilon_or_blur,
            "result": result,
            "exact_cost": exact_cost,
            "cost_type": cost_type,
            "hardware": hardware,
        }
    )


def run_accuracy_phase():
    print("\n=== Table 1: Accuracy ===", flush=True)
    print(
        "\nNote: Sinkhorn eps_reg values span high/medium/low regularization "
        "relative to mean pairwise L2 ~0.52 in normalized [0,1]^2 uniform data.\n"
        "Our solver eps=0.01 is an additive per-pair error bound -- a different "
        "quantity. Sinkhorn costs are soft-plan expected costs biased upward by "
        "entropic smoothing. Delta from Exact includes this bias.",
        flush=True,
    )
    rows = []

    for n in ACCURACY_SIZES:
        print(f"\n=== Accuracy N = {n:,} ===", flush=True)
        for seed in ACCURACY_SEEDS:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            try:
                A, B, diameter = make_data(n)
            except torch.cuda.OutOfMemoryError:
                cleanup()
                print(f"  DATA OOM at N={n}, seed={seed}; skipping.", flush=True)
                continue
            except Exception as exc:
                cleanup()
                print(f"  DATA ERROR at N={n}, seed={seed}: {exc}; skipping.", flush=True)
                continue

            print(f"--- Accuracy N = {n:,}, seed = {seed} ---", flush=True)
            alloc, reserved = gpu_mem_gb()
            print(
                f"GPU Memory: {alloc:.2f} GB allocated, {reserved:.2f} GB reserved",
                flush=True,
            )

            cleanup()
            exact = run_exact_emd(A, B, diameter)
            exact_cost = exact["cost"] if exact["status"] == "ok" else math.nan
            cleanup()
            append_accuracy_row(
                rows,
                n,
                seed,
                "Exact EMD",
                "",
                exact,
                exact_cost,
                "hard_matching_cost",
                "CPU (single-thread C++)",
            )

            for reg in SINKHORN_REGS:
                cleanup()
                sinkhorn = run_sinkhorn_pot(A, B, diameter, reg)
                append_accuracy_row(
                    rows,
                    n,
                    seed,
                    "Sinkhorn (POT)",
                    reg,
                    sinkhorn,
                    exact_cost,
                    "soft_plan_cost",
                    "GPU (RTX 2060 Super, 8GB VRAM)",
                )

            cleanup()
            sol2 = run_solver("2-Level", SimpleGPUSolver, A, B, diameter)
            append_accuracy_row(
                rows,
                n,
                seed,
                "2-Level Push-Relabel",
                EPSILON,
                sol2,
                exact_cost,
                "hard_matching_cost",
                "GPU (RTX 2060 Super, 8GB VRAM)",
            )

            cleanup()
            sol3 = run_solver("3-Level", ThreeLevelGPUSolver, A, B, diameter)
            append_accuracy_row(
                rows,
                n,
                seed,
                "3-Level Push-Relabel",
                EPSILON,
                sol3,
                exact_cost,
                "hard_matching_cost",
                "GPU (RTX 2060 Super, 8GB VRAM)",
            )

            del A, B
            cleanup()

    print_accuracy_table(rows)
    return rows


def run_scalability_phase():
    print("\n=== Table 2: Scalability ===", flush=True)
    print("Sinkhorn (POT): N/A (N x N matrix) for this phase.", flush=True)
    rows = []
    sol2_active = True
    sol3_active = True
    stop_scalability = False

    for n in scalability_n_values():
        print(f"\n=== Scalability N = {n:,} ===", flush=True)
        for seed in SCALABILITY_SEEDS:
            print(f"\n--- Scalability N = {n:,}, seed = {seed} ---", flush=True)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            alloc, reserved = gpu_mem_gb()
            print(f"GPU Memory: {alloc:.2f} GB allocated, {reserved:.2f} GB reserved", flush=True)

            try:
                A, B, diameter = make_data(n)
            except torch.cuda.OutOfMemoryError:
                cleanup()
                print(
                    f"DATA OOM while generating or normalizing N={n:,}, seed={seed}; stopping.",
                    flush=True,
                )
                stop_scalability = True
                break
            except Exception as exc:
                cleanup()
                print(f"DATA ERROR at N={n:,}, seed={seed}: {exc}; stopping.", flush=True)
                stop_scalability = True
                break

            if sol2_active:
                cleanup()
                sol2 = run_solver("2-Level", SimpleGPUSolver, A, B, diameter)
                if sol2["status"] == "oom":
                    sol2_active = False
            else:
                sol2 = {"status": "oom", "time": math.nan, "cost": math.nan}
                print("  [2-Level] OOM", flush=True)

            if sol3_active:
                cleanup()
                sol3 = run_solver("3-Level", ThreeLevelGPUSolver, A, B, diameter)
                if sol3["status"] == "oom":
                    sol3_active = False
            else:
                sol3 = {"status": "oom", "time": math.nan, "cost": math.nan}
                print("  [3-Level] OOM", flush=True)

            rows.append({"n": n, "seed": seed, "sol2": sol2, "sol3": sol3})
            print(
                f"Row: {n:,}, seed {seed} | "
                f"2-Level {progress_pair(sol2)} | 3-Level {progress_pair(sol3)}",
                flush=True,
            )

            stop_for_error = sol2["status"] == "error" or sol3["status"] == "error"
            del A, B
            cleanup()
            if stop_for_error:
                print("Unrecoverable solver error recorded; stopping.", flush=True)
                stop_scalability = True
                break
            if not sol2_active and not sol3_active:
                print("Both push-relabel solvers have OOMed; stopping.", flush=True)
                stop_scalability = True
                break
        if stop_scalability:
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
