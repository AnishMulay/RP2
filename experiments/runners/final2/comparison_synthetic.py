#!/usr/bin/env python3
"""
Standalone comparison sweep on synthetic 2D point clouds.

Table 1 compares exact EMD, POT Sinkhorn, GeomLoss, and push-relabel on
small point clouds. Table 2 compares scalable methods on larger point clouds.
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
SEED = 42

DIAMETER_TILE = 1024
ACCURACY_SIZES = [1_000, 5_000, 10_000]
SINKHORN_REGS = [0.1, 0.01, 0.001]
GEOMLOSS_BLURS = [0.1, 0.05]


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

        C = torch.cdist(
            B_cuda,
            A_cuda,
            p=2,
            compute_mode="use_mm_for_euclid_dist_if_necessary",
        )

        sync()
        t0 = time.time()

        T = pot.sinkhorn(
            a,
            b,
            C,
            reg=reg,
            numItermax=10000,
            stopThr=1e-9,
            backend="torch",
        )

        sync()
        elapsed = time.time() - t0

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


def run_geomloss(A_cuda, B_cuda, diameter, blur):
    """
    GeomLoss SamplesLoss Sinkhorn divergence (debiased, L2 cost).
    Output is the Sinkhorn divergence scalar, not average matched cost.
    """
    try:
        from geomloss import SamplesLoss

        loss_fn = SamplesLoss(
            loss="sinkhorn",
            p=2,
            blur=blur,
            debias=True,
            scaling=0.9,
            backend="online",
        )

        sync()
        t0 = time.time()

        div = loss_fn(B_cuda, A_cuda)
        div_val = div.item()

        sync()
        elapsed = time.time() - t0

        div_scaled = div_val * diameter

        print(
            f"  [GeomLoss blur={blur}] Time: {elapsed:.2f}s | "
            f"Sinkhorn Divergence: {div_scaled:.5f}",
            flush=True,
        )
        return {"status": "ok", "cost": div_scaled, "time": elapsed}

    except ImportError:
        print(f"  [GeomLoss blur={blur}] SKIP - geomloss/pykeops not installed", flush=True)
        return {"status": "skip", "cost": math.nan, "time": math.nan}
    except torch.cuda.OutOfMemoryError:
        cleanup()
        print(f"  [GeomLoss blur={blur}] OOM", flush=True)
        return {"status": "oom", "cost": math.nan, "time": math.nan}
    except Exception as exc:
        cleanup()
        print(f"  [GeomLoss blur={blur}] ERROR: {exc}", flush=True)
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
    return result["status"].upper()


def delta_from_exact(result, exact_cost):
    if result["status"] != "ok" or exact_cost is None or not math.isfinite(exact_cost):
        return math.nan
    return (result["cost"] - exact_cost) / exact_cost * 100.0


def delta_value(result, exact_cost):
    delta = delta_from_exact(result, exact_cost)
    if math.isfinite(delta):
        return f"{delta:.2f}"
    return "N/A"


def csv_float(value):
    if value is None or not math.isfinite(value):
        return ""
    return f"{value:.10g}"


def progress_pair(result):
    if result["status"] == "ok":
        return f"{result_value(result, 'time')}s / {result_value(result, 'cost')}"
    return f"{result['status'].upper()} / {result['status'].upper()}"


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
        "eps / blur",
        "Time (s)",
        "Avg Cost or Divergence",
        "Delta from Exact (%)",
    ]
    table = []
    for row in rows:
        exact_cost = row["exact_cost"]
        table.append(
            [
                f"{row['n']:,}",
                row["method"],
                row["epsilon_or_blur"],
                result_value(row["result"], "time"),
                result_value(row["result"], "cost"),
                delta_value(row["result"], exact_cost),
            ]
        )
    print_table("Table 1 - Accuracy", headers, table)
    print(
        "\nNotes: Sinkhorn costs are soft plan costs and biased upper estimates of OT. "
        "GeomLoss rows report Sinkhorn divergence, not average transport cost. "
        "Push-relabel rows report hard matching average L2 cost with additive epsilon guarantee.",
        flush=True,
    )


def print_scalability_table(rows):
    headers = [
        "N",
        "GeomLoss Time (s)",
        "GeomLoss Div",
        "2-Level Time (s)",
        "2-Level Cost",
        "3-Level Time (s)",
        "3-Level Cost",
    ]
    table = []
    for row in rows:
        table.append(
            [
                f"{row['n']:,}",
                result_value(row["geomloss"], "time"),
                result_value(row["geomloss"], "cost"),
                result_value(row["sol2"], "time"),
                result_value(row["sol2"], "cost"),
                result_value(row["sol3"], "time"),
                result_value(row["sol3"], "cost"),
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
                "cost_or_divergence",
                "delta_from_exact_pct",
                "cost_type",
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
                    csv_float(result.get("time", math.nan)),
                    csv_float(result.get("cost", math.nan)),
                    csv_float(delta),
                    row["cost_type"],
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
                "geomloss_time_s",
                "geomloss_divergence",
                "geomloss_status",
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
            writer.writerow(
                [
                    row["n"],
                    csv_float(row["geomloss"].get("time", math.nan)),
                    csv_float(row["geomloss"].get("cost", math.nan)),
                    row["geomloss"]["status"],
                    csv_float(row["sol2"].get("time", math.nan)),
                    csv_float(row["sol2"].get("cost", math.nan)),
                    row["sol2"]["status"],
                    csv_float(row["sol3"].get("time", math.nan)),
                    csv_float(row["sol3"].get("cost", math.nan)),
                    row["sol3"]["status"],
                    "N/A (N x N matrix)",
                ]
            )
    print(f"\nSaved scalability CSV: {path}", flush=True)
    return path


def append_accuracy_row(rows, n, method, epsilon_or_blur, result, exact_cost, cost_type):
    rows.append(
        {
            "n": n,
            "method": method,
            "epsilon_or_blur": epsilon_or_blur,
            "result": result,
            "exact_cost": exact_cost,
            "cost_type": cost_type,
        }
    )


def run_accuracy_phase():
    print("\n=== Table 1: Accuracy ===", flush=True)
    rows = []

    for n in ACCURACY_SIZES:
        print(f"\n=== Accuracy N = {n:,} ===", flush=True)
        alloc, reserved = gpu_mem_gb()
        print(f"GPU Memory: {alloc:.2f} GB allocated, {reserved:.2f} GB reserved", flush=True)

        try:
            A, B, diameter = make_data(n)
        except torch.cuda.OutOfMemoryError:
            cleanup()
            print(f"DATA OOM while generating or normalizing accuracy N={n:,}; skipping.", flush=True)
            continue
        except Exception as exc:
            cleanup()
            print(f"DATA ERROR at accuracy N={n:,}: {exc}; skipping.", flush=True)
            continue

        cleanup()
        exact = run_exact_emd(A, B, diameter)
        exact_cost = exact["cost"] if exact["status"] == "ok" else math.nan
        append_accuracy_row(rows, n, "Exact EMD", "", exact, exact_cost, "hard_matching_cost")

        for reg in SINKHORN_REGS:
            cleanup()
            sinkhorn = run_sinkhorn_pot(A, B, diameter, reg)
            append_accuracy_row(
                rows,
                n,
                "Sinkhorn (POT)",
                reg,
                sinkhorn,
                exact_cost,
                "soft_plan_cost",
            )

        for blur in GEOMLOSS_BLURS:
            cleanup()
            geomloss = run_geomloss(A, B, diameter, blur)
            append_accuracy_row(
                rows,
                n,
                "GeomLoss Div. [divergence]",
                blur,
                geomloss,
                exact_cost,
                "sinkhorn_divergence",
            )

        cleanup()
        sol2 = run_solver("2-Level", SimpleGPUSolver, A, B, diameter)
        append_accuracy_row(rows, n, "2-Level Push-Relabel", EPSILON, sol2, exact_cost, "hard_matching_cost")

        cleanup()
        sol3 = run_solver("3-Level", ThreeLevelGPUSolver, A, B, diameter)
        append_accuracy_row(rows, n, "3-Level Push-Relabel", EPSILON, sol3, exact_cost, "hard_matching_cost")

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

    for n in scalability_n_values():
        print(f"\n=== Scalability N = {n:,} ===", flush=True)
        alloc, reserved = gpu_mem_gb()
        print(f"GPU Memory: {alloc:.2f} GB allocated, {reserved:.2f} GB reserved", flush=True)

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

        cleanup()
        geomloss = run_geomloss(A, B, diameter, 0.05)

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

        rows.append({"n": n, "geomloss": geomloss, "sol2": sol2, "sol3": sol3})
        print(
            f"Row: {n:,} | GeomLoss {progress_pair(geomloss)} | "
            f"2-Level {progress_pair(sol2)} | 3-Level {progress_pair(sol3)}",
            flush=True,
        )

        stop_for_error = sol2["status"] == "error" or sol3["status"] == "error"
        del A, B
        cleanup()
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
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    accuracy_rows = run_accuracy_phase()
    scalability_rows = run_scalability_phase()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_accuracy_csv(accuracy_rows, stamp)
    save_scalability_csv(scalability_rows, stamp)


if __name__ == "__main__":
    main()
