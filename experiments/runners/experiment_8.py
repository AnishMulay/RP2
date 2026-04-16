#!/usr/bin/env python3

import math
import pathlib
import sys
import time

import numpy as np
import torch


BASE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    import ot
except ImportError:
    ot = None

from clustered_push_relabel.solvers.simple_bipartite import SimpleGPUSolver


N_VALUES = [
    1_000,
    2_000,
    3_000,
    4_000,
    5_000,
    6_000,
    7_000,
    8_000,
    9_000,
    10_000,
]
EPSILON = 0.001
SYNTHETIC_DIM = 2
SEED = 42
BATCH_SIZE = 512
WARMUP_RUNS = 0
TIMED_RUNS = 1
METHODS = ("Exact", "Simple")


def generate_synthetic_2d(n, device):
    red = torch.rand((n, SYNTHETIC_DIM), dtype=torch.float32)
    blue = torch.rand((n, SYNTHETIC_DIM), dtype=torch.float32)
    P_red = red.to(device)
    P_blue = blue.to(P_red.device)
    return P_red, P_blue


def normalize_points(P_red, P_blue):
    """
    Normalize red and blue points jointly by the diameter of their union.
    Returns (P_red_norm, P_blue_norm, diameter).
    diameter is a Python float for passing to SimpleGPUSolver.
    """
    all_points = torch.cat([P_red, P_blue], dim=0)
    mean = all_points.mean(dim=0, keepdim=True)
    centered = all_points - mean
    diameter = float(2.0 * centered.norm(dim=1).max().item())
    diameter = max(diameter, 1e-8)
    return P_red / diameter, P_blue / diameter, diameter


def synchronize_if_cuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def empty_cache_if_cuda(device):
    if device.type == "cuda":
        torch.cuda.empty_cache()


def compute_cost_matrix_L2(red, blue):
    """Compute full Euclidean distance matrix between two point sets on CPU."""
    X = torch.as_tensor(red, dtype=torch.float64).contiguous()
    Y = torch.as_tensor(blue, dtype=torch.float64).contiguous()
    return torch.cdist(X, Y, p=2).cpu().numpy()


def matching_from_plan(plan):
    """Convert a POT red-by-blue transport plan into match_B[blue] = red."""
    return torch.from_numpy(plan.argmax(axis=0).astype(np.int64, copy=False))


def matching_costs(P_red, P_blue, match_B):
    match_B = match_B.to(device=P_red.device, dtype=torch.long)
    if match_B.numel() != P_blue.shape[0] or (match_B < 0).any():
        raise RuntimeError("matching is incomplete")

    matched_red = P_red[match_B]
    dists = torch.norm(P_blue - matched_red, p=2, dim=1)
    total_cost = dists.sum().item()
    avg_cost = total_cost / P_blue.shape[0]
    return total_cost, avg_cost


def solver_matching(solver):
    if hasattr(solver, "match_B"):
        return solver.match_B
    return solver.MB


def benchmark_exact(P_red, P_blue):
    if ot is None:
        raise RuntimeError("POT is not installed; exact solver unavailable.")

    red_cpu = P_red.detach().cpu()
    blue_cpu = P_blue.detach().cpu()
    n = red_cpu.shape[0]
    C = compute_cost_matrix_L2(red_cpu, blue_cpu)
    a = np.full(n, 1.0 / n, dtype=np.float64)
    b = np.full(n, 1.0 / n, dtype=np.float64)

    for _ in range(WARMUP_RUNS):
        plan = ot.emd(a, b, C, numItermax=10**6)
        del plan

    for _ in range(TIMED_RUNS):
        t0 = time.perf_counter()
        plan = ot.emd(a, b, C, numItermax=10**6)
        t1 = time.perf_counter()

    match_B = matching_from_plan(plan)
    total_cost, avg_cost = matching_costs(red_cpu, blue_cpu, match_B)
    del plan, match_B, C
    return (t1 - t0) * 1000.0, total_cost, avg_cost


def benchmark_simple(P_red, P_blue, device):
    P_red_norm, P_blue_norm, diameter = normalize_points(P_red, P_blue)

    for _ in range(WARMUP_RUNS):
        empty_cache_if_cuda(device)
        solver = SimpleGPUSolver(
            P_red_norm,
            P_blue_norm,
            EPSILON,
            batch_size=BATCH_SIZE,
            verbose=False,
            diameter=diameter,
        )
        synchronize_if_cuda(device)
        solver.solve()
        synchronize_if_cuda(device)
        del solver

    empty_cache_if_cuda(device)
    solver = SimpleGPUSolver(
        P_red_norm,
        P_blue_norm,
        EPSILON,
        batch_size=BATCH_SIZE,
        verbose=False,
        diameter=diameter,
    )

    for _ in range(TIMED_RUNS):
        synchronize_if_cuda(device)
        t0 = time.perf_counter()
        result = solver.solve()
        synchronize_if_cuda(device)
        t1 = time.perf_counter()

    match_B = result if result is not None else solver_matching(solver)
    phases = solver.iterations
    total_cost, avg_cost = matching_costs(P_red_norm, P_blue_norm, match_B)
    total_cost *= diameter
    avg_cost *= diameter
    del solver, match_B, P_red_norm, P_blue_norm
    return (t1 - t0) * 1000.0, total_cost, avg_cost, phases


def result_na():
    return {
        "time_ms": math.nan,
        "total_cost": math.nan,
        "avg_cost": math.nan,
        "phases": math.nan,
    }


def run_method(n, method_name, P_red, P_blue, device):
    try:
        print(f"  Running {method_name} for N={n:,}...", flush=True)
        phases = math.nan
        if method_name == "Exact":
            time_ms, total_cost, avg_cost = benchmark_exact(P_red, P_blue)
        elif method_name == "Simple":
            time_ms, total_cost, avg_cost, phases = benchmark_simple(
                P_red, P_blue, device
            )
        else:
            raise ValueError(f"Unknown method: {method_name}")
        print(
            f"  Completed {method_name} for N={n:,}: "
            f"{time_ms:.1f} ms, avg_cost={avg_cost:.4f}",
            flush=True,
        )
        return {
            "time_ms": time_ms,
            "total_cost": total_cost,
            "avg_cost": avg_cost,
            "phases": phases,
        }
    except Exception as exc:
        print(f"Warning: N={n:,} {method_name} failed: {exc}", flush=True)
        empty_cache_if_cuda(device)
        return result_na()


def is_available(value):
    return value == value


def format_time(value):
    if not is_available(value):
        return "N/A"
    return f"{value:.1f}"


def format_avg_cost(value):
    if not is_available(value):
        return "N/A"
    return f"{value:.4f}"


def format_phases(value):
    if not is_available(value):
        return "N/A"
    return f"{int(value)}"


def format_speedup(exact_result, simple_result):
    exact_time = exact_result["time_ms"]
    simple_time = simple_result["time_ms"]
    if (
        not is_available(exact_time)
        or not is_available(simple_time)
        or simple_time == 0
    ):
        return "N/A"
    return f"{exact_time / simple_time:.2f}x"


def format_add_err(exact_result, simple_result):
    exact_avg = exact_result["avg_cost"]
    simple_avg = simple_result["avg_cost"]
    if not is_available(exact_avg) or not is_available(simple_avg):
        return "N/A"
    return f"{simple_avg - exact_avg:.4f}"


def print_table(rows):
    print()
    print(
        "  N    | ExactT(ms) | SimpleT(ms) | Phases | Speedup | "
        "ExactAvg | SimpleAvg | AvgAddErr"
    )
    print("-------|------------|-------------|--------|---------|----------|-----------|----------")

    for row in rows:
        n = row["n"]
        exact = row["results"]["Exact"]
        simple = row["results"]["Simple"]

        print(
            f"{n:>6,} | "
            f"{format_time(exact['time_ms']):>10} | "
            f"{format_time(simple['time_ms']):>11} | "
            f"{format_phases(simple['phases']):>6} | "
            f"{format_speedup(exact, simple):>7} | "
            f"{format_avg_cost(exact['avg_cost']):>8} | "
            f"{format_avg_cost(simple['avg_cost']):>9} | "
            f"{format_add_err(exact, simple):>9}"
        )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    print("Experiment 8: Simple vs Exact, Synthetic 2D", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Seed: {SEED}", flush=True)
    print(f"N values: {', '.join(f'{n:,}' for n in N_VALUES)}", flush=True)
    print(f"Warmup runs: {WARMUP_RUNS}  Timed runs: {TIMED_RUNS}", flush=True)
    print(f"Methods: {', '.join(METHODS)}", flush=True)

    rows = []
    for n in N_VALUES:
        print(f"\nPreparing Synthetic 2D data for N={n:,}...", flush=True)
        try:
            P_red, P_blue = generate_synthetic_2d(n, device)
        except Exception as exc:
            print(f"Warning: N={n:,} data generation failed: {exc}", flush=True)
            rows.append(
                {
                    "n": n,
                    "results": {method_name: result_na() for method_name in METHODS},
                }
            )
            empty_cache_if_cuda(device)
            continue

        print(f"Generated Synthetic 2D data for N={n:,}.", flush=True)
        results = {}
        for method_name in METHODS:
            results[method_name] = run_method(n, method_name, P_red, P_blue, device)

        rows.append({"n": n, "results": results})
        del P_red, P_blue
        empty_cache_if_cuda(device)

    print_table(rows)


if __name__ == "__main__":
    main()
