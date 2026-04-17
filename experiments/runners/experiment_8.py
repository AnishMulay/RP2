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
from clustered_push_relabel.clustering.simple import SimpleClustering


SYNTHETIC_DIM = 2
SEED = 42
BATCH_SIZE = 512
WARMUP_RUNS = 0
TIMED_RUNS = 1
METHODS = ("Exact", "ProxyExact", "Simple")
SAMPLE_FACTOR_DEFAULT = 1.0


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


def build_proxy_matrix(clustering, N, epsilon):
    """
    Build an (N, N) proxy cost matrix from SimpleClustering output.

    For every (b, a) pair the proxy cost is:
        d_min_b[b] + DR[nearest_s[b], a]   (triangle proxy through nearest center)

    For pairs where a is in b's adjacency list, overwrite with the exact
    stored float distance adj_dist_float[entry].

    NOTE: Allocates an (N x N) float32 tensor on GPU. Only valid for N up to
    roughly 10,000 (400 MB at float32). Do not use for larger N.

    Returns a CPU float64 numpy array of shape (N, N) ready for ot.emd.
    """
    device = clustering["DR"].device
    DR = clustering["DR"]           # (S, N) float
    d_min_b = clustering["d_min_b"] # (N,)   float
    nearest_s = clustering["nearest_s"]  # (N,) int64
    adj_ptr = clustering["adj_ptr"]      # (N+1,) int64
    adj_col = clustering["adj_col"]      # (M,) int64
    adj_dist_float = clustering["adj_dist_float"]  # (M,) float

    # Triangle proxy for all (b, a): d_min_b[b] + DR[nearest_s[b], a]
    # DR[nearest_s, :] selects one row of DR per blue -> (N, N)
    C = (d_min_b.unsqueeze(1) + DR[nearest_s, :]).T  # (N_red × N_blue)

    # Overwrite adjacency-list entries with exact stored distances
    if adj_col.numel() > 0:
        b_indices = torch.repeat_interleave(
            torch.arange(N, device=device, dtype=torch.long),
            adj_ptr[1:] - adj_ptr[:-1],
        )
        C[adj_col, b_indices] = adj_dist_float

    return C.cpu().to(torch.float64).numpy()


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


def benchmark_proxy_exact(P_red, P_blue, device, epsilon, sample_factor):
    if ot is None:
        raise RuntimeError("POT is not installed; exact solver unavailable.")

    P_red_norm, P_blue_norm, diameter = normalize_points(P_red, P_blue)
    N = P_red_norm.shape[0]

    cluster_engine = SimpleClustering(
        epsilon=epsilon, tile_size=BATCH_SIZE, sample_factor=sample_factor
    )
    clustering = cluster_engine.run(P_red_norm, P_blue_norm)
    if device.type == "cuda":
        torch.cuda.synchronize()

    C = build_proxy_matrix(clustering, N, epsilon)

    a = np.full(N, 1.0 / N, dtype=np.float64)
    b = np.full(N, 1.0 / N, dtype=np.float64)

    t0 = time.perf_counter()
    plan = ot.emd(a, b, C, numItermax=10**6)
    t1 = time.perf_counter()

    match_B = matching_from_plan(plan)
    total_cost, avg_cost = matching_costs(P_red_norm, P_blue_norm, match_B)
    total_cost *= diameter
    avg_cost *= diameter

    del plan, match_B, C, clustering, cluster_engine
    return (t1 - t0) * 1000.0, total_cost, avg_cost


def benchmark_simple(P_red, P_blue, device, epsilon, sample_factor):
    P_red_norm, P_blue_norm, diameter = normalize_points(P_red, P_blue)

    for _ in range(WARMUP_RUNS):
        empty_cache_if_cuda(device)
        solver = SimpleGPUSolver(
            P_red_norm,
            P_blue_norm,
            epsilon,
            batch_size=BATCH_SIZE,
            verbose=False,
            diameter=diameter,
            sample_factor=sample_factor,
        )
        synchronize_if_cuda(device)
        solver.solve()
        synchronize_if_cuda(device)
        del solver

    empty_cache_if_cuda(device)
    solver = SimpleGPUSolver(
        P_red_norm,
        P_blue_norm,
        epsilon,
        batch_size=BATCH_SIZE,
        verbose=False,
        diameter=diameter,
        sample_factor=sample_factor,
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
    solver.verbose = True
    feasibility = solver.verify_solution()
    solver.verbose = False
    del solver, match_B, P_red_norm, P_blue_norm
    return (t1 - t0) * 1000.0, total_cost, avg_cost, phases, feasibility


def result_na():
    return {
        "time_ms": math.nan,
        "total_cost": math.nan,
        "avg_cost": math.nan,
        "phases": math.nan,
        "feasibility": None,
    }


def run_method(n, method_name, P_red, P_blue, device, epsilon, sample_factor):
    try:
        print(f"  Running {method_name} for N={n:,}...", flush=True)
        phases = math.nan
        feasibility = None
        if method_name == "Exact":
            time_ms, total_cost, avg_cost = benchmark_exact(P_red, P_blue)
        elif method_name == "ProxyExact":
            time_ms, total_cost, avg_cost = benchmark_proxy_exact(
                P_red, P_blue, device, epsilon, sample_factor
            )
        elif method_name == "Simple":
            time_ms, total_cost, avg_cost, phases, feasibility = benchmark_simple(
                P_red, P_blue, device, epsilon, sample_factor
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
            "feasibility": feasibility,
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
        "  N    | ExactT(ms) | ProxyT(ms) | SimpleT(ms) | Phases | Speedup | "
        "ExactAvg | ProxyAvg | SimpleAvg | ProxyAddErr | SimpleAddErr"
    )
    print(
        "-------|------------|------------|-------------|--------|---------|"
        "----------|----------|-----------|-------------|-------------"
    )

    for row in rows:
        n = row["n"]
        exact = row["results"]["Exact"]
        proxy = row["results"]["ProxyExact"]
        simple = row["results"]["Simple"]

        print(
            f"{n:>6,} | "
            f"{format_time(exact['time_ms']):>10} | "
            f"{format_time(proxy['time_ms']):>10} | "
            f"{format_time(simple['time_ms']):>11} | "
            f"{format_phases(simple['phases']):>6} | "
            f"{format_speedup(exact, simple):>7} | "
            f"{format_avg_cost(exact['avg_cost']):>8} | "
            f"{format_avg_cost(proxy['avg_cost']):>8} | "
            f"{format_avg_cost(simple['avg_cost']):>9} | "
            f"{format_add_err(exact, proxy):>11} | "
            f"{format_add_err(proxy, simple):>11}"
        )


def print_feasibility_table(rows):
    print()
    print("Feasibility Verification Table (Simple solver)")
    print(
        "  N    | Ch1 Violations | Ch1 WorstSlack | "
        "Ch2 Total | Ch2 Violations | Ch2 SlackRange | "
        "Ch3 Total | Ch3 Violations | Ch3 SlackRange"
    )
    print("-" * 120)

    for row in rows:
        n = row["n"]
        f = row["results"]["Simple"].get("feasibility")
        if f is None:
            print(f"{n:>6,} | {'N/A':>14} | {'N/A':>14} | {'N/A':>9} | {'N/A':>14} | {'N/A':>14} | {'N/A':>9} | {'N/A':>14} | {'N/A':>14}")
            continue

        def fmt_range(mn, mx):
            if mn is None or mx is None:
                return "N/A"
            return f"[{mn}, {mx}]"

        print(
            f"{n:>6,} | "
            f"{f['check1_violations']:>14} | "
            f"{f['check1_worst_slack']:>14} | "
            f"{f['check2_total']:>9} | "
            f"{f['check2_violations']:>14} | "
            f"{fmt_range(f['check2_min'], f['check2_max']):>14} | "
            f"{f['check3_total']:>9} | "
            f"{f['check3_violations']:>14} | "
            f"{fmt_range(f['check3_min'], f['check3_max']):>14}"
        )


def pick(prompt, options):
    """
    Display a numbered menu and return the user's chosen value.
    options is a list of (label, value) tuples.
    Loops until a valid integer is entered.
    """
    print(f"\n{prompt}")
    for i, (label, _) in enumerate(options, 1):
        print(f"  {i}. {label}")
    while True:
        raw = input("Enter number: ").strip()
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(options):
                chosen_label, chosen_value = options[idx - 1]
                print(f"  → {chosen_label}")
                return chosen_value
        print(f"  Please enter a number between 1 and {len(options)}.")


def configure_experiment():
    """
    Interactive CLI picker for experiment configuration.
    Returns (n_values, epsilon, sample_factor).
    """
    print("\n" + "=" * 50)
    print("  Experiment 8 Configuration")
    print("=" * 50)

    n_values = pick(
        "Select N values:",
        [
            ("1,000 → 10,000 in steps of 1,000",
             list(range(1_000, 11_000, 1_000))),
            ("5,000 only",
             [5_000]),
        ],
    )

    epsilon = pick(
        "Select epsilon:",
        [
            ("0.01",  0.01),
            ("0.005", 0.005),
            ("0.001", 0.001),
        ],
    )

    sample_factor = pick(
        "Select sampling probability:",
        [
            ("1 / sqrt(N)    (default)",  SAMPLE_FACTOR_DEFAULT),
            ("1 / (2*sqrt(N))  (reduced)", 2.0),
        ],
    )

    print("\nConfiguration:")
    print(f"  N values      : {n_values}")
    print(f"  Epsilon       : {epsilon}")
    print(f"  Sample factor : 1 / ({int(sample_factor)}*sqrt(N))")
    print()
    return n_values, epsilon, sample_factor


def main():
    N_VALUES, EPSILON, SAMPLE_FACTOR = configure_experiment()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    print("Experiment 8: Simple vs Exact, Synthetic 2D", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Seed: {SEED}", flush=True)
    print(f"N values: {', '.join(f'{n:,}' for n in N_VALUES)}", flush=True)
    print(f"Epsilon: {EPSILON}", flush=True)
    print(f"Sample factor: 1 / ({int(SAMPLE_FACTOR)}*sqrt(N))", flush=True)
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
            results[method_name] = run_method(
                n,
                method_name,
                P_red,
                P_blue,
                device,
                epsilon=EPSILON,
                sample_factor=SAMPLE_FACTOR,
            )

        rows.append({"n": n, "results": results})
        del P_red, P_blue
        empty_cache_if_cuda(device)

    print_table(rows)
    print_feasibility_table(rows)


if __name__ == "__main__":
    main()
