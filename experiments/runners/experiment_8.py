#!/usr/bin/env python3

import inspect
import math
import multiprocessing as mp
import pathlib
import queue
import statistics
import sys
import time
import traceback

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

from clustered_push_relabel.solvers.color_aware_bipartite import (
    ColorAwareTwoLevelSolver,
)
from clustered_push_relabel.solvers.simple_bipartite import SimpleGPUSolver


N_VALUES = [10_000, 20_000, 50_000]
EPSILON = 0.01
SYNTHETIC_DIM = 2
SEED = 42
BATCH_SIZE = 512
WARMUP_RUNS = 1
TIMED_RUNS = 3
EXACT_TIMEOUT_SECONDS = 600
METHODS = ["Exact", "ColorAware", "Simple"]
PHASE_METHODS = ["ColorAware", "Simple"]

SIMPLE_PHASE_KEYS = [
    "t_group",
    "set1_v_lookup",
    "set2_adj_scan",
    "proposal",
    "conflict_match",
    "dual_v_update",
]

SIMPLE_PHASE_LABELS = [
    ("t_group", "t-compute + group"),
    ("set1_v_lookup", "Set1 V-lookup"),
    ("set2_adj_scan", "Set2 adj-scan"),
    ("proposal", "proposal"),
    ("conflict_match", "conflict + match"),
    ("dual_v_update", "dual + V-update"),
]


def generate_synthetic_2d(n, device):
    red = torch.rand((n, SYNTHETIC_DIM), dtype=torch.float32)
    blue = torch.rand((n, SYNTHETIC_DIM), dtype=torch.float32)
    P_red = red.to(device)
    P_blue = blue.to(P_red.device)
    return P_red, P_blue


def color_aware_normalized_inputs(P_red, P_blue):
    # ColorAwareTwoLevelSolver divides both point sets by the L2 diameter of
    # the joint bounding box before clustering. SimpleGPUSolver does not do
    # that internally, so Simple receives the same normalized coordinates.
    P_all = torch.cat([P_red, P_blue], dim=0)
    delta = (
        (P_all.max(dim=0).values - P_all.min(dim=0).values).pow(2).sum()
    ).sqrt()
    delta = delta.clamp(min=1e-8)
    return P_red.float() / delta, P_blue.float() / delta, delta


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


def average_l2_matching_cost(P_red, P_blue, matching):
    matched_red = P_red[matching.to(device=P_red.device, dtype=torch.long)]
    return torch.norm(P_blue - matched_red, p=2, dim=1).mean().item()


def solver_matching(solver):
    if hasattr(solver, "match_B"):
        return solver.match_B
    return solver.MB


def _benchmark_exact_impl(red_cpu, blue_cpu):
    if ot is None:
        raise RuntimeError("POT is not installed; exact solver unavailable.")

    n = red_cpu.shape[0]
    C = compute_cost_matrix_L2(red_cpu, blue_cpu)
    a = np.full(n, 1.0 / n, dtype=np.float64)
    b = np.full(n, 1.0 / n, dtype=np.float64)

    for _ in range(WARMUP_RUNS):
        plan = ot.emd(a, b, C, numItermax=10**6)
        del plan

    times_ms = []
    costs = []
    for _ in range(TIMED_RUNS):
        t0 = time.perf_counter()
        plan = ot.emd(a, b, C, numItermax=10**6)
        t1 = time.perf_counter()
        match_B = matching_from_plan(plan)
        costs.append(average_l2_matching_cost(red_cpu, blue_cpu, match_B))
        times_ms.append((t1 - t0) * 1000.0)
        del plan, match_B

    return statistics.median(times_ms), statistics.median(costs)


def _exact_worker(red_np, blue_np, result_queue):
    try:
        red_cpu = torch.from_numpy(red_np).float()
        blue_cpu = torch.from_numpy(blue_np).float()
        time_ms, cost = _benchmark_exact_impl(red_cpu, blue_cpu)
        result_queue.put({"ok": True, "time_ms": time_ms, "cost": cost})
    except BaseException as exc:
        result_queue.put(
            {
                "ok": False,
                "error": repr(exc),
                "traceback": traceback.format_exc(limit=8),
            }
        )


def benchmark_exact(P_red, P_blue):
    red_np = P_red.detach().cpu().numpy().astype(np.float32, copy=True)
    blue_np = P_blue.detach().cpu().numpy().astype(np.float32, copy=True)

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue(maxsize=1)
    process = ctx.Process(target=_exact_worker, args=(red_np, blue_np, result_queue))
    process.start()
    process.join(EXACT_TIMEOUT_SECONDS)

    if process.is_alive():
        process.terminate()
        process.join()
        raise TimeoutError(
            f"Exact solver exceeded {EXACT_TIMEOUT_SECONDS}s timeout."
        )

    try:
        result = result_queue.get_nowait()
    except queue.Empty:
        if process.exitcode != 0:
            raise RuntimeError(f"Exact solver process exited with {process.exitcode}.")
        raise RuntimeError("Exact solver process produced no result.")

    if not result["ok"]:
        raise RuntimeError(result["error"])
    return result["time_ms"], result["cost"]


def _iteration_increment_line(method):
    source_lines, start_line = inspect.getsourcelines(method)
    for offset, line in enumerate(source_lines):
        if "iteration += 1" in line:
            return start_line + offset
    return None


def solve_with_phase_counter(solver):
    target_code = solver.solve.__code__
    target_line = _iteration_increment_line(solver.solve)
    if target_line is None:
        solver.solve()
        return getattr(solver, "iterations", math.nan)

    phases = 0

    def local_trace(frame, event, arg):
        nonlocal phases
        if event == "line" and frame.f_lineno == target_line:
            phases += 1
        return local_trace

    def global_trace(frame, event, arg):
        if event == "call" and frame.f_code is target_code:
            return local_trace
        return None

    old_trace = sys.gettrace()
    sys.settrace(global_trace)
    try:
        solver.solve()
    finally:
        sys.settrace(old_trace)

    return getattr(solver, "iterations", phases)


class ProfiledSimpleGPUSolver(SimpleGPUSolver):
    def _sync(self):
        synchronize_if_cuda(self.device)

    def _add_timing(self, phase, key, start):
        self._sync()
        phase[key] += time.perf_counter() - start

    def _set1_groups_profiled(self, B_free, phase):
        self._sync()
        t0 = time.perf_counter()
        free_s = self.nearest_s[B_free]
        free_t = 1 - self.y_B[B_free] - self.d_min_b_int[B_free]

        order = torch.argsort(free_s)
        sorted_pairs = torch.stack(
            (free_s[order], free_t[order].to(torch.long)),
            dim=1,
        )
        unique_pairs, inverse_sorted = torch.unique(
            sorted_pairs, dim=0, return_inverse=True
        )

        pair_inverse = torch.empty_like(inverse_sorted)
        pair_inverse[order] = inverse_sorted
        self._add_timing(phase, "t_group", t0)

        self._sync()
        t0 = time.perf_counter()
        num_pairs = unique_pairs.shape[0]
        set1_counts = torch.empty(num_pairs, device=self.device, dtype=torch.long)
        set1_value_parts = []
        for start in range(0, unique_pairs.shape[0], self.set1_pair_batch):
            end = min(start + self.set1_pair_batch, unique_pairs.shape[0])
            s = unique_pairs[start:end, 0].to(torch.long)
            t = unique_pairs[start:end, 1].to(torch.int32)
            matches = self.V[s] == t.unsqueeze(1)
            set1_counts[start:end] = matches.sum(dim=1)

            _, a_idx = matches.nonzero(as_tuple=True)
            if a_idx.numel() != 0:
                set1_value_parts.append(a_idx)

        set1_offsets = torch.empty(num_pairs + 1, device=self.device, dtype=torch.long)
        set1_offsets[0] = 0
        set1_offsets[1:] = torch.cumsum(set1_counts, dim=0)

        if set1_value_parts:
            set1_values = torch.cat(set1_value_parts)
        else:
            set1_values = torch.empty(0, device=self.device, dtype=torch.long)
        self._add_timing(phase, "set1_v_lookup", t0)

        return pair_inverse, set1_counts, set1_offsets, set1_values

    def solve(self):
        N = self.N
        device = self.device
        B_free = torch.arange(N, device=device, dtype=torch.long)
        iteration = 0
        self.phase_timings = []

        def _print_progress(iteration, free_before, free_after, status):
            if self.verbose and iteration % 100 == 0:
                print(
                    f"[Simple iter {iteration}] free_before={free_before} "
                    f"free_after={free_after} status={status}",
                    flush=True,
                )

        def _new_phase():
            return {key: 0.0 for key in SIMPLE_PHASE_KEYS}

        def _record_phase(phase):
            self.phase_timings.append(phase)

        while True:
            num_free = B_free.numel()
            if num_free <= self.epsilon_int:
                break
            if iteration >= self.max_iters:
                break
            iteration += 1
            phase = _new_phase()

            pair_inverse, set1_counts, set1_offsets, set1_values = (
                self._set1_groups_profiled(B_free, phase)
            )
            set1_has = set1_counts[pair_inverse] > 0

            self._sync()
            t0 = time.perf_counter()
            set2_has, set2_choice = self._set2_choices(B_free)
            self._add_timing(phase, "set2_adj_scan", t0)

            self._sync()
            t0 = time.perf_counter()
            rand_pick = torch.rand(num_free, device=device) < 0.5
            choose_set1 = set1_has & (~set2_has | rand_pick)
            choose_set2 = set2_has & (~set1_has | ~rand_pick)

            proposal_a_parts = []
            proposal_b_parts = []

            b1, a1 = self._sample_set1_choices(
                B_free,
                pair_inverse,
                set1_counts,
                set1_offsets,
                set1_values,
                choose_set1,
            )
            if a1.numel() != 0:
                proposal_a_parts.append(a1)
                proposal_b_parts.append(b1)

            b2 = B_free[choose_set2]
            a2 = set2_choice[choose_set2]
            if a2.numel() != 0:
                proposal_a_parts.append(a2)
                proposal_b_parts.append(b2)
            self._add_timing(phase, "proposal", t0)

            if not proposal_a_parts:
                self._sync()
                t0 = time.perf_counter()
                self.y_B[B_free] += 1
                self._add_timing(phase, "dual_v_update", t0)
                _print_progress(iteration, num_free, num_free, "no_proposals")
                _record_phase(phase)
                continue

            proposal_a = torch.cat(proposal_a_parts)
            proposal_b = torch.cat(proposal_b_parts)

            self._sync()
            t0 = time.perf_counter()
            r_new, b_new = self._resolve_conflicts(proposal_a, proposal_b)
            self._add_timing(phase, "conflict_match", t0)

            if r_new.numel() == 0:
                self._sync()
                t0 = time.perf_counter()
                self.y_B[B_free] += 1
                self._add_timing(phase, "dual_v_update", t0)
                _print_progress(iteration, num_free, num_free, "no_accepts")
                _record_phase(phase)
                continue

            self._sync()
            t0 = time.perf_counter()
            F_B_new = self._update_matching(B_free, r_new, b_new)
            self._add_timing(phase, "conflict_match", t0)

            self._sync()
            t0 = time.perf_counter()
            self.y_B[F_B_new] += 1
            self.y_A[r_new] -= 1
            self.V[:, r_new] -= 1
            self._add_timing(phase, "dual_v_update", t0)

            _print_progress(iteration, num_free, F_B_new.numel(), "ok")
            B_free = F_B_new
            _record_phase(phase)

        self.iterations = iteration
        self.cleanup_remaining_points()
        if self.verbose:
            print(f"[Simple] Matched: {(self.match_B != -1).sum().item()}/{self.N}")
            self.calculate_final_stats()
        return self.match_B


def median_phase_breakdown(phase_timings):
    if not phase_timings:
        return None
    breakdown = {}
    for key in SIMPLE_PHASE_KEYS:
        breakdown[key] = statistics.median(
            phase[key] * 1000.0 for phase in phase_timings
        )
    breakdown["total"] = sum(breakdown[key] for key in SIMPLE_PHASE_KEYS)
    breakdown["phase_count"] = len(phase_timings)
    return breakdown


def benchmark_solver(
    P_solve_red,
    P_solve_blue,
    P_cost_red,
    P_cost_blue,
    device,
    factory,
    count_phases=False,
    collect_simple_breakdown=False,
):
    warmup_phase_count = math.nan
    for _ in range(WARMUP_RUNS):
        empty_cache_if_cuda(device)
        solver = factory()
        synchronize_if_cuda(device)
        if count_phases:
            warmup_phase_count = solve_with_phase_counter(solver)
        else:
            solver.solve()
            warmup_phase_count = getattr(solver, "iterations", math.nan)
        synchronize_if_cuda(device)
        del solver

    times_ms = []
    costs = []
    phase_counts = []
    phase_timings = []
    for _ in range(TIMED_RUNS):
        empty_cache_if_cuda(device)
        solver = factory()
        synchronize_if_cuda(device)
        t0 = time.perf_counter()
        solver.solve()
        synchronize_if_cuda(device)
        t1 = time.perf_counter()
        costs.append(average_l2_matching_cost(P_cost_red, P_cost_blue, solver_matching(solver)))
        times_ms.append((t1 - t0) * 1000.0)
        if hasattr(solver, "iterations"):
            phase_counts.append(solver.iterations)
        if collect_simple_breakdown and hasattr(solver, "phase_timings"):
            phase_timings.extend(solver.phase_timings)
        del solver

    if phase_counts:
        phase_count = statistics.median(phase_counts)
    else:
        phase_count = warmup_phase_count

    return (
        statistics.median(times_ms),
        statistics.median(costs),
        phase_count,
        median_phase_breakdown(phase_timings),
    )


def result_na():
    return {
        "time_ms": math.nan,
        "cost": math.nan,
        "status": "fail",
        "phase_count": math.nan,
        "phase_breakdown": None,
    }


def run_method(method_name, P_red, P_blue, P_red_simple, P_blue_simple, device):
    try:
        if method_name == "Exact":
            time_ms, cost = benchmark_exact(P_red, P_blue)
            return {
                "time_ms": time_ms,
                "cost": cost,
                "status": "success",
                "phase_count": math.nan,
                "phase_breakdown": None,
            }
        if method_name == "ColorAware":
            time_ms, cost, phase_count, breakdown = benchmark_solver(
                P_red,
                P_blue,
                P_red,
                P_blue,
                device,
                lambda: ColorAwareTwoLevelSolver(
                    P_red, P_blue, EPSILON, metric="L2", verbose=True
                ),
                count_phases=True,
            )
            return {
                "time_ms": time_ms,
                "cost": cost,
                "status": "success",
                "phase_count": phase_count,
                "phase_breakdown": breakdown,
            }
        if method_name == "Simple":
            time_ms, cost, phase_count, breakdown = benchmark_solver(
                P_red_simple,
                P_blue_simple,
                P_red,
                P_blue,
                device,
                lambda: ProfiledSimpleGPUSolver(
                    P_red_simple,
                    P_blue_simple,
                    EPSILON,
                    batch_size=BATCH_SIZE,
                    verbose=True,
                ),
                collect_simple_breakdown=True,
            )
            return {
                "time_ms": time_ms,
                "cost": cost,
                "status": "success",
                "phase_count": phase_count,
                "phase_breakdown": breakdown,
            }
        raise ValueError(f"Unknown method: {method_name}")
    except Exception as exc:
        print(f"Warning: {method_name} failed: {exc}", flush=True)
        empty_cache_if_cuda(device)
        return result_na()


def is_available(value):
    return value == value


def format_time(value):
    if not is_available(value):
        return "N/A"
    return f"{value:.1f} ms"


def format_cost(value):
    if not is_available(value):
        return "N/A"
    return f"{value:.4f}"


def format_ratio(value):
    if not is_available(value):
        return "N/A"
    return f"{value:.3f}"


def format_phase_count(value):
    if not is_available(value):
        return "N/A"
    return f"{int(round(value))}"


def print_method_table(rows, title, metric_key, formatter):
    print()
    print(title)
    print(f"{'N':>8} | {'Exact':>11} | {'ColorAware':>11} | {'Simple':>11}")
    print("---------|-------------|-------------|-------------")
    for row in rows:
        cells = [
            f"{row['n']:>8,}",
            f"{formatter(row['results']['Exact'][metric_key]):>11}",
            f"{formatter(row['results']['ColorAware'][metric_key]):>11}",
            f"{formatter(row['results']['Simple'][metric_key]):>11}",
        ]
        print(" | ".join(cells))


def print_ratio_table(rows):
    print()
    print("Approximation Ratio vs Exact")
    print(f"{'N':>8} | {'ColorAware':>11} | {'Simple':>11}")
    print("---------|-------------|-------------")
    printed = False
    for row in rows:
        exact_cost = row["results"]["Exact"]["cost"]
        if not is_available(exact_cost):
            continue
        ratios = {}
        for method_name in ["ColorAware", "Simple"]:
            method_cost = row["results"][method_name]["cost"]
            if is_available(method_cost):
                ratios[method_name] = method_cost / exact_cost
            else:
                ratios[method_name] = math.nan
        print(
            f"{row['n']:>8,} | "
            f"{format_ratio(ratios['ColorAware']):>11} | "
            f"{format_ratio(ratios['Simple']):>11}"
        )
        printed = True
    if not printed:
        print("No exact results available; ratios skipped.")


def print_phase_count_table(rows):
    print()
    print("Phase Count")
    print(f"{'N':>8} | {'ColorAware':>11} | {'Simple':>11}")
    print("---------|-------------|-------------")
    for row in rows:
        print(
            f"{row['n']:>8,} | "
            f"{format_phase_count(row['results']['ColorAware']['phase_count']):>11} | "
            f"{format_phase_count(row['results']['Simple']['phase_count']):>11}"
        )


def print_simple_breakdowns(rows):
    print()
    for row in rows:
        breakdown = row["results"]["Simple"]["phase_breakdown"]
        if breakdown is None:
            print(f"Simple phase breakdown  N={row['n']:,}: N/A")
            continue
        phase_count = int(breakdown["phase_count"])
        print(
            f"Simple phase breakdown  N={row['n']:,}  "
            f"(median across {phase_count} phases):"
        )
        for key, label in SIMPLE_PHASE_LABELS:
            print(f"  {label:<20}: {breakdown[key]:>7.2f} ms")
        print("  --------------------")
        print(f"  {'total per phase':<20}: {breakdown['total']:>7.2f} ms")
        print()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    print(f"Device: {device}  epsilon={EPSILON}  batch_size={BATCH_SIZE}")
    print(f"Warmup runs: {WARMUP_RUNS}  timed runs: {TIMED_RUNS}")
    print(f"Exact timeout: {EXACT_TIMEOUT_SECONDS}s")

    rows = []
    for n in N_VALUES:
        print(f"\nPreparing Synthetic N={n:,}", flush=True)
        P_red, P_blue = generate_synthetic_2d(n, device)
        P_red_simple, P_blue_simple, delta = color_aware_normalized_inputs(P_red, P_blue)
        print(
            f"  ColorAware normalization delta={float(delta.detach().cpu()):.6f}",
            flush=True,
        )
        row = {"dataset": "Synthetic", "n": n, "results": {}}

        for method_name in METHODS:
            print(f"  Running {method_name}...", flush=True)
            row["results"][method_name] = run_method(
                method_name, P_red, P_blue, P_red_simple, P_blue_simple, device
            )

        rows.append(row)
        del P_red, P_blue, P_red_simple, P_blue_simple
        empty_cache_if_cuda(device)

    print_method_table(rows, "Wall-clock Time (ms)", "time_ms", format_time)
    print_method_table(
        rows,
        "Average Matching Cost (L2 per pair)",
        "cost",
        format_cost,
    )
    print_ratio_table(rows)
    print_phase_count_table(rows)
    print_simple_breakdowns(rows)


if __name__ == "__main__":
    main()
