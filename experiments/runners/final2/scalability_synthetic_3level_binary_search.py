#!/usr/bin/env python3
"""
Binary-search GPU scalability limit for the 3-level solver on synthetic 2D data.

The search predicate is strict: a trial passes only when the full 3-level run
finishes without CUDA out-of-memory. A Markdown report is refreshed after each
trial so partial results survive long runs, scheduler kills, and OOM failures.
"""

import argparse
import gc
import json
import math
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from clustered_push_relabel.solvers.three_level_bipartite import ThreeLevelGPUSolver

DEFAULT_LOW = 0
DEFAULT_HIGH = 2_000_000
DEFAULT_EPSILON = 0.01
DEFAULT_BATCH_SIZE = 2048
DEFAULT_CLUSTERING_TILE_SIZE = 512
DEFAULT_MAX_ITERS = 999_999_999
DEFAULT_SEED = 42
SYNTHETIC_2D_DIAMETER = math.sqrt(2.0)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def cleanup(device):
    gc.collect()
    if device.type == "cuda":
        try:
            torch.cuda.empty_cache()
        except RuntimeError:
            pass
        try:
            torch.cuda.ipc_collect()
        except RuntimeError:
            pass
        try:
            torch.cuda.synchronize(device)
        except RuntimeError:
            pass


def reset_peak_memory(device):
    if device.type != "cuda":
        return
    torch.cuda.reset_peak_memory_stats(device)
    try:
        torch.cuda.reset_accumulated_memory_stats(device)
    except AttributeError:
        pass


def gpu_mem_gb(device):
    if device.type != "cuda":
        return 0.0, 0.0
    return (
        torch.cuda.memory_allocated(device) / 1024**3,
        torch.cuda.memory_reserved(device) / 1024**3,
    )


def peak_gpu_mem_gb(device):
    if device.type != "cuda":
        return math.nan
    return torch.cuda.max_memory_allocated(device) / 1024**3


def is_cuda_oom(exc):
    return isinstance(exc, torch.cuda.OutOfMemoryError) or (
        isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()
    )


def is_missing(value):
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return value is None


def fmt_int(value):
    return "N/A" if is_missing(value) else f"{int(value):,}"


def fmt_seconds(value):
    return "N/A" if is_missing(value) else f"{float(value):.2f}s"


def fmt_gib(value):
    return "N/A" if is_missing(value) else f"{float(value):.2f} GiB"


def fmt_float(value, digits=6):
    return "N/A" if is_missing(value) else f"{float(value):.{digits}f}"


def fmt_status(row):
    if row["status"] == "ok":
        return "PASS"
    if row["status"] == "oom":
        return "OOM"
    if row["status"] == "error":
        return "ERROR"
    return str(row["status"]).upper()


def markdown_escape(value):
    text = str(value)
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("\n", "<br>")


def make_data(n, seed, device):
    gen_a = torch.Generator(device=device)
    gen_b = torch.Generator(device=device)
    gen_a.manual_seed(seed)
    gen_b.manual_seed(seed + 1)
    A = torch.rand(n, 2, device=device, dtype=torch.float32, generator=gen_a)
    B = torch.rand(n, 2, device=device, dtype=torch.float32, generator=gen_b)

    # Synthetic data lives in [0, 1]^2, so sqrt(2) is a cheap safe diameter.
    A = A / SYNTHETIC_2D_DIAMETER
    B = B / SYNTHETIC_2D_DIAMETER
    return A, B


def avg_matching_error(A, B, match_B):
    match_B = match_B.to(device=A.device, dtype=torch.long)
    matched_mask = match_B >= 0
    if not bool(matched_mask.any()):
        return math.nan
    matched_a = match_B[matched_mask]
    normalized = torch.norm(B[matched_mask] - A[matched_a], p=2, dim=1).mean().item()
    return normalized * SYNTHETIC_2D_DIAMETER


def base_row(args, trial, n, low, high):
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "trial": trial,
        "n": n,
        "low_before": low,
        "high_before": high,
        "low_after": low,
        "high_after": high,
        "pass": False,
        "status": "pending",
        "phase": "start",
        "cluster_time_s": math.nan,
        "solve_time_s": math.nan,
        "total_time_s": math.nan,
        "avg_error": math.nan,
        "peak_gpu_mem_gb": math.nan,
        "mem_alloc_before_gb": math.nan,
        "mem_reserved_before_gb": math.nan,
        "mem_alloc_after_gb": math.nan,
        "mem_reserved_after_gb": math.nan,
        "s1": math.nan,
        "s2": math.nan,
        "adj_b_edges": math.nan,
        "adj_a1_edges": math.nan,
        "iterations": math.nan,
        "matched": math.nan,
        "unmatched": math.nan,
        "epsilon": args.epsilon,
        "batch_size": args.batch_size,
        "clustering_tile_size": args.clustering_tile_size,
        "max_iters": args.max_iters,
        "seed": args.seed,
        "error": "",
    }


def run_trial(n, trial, low, high, args, device):
    row = base_row(args, trial, n, low, high)
    solver = None
    A = None
    B = None
    phase = "data"
    t_cluster = math.nan
    t_solve = math.nan

    cleanup(device)
    reset_peak_memory(device)
    row["mem_alloc_before_gb"], row["mem_reserved_before_gb"] = gpu_mem_gb(device)

    print("-" * 80, flush=True)
    print(
        f"Trial {trial}: N={n:,}  search=[{low:,}, {high:,}]  "
        f"mem_before={row['mem_alloc_before_gb']:.2f}/{row['mem_reserved_before_gb']:.2f} GiB",
        flush=True,
    )

    try:
        with torch.no_grad():
            A, B = make_data(n, args.seed, device)

            phase = "clustering"
            sync(device)
            t0 = time.perf_counter()
            solver = ThreeLevelGPUSolver(
                A,
                B,
                epsilon=args.epsilon,
                batch_size=args.batch_size,
                tile_size=args.batch_size,
                clustering_tile_size=args.clustering_tile_size,
                verbose=args.verbose,
                diameter=1.0,
                max_iters=args.max_iters,
                sample_factor=args.sample_factor,
                set1_pair_batch=args.set1_pair_batch,
            )
            sync(device)
            t_cluster = time.perf_counter() - t0
            row["cluster_time_s"] = t_cluster
            row["s1"] = int(solver.S1)
            row["s2"] = int(solver.S2)
            row["adj_b_edges"] = int(solver.adj_B_ptr[-1].item())
            row["adj_a1_edges"] = int(solver.adj_A1_ptr[-1].item())
            print(
                f"  clustering ok: {t_cluster:.2f}s  "
                f"S1={row['s1']:,} S2={row['s2']:,} "
                f"AdjB={row['adj_b_edges']:,} AdjA1={row['adj_a1_edges']:,}",
                flush=True,
            )

            phase = "solve"
            t0 = time.perf_counter()
            match_B = solver.solve()
            sync(device)
            t_solve = time.perf_counter() - t0
            row["solve_time_s"] = t_solve
            row["iterations"] = int(solver.iterations)

            phase = "accuracy"
            if match_B is None:
                match_B = solver.match_B
            row["avg_error"] = avg_matching_error(A, B, match_B)
            row["unmatched"] = int((match_B == -1).sum().item())
            row["matched"] = int(n - row["unmatched"])
            row["total_time_s"] = t_cluster + t_solve
            row["peak_gpu_mem_gb"] = peak_gpu_mem_gb(device)
            row["pass"] = True
            row["status"] = "ok"
            row["phase"] = "done"
            print(
                f"  solve ok: {t_solve:.2f}s  total={row['total_time_s']:.2f}s  "
                f"avg_error={row['avg_error']:.6f}  peak={row['peak_gpu_mem_gb']:.2f} GiB",
                flush=True,
            )
    except Exception as exc:
        row["phase"] = phase
        row["peak_gpu_mem_gb"] = peak_gpu_mem_gb(device)
        if is_cuda_oom(exc):
            row["status"] = "oom"
            row["error"] = str(exc).splitlines()[0][:500]
            print(f"  CUDA OOM during {phase}; peak={row['peak_gpu_mem_gb']:.2f} GiB", flush=True)
        else:
            row["status"] = "error"
            row["error"] = "".join(traceback.format_exception_only(type(exc), exc)).strip()[:500]
            print(f"  ERROR during {phase}: {row['error']}", flush=True)
    finally:
        if solver is not None:
            del solver
            solver = None
        if A is not None:
            del A
            A = None
        if B is not None:
            del B
            B = None
        cleanup(device)
        row["mem_alloc_after_gb"], row["mem_reserved_after_gb"] = gpu_mem_gb(device)
        print(
            f"  cleanup: mem_after={row['mem_alloc_after_gb']:.2f}/"
            f"{row['mem_reserved_after_gb']:.2f} GiB",
            flush=True,
        )

    return row


def markdown_table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(markdown_escape(cell) for cell in row) + " |")
    return "\n".join(lines)


def render_trial_table(rows):
    headers = [
        "Trial",
        "N",
        "Result",
        "Phase",
        "Search Before",
        "Search After",
        "Cluster",
        "Solve",
        "Total",
        "Avg Error",
        "Peak GPU",
        "S1",
        "S2",
        "Adj B",
        "Adj A1",
        "Iters",
        "Matched",
        "Unmatched",
    ]
    table_rows = []
    for row in rows:
        table_rows.append(
            [
                fmt_int(row["trial"]),
                fmt_int(row["n"]),
                fmt_status(row),
                row["phase"],
                f"[{fmt_int(row['low_before'])}, {fmt_int(row['high_before'])}]",
                f"[{fmt_int(row['low_after'])}, {fmt_int(row['high_after'])}]",
                fmt_seconds(row["cluster_time_s"]),
                fmt_seconds(row["solve_time_s"]),
                fmt_seconds(row["total_time_s"]),
                fmt_float(row["avg_error"]),
                fmt_gib(row["peak_gpu_mem_gb"]),
                fmt_int(row["s1"]),
                fmt_int(row["s2"]),
                fmt_int(row["adj_b_edges"]),
                fmt_int(row["adj_a1_edges"]),
                fmt_int(row["iterations"]),
                fmt_int(row["matched"]),
                fmt_int(row["unmatched"]),
            ]
        )
    return markdown_table(headers, table_rows)


def render_error_notes(rows):
    notes = []
    for row in rows:
        if row.get("error"):
            notes.append(
                [
                    fmt_int(row["trial"]),
                    fmt_int(row["n"]),
                    fmt_status(row),
                    row["phase"],
                    row["error"],
                ]
            )
    if not notes:
        return "No OOM/error details recorded yet."
    return markdown_table(["Trial", "N", "Result", "Phase", "Detail"], notes)


def render_markdown_report(state, rows):
    best = state.get("best_ok")
    first_oom = state.get("first_oom")
    complete = "yes" if state.get("complete") else "no"
    stopped = "yes" if state.get("stopped_on_error") else "no"
    args = state["args"]

    summary_rows = [
        ["Updated", datetime.now().isoformat(timespec="seconds")],
        ["Trials completed", fmt_int(state["trial_count"])],
        ["Complete", complete],
        ["Stopped on non-OOM error", stopped],
        ["Best passing N", "N/A" if best is None else fmt_int(best)],
        ["First OOM N seen", "N/A" if first_oom is None else fmt_int(first_oom)],
        ["Current search interval", f"[{fmt_int(state['low'])}, {fmt_int(state['high'])}]"],
        ["Initial search interval", f"[{fmt_int(state['initial_low'])}, {fmt_int(state['initial_high'])}]"],
    ]

    config_rows = [
        ["epsilon", args["epsilon"]],
        ["batch_size", fmt_int(args["batch_size"])],
        ["clustering_tile_size", fmt_int(args["clustering_tile_size"])],
        ["max_iters", fmt_int(args["max_iters"])],
        ["seed", fmt_int(args["seed"])],
        ["sample_factor", args["sample_factor"]],
        ["set1_pair_batch", fmt_int(args["set1_pair_batch"])],
    ]

    lines = [
        "# Synthetic 2D 3-Level Scalability Binary Search",
        "",
        "Pass condition: the full 3-level GPU run completes without CUDA out-of-memory.",
        "The report is rewritten atomically after every trial, so it remains readable during long runs.",
        "",
        "## Search State",
        "",
        markdown_table(["Metric", "Value"], summary_rows),
        "",
        "## Configuration",
        "",
        markdown_table(["Setting", "Value"], config_rows),
        "",
        "## Trial Results",
        "",
        render_trial_table(rows) if rows else "No trials have completed yet.",
        "",
        "## OOM And Error Details",
        "",
        render_error_notes(rows),
        "",
        "## Memory Safeguards",
        "",
        "- `gc.collect()` and `torch.cuda.empty_cache()` run before and after every trial.",
        "- `torch.cuda.ipc_collect()` is attempted after every trial.",
        "- CUDA synchronization is used around timed regions and cleanup.",
        "- Peak CUDA memory stats are reset at the start of every trial.",
        "",
    ]
    return "\n".join(lines)


def write_markdown_report(path, state, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w") as f:
        f.write(render_markdown_report(state, rows))
        f.flush()
        os.fsync(f.fileno())
    tmp_path.replace(path)


def write_checkpoint(path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w") as f:
        json.dump(state, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    tmp_path.replace(path)


def update_bounds(row, low, high):
    if row["status"] == "ok":
        low = int(row["n"])
    elif row["status"] == "oom":
        high = int(row["n"]) - 1
    row["low_after"] = low
    row["high_after"] = high
    return low, high


def print_search_summary(state):
    best = state.get("best_ok")
    first_oom = state.get("first_oom")
    print("\nBinary Search Summary", flush=True)
    print(f"  trials: {state['trial_count']}", flush=True)
    print(f"  best passing N: {best if best is not None else 'N/A'}", flush=True)
    print(f"  first OOM N seen: {first_oom if first_oom is not None else 'N/A'}", flush=True)
    print(f"  final search interval: [{state['low']:,}, {state['high']:,}]", flush=True)
    print(f"  Markdown: {state['markdown_path']}", flush=True)
    print(f"  checkpoint: {state['checkpoint_path']}", flush=True)


def binary_search(args, device):
    low = int(args.low)
    high = int(args.high)
    if low < 0:
        raise ValueError("--low must be non-negative")
    if high < low:
        raise ValueError("--high must be >= --low")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(args.results_dir)
    markdown_path = Path(args.output) if args.output else results_dir / f"synthetic_3level_binary_search_{stamp}.md"
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else markdown_path.with_suffix(".json")

    state = {
        "low": low,
        "high": high,
        "initial_low": low,
        "initial_high": high,
        "best_ok": None if low == 0 else low,
        "first_oom": None,
        "trial_count": 0,
        "complete": False,
        "stopped_on_error": False,
        "markdown_path": str(markdown_path),
        "checkpoint_path": str(checkpoint_path),
        "args": vars(args),
    }

    print(
        f"Device: {device}  GPU: {torch.cuda.get_device_name(device)}  "
        f"epsilon={args.epsilon}  range=[{low:,}, {high:,}]",
        flush=True,
    )
    print(f"Writing every trial to: {markdown_path}", flush=True)

    rows = []
    write_markdown_report(markdown_path, state, rows)
    while low < high:
        trial = state["trial_count"] + 1
        mid = (low + high + 1) // 2
        row = run_trial(mid, trial, low, high, args, device)

        if row["status"] in {"ok", "oom"}:
            low, high = update_bounds(row, low, high)
            if row["status"] == "ok":
                state["best_ok"] = int(row["n"])
            elif state["first_oom"] is None or int(row["n"]) < int(state["first_oom"]):
                state["first_oom"] = int(row["n"])
        else:
            row["low_after"] = low
            row["high_after"] = high
            state["stopped_on_error"] = True

        rows.append(row)
        state["trial_count"] = trial
        state["low"] = low
        state["high"] = high
        state["complete"] = low >= high and not state["stopped_on_error"]
        write_markdown_report(markdown_path, state, rows)
        write_checkpoint(checkpoint_path, state)

        print(
            f"  persisted trial {trial}; status={row['status']}  "
            f"new_search=[{low:,}, {high:,}]",
            flush=True,
        )

        if row["status"] == "error":
            print("Stopping because a non-OOM error occurred.", flush=True)
            break

    print_search_summary(state)
    return state


def parse_args():
    parser = argparse.ArgumentParser(
        description="Find the largest synthetic 2D N that the 3-level GPU solver can run without CUDA OOM."
    )
    parser.add_argument("--low", type=int, default=DEFAULT_LOW, help="Lower search bound; treated as passing.")
    parser.add_argument("--high", type=int, default=DEFAULT_HIGH, help="Upper search bound.")
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--clustering-tile-size", type=int, default=DEFAULT_CLUSTERING_TILE_SIZE)
    parser.add_argument("--max-iters", type=int, default=DEFAULT_MAX_ITERS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--sample-factor", type=float, default=1.0)
    parser.add_argument("--set1-pair-batch", type=int, default=64)
    parser.add_argument("--results-dir", default=str(SCRIPT_DIR / "results"))
    parser.add_argument("--output", default=None, help="Optional Markdown output path.")
    parser.add_argument("--checkpoint", default=None, help="Optional JSON checkpoint path.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        print("CUDA is required for this experiment.", flush=True)
        return 1

    device = torch.device("cuda:0")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    state = binary_search(args, device)
    return 1 if state["stopped_on_error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
