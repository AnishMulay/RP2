#!/usr/bin/env python3

import gc
import re
import sys
import time
from pathlib import Path

import torch
from torch import Tensor


BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from clustered_push_relabel.clustering.simple_three_level import ThreeLevelClustering
from clustered_push_relabel.solvers.three_level_bipartite import ThreeLevelGPUSolver


N_START = 50_000
N_STEP = 50_000
N_MAX = 1_000_000
EPSILON = 0.01
BATCH_SIZE = 2048
SEED = 42
MAX_ITERS = 100_000
TILE_SIZE = 2048
CLUSTERING_TILE_SIZE = 512


def gpu_mem_mb() -> tuple[float, float]:
    """Returns (allocated_MB, reserved_MB). Returns (0, 0) on CPU."""
    if not torch.cuda.is_available():
        return 0.0, 0.0
    return (
        torch.cuda.memory_allocated() / 1024**2,
        torch.cuda.memory_reserved() / 1024**2,
    )


def gpu_mem_str() -> str:
    alloc, res = gpu_mem_mb()
    return f"alloc={alloc:.0f}MB res={res:.0f}MB"


def synchronize_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def empty_cache_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()


def generate_uniform_2d(n: int, seed: int, device: torch.device) -> tuple[Tensor, Tensor]:
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    A = torch.rand(n, 2, device=device, dtype=torch.float32, generator=rng)
    B = torch.rand(n, 2, device=device, dtype=torch.float32, generator=rng)
    return A, B


def seed_trial(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_compact_count(value: float) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:.0f}"


def print_expected_sizes(n: int) -> None:
    s2 = n ** (1.0 / 3.0)
    s1 = n ** (2.0 / 3.0)
    n43 = n ** (4.0 / 3.0)
    print(
        f"[N={n}] expected: S2≈{int(round(s2))}  "
        f"S1≈{int(round(s1))}  "
        f"MA1≈{format_compact_count(n43)}  "
        f"MB≈{format_compact_count(n43)}  "
        f"DR≈{format_compact_count(n43)}",
        flush=True,
    )


def print_mem_boundary(n: int, label: str) -> tuple[float, float]:
    alloc, res = gpu_mem_mb()
    print(f"[N={n}] {label:<20} alloc={alloc:.0f}MB res={res:.0f}MB", flush=True)
    return alloc, res


def sanitize_reason(exc: BaseException) -> str:
    text = str(exc).strip().lower()
    if "max_iters" in text:
        return "max_iters"
    if "unmatched" in text:
        return "unmatched"
    if not text:
        return exc.__class__.__name__.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text[:32] or exc.__class__.__name__.lower()


def fmt_int(value: int | None, width: int) -> str:
    if value is None:
        return f"{'N/A':>{width}}"
    return f"{value:>{width},}"


def fmt_float(value: float | None, width: int, digits: int = 1) -> str:
    if value is None:
        return f"{'N/A':>{width}}"
    return f"{value:>{width}.{digits}f}"


def fmt_sci(value: int | None, width: int) -> str:
    if value is None:
        return f"{'N/A':>{width}}"
    return f"{value:>{width}.2e}"


def run_trial(n: int, device: torch.device) -> tuple[dict[str, object], bool]:
    result: dict[str, object] = {
        "n": n,
        "S1": None,
        "S2": None,
        "MA1": None,
        "MB": None,
        "t_cluster_ms": None,
        "t_solve_ms": None,
        "mem_after_cluster_mb": None,
        "mem_after_solve_mb": None,
        "iters": None,
        "status": "pending",
    }

    solver = None
    A = None
    B = None
    oom_detected = False
    oom_stage = "clustering"

    gc.collect()
    empty_cache_if_cuda(device)
    synchronize_if_cuda(device)

    print("-" * 73, flush=True)
    print(f"N = {n:>10,}", flush=True)
    print_expected_sizes(n)
    print_mem_boundary(n, "mem_before_trial:")

    try:
        seed_trial(SEED)
        A, B = generate_uniform_2d(n, SEED, device)

        print(f"[N={n}] clustering...", flush=True)
        synchronize_if_cuda(device)
        t0 = time.perf_counter()
        solver = ThreeLevelGPUSolver(
            A,
            B,
            epsilon=EPSILON,
            batch_size=BATCH_SIZE,
            tile_size=TILE_SIZE,
            clustering_tile_size=CLUSTERING_TILE_SIZE,
            verbose=False,
            max_iters=MAX_ITERS,
            clustering_class=ThreeLevelClustering,
        )
        synchronize_if_cuda(device)
        result["t_cluster_ms"] = (time.perf_counter() - t0) * 1000.0
        _, mem_after_cluster_res = print_mem_boundary(n, "mem_after_cluster:")
        result["mem_after_cluster_mb"] = mem_after_cluster_res

        solver.debug_audit = False
        result["S1"] = int(solver.S1)
        result["S2"] = int(solver.S2)
        result["MB"] = int(solver.adj_B_ptr[-1].item())
        result["MA1"] = int(solver.adj_A1_ptr[-1].item())

        if int(solver.nearest_s1.shape[0]) != n:
            raise RuntimeError(
                f"nearest_s1 sanity failed: got {solver.nearest_s1.shape[0]}, expected {n}"
            )

        print(
            f"[N={n}] S1={result['S1']:,}  S2={result['S2']:,}  "
            f"MB={result['MB']:,}  MA1={result['MA1']:,}",
            flush=True,
        )

        oom_stage = "solve"
        print(f"[N={n}] solving...", flush=True)
        synchronize_if_cuda(device)
        t0 = time.perf_counter()
        solver.solve()
        synchronize_if_cuda(device)
        result["t_solve_ms"] = (time.perf_counter() - t0) * 1000.0
        _, mem_after_solve_res = print_mem_boundary(n, "mem_after_solve:")
        result["mem_after_solve_mb"] = mem_after_solve_res

        unmatched = int((solver.match_B == -1).sum().item())
        matched = n - unmatched
        if unmatched > EPSILON * n:
            raise RuntimeError(
                f"unmatched={unmatched} exceeds epsilon*N={EPSILON * n:.0f}"
            )

        result["iters"] = int(solver.iterations)
        result["status"] = "ok"
        print(
            f"[N={n}] iters={result['iters']}  matched={matched}/{n}  status=ok",
            flush=True,
        )
    except torch.cuda.OutOfMemoryError:
        oom_detected = True
        result["status"] = "oom_cluster" if oom_stage == "clustering" else "oom_solve"
        print(f"[N={n}] CUDA OOM during {oom_stage}.", flush=True)
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            oom_detected = True
            result["status"] = "oom_cluster" if oom_stage == "clustering" else "oom_solve"
            print(f"[N={n}] CUDA OOM during {oom_stage}.", flush=True)
        else:
            reason = sanitize_reason(exc)
            result["status"] = f"skip:{reason}"
            print(f"[N={n}] SKIP: {exc}", flush=True)
    except Exception as exc:
        reason = sanitize_reason(exc)
        result["status"] = f"skip:{reason}"
        print(f"[N={n}] SKIP: {exc}", flush=True)
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
        gc.collect()
        empty_cache_if_cuda(device)
        synchronize_if_cuda(device)
        alloc_cleanup, res_cleanup = gpu_mem_mb()
        print(
            f"[N={n}] mem_after_cleanup:  alloc={alloc_cleanup:.0f}MB "
            f"res={res_cleanup:.0f}MB",
            flush=True,
        )

    return result, oom_detected


def print_results_table(rows: list[dict[str, object]]) -> None:
    print()
    print("Results table")
    print("Memory columns use reserved MB; per-trial logs show alloc/res.")

    widths = {
        "n": 9,
        "s1": 8,
        "s2": 8,
        "ma1": 12,
        "mb": 12,
        "t_cluster": 12,
        "t_solve": 12,
        "mem_cluster": 12,
        "mem_solve": 12,
        "iters": 8,
    }

    headers = [
        ("N", widths["n"]),
        ("S1", widths["s1"]),
        ("S2", widths["s2"]),
        ("MA1", widths["ma1"]),
        ("MB", widths["mb"]),
        ("t_cluster_ms", widths["t_cluster"]),
        ("t_solve_ms", widths["t_solve"]),
        ("mem_after_cluster_MB", widths["mem_cluster"]),
        ("mem_after_solve_MB", widths["mem_solve"]),
        ("iters", widths["iters"]),
        ("status", len("mem_after_solve_MB")),
    ]

    header_line = " | ".join(f"{label:>{width}}" for label, width in headers[:-1])
    header_line = f"{header_line} | status"
    separator = "-+-".join("-" * width for _, width in headers[:-1]) + "-+-" + "-" * len("status")

    print(header_line)
    print(separator)
    for row in rows:
        cells = [
            fmt_int(row["n"], widths["n"]),
            fmt_int(row["S1"], widths["s1"]),
            fmt_int(row["S2"], widths["s2"]),
            fmt_sci(row["MA1"], widths["ma1"]),
            fmt_sci(row["MB"], widths["mb"]),
            fmt_float(row["t_cluster_ms"], widths["t_cluster"]),
            fmt_float(row["t_solve_ms"], widths["t_solve"]),
            fmt_float(row["mem_after_cluster_mb"], widths["mem_cluster"], digits=0),
            fmt_float(row["mem_after_solve_mb"], widths["mem_solve"], digits=0),
            fmt_int(row["iters"], widths["iters"]),
            str(row["status"]),
        ]
        print(" | ".join(cells))


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA is required for this experiment.")
        return

    device = torch.device("cuda:0")
    print(
        f"Device: {device}  epsilon={EPSILON}  "
        f"solver_tile={TILE_SIZE}  cluster_tile={CLUSTERING_TILE_SIZE}",
        flush=True,
    )

    rows: list[dict[str, object]] = []
    for n in range(N_START, N_MAX + 1, N_STEP):
        row, oom_detected = run_trial(n, device)
        rows.append(row)
        if oom_detected:
            break

    print_results_table(rows)


if __name__ == "__main__":
    main()
