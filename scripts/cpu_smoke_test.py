#!/usr/bin/env python3
"""
CPU portability smoke test (rebuttal Task 4).

Runs the 2-Level pipeline (SimpleClustering + SimpleGPUSolver) and the
3-Level pipeline (ThreeLevelClustering + ThreeLevelGPUSolver) end-to-end on
CPU tensors at a small N, using the CUDA-or-CPU relaxation added to each
_validate()/__init__ device check. Reports success/failure and wall-clock
time for each pipeline; does not touch or depend on any GPU-path default
behavior.

Usage:
    python scripts/cpu_smoke_test.py [--n 8000] [--epsilon 0.02]
"""
import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch

from clustered_push_relabel.solvers.simple_bipartite import SimpleGPUSolver
from clustered_push_relabel.solvers.three_level_bipartite import ThreeLevelGPUSolver


def make_data(n, seed, device):
    g_a = torch.Generator(device=device)
    g_a.manual_seed(seed)
    g_b = torch.Generator(device=device)
    g_b.manual_seed(seed + 1)
    A = torch.rand(n, 2, device=device, dtype=torch.float32, generator=g_a)
    B = torch.rand(n, 2, device=device, dtype=torch.float32, generator=g_b)
    return A, B


def avg_cost(A, B, match_B):
    match_B = match_B.to(torch.long)
    matched = match_B >= 0
    if not bool(matched.any()):
        return float("nan")
    return torch.norm(B[matched] - A[match_B[matched]], p=2, dim=1).mean().item()


def run_pipeline(name, solver_cls, A, B, epsilon, batch_size, extra_kwargs=None):
    extra_kwargs = extra_kwargs or {}
    print(f"\n--- {name} (CPU, N={A.shape[0]:,}) ---", flush=True)
    t0 = time.perf_counter()
    try:
        solver = solver_cls(A, B, epsilon=epsilon, batch_size=batch_size, verbose=False, **extra_kwargs)
        t_init = time.perf_counter() - t0
        t1 = time.perf_counter()
        match_B = solver.solve()
        t_solve = time.perf_counter() - t1
        total = time.perf_counter() - t0
        cost = avg_cost(A, B, match_B)
        matched = int((match_B != -1).sum().item())
        print(
            f"  status=OK  init={t_init:.2f}s  solve={t_solve:.2f}s  total={total:.2f}s  "
            f"matched={matched}/{A.shape[0]}  avg_cost={cost:.6f}",
            flush=True,
        )
        return {"status": "ok", "init_s": t_init, "solve_s": t_solve, "total_s": total,
                "matched": matched, "n": A.shape[0], "avg_cost": cost}
    except Exception as exc:  # noqa: BLE001 - smoke test surfaces any failure
        total = time.perf_counter() - t0
        print(f"  status=FAIL  after={total:.2f}s  error={exc!r}", flush=True)
        return {"status": "fail", "total_s": total, "n": A.shape[0], "error": repr(exc)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=8000, help="Number of points per side (small; CPU-scale).")
    parser.add_argument("--epsilon", type=float, default=0.02)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cpu")
    print(f"CPU smoke test: torch={torch.__version__}  device={device}  N={args.n:,}  epsilon={args.epsilon}")

    torch.manual_seed(args.seed)
    A, B = make_data(args.n, args.seed, device)

    results = {}
    torch.manual_seed(args.seed)
    results["2-level (Simple)"] = run_pipeline(
        "2-Level pipeline (SimpleClustering + SimpleGPUSolver)",
        SimpleGPUSolver, A, B, args.epsilon, args.batch_size,
    )

    torch.manual_seed(args.seed)
    results["3-level (ThreeLevel)"] = run_pipeline(
        "3-Level pipeline (ThreeLevelClustering + ThreeLevelGPUSolver)",
        ThreeLevelGPUSolver, A, B, args.epsilon, args.batch_size,
        extra_kwargs={"clustering_tile_size": args.batch_size},
    )

    print("\n=== Summary ===")
    all_ok = True
    for name, row in results.items():
        status = row["status"]
        all_ok = all_ok and status == "ok"
        print(f"  {name:<28} status={status:<4} total={row['total_s']:.2f}s")

    if not all_ok:
        print("\nOne or more pipelines FAILED on CPU.")
        return 1
    print("\nAll pipelines completed successfully on CPU.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
