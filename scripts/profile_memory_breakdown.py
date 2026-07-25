#!/usr/bin/env python3
"""
Stage-level GPU peak-memory breakdown (rebuttal Task 3 / Task 6).

Runs the 2-Level (SimpleClustering + SimpleGPUSolver) or 3-Level
(ThreeLevelClustering + ThreeLevelGPUSolver) pipeline with profile_memory=True
and prints a stage -> peak-GiB table.

Defaults match the configuration actually used by
experiments/runners/final2/scalability_synthetic_3level_binary_search.py,
which is the script that produced the paper's large-N 3-level scalability
result (see NOTES_FOR_REBUTTAL.md, Task 2): epsilon=0.01, tile_size=2048,
clustering_tile_size=512, sample_factor=1.0, seed=42, synthetic 2D data in
[0, 1]^2 / sqrt(2). --n has no single fixed value in that script (it comes
from a binary search over N); default here (650,000) approximates the
reviewer-cited scalability point and should be treated as a starting point,
not a guaranteed-to-fit value on a given GPU.

Usage:
    python scripts/profile_memory_breakdown.py --mode 3level --n 650000
    python scripts/profile_memory_breakdown.py --mode 2level --n 200000

Requires a CUDA GPU: profile_memory=True is a documented no-op on CPU
(torch.cuda.* calls are skipped), so this script will report an all-zero /
empty table if run without a GPU. Use scripts/cpu_smoke_test.py to validate
CPU portability instead.
"""
import argparse
import gc
import math
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

SYNTHETIC_2D_DIAMETER = math.sqrt(2.0)


def make_data(n, seed, device):
    gen_a = torch.Generator(device=device)
    gen_b = torch.Generator(device=device)
    gen_a.manual_seed(seed)
    gen_b.manual_seed(seed + 1)
    A = torch.rand(n, 2, device=device, dtype=torch.float32, generator=gen_a)
    B = torch.rand(n, 2, device=device, dtype=torch.float32, generator=gen_b)
    A = A / SYNTHETIC_2D_DIAMETER
    B = B / SYNTHETIC_2D_DIAMETER
    return A, B


def print_table(title, memory_profile):
    print(f"\n=== {title} ===")
    if not memory_profile:
        print("  (empty — profile_memory=True is a no-op without a CUDA device)")
        return
    width = max(len(k) for k in memory_profile) + 2
    print(f"  {'stage':<{width}} peak (GiB)")
    print(f"  {'-' * width} ----------")
    for stage, gb in memory_profile.items():
        print(f"  {stage:<{width}} {gb:>10.4f}")


def run_3level(args, device):
    A, B = make_data(args.n, args.seed, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    solver = ThreeLevelGPUSolver(
        A, B,
        epsilon=args.epsilon,
        batch_size=args.tile_size,
        tile_size=args.tile_size,
        clustering_tile_size=args.clustering_tile_size,
        verbose=args.verbose,
        diameter=1.0,
        max_iters=args.max_iters,
        sample_factor=args.sample_factor,
        profile_memory=True,
    )
    t_init = time.perf_counter() - t0
    t1 = time.perf_counter()
    solver.solve()
    t_solve = time.perf_counter() - t1
    print(f"\n[3-level] init={t_init:.2f}s solve={t_solve:.2f}s total={t_init + t_solve:.2f}s "
          f"(instrumented run — do not compare directly to unstrumented paper timings)")
    print_table("3-Level stage-level peak GPU memory (GiB)", solver.memory_profile)
    if device.type == "cuda":
        print(f"\n  overall process peak (torch.cuda.max_memory_allocated over whole run): "
              f"{torch.cuda.max_memory_allocated(device) / 1024**3:.4f} GiB")
    del solver, A, B
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def run_2level(args, device):
    A, B = make_data(args.n, args.seed, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    solver = SimpleGPUSolver(
        A, B,
        epsilon=args.epsilon,
        batch_size=args.tile_size,
        verbose=args.verbose,
        max_iters=args.max_iters,
        sample_factor=args.sample_factor,
        profile_memory=True,
    )
    t_init = time.perf_counter() - t0
    t1 = time.perf_counter()
    solver.solve()
    t_solve = time.perf_counter() - t1
    print(f"\n[2-level] init={t_init:.2f}s solve={t_solve:.2f}s total={t_init + t_solve:.2f}s "
          f"(instrumented run — do not compare directly to unstrumented paper timings)")
    print_table("2-Level stage-level peak GPU memory (GiB)", solver.memory_profile)
    if device.type == "cuda":
        print(f"\n  overall process peak (torch.cuda.max_memory_allocated over whole run): "
              f"{torch.cuda.max_memory_allocated(device) / 1024**3:.4f} GiB")
    del solver, A, B
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["2level", "3level", "both"], default="3level")
    parser.add_argument("--n", type=int, default=650_000,
                         help="Points per side. Default approximates the paper's cited "
                              "3-level scalability point; lower it if you OOM.")
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--tile-size", type=int, default=2048, help="Solver batch/tile size.")
    parser.add_argument("--clustering-tile-size", type=int, default=512,
                         help="3-level clustering tile size (matches the binary-search driver default).")
    parser.add_argument("--sample-factor", type=float, default=1.0,
                         help="See NOTES_FOR_REBUTTAL.md Task 2: every real experiment in this "
                              "repo uses the default of 1.0.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iters", type=int, default=999_999_999)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("WARNING: no CUDA device available. profile_memory=True stages will be empty "
              "no-ops on this machine — run this script on the target GPU for real numbers.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Device: cuda — {torch.cuda.get_device_name(device)} "
              f"({torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GiB)")

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    print(f"Config: N={args.n:,} epsilon={args.epsilon} tile_size={args.tile_size} "
          f"clustering_tile_size={args.clustering_tile_size} sample_factor={args.sample_factor} "
          f"seed={args.seed}")

    if args.mode in ("3level", "both"):
        run_3level(args, device)
    if args.mode in ("2level", "both"):
        run_2level(args, device)


if __name__ == "__main__":
    main()
