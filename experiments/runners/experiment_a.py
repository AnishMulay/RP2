#!/usr/bin/env python3
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from clustered_push_relabel.solvers.simple_bipartite import SimpleGPUSolver


N_VALUES = [100_000, 150_000, 200_000, 250_000, 300_000, 350_000, 400_000]
EPSILON = 0.01
SYNTHETIC_DIM = 2
SEED = 42
BATCH_SIZE = 512


def ckpt(n, tag):
    mem = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(
        f"  [N={n:,}] CKPT {tag:<45}  alloc={mem:.2f}GB  reserved={reserved:.2f}GB",
        flush=True,
    )


def print_summary(results):
    print("\nSuccessful runs:", flush=True)
    header = (
        f"{'N':>9} | {'Cluster (ms)':>12} | {'Solve (ms)':>10} | {'Total (ms)':>10} | "
        f"{'Iterations':>10} | {'Avg Cost':>8} | {'Peak GPU (GB)':>13}"
    )
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for row in results:
        print(
            f"{row['n']:>9,} | "
            f"{row['clustering_ms']:>12.1f} | "
            f"{row['solve_ms']:>10.1f} | "
            f"{row['total_ms']:>10.1f} | "
            f"{row['iterations']:>10,} | "
            f"{row['avg_cost']:>8.4f} | "
            f"{row['peak_gb']:>13.2f}",
            flush=True,
        )


def main(device):
    if device.type != "cuda":
        return

    results = []

    for i, n in enumerate(N_VALUES):
        print(f"\n{'=' * 60}", flush=True)
        print(f"  Starting N={n:,}", flush=True)
        print(f"{'=' * 60}", flush=True)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        solver = None
        P_red = None
        P_blue = None

        try:
            torch.manual_seed(SEED + i)

            ckpt(n, "1/8  generating point clouds")
            P_red = torch.rand((n, SYNTHETIC_DIM), dtype=torch.float32, device=device)
            P_blue = torch.rand((n, SYNTHETIC_DIM), dtype=torch.float32, device=device)

            ckpt(n, "2/8  computing diameter")
            sample_for_diam = torch.cat([P_red[:500], P_blue[:500]]).cpu()
            diameter = float(torch.cdist(sample_for_diam, sample_for_diam).max().item())
            # Use a 500-point subsample to avoid an O(N^2) diameter allocation.

            ckpt(n, "3/8  starting SimpleGPUSolver init")
            t0 = time.perf_counter()
            # SimpleGPUSolver.__init__ calls SimpleClustering.run internally.
            # That path allocates DR/DB, V, and CSR buffers; we cannot inject
            # finer checkpoints inside simple.py, so this external boundary is
            # the closest reliable marker for clustering-phase failures.
            solver = SimpleGPUSolver(
                P_red,
                P_blue,
                EPSILON,
                batch_size=BATCH_SIZE,
                verbose=True,
                diameter=diameter,
            )
            torch.cuda.synchronize()
            t1 = time.perf_counter()

            ckpt(n, "4/8  clustering complete / solve starting")
            solver.solve()
            torch.cuda.synchronize()
            t2 = time.perf_counter()

            ckpt(n, "5/8  solve() returned")
            ckpt(n, "6/8  computing matching cost")
            matched_red = P_red[solver.match_B]
            avg_cost = torch.norm(P_blue - matched_red, p=2, dim=1).mean().item()

            clustering_ms = (t1 - t0) * 1000
            solve_ms = (t2 - t1) * 1000
            total_ms = (t2 - t0) * 1000
            peak_gb = torch.cuda.max_memory_allocated(device) / 1024**3

            ckpt(n, "7/8  recording results")
            results.append(
                {
                    "n": n,
                    "clustering_ms": clustering_ms,
                    "solve_ms": solve_ms,
                    "total_ms": total_ms,
                    "iterations": solver.iterations,
                    "avg_cost": avg_cost,
                    "peak_gb": peak_gb,
                }
            )

            print(f"\n  RESULT N={n:,}", flush=True)
            print(f"    Clustering : {clustering_ms:>9.1f} ms", flush=True)
            print(f"    Solve      : {solve_ms:>9.1f} ms", flush=True)
            print(f"    Total      : {total_ms:>9.1f} ms", flush=True)
            print(f"    Iterations : {solver.iterations:,}", flush=True)
            print(f"    Avg L2 cost: {avg_cost:.4f}", flush=True)
            print(f"    Peak GPU   : {peak_gb:.2f} GB", flush=True)

        except torch.cuda.OutOfMemoryError as e:
            print(f"\n  *** OOM at N={n:,} ***", flush=True)
            print(f"  Last checkpoint was the one printed above.", flush=True)
            print(f"  Error: {e}", flush=True)
            print(
                f"  Allocated at crash: "
                f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB",
                flush=True,
            )
            print(
                f"  Reserved at crash:  "
                f"{torch.cuda.memory_reserved() / 1024**3:.2f} GB",
                flush=True,
            )
            break
        except Exception as e:
            print(
                f"\n  *** FAILED at N={n:,}: {type(e).__name__}: {e} ***",
                flush=True,
            )
            break
        finally:
            ckpt(n, "8/8  cleanup")
            del solver, P_red, P_blue
            torch.cuda.empty_cache()

    print_summary(results)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: No CUDA device found. This experiment is designed for GPU.")
    main(device)
