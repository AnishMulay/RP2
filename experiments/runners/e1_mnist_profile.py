#!/usr/bin/env python3
"""
Profiling variant of e1_mnist_vs_exact.py.

Runs only n=1000 and n=5000 (profiling overhead makes large sizes less useful).
At the end of each solve, prints a phase-level timing breakdown showing:
  operation | total time (s) | % of solver time | avg per phase (ms)

Existing correctness output is preserved so results can be verified visually.

Usage:
  python -u experiments/runners/e1_mnist_profile.py --epsilon 0.01 --k 4 --trials 1 --seed 42
"""
import pathlib
import os
import sys
import csv
import time
import math
import argparse
import numpy as np
import torch

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
os.environ.setdefault("JAX_ENABLE_X64", "True")

try:
    import ot
except ImportError:
    ot = None

import clustered_push_relabel.solvers.bipartite as _bip_module
from clustered_push_relabel.solvers.bipartite import (
    TwoLevelBipartiteSolver as TwoLevelSolver,
    KLevelBipartiteSolver as KLevelSolver,
)

# ── Enable profiling instrumentation ────────────────────────────────────────
_bip_module.PROFILING = True

# ── Reuse helpers from the original runner ──────────────────────────────────
def resolve_mnist_paths(data_dir=None):
    candidate_dirs = []
    if data_dir:
        candidate_dirs.append(pathlib.Path(data_dir))
    env_data_dir = os.environ.get("MNIST_DATA_DIR")
    if env_data_dir:
        candidate_dirs.append(pathlib.Path(env_data_dir))
    candidate_dirs.extend([
        BASE_DIR / "data",
        BASE_DIR / "experiments" / "data",
        pathlib.Path.cwd() / "data",
        pathlib.Path.cwd() / "experiments" / "data",
    ])
    seen = set()
    deduped = []
    for d in candidate_dirs:
        r = d.expanduser().resolve()
        if r not in seen:
            deduped.append(r)
            seen.add(r)
    for d in deduped:
        img_path = d / "train-images-idx3-ubyte.gz"
        lbl_path = d / "train-labels-idx1-ubyte.gz"
        if img_path.is_file() and lbl_path.is_file():
            return img_path, lbl_path
    searched = ", ".join(str(p) for p in deduped)
    raise FileNotFoundError(
        f"MNIST data files not found. Searched: {searched}."
    )


def load_mnist_flat(n_samples, seed=0, data_dir=None):
    import gzip
    img_path, _ = resolve_mnist_paths(data_dir=data_dir)
    with gzip.open(img_path, "rb") as f:
        images = np.frombuffer(f.read(), dtype=np.uint8, offset=16).reshape(-1, 784)
    images = images.astype(np.float32) / 255.0
    total = images.shape[0]
    if 2 * n_samples > total:
        raise ValueError(f"Not enough MNIST images for {2*n_samples} samples.")
    rng = np.random.RandomState(seed)
    perm = rng.permutation(total)
    red = torch.from_numpy(images[perm[:n_samples]]).float()
    blue = torch.from_numpy(images[perm[n_samples:2 * n_samples]]).float()
    return red, blue


def compute_cost_matrix_L1(red, blue):
    X = torch.as_tensor(red, dtype=torch.float64).contiguous()
    Y = torch.as_tensor(blue, dtype=torch.float64).contiguous()
    return torch.cdist(X, Y, p=1).cpu().numpy()


def average_l1_matching_cost(P_red, P_blue, matching):
    matched_red = P_red[matching.to(torch.long)]
    return (P_blue - matched_red).abs().sum(dim=1).mean().item()


def synchronize_if_cuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def reset_peak_memory_stats_if_cuda(device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()


def peak_memory_allocated_mb(device):
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


# ── Profiling summary printer ────────────────────────────────────────────────
_PHASE_LABELS = {
    'A_maint':     'A  Maintenance (scatter_reduce yA/L_A)',
    'B_index':     'B1 Push – ragged index construction',
    'B_gather':    'B2 Push – CSR gather (c_ids / b_levels)',
    'B_slack':     'B3 Push – slack arithmetic + filter',
    'C_free_reds': 'C1 Resolve – free-red mask filter',
    'C_sort_rank': 'C2 Resolve – argsort/bincount/rank',
    'C_commit':    'C3 Resolve – slack check + MB/MA write',
    'D_relabel':   'D  Relabel (yB↑ yA↓ B_free update)',
}

def print_profiling_summary(prof, solver_time_s, solver_label):
    iters = max(prof['iterations'], 1)
    keys = list(_PHASE_LABELS.keys())
    total_accounted = sum(prof[k] for k in keys)

    col_w = 42
    print()
    print(f"{'─'*80}")
    print(f"  PROFILING SUMMARY — {solver_label}")
    print(f"  Solver wall time : {solver_time_s:.3f} s    Iterations: {iters}")
    print(f"  Time accounted   : {total_accounted:.3f} s  "
          f"({100*total_accounted/solver_time_s:.1f}% of solver time)")
    print(f"{'─'*80}")
    header = f"  {'Operation':<{col_w}}  {'Total(s)':>9}  {'%solver':>8}  {'Avg/iter(ms)':>13}"
    print(header)
    print(f"  {'─'*col_w}  {'─'*9}  {'─'*8}  {'─'*13}")
    for k in keys:
        t = prof[k]
        pct = 100.0 * t / solver_time_s if solver_time_s > 0 else 0.0
        avg_ms = 1000.0 * t / iters
        label = _PHASE_LABELS[k]
        print(f"  {label:<{col_w}}  {t:>9.3f}  {pct:>7.1f}%  {avg_ms:>12.3f}")
    print(f"  {'─'*col_w}  {'─'*9}  {'─'*8}  {'─'*13}")
    unaccounted = solver_time_s - total_accounted
    pct_u = 100.0 * unaccounted / solver_time_s if solver_time_s > 0 else 0.0
    avg_u = 1000.0 * unaccounted / iters
    print(f"  {'(overhead / unaccounted)':<{col_w}}  {unaccounted:>9.3f}  {pct_u:>7.1f}%  {avg_u:>12.3f}")
    print(f"{'─'*80}")
    print()


# ── Main ─────────────────────────────────────────────────────────────────────
DEFAULT_N_VALUES = [1000, 5000]


def main():
    parser = argparse.ArgumentParser(description="E1 MNIST profiling runner")
    parser.add_argument("--n_values", type=int, nargs='+', default=DEFAULT_N_VALUES)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--csv", type=str, default="results_e1_mnist_profile.csv")
    parser.add_argument("--data_dir", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"PROFILING enabled in bipartite.py: {_bip_module.PROFILING}")

    headers = ["dataset", "n", "dim", "epsilon", "k", "trial", "algo", "status",
               "total_time_s", "cluster_time_s", "solver_time_s", "peak_gpu_mem_mb",
               "cost", "abs_error", "rel_error"]
    out_path = args.csv
    write_header = not os.path.isfile(out_path)
    f = open(out_path, "a", newline="")
    writer = csv.writer(f)
    if write_header:
        writer.writerow(headers)

    dataset_label = "MNIST"
    base_seed = args.seed

    for n in args.n_values:
        for t in range(args.trials):
            trial_seed = base_seed + t
            print(f"\n{'='*60}")
            print(f"Running MNIST n={n}, trial {t+1}, seed={trial_seed}")
            print(f"{'='*60}")

            red, blue = load_mnist_flat(n, seed=trial_seed, data_dir=args.data_dir)
            P_red = red.to(device)
            P_blue = blue.to(device)
            dim = 784

            exact_cost = None
            exact_time = None
            if ot is None:
                print("POT not installed, skipping exact OT.")
            else:
                try:
                    C = compute_cost_matrix_L1(red.numpy(), blue.numpy())
                    a = np.full(n, 1.0 / n, dtype=np.float64)
                    b = np.full(n, 1.0 / n, dtype=np.float64)
                    t0 = time.time()
                    P_plan = ot.emd(a, b, C, numItermax=10 ** 6)
                    t1 = time.time()
                    exact_time = t1 - t0
                    exact_cost = float((P_plan * C).sum())
                    print(f"Exact cost = {exact_cost:.4f}  (time {exact_time:.2f}s)")
                    writer.writerow([dataset_label, n, dim, args.epsilon, args.k, t+1,
                                     "POT-Exact", "success", f"{exact_time:.6f}",
                                     0.0, 0.0, 0.0, f"{exact_cost:.6f}", "", ""])
                    f.flush()
                except Exception as e:
                    print(f"[Exact] Failed for n={n}: {e}")
                    writer.writerow([dataset_label, n, dim, args.epsilon, args.k, t+1,
                                     "POT-Exact", "fail", "", "", "", "", "", "", ""])
                    f.flush()

            # ── 2-Level solver ───────────────────────────────────────────
            try:
                reset_peak_memory_stats_if_cuda(device)
                t_start = time.time()
                solver2 = TwoLevelSolver(P_red, P_blue, args.epsilon, metric="L1")
                synchronize_if_cuda(device)
                t_mid = time.time()
                solver2.solve()
                synchronize_if_cuda(device)
                t_end = time.time()
                total_time = t_end - t_start
                clust_time = t_mid - t_start
                solve_time = t_end - t_mid
                cost_val = average_l1_matching_cost(P_red, P_blue, solver2.MB)
                peak_mem = peak_memory_allocated_mb(device)
                status = "success"
                prof2 = getattr(solver2, '_prof', None)
            except Exception as e:
                print(f"[2-Level] Exception: {e}")
                import traceback; traceback.print_exc()
                status = "fail"
                total_time = clust_time = solve_time = float('nan')
                cost_val = float('nan')
                peak_mem = float('nan')
                prof2 = None
            finally:
                try:
                    del solver2
                except Exception:
                    pass

            if not math.isnan(solve_time) and prof2 is not None:
                print(f"[2-Level] cost={cost_val:.4f}  solver_time={solve_time:.2f}s")
                print_profiling_summary(prof2, solve_time, f"2-Level  n={n}  trial={t+1}")

            abs_err = (cost_val - exact_cost) if (exact_cost is not None and not math.isnan(cost_val)) else ""
            rel_err = ((cost_val / (exact_cost or 1) - 1) * 100) if abs_err != "" else ""
            writer.writerow([dataset_label, n, dim, args.epsilon, args.k, t+1,
                             "2-Level", status,
                             f"{total_time:.6f}" if not math.isnan(total_time) else "",
                             f"{clust_time:.6f}" if not math.isnan(clust_time) else "",
                             f"{solve_time:.6f}" if not math.isnan(solve_time) else "",
                             f"{peak_mem:.2f}" if not math.isnan(peak_mem) else "",
                             f"{cost_val:.6f}" if not math.isnan(cost_val) else "",
                             f"{abs_err:.6f}" if abs_err != "" else "",
                             f"{rel_err:.2f}" if rel_err != "" else ""])
            f.flush()

            # ── k-Level solver ───────────────────────────────────────────
            try:
                reset_peak_memory_stats_if_cuda(device)
                t_start = time.time()
                solverK = KLevelSolver(P_red, P_blue, args.epsilon, k=args.k, metric="L1")
                synchronize_if_cuda(device)
                t_mid = time.time()
                solverK.solve()
                synchronize_if_cuda(device)
                t_end = time.time()
                total_timeK = t_end - t_start
                clust_timeK = t_mid - t_start
                solve_timeK = t_end - t_mid
                cost_valK = average_l1_matching_cost(P_red, P_blue, solverK.MB)
                peak_memK = peak_memory_allocated_mb(device)
                statusK = "success"
                profK = getattr(solverK, '_prof', None)
            except Exception as e:
                print(f"[k-Level] Exception: {e}")
                import traceback; traceback.print_exc()
                statusK = "fail"
                total_timeK = clust_timeK = solve_timeK = float('nan')
                cost_valK = float('nan')
                peak_memK = float('nan')
                profK = None
            finally:
                try:
                    del solverK
                except Exception:
                    pass

            if not math.isnan(solve_timeK) and profK is not None:
                print(f"[k-Level] cost={cost_valK:.4f}  solver_time={solve_timeK:.2f}s")
                print_profiling_summary(profK, solve_timeK, f"k-Level  n={n}  trial={t+1}")

            abs_errK = (cost_valK - exact_cost) if (exact_cost is not None and not math.isnan(cost_valK)) else ""
            rel_errK = ((cost_valK / (exact_cost or 1) - 1) * 100) if abs_errK != "" else ""
            writer.writerow([dataset_label, n, dim, args.epsilon, args.k, t+1,
                             "k-Level", statusK,
                             f"{total_timeK:.6f}" if not math.isnan(total_timeK) else "",
                             f"{clust_timeK:.6f}" if not math.isnan(clust_timeK) else "",
                             f"{solve_timeK:.6f}" if not math.isnan(solve_timeK) else "",
                             f"{peak_memK:.2f}" if not math.isnan(peak_memK) else "",
                             f"{cost_valK:.6f}" if not math.isnan(cost_valK) else "",
                             f"{abs_errK:.6f}" if abs_errK != "" else "",
                             f"{rel_errK:.2f}" if rel_errK != "" else ""])
            f.flush()

    f.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
