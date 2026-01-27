#!/usr/bin/env python3
import os, csv, time, math
import argparse
import numpy as np
import torch
try:
    from clustered_push_relabel import GPUClusteredSolver as TwoLevelSolver
    from k_level_clustered_push_relabel import GPUClusteredSolver as KLevelSolver
except ImportError as e:
    raise ImportError(f"Solver import error: {e}")

def main():
    parser = argparse.ArgumentParser(description="E3 Scalability Synthetic")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--n_values", type=int, nargs='+')
    group.add_argument("--min_n", type=int)
    parser.add_argument("--max_n", type=int)
    parser.add_argument("--step", type=int)
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--csv", type=str, default="results_e3_scaling.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.n_values:
        n_list = args.n_values
    else:
        if args.min_n is None or args.max_n is None or args.step is None:
            parser.error("Provide --n_values or --min_n/--max_n/--step")
        n_list = list(range(args.min_n, args.max_n + 1, args.step))
    headers = ["dataset", "n", "dim", "epsilon", "k", "algo", "status",
               "total_time_s", "cluster_time_s", "solver_time_s", "peak_gpu_mem_mb", "cost"]
    out_path = args.csv
    write_header = not os.path.isfile(out_path)
    f = open(out_path, "a", newline="")
    writer = csv.writer(f)
    if write_header:
        writer.writerow(headers)
    dataset = "synthetic"
    rng = np.random.RandomState(args.seed)
    torch.manual_seed(args.seed)
    max_n_reached = 0
    for n in n_list:
        print(f"Running n={n}...")
        red = torch.rand((n, args.dim), dtype=torch.float32)
        blue = torch.rand((n, args.dim), dtype=torch.float32)
        P_red = red.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        P_blue = blue.to(P_red.device)
        # 2-Level
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            solver2 = TwoLevelSolver(P_red, P_blue, args.epsilon)
            torch.cuda.synchronize()
            t1 = time.time()
            solver2.solve()
            torch.cuda.synchronize()
            t2 = time.time()
            total_time = t2 - t0
            clust_time = t1 - t0
            solver_time = t2 - t1
            MB = solver2.MB
            # Compute cost (Euclidean)
            diff = P_blue - P_red[MB]
            dists = torch.norm(diff, p=2, dim=1)
            cost_val = dists.sum().item()
            peak_mem = torch.cuda.max_memory_allocated()/(1024**2)
            status = "success"
            max_n_reached = n
        except RuntimeError as e:
            # Catch OOM or other runtime errors
            err_msg = str(e).lower()
            if "out of memory" in err_msg:
                status = "oom"
            else:
                status = "fail"
            total_time = clust_time = solver_time = float('nan')
            cost_val = float('nan')
            peak_mem = float('nan')
            print(f"2-Level failed at n={n}: {e}")
            # Stop further scaling if out-of-memory or failure
            writer.writerow([dataset, n, args.dim, args.epsilon, args.k,
                             "2-Level", status, "", "", "", "", ""])
            f.flush()
        finally:
            try: del solver2
            except: pass
        writer.writerow([dataset, n, args.dim, args.epsilon, args.k,
                         "2-Level", status,
                         f"{total_time:.6f}" if not math.isnan(total_time) else "",
                         f"{clust_time:.6f}" if not math.isnan(clust_time) else "",
                         f"{solver_time:.6f}" if not math.isnan(solver_time) else "",
                         f"{peak_mem:.2f}" if not math.isnan(peak_mem) else "",
                         f"{cost_val:.6f}" if not math.isnan(cost_val) else ""])
        f.flush()
        # k-Level
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            solverK = KLevelSolver(P_red, P_blue, args.epsilon, k=args.k)
            torch.cuda.synchronize()
            t1 = time.time()
            solverK.solve()
            torch.cuda.synchronize()
            t2 = time.time()
            total_timeK = t2 - t0
            clust_timeK = t1 - t0
            solver_timeK = t2 - t1
            MBk = solverK.MB
            diff_k = P_blue - P_red[MBk]
            dists_k = torch.norm(diff_k, p=2, dim=1)
            cost_valK = dists_k.sum().item()
            peak_memK = torch.cuda.max_memory_allocated()/(1024**2)
            statusK = "success"
        except RuntimeError as e:
            err_msg = str(e).lower()
            if "out of memory" in err_msg:
                statusK = "oom"
            else:
                statusK = "fail"
            total_timeK = clust_timeK = solver_timeK = float('nan')
            cost_valK = float('nan')
            peak_memK = float('nan')
            print(f"k-Level failed at n={n}: {e}")
            writer.writerow([dataset, n, args.dim, args.epsilon, args.k,
                             "k-Level", statusK, "", "", "", "", ""])
            f.flush()
            break
        finally:
            try: del solverK
            except: pass
        writer.writerow([dataset, n, args.dim, args.epsilon, args.k,
                         "k-Level", statusK,
                         f"{total_timeK:.6f}" if not math.isnan(total_timeK) else "",
                         f"{clust_timeK:.6f}" if not math.isnan(clust_timeK) else "",
                         f"{solver_timeK:.6f}" if not math.isnan(solver_timeK) else "",
                         f"{peak_memK:.2f}" if not math.isnan(peak_memK) else "",
                         f"{cost_valK:.6f}" if not math.isnan(cost_valK) else ""])
        f.flush()
    f.close()
    if max_n_reached:
        print(f"Max N solved successfully: {max_n_reached}")

if __name__ == "__main__":
    main()
