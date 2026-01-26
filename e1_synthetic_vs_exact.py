#!/usr/bin/env python3
import os, csv, time, math
import argparse
import numpy as np
import torch
# Prevent JAX from preallocating GPU memory (in case we use Sinkhorn here)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
os.environ.setdefault("JAX_ENABLE_X64", "True")
try:
    import ot  # POT for exact EMD and Sinkhorn
except ImportError:
    ot = None
# Import custom GPU solvers
try:
    from clustered_push_relabel import GPUClusteredSolver as TwoLevelSolver
    from k_level_clustered_push_relabel import GPUClusteredSolver as KLevelSolver
except ImportError as e:
    raise ImportError(f"Error importing solvers: {e}")

def compute_cost_matrix_L2(red, blue):
    """Compute full Euclidean distance matrix (L2) for two sets of points (CPU)."""
    # red, blue are numpy arrays of shape (n, dim)
    # We use the formula ||x-y|| = sqrt(||x||^2 + ||y||^2 - 2 x·y)
    # But to get exact distances we will sqrt later; for EMD we can provide squared or linear cost directly.
    # Here, we compute squared distances matrix and will take sqrt for cost evaluation.
    X = red.astype(np.float64)
    Y = blue.astype(np.float64)
    n = X.shape[0]
    # Efficient computation in chunks to conserve memory
    C_squared = np.empty((n, n), dtype=np.float64)
    # precompute norms
    x_norm = (X**2).sum(axis=1)
    y_norm = (Y**2).sum(axis=1)
    for i in range(0, n, 1000):
        i_end = min(n, i+1000)
        # broadcasting: compute squared dist for chunk i:i_end
        # shape (chunk_size, n)
        dist2 = x_norm[i:i_end, None] + y_norm[None, :] - 2 * X[i:i_end].dot(Y.T)
        # Numerical precision: ensure no negative due to float error
        dist2 = np.clip(dist2, 0.0, None)
        C_squared[i:i_end] = dist2
    return C_squared

def main():
    parser = argparse.ArgumentParser(description="E1 Synthetic vs Exact")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--n_values", type=int, nargs='+',
                       help="List of N values to run (overrides min/max/step)")
    group.add_argument("--min_n", type=int, help="Min N (inclusive, use with --max_n and --step)")
    parser.add_argument("--max_n", type=int, help="Max N (inclusive)")
    parser.add_argument("--step", type=int, help="Step size for N")
    parser.add_argument("--dim", type=int, default=2, help="Dimension of points (default 2D)")
    parser.add_argument("--epsilon", type=float, default=0.01, help="Regularization epsilon (for Sinkhorn if used)")
    parser.add_argument("--k", type=int, default=4, help="Number of levels for k-Level solver")
    parser.add_argument("--trials", type=int, default=1, help="Number of random trials per N")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--csv", type=str, default="results_e1_synth.csv", help="Output CSV file")
    parser.add_argument("--with_sinkhorn", action="store_true",
                        help="Include Sinkhorn (OTT/POT) in the comparisons")
    args = parser.parse_args()

    # Determine list of n values
    if args.n_values:
        n_list = args.n_values
    else:
        if args.min_n is None or args.max_n is None or args.step is None:
            parser.error("Specify --n_values or --min_n/--max_n/--step")
        n_list = list(range(args.min_n, args.max_n + 1, args.step))
    # Setup output CSV
    headers = ["dataset", "n", "dim", "epsilon", "k", "trial", "algo", "status",
               "total_time_s", "cluster_time_s", "solver_time_s", "peak_gpu_mem_mb",
               "cost", "abs_error", "rel_error"]
    out_path = args.csv
    write_header = not os.path.exists(out_path)
    f = open(out_path, "a", newline="")
    writer = csv.writer(f)
    if write_header:
        writer.writerow(headers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_label = "synthetic"
    rng = np.random.RandomState(args.seed)  # NumPy RNG for data generation
    torch.manual_seed(args.seed)
    for n in n_list:
        for t in range(args.trials):
            trial_id = t + 1
            current_seed = args.seed + t
            rng.seed(current_seed)
            torch.manual_seed(current_seed)
            # Generate random points in [0,1]^dim
            # Use double precision for data to avoid precision issues in cost calc
            red = torch.rand((n, args.dim), dtype=torch.float64)
            blue = torch.rand((n, args.dim), dtype=torch.float64)
            # Move to GPU for our methods (POT will use CPU numpy)
            P_red = red.float().to(device)
            P_blue = blue.float().to(device)
            # Compute exact OT using POT (network simplex)
            exact_cost = None
            exact_time = None
            if ot is None:
                print("POT library not installed; skipping exact EMD.")
            else:
                # Construct cost matrix (squared distances, then take sqrt for cost sum)
                try:
                    # Compute on CPU (may be memory heavy for large n)
                    t0 = time.time()
                    C_squared = compute_cost_matrix_L2(red.numpy(), blue.numpy())
                    # Solve OT
                    a = np.full(n, 1.0/n, dtype=np.float64)
                    b = np.full(n, 1.0/n, dtype=np.float64)
                    # Use `ot.emd` for exact transport plan
                    P_plan = ot.emd(a, b, C_squared, numItermax=10**6)
                    t1 = time.time()
                    exact_time = t1 - t0
                    # Compute cost: sum of sqrt(dist^2) * flow
                    # Since P_plan is likely a permutation matrix (or close), we do elementwise multiply
                    total_cost = 0.0
                    # We sum in chunks to avoid large intermediate arrays if n is big
                    # But here n is moderate.
                    if P_plan.size > 0:
                        # Where P_plan > 0, that indicates a match. We can vectorize by taking argmax per row.
                        # But to be safe, sum over all i,j.
                        # We take sqrt of each nonzero cost.
                        # (This double loops for clarity; for moderate n it's okay.)
                        ii, jj = np.nonzero(P_plan)
                        for (i, j) in zip(ii, jj):
                            if P_plan[i, j] > 0:
                                total_cost += math.sqrt(C_squared[i, j]) * P_plan[i, j]
                    exact_cost = total_cost
                    print(f"Exact OT for n={n}: cost={exact_cost:.4f}, time={exact_time:.2f}s")
                except MemoryError as e:
                    print(f"[Exact] Memory error at n={n}: {e}")
                    exact_time = None
                    exact_cost = None
            # Prepare to record baseline (exact) row
            if exact_cost is not None:
                writer.writerow([dataset_label, n, args.dim, args.epsilon, args.k, trial_id,
                                 "POT-Exact", "success", f"{exact_time:.6f}",
                                 0.0, 0.0, 0.0,
                                 f"{exact_cost:.6f}", "", ""])
                f.flush()
            else:
                # If no exact solution (maybe n too large for exact), skip error calc for approximations
                print(f"Exact solution not available for n={n}.")
            # Now run approximate methods
            # Optionally, Sinkhorn baseline (POT’s or OTT’s)
            if args.with_sinkhorn and ot is not None:
                # Use POT’s sinkhorn with log stabilization for comparison
                try:
                    # Compute cost matrix again (or reuse squared matrix if available)
                    if exact_cost is None or C_squared is None:
                        C_squared = compute_cost_matrix_L2(red.numpy(), blue.numpy())
                    # POT Sinkhorn (regularized OT)
                    reg = args.epsilon
                    t0 = time.time()
                    P_sink = ot.sinkhorn(a, b, C_squared, reg, method='sinkhorn', numItermax=100000)
                    t1 = time.time()
                    # Calculate soft cost (reg OT cost) and hard cost by rounding
                    soft_cost = float((P_sink * np.sqrt(C_squared)).sum())  # using sqrt since C_squared matrix
                    # Greedy matching: argmax per row of coupling
                    matches = P_sink.argmax(axis=1)
                    hard_cost = 0.0
                    for i in range(n):
                        j = matches[i]
                        hard_cost += math.sqrt(C_squared[i, j])
                    sink_time = t1 - t0
                    # Memory usage: cost matrix was size n^2 in memory (approx)
                    mem_est = (n**2 * 8) / (1024**2)  # bytes to MB for double matrix
                    writer.writerow([dataset_label, n, args.dim, args.epsilon, args.k, trial_id,
                                     "Sinkhorn", "success", f"{sink_time:.6f}",
                                     0.0, f"{sink_time:.6f}", f"{mem_est:.2f}",
                                     f"{hard_cost:.6f}",
                                     f"{(hard_cost - (exact_cost or hard_cost)):.6f}" if exact_cost else "",
                                     f"{((hard_cost/(exact_cost or hard_cost) - 1)*100):.2f}" if exact_cost else ""])
                    f.flush()
                except Exception as e:
                    print(f"[Sinkhorn] Failed at n={n}: {e}")
                    writer.writerow([dataset_label, n, args.dim, args.epsilon, args.k, trial_id,
                                     "Sinkhorn", "fail", "", "", "", "",
                                     "", "", ""])
                    f.flush()
            # 2-Level Push-Relabel
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                t_start = time.time()
                solver = TwoLevelSolver(P_red, P_blue, args.epsilon)
                torch.cuda.synchronize()
                t_cluster = time.time()
                solver.solve()
                torch.cuda.synchronize()
                t_end = time.time()
                total_time = t_end - t_start
                cluster_time = t_cluster - t_start
                solver_time = t_end - t_cluster
                # Compute matching cost
                # solver.MB gives matching: MB[j] = i matched to that blue j
                MB = solver.MB
                red_idx = MB.cpu()  # get indices on CPU
                dists = torch.norm(P_blue - P_red[red_idx], p=2, dim=1)
                match_cost = dists.sum().item()
                peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
                status = "success"
            except Exception as e:
                status = "fail"
                total_time = cluster_time = solver_time = float('nan')
                match_cost = float('nan')
                peak_mem = float('nan')
                print(f"[2-Level] Failed at n={n}: {e}")
            finally:
                # Clean up solver to free GPU memory
                try:
                    del solver
                except UnboundLocalError:
                    pass
            # Log 2-Level results
            abs_err = (match_cost - exact_cost) if (exact_cost is not None and not math.isnan(match_cost)) else ""
            rel_err = ((match_cost/(exact_cost if exact_cost else 1) - 1)*100) if (exact_cost is not None and not math.isnan(match_cost)) else ""
            writer.writerow([dataset_label, n, args.dim, args.epsilon, args.k, trial_id,
                             "2-Level", status,
                             f"{total_time:.6f}" if not math.isnan(total_time) else "",
                             f"{cluster_time:.6f}" if not math.isnan(cluster_time) else "",
                             f"{solver_time:.6f}" if not math.isnan(solver_time) else "",
                             f"{peak_mem:.2f}" if not math.isnan(peak_mem) else "",
                             f"{match_cost:.6f}" if not math.isnan(match_cost) else "",
                             f"{abs_err:.6f}" if abs_err != "" else "",
                             f"{rel_err:.2f}" if rel_err != "" else ""])
            f.flush()
            # k-Level Push-Relabel
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                t_start = time.time()
                solver_k = KLevelSolver(P_red, P_blue, args.epsilon, k=args.k)
                torch.cuda.synchronize()
                t_cluster = time.time()
                solver_k.solve()
                torch.cuda.synchronize()
                t_end = time.time()
                total_time_k = t_end - t_start
                cluster_time_k = t_cluster - t_start
                solver_time_k = t_end - t_cluster
                MB_k = solver_k.MB
                red_idx_k = MB_k.cpu()
                dists_k = torch.norm(P_blue - P_red[red_idx_k], p=2, dim=1)
                match_cost_k = dists_k.sum().item()
                peak_mem_k = torch.cuda.max_memory_allocated() / (1024**2)
                status_k = "success"
            except Exception as e:
                status_k = "fail"
                total_time_k = cluster_time_k = solver_time_k = float('nan')
                match_cost_k = float('nan')
                peak_mem_k = float('nan')
                print(f"[k-Level] Failed at n={n}: {e}")
            finally:
                try:
                    del solver_k
                except UnboundLocalError:
                    pass
            abs_err_k = (match_cost_k - exact_cost) if (exact_cost is not None and not math.isnan(match_cost_k)) else ""
            rel_err_k = ((match_cost_k/(exact_cost if exact_cost else 1) - 1)*100) if (exact_cost is not None and not math.isnan(match_cost_k)) else ""
            writer.writerow([dataset_label, n, args.dim, args.epsilon, args.k, trial_id,
                             "k-Level", status_k,
                             f"{total_time_k:.6f}" if not math.isnan(total_time_k) else "",
                             f"{cluster_time_k:.6f}" if not math.isnan(cluster_time_k) else "",
                             f"{solver_time_k:.6f}" if not math.isnan(solver_time_k) else "",
                             f"{peak_mem_k:.2f}" if not math.isnan(peak_mem_k) else "",
                             f"{match_cost_k:.6f}" if not math.isnan(match_cost_k) else "",
                             f"{abs_err_k:.6f}" if abs_err_k != "" else "",
                             f"{rel_err_k:.2f}" if rel_err_k != "" else ""])
            f.flush()
            # End of one trial
        # End of trials for this n
    f.close()

if __name__ == "__main__":
    main()
