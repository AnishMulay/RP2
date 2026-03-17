import pathlib
import os
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
#!/usr/bin/env python3
import os, csv, time
import argparse
import math
import numpy as np
import torch
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
os.environ.setdefault("JAX_ENABLE_X64", "True")
try:
    import ot  # POT
except ImportError:
    ot = None
try:
    from clustered_push_relabel.solvers.bipartite import TwoLevelBipartiteSolver as TwoLevelSolver
    from clustered_push_relabel.solvers.bipartite import KLevelBipartiteSolver as KLevelSolver
except ImportError as e:
    raise ImportError(f"Error importing solvers: {e}")

# MNIST data loader
def resolve_mnist_paths(data_dir=None):
    """Resolve MNIST image/label file paths from an explicit dir or common repo locations."""
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
    deduped_dirs = []
    for directory in candidate_dirs:
        resolved = directory.expanduser().resolve()
        if resolved not in seen:
            deduped_dirs.append(resolved)
            seen.add(resolved)

    for directory in deduped_dirs:
        img_path = directory / "train-images-idx3-ubyte.gz"
        lbl_path = directory / "train-labels-idx1-ubyte.gz"
        if img_path.is_file() and lbl_path.is_file():
            return img_path, lbl_path

    searched = ", ".join(str(path) for path in deduped_dirs)
    raise FileNotFoundError(
        "MNIST data files not found. Searched: "
        f"{searched}. Expected train-images-idx3-ubyte.gz and "
        "train-labels-idx1-ubyte.gz."
    )

def load_mnist_flat(n_samples, seed=0, data_dir=None):
    """Load MNIST images, normalize pixels to [0,1], sample n_samples for red and blue."""
    import gzip
    img_path, lbl_path = resolve_mnist_paths(data_dir=data_dir)
    # Load images
    with gzip.open(img_path, "rb") as f:
        images = np.frombuffer(f.read(), dtype=np.uint8, offset=16).reshape(-1, 784)
    # Load labels (in case needed for potential analysis, though not used in this experiment)
    with gzip.open(lbl_path, "rb") as f:
        labels = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
    # Normalize pixel values to [0,1]
    images = images.astype(np.float32) / 255.0
    total = images.shape[0]
    if 2 * n_samples > total:
        raise ValueError(f"Not enough MNIST images to sample {2*n_samples} (only {total} available).")
    rng = np.random.RandomState(seed)
    perm = rng.permutation(total)
    red_idx = perm[:n_samples]
    blue_idx = perm[n_samples:2*n_samples]
    red = torch.from_numpy(images[red_idx]).float()
    blue = torch.from_numpy(images[blue_idx]).float()
    return red, blue  # return flat 784-D tensors

def compute_cost_matrix_L1(red, blue):
    """Compute full L1 distance matrix between two sets of vectors (CPU)."""
    X = red.astype(np.float64)
    Y = blue.astype(np.float64)
    n = X.shape[0]
    C = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        # Using broadcasting: L1 distance of X[i] to all Y
        C[i, :] = np.abs(X[i] - Y).sum(axis=1)
    return C

def main():
    parser = argparse.ArgumentParser(description="E1 MNIST vs Exact")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--n_values", type=int, nargs='+', help="List of N values to test")
    group.add_argument("--min_n", type=int, help="Min N (use with --max_n, --step)")
    parser.add_argument("--max_n", type=int, help="Max N")
    parser.add_argument("--step", type=int, help="Step size")
    parser.add_argument("--epsilon", type=float, default=0.05, help="Sinkhorn regularization (default 0.05 for L1 costs)")
    parser.add_argument("--k", type=int, default=4, help="Number of clustering levels (default 4)")
    parser.add_argument("--trials", type=int, default=1, help="Trials per N (different random samples)")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--csv", type=str, default="results_e1_mnist.csv", help="Output CSV file")
    parser.add_argument("--data_dir", type=str, default=None, help="Directory containing MNIST .gz files")
    parser.add_argument("--with_sinkhorn", action="store_true", help="Include entropic Sinkhorn baseline")
    args = parser.parse_args()

    # Create list of n values
    if args.n_values:
        n_list = args.n_values
    else:
        if args.min_n is None or args.max_n is None or args.step is None:
            parser.error("Specify --n_values or --min_n/--max_n/--step")
        n_list = list(range(args.min_n, args.max_n + 1, args.step))
    # CSV setup
    headers = ["dataset", "n", "dim", "epsilon", "k", "trial", "algo", "status",
               "total_time_s", "cluster_time_s", "solver_time_s", "peak_gpu_mem_mb",
               "cost", "abs_error", "rel_error"]
    out_path = args.csv
    write_header = not os.path.isfile(out_path)
    f = open(out_path, "a", newline="")
    writer = csv.writer(f)
    if write_header:
        writer.writerow(headers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_label = "MNIST"
    base_seed = args.seed
    for n in n_list:
        for t in range(args.trials):
            trial_seed = base_seed + t
            print(f"\nRunning MNIST n={n}, trial {t+1}")
            # Load data
            red, blue = load_mnist_flat(n, seed=trial_seed, data_dir=args.data_dir)
            # Move to GPU for our solvers
            P_red = red.to(device)
            P_blue = blue.to(device)
            dim = 784
            exact_cost = None
            exact_time = None
            # Exact OT via network simplex (POT)
            if ot is None:
                print("POT not installed, skipping exact OT.")
            else:
                try:
                    C = compute_cost_matrix_L1(red.numpy(), blue.numpy())
                    a = np.full(n, 1.0/n, dtype=np.float64)
                    b = np.full(n, 1.0/n, dtype=np.float64)
                    t0 = time.time()
                    P_plan = ot.emd(a, b, C, numItermax=10**6)
                    t1 = time.time()
                    exact_time = t1 - t0
                    # Calculate total cost from plan
                    total_cost = float((P_plan * C).sum())
                    exact_cost = total_cost
                    print(f"Exact cost = {exact_cost:.4f} (time {exact_time:.2f}s)")
                    writer.writerow([dataset_label, n, dim, args.epsilon, args.k, t+1,
                                     "POT-Exact", "success", f"{exact_time:.6f}",
                                     0.0, 0.0, 0.0,
                                     f"{exact_cost:.6f}", "", ""])
                    f.flush()
                except Exception as e:
                    print(f"[Exact] Failed for n={n}: {e}")
                    writer.writerow([dataset_label, n, dim, args.epsilon, args.k, t+1,
                                     "POT-Exact", "fail", "", "", "", "", "", "", ""])
                    f.flush()
            # Optional Sinkhorn baseline (OTT or POT)
            if args.with_sinkhorn and ot is not None:
                try:
                    C = C if 'C' in locals() else compute_cost_matrix_L1(red.numpy(), blue.numpy())
                    reg = args.epsilon
                    t0 = time.time()
                    P_sink = ot.sinkhorn(np.ones(n)/n, np.ones(n)/n, C, reg=reg, method='sinkhorn', numItermax=100000)
                    t1 = time.time()
                    sink_time = t1 - t0
                    # Soft cost (reg OT objective) and hard matching cost
                    soft_cost = float((P_sink * C).sum())
                    matches = P_sink.argmax(axis=1)
                    hard_cost = 0.0
                    unique_matches = len(np.unique(matches))
                    for i in range(n):
                        j = matches[i]
                        hard_cost += C[i, j]
                    status = "success"
                    mem_est = (n**2 * 8) / (1024**2)  # coupling matrix size in MB
                except Exception as e:
                    print(f"[Sinkhorn] Failed: {e}")
                    status = "fail"
                    sink_time = float('nan')
                    soft_cost = hard_cost = float('nan')
                    mem_est = float('nan')
                    unique_matches = 0
                # Log Sinkhorn
                abs_err = (hard_cost - exact_cost) if (exact_cost is not None and not math.isnan(hard_cost)) else ""
                rel_err = ((hard_cost/(exact_cost if exact_cost else 1) - 1)*100) if (exact_cost is not None and not math.isnan(hard_cost)) else ""
                writer.writerow([dataset_label, n, dim, args.epsilon, args.k, t+1,
                                 "Sinkhorn", status,
                                 f"{sink_time:.6f}" if sink_time == sink_time else "",
                                 0.0,
                                 f"{sink_time:.6f}" if sink_time == sink_time else "",
                                 f"{mem_est:.2f}" if mem_est == mem_est else "",
                                 f"{hard_cost:.6f}" if hard_cost == hard_cost else "",
                                 f"{(hard_cost - exact_cost):.6f}" if abs_err != "" else "",
                                 f"{rel_err:.2f}" if rel_err != "" else ""])
                f.flush()
            # 2-Level solver
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                t_start = time.time()
                solver2 = TwoLevelSolver(P_red, P_blue, args.epsilon, metric="L1")
                torch.cuda.synchronize()
                t_mid = time.time()
                solver2.solve()
                torch.cuda.synchronize()
                t_end = time.time()
                total_time = t_end - t_start
                clust_time = t_mid - t_start
                solve_time = t_end - t_mid
                # Calculate cost of matching
                MB = solver2.MB.cpu().numpy()
                # MB[j] = i matched
                cost_val = 0.0
                for j in range(n):
                    i = MB[j]
                    cost_val += float(torch.norm(P_blue[j] - P_red[i], p=1).cpu().item())
                cost_val /= n
                peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
                status = "success"
            except Exception as e:
                print(f"[2-Level] Exception: {e}")
                status = "fail"
                total_time = clust_time = solve_time = float('nan')
                cost_val = float('nan')
                peak_mem = float('nan')
            finally:
                try: del solver2
                except: pass
            abs_err = (cost_val - exact_cost) if (exact_cost is not None and not math.isnan(cost_val)) else ""
            rel_err = ((cost_val/(exact_cost if exact_cost else 1) - 1)*100) if (exact_cost is not None and not math.isnan(cost_val)) else ""
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
            # k-Level solver
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                t_start = time.time()
                solverK = KLevelSolver(P_red, P_blue, args.epsilon, k=args.k, metric="L1")
                torch.cuda.synchronize()
                t_mid = time.time()
                solverK.solve()
                torch.cuda.synchronize()
                t_end = time.time()
                total_timeK = t_end - t_start
                clust_timeK = t_mid - t_start
                solve_timeK = t_end - t_mid
                MBk = solverK.MB.cpu().numpy()
                cost_valK = 0.0
                for j in range(n):
                    i = MBk[j]
                    cost_valK += float(torch.norm(P_blue[j] - P_red[i], p=1).cpu().item())
                cost_valK /= n
                peak_memK = torch.cuda.max_memory_allocated() / (1024**2)
                statusK = "success"
            except Exception as e:
                print(f"[k-Level] Exception: {e}")
                statusK = "fail"
                total_timeK = clust_timeK = solve_timeK = float('nan')
                cost_valK = float('nan')
                peak_memK = float('nan')
            finally:
                try: del solverK
                except: pass
            abs_errK = (cost_valK - exact_cost) if (exact_cost is not None and not math.isnan(cost_valK)) else ""
            rel_errK = ((cost_valK/(exact_cost if exact_cost else 1) - 1)*100) if (exact_cost is not None and not math.isnan(cost_valK)) else ""
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
        # end trials
    # end for n
    f.close()

if __name__ == "__main__":
    main()
