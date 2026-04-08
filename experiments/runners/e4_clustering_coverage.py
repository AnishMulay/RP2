#!/usr/bin/env python3
"""
E4: Clustering Coverage Experiment
===================================
For each N in 1000..20000:
  1. Load N red and N blue MNIST images (784-D, L1 metric).
  2. Run the exact POT solver to get a perfect matching M*.
  3. Run the ColorAwareClustering phase only (no push-relabel).
  4. For each matched pair (a, b) in M*, check whether the clustering
     "covers" that edge, i.e.:
       - blue point b (P_all index N+b) is a member of red center a's cluster, OR
       - red point a (P_all index a)    is a member of blue center b's cluster.
     If either holds, the distance is approximated within (1+ε).
  5. Report: N, total pairs, covered pairs, fraction covered, cluster time, exact time.
"""

import pathlib, os, csv, time, math, argparse
import numpy as np
import torch

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent

try:
    import ot
except ImportError:
    raise ImportError("POT library is required: pip install POT")

try:
    from clustered_push_relabel.clustering.color_aware_two_level import ColorAwareClustering
except ImportError as e:
    raise ImportError(f"Could not import ColorAwareClustering: {e}")


# ---------------------------------------------------------------------------
# MNIST helpers (reused from e2)
# ---------------------------------------------------------------------------

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
    seen, deduped = set(), []
    for d in candidate_dirs:
        r = d.expanduser().resolve()
        if r not in seen:
            deduped.append(r); seen.add(r)
    for d in deduped:
        img = d / "train-images-idx3-ubyte.gz"
        lbl = d / "train-labels-idx1-ubyte.gz"
        if img.is_file() and lbl.is_file():
            return img, lbl
    raise FileNotFoundError(
        "MNIST data not found. Searched: " + ", ".join(str(d) for d in deduped)
    )


def load_mnist_flat(n_samples, seed=0, data_dir=None):
    import gzip
    img_path, _ = resolve_mnist_paths(data_dir=data_dir)
    with gzip.open(img_path, "rb") as f:
        images = np.frombuffer(f.read(), dtype=np.uint8, offset=16).reshape(-1, 784)
    images = images.astype(np.float32) / 255.0
    if 2 * n_samples > images.shape[0]:
        raise ValueError(f"Need {2*n_samples} images, only {images.shape[0]} available.")
    rng = np.random.RandomState(seed)
    perm = rng.permutation(images.shape[0])
    red  = torch.from_numpy(images[perm[:n_samples]]).float()
    blue = torch.from_numpy(images[perm[n_samples:2*n_samples]]).float()
    return red, blue


# ---------------------------------------------------------------------------
# Exact matching helper
# ---------------------------------------------------------------------------

def run_exact_matching(red_np, blue_np):
    """
    Returns:
      matching: int array of length N where matching[b] = a
                (blue index b is matched to red index a).
      total_cost: float, total L1 transport cost (unnormalized).
      elapsed: float seconds (solver only, not cost matrix build).
    """
    N = red_np.shape[0]
    C = np.abs(red_np[:, None, :] - blue_np[None, :, :]).sum(axis=2).astype(np.float64)
    a = np.full(N, 1.0 / N, dtype=np.float64)
    b = np.full(N, 1.0 / N, dtype=np.float64)

    t0 = time.time()
    plan = ot.emd(a, b, C, numItermax=10**7)
    elapsed = time.time() - t0

    # plan is (N x N) scaled by 1/N; argmax per row gives the matching
    # matching[i] = j means red i is matched to blue j
    matching_rb = plan.argmax(axis=1)  # shape (N,), red->blue
    total_cost = float((plan * C).sum())
    return matching_rb, total_cost, elapsed


# ---------------------------------------------------------------------------
# Coverage check
# ---------------------------------------------------------------------------

def build_cluster_sets(r_c, r_p, b_c, b_p):
    """
    Build two sets for O(1) membership lookup.

    red_set:  { (red_center_idx, p_all_idx) }
    blue_set: { (blue_center_idx, p_all_idx) }

    Both tensors are already on CPU after .cpu() calls below.
    """
    r_c_cpu = r_c.cpu()
    r_p_cpu = r_p.cpu()
    b_c_cpu = b_c.cpu()
    b_p_cpu = b_p.cpu()

    red_set  = set(zip(r_c_cpu.tolist(), r_p_cpu.tolist()))
    blue_set = set(zip(b_c_cpu.tolist(), b_p_cpu.tolist()))
    return red_set, blue_set


def compute_coverage(matching_rb, red_set, blue_set, N):
    """
    For each matched pair (red=a, blue=b) from matching_rb:
      covered if:
        (a, N+b) in red_set   -- blue point b is in red center a's cluster
        OR
        (b, a)   in blue_set  -- red point a (P_all index = a) is in blue center b's cluster

    Returns (n_covered, n_total).
    """
    n_total   = len(matching_rb)
    n_covered = 0
    for a, b in enumerate(matching_rb):
        b = int(b)
        if (a, N + b) in red_set or (b, a) in blue_set:
            n_covered += 1
    return n_covered, n_total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DEFAULT_N_VALUES = list(range(1000, 20001, 1000))

def main():
    parser = argparse.ArgumentParser(description="E3: Clustering Coverage vs Exact Matching")
    parser.add_argument("--n_values",   type=int, nargs="+", default=None)
    parser.add_argument("--epsilon",    type=float, default=0.05,
                        help="Clustering epsilon (normalized, same as E2)")
    parser.add_argument("--trials",     type=int, default=1)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--csv",        type=str, default="results_e3_coverage.csv")
    parser.add_argument("--data_dir",   type=str, default=None)
    parser.add_argument("--metric",     type=str, default="L1", choices=["L1", "L2"])
    args = parser.parse_args()

    n_list = args.n_values if args.n_values else DEFAULT_N_VALUES

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    headers = [
        "n", "trial", "epsilon", "metric",
        "n_total_pairs", "n_covered", "frac_covered",
        "cluster_time_s", "exact_time_s", "exact_avg_cost",
    ]
    write_header = not os.path.isfile(args.csv)
    f_out = open(args.csv, "a", newline="")
    writer = csv.writer(f_out)
    if write_header:
        writer.writerow(headers)

    # Summary accumulators
    summary = {n: {"frac_covered": [], "cluster_time": [], "exact_time": []} for n in n_list}

    for n in n_list:
        for t in range(args.trials):
            trial_seed = args.seed + t
            print(f"\n[n={n:>6}  trial={t+1}/{args.trials}] Loading MNIST ...", flush=True)
            red, blue = load_mnist_flat(n, seed=trial_seed, data_dir=args.data_dir)

            # --- Step 1: Exact matching ---
            print(f"  Running exact POT solver ...", flush=True)
            red_np  = red.numpy()
            blue_np = blue.numpy()
            matching_rb, exact_total_cost, exact_time = run_exact_matching(red_np, blue_np)
            exact_avg_cost = exact_total_cost  # already total; divide by n for avg
            print(f"  Exact done: avg cost={exact_total_cost/n:.4f}  time={exact_time:.3f}s")

            # --- Step 2: Clustering only ---
            print(f"  Running clustering (ε={args.epsilon}) ...", flush=True)

            # Normalize by diameter before clustering (mirrors solver behaviour)
            P_red_dev  = red.to(device)
            P_blue_dev = blue.to(device)
            P_all      = torch.cat([P_red_dev, P_blue_dev], dim=0)
            if args.metric == "L1":
                diameter = (P_all.max(dim=0).values - P_all.min(dim=0).values).sum().item()
            else:
                diameter = torch.cdist(
                    P_all.unsqueeze(0), P_all.unsqueeze(0)
                ).max().item()
            diameter = max(diameter, 1e-9)

            P_red_norm  = P_red_dev  / diameter
            P_blue_norm = P_blue_dev / diameter

            clustering = ColorAwareClustering(epsilon=args.epsilon, metric=args.metric)

            t0 = time.time()
            (b_c, b_l, b_p), (r_c, r_l, r_p) = clustering.run(P_red_norm, P_blue_norm)
            if device.type == "cuda":
                torch.cuda.synchronize()
            cluster_time = time.time() - t0
            print(f"  Clustering done: {r_p.numel()} red edges, {b_p.numel()} blue edges"
                  f"  time={cluster_time:.3f}s", flush=True)

            # --- Step 3: Build lookup sets ---
            red_set, blue_set = build_cluster_sets(r_c, r_p, b_c, b_p)

            # --- Step 4: Check coverage ---
            n_covered, n_total = compute_coverage(matching_rb, red_set, blue_set, n)
            frac_covered = n_covered / n_total
            print(f"  Coverage: {n_covered}/{n_total} = {frac_covered*100:.2f}%", flush=True)

            # Accumulate summary
            summary[n]["frac_covered"].append(frac_covered)
            summary[n]["cluster_time"].append(cluster_time)
            summary[n]["exact_time"].append(exact_time)

            writer.writerow([
                n, t+1, args.epsilon, args.metric,
                n_total, n_covered, f"{frac_covered:.6f}",
                f"{cluster_time:.6f}", f"{exact_time:.6f}",
                f"{exact_total_cost/n:.6f}",
            ])
            f_out.flush()

            # Cleanup GPU tensors
            del P_red_dev, P_blue_dev, P_all, P_red_norm, P_blue_norm
            del r_c, r_l, r_p, b_c, b_l, b_p
            del red_set, blue_set
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # --- Summary table ---
    print("\n" + "=" * 80)
    print(f"{'n':>8}  {'frac_covered_%':>16}  {'cluster_time_s':>16}  {'exact_time_s':>14}")
    print("-" * 80)
    for n in n_list:
        row = summary[n]
        def avg(lst): return sum(lst) / len(lst) if lst else float("nan")
        fc = avg(row["frac_covered"])
        ct = avg(row["cluster_time"])
        et = avg(row["exact_time"])
        print(f"{n:>8}  {fc*100:>15.2f}%  {ct:>16.4f}  {et:>14.4f}")
    print("=" * 80)

    f_out.close()
    print(f"\nResults written to {args.csv}")


if __name__ == "__main__":
    main()