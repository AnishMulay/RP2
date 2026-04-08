#!/usr/bin/env python3
"""
E4: Clustering Coverage Experiment
====================================
For each N in 1000..20000:
  1. Load N red and N blue MNIST images (784-D, L1 metric).
  2. Run the exact POT solver to get the optimal matching M*.
  3. Run the ColorAwareClustering phase only (no push-relabel).
  4. For each matched pair (red=a, blue=b) in M*, check whether the clustering
     "covers" that edge:
       - blue point b (P_all index N+b) is a member of red center a's cluster, OR
       - red point a  (P_all index a)   is a member of blue center b's cluster.
     If either holds, the distance proxy is within (1+ε) of the true distance.
  5. Report: N, total pairs, covered pairs, fraction covered, cluster time, exact time.

Memory notes:
  - Cost matrix C is (N x N) float64 — unavoidable for exact OT.
    At N=20000 this is ~3.2 GB.
  - torch.cdist(p=1) computes the same L1 distances as the naive numpy broadcast
    but never materialises the (N, N, D) intermediate (~73 GB at N=5000, D=784).
    Correctness is identical.
"""

import pathlib, os, csv, time, argparse
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
# MNIST helpers
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
            deduped.append(r)
            seen.add(r)
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
        raise ValueError(
            f"Need {2 * n_samples} images, only {images.shape[0]} available."
        )
    rng = np.random.RandomState(seed)
    perm = rng.permutation(images.shape[0])
    red  = torch.from_numpy(images[perm[:n_samples]]).float()
    blue = torch.from_numpy(images[perm[n_samples:2 * n_samples]]).float()
    return red, blue


# ---------------------------------------------------------------------------
# Exact matching
# ---------------------------------------------------------------------------

def run_exact_matching(red: torch.Tensor, blue: torch.Tensor):
    """
    Compute the exact optimal L1 matching between red and blue point sets.

    Uses torch.cdist(p=1) to build the (N x N) cost matrix without ever
    creating an (N, N, D) intermediate array.  The result is mathematically
    identical to sum_d |r_i[d] - b_j[d]| for all (i, j).

    Returns:
      matching_rb : np.ndarray shape (N,) int64
                    matching_rb[a] = b  means red a is matched to blue b.
      elapsed     : float, seconds for the EMD solver only.
    """
    N = red.shape[0]
    print(f"    Building ({N}x{N}) L1 cost matrix via torch.cdist ...", flush=True)

    # torch.cdist never allocates (N, N, D); output is (N, N) only.
    C = torch.cdist(red.double(), blue.double(), p=1).numpy()  # (N, N) float64

    a = np.full(N, 1.0 / N, dtype=np.float64)
    b = np.full(N, 1.0 / N, dtype=np.float64)

    print(f"    Running EMD solver ...", flush=True)
    t0 = time.time()
    plan = ot.emd(a, b, C, numItermax=10 ** 7)   # (N, N) float64
    elapsed = time.time() - t0

    # plan is a near-permutation matrix scaled by 1/N.
    # argmax over columns gives the unique matched blue index for each red.
    matching_rb = plan.argmax(axis=1).astype(np.int64)  # shape (N,)

    del C, plan   # free memory before clustering phase
    return matching_rb, elapsed


# ---------------------------------------------------------------------------
# Clustering coverage check
# ---------------------------------------------------------------------------

def build_cluster_sets(r_c, r_p, b_c, b_p):
    """
    Build two Python sets for O(1) membership lookup.

    red_set  : { (red_center_idx,  p_all_idx) }
    blue_set : { (blue_center_idx, p_all_idx) }

    P_all indexing: red points are indices 0..N-1,
                    blue points are indices N..2N-1.
    """
    red_set  = set(zip(r_c.cpu().tolist(), r_p.cpu().tolist()))
    blue_set = set(zip(b_c.cpu().tolist(), b_p.cpu().tolist()))
    return red_set, blue_set


def compute_coverage(matching_rb: np.ndarray, red_set: set, blue_set: set, N: int):
    """
    For each matched pair (red=a, blue=b):
      covered if:
        (a, N+b) in red_set   -- blue point b is a member of red center a's cluster
        OR
        (b, a)   in blue_set  -- red point a (P_all idx = a) is a member of
                                  blue center b's cluster.

    Returns (n_covered, n_total).
    """
    n_total = len(matching_rb)
    n_covered = sum(
        1
        for a, b in enumerate(matching_rb.tolist())
        if (a, N + b) in red_set or (b, a) in blue_set
    )
    return n_covered, n_total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DEFAULT_N_VALUES = list(range(1000, 20001, 1000))


def main():
    parser = argparse.ArgumentParser(
        description="E3: Clustering Coverage vs Exact Matching"
    )
    parser.add_argument("--n_values",  type=int, nargs="+", default=None,
                        help="List of N values (default: 1000..20000 step 1000)")
    parser.add_argument("--epsilon",   type=float, default=0.05,
                        help="Clustering epsilon (normalized; same as E2)")
    parser.add_argument("--trials",    type=int, default=1,
                        help="Independent trials per N")
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--csv",       type=str, default="results_e3_coverage.csv")
    parser.add_argument("--data_dir",  type=str, default=None)
    parser.add_argument("--metric",    type=str, default="L1", choices=["L1", "L2"])
    args = parser.parse_args()

    n_list = args.n_values if args.n_values else DEFAULT_N_VALUES
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  epsilon={args.epsilon}  metric={args.metric}\n")

    headers = [
        "n", "trial", "epsilon", "metric",
        "n_total_pairs", "n_covered", "frac_covered",
        "cluster_time_s", "exact_time_s",
    ]
    write_header = not os.path.isfile(args.csv)
    f_out = open(args.csv, "a", newline="")
    writer = csv.writer(f_out)
    if write_header:
        writer.writerow(headers)

    summary = {
        n: {"frac_covered": [], "cluster_time": [], "exact_time": []}
        for n in n_list
    }

    for n in n_list:
        for t in range(args.trials):
            trial_seed = args.seed + t
            print(f"\n[n={n:>6}  trial={t+1}/{args.trials}]", flush=True)

            # --- Load data ---
            red, blue = load_mnist_flat(n, seed=trial_seed, data_dir=args.data_dir)

            # --- Exact matching (CPU) ---
            matching_rb, exact_time = run_exact_matching(red, blue)
            print(f"  Exact done in {exact_time:.3f}s", flush=True)

            # --- Clustering only ---
            print(f"  Running clustering (ε={args.epsilon}) ...", flush=True)
            P_red_dev  = red.to(device)
            P_blue_dev = blue.to(device)

            # Normalize by L1 diameter — mirrors what the solver does internally
            P_all = torch.cat([P_red_dev, P_blue_dev], dim=0)
            diameter = (
                P_all.max(dim=0).values - P_all.min(dim=0).values
            ).sum().item()
            diameter = max(diameter, 1e-9)

            P_red_norm  = P_red_dev  / diameter
            P_blue_norm = P_blue_dev / diameter

            clustering = ColorAwareClustering(epsilon=args.epsilon, metric=args.metric)
            t0 = time.time()
            (b_c, b_l, b_p), (r_c, r_l, r_p) = clustering.run(P_red_norm, P_blue_norm)
            if device.type == "cuda":
                torch.cuda.synchronize()
            cluster_time = time.time() - t0
            print(
                f"  Clustering done in {cluster_time:.3f}s"
                f"  ({r_p.numel()} red edges, {b_p.numel()} blue edges)",
                flush=True,
            )

            # --- Coverage check ---
            red_set, blue_set = build_cluster_sets(r_c, r_p, b_c, b_p)
            n_covered, n_total = compute_coverage(matching_rb, red_set, blue_set, n)
            frac_covered = n_covered / n_total
            print(
                f"  Coverage: {n_covered}/{n_total} = {frac_covered * 100:.2f}%",
                flush=True,
            )

            summary[n]["frac_covered"].append(frac_covered)
            summary[n]["cluster_time"].append(cluster_time)
            summary[n]["exact_time"].append(exact_time)

            writer.writerow([
                n, t + 1, args.epsilon, args.metric,
                n_total, n_covered, f"{frac_covered:.6f}",
                f"{cluster_time:.6f}", f"{exact_time:.6f}",
            ])
            f_out.flush()

            # Cleanup GPU memory between iterations
            del P_red_dev, P_blue_dev, P_all, P_red_norm, P_blue_norm
            del r_c, r_l, r_p, b_c, b_l, b_p
            del red_set, blue_set
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # --- Summary table ---
    def avg(lst):
        return sum(lst) / len(lst) if lst else float("nan")

    print("\n" + "=" * 72)
    print(
        f"{'n':>8}  {'frac_covered_%':>16}  "
        f"{'cluster_time_s':>16}  {'exact_time_s':>14}"
    )
    print("-" * 72)
    for n in n_list:
        row = summary[n]
        print(
            f"{n:>8}  {avg(row['frac_covered']) * 100:>15.2f}%  "
            f"{avg(row['cluster_time']):>16.4f}  "
            f"{avg(row['exact_time']):>14.4f}"
        )
    print("=" * 72)

    f_out.close()
    print(f"\nResults written to {args.csv}")


if __name__ == "__main__":
    main()