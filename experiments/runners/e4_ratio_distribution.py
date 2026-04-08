#!/usr/bin/env python3
"""
E4: Proxy Distance Ratio Distribution
=======================================
For a fixed N=5000, epsilon=0.01:

1. Load N red and N blue MNIST images (784-D, L1 metric).
2. Run ColorAwareClustering only.
3. Build an inverted index: for every point p (red or blue), map to
   all (center_type, center_idx, level) entries where p participates.
4. For each blue point b, find its true k=floor(sqrt(N)) nearest red
   neighbors by actual L1 distance.
5. For each such (blue_b, red_a) pair:
     - proxy = min over all shared centers q of  2 * max(k_b_q, k_a_q) * eps
     - ratio = proxy / true_L1_dist
   Pairs with no shared center are counted separately (uncovered).
6. Repeat symmetrically: for each red point a, find its k nearest blue
   neighbors and compute ratios.
7. Plot both ratio distributions as overlapping histograms + KDE,
   and print a summary statistics table.

Memory notes:
  - kNN via torch.cdist on GPU: (N x N) output only, no (N,N,D) intermediate.
    At N=5000, D=784: output is 5000x5000x4 bytes = 100 MB (float32).
  - Inverted index: O(N * sqrt(N)) entries total ~ 350K entries, trivially small.
"""

import pathlib, os, math, time, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")   # no display needed on cluster
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent

try:
    from clustered_push_relabel.clustering.color_aware_two_level import ColorAwareClustering
except ImportError as e:
    raise ImportError(f"Could not import ColorAwareClustering: {e}")


# ---------------------------------------------------------------------------
# MNIST loader
# ---------------------------------------------------------------------------

def resolve_mnist_paths(data_dir=None):
    candidate_dirs = []
    if data_dir:
        candidate_dirs.append(pathlib.Path(data_dir))
    env = os.environ.get("MNIST_DATA_DIR")
    if env:
        candidate_dirs.append(pathlib.Path(env))
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
# k-nearest neighbours via torch.cdist  (no (N,N,D) intermediate)
# ---------------------------------------------------------------------------

def knn_l1(query: torch.Tensor, targets: torch.Tensor, k: int):
    """
    For each row in `query`, return the indices of its k nearest rows in
    `targets` under L1, and the corresponding L1 distances.

    Uses torch.cdist(p=1) which never materialises a (Q, T, D) tensor.
    Output: (Q, k) indices and (Q, k) distances, both on CPU as numpy arrays.
    At N=5000, D=784: the (N x N) float32 distance matrix is ~100 MB.
    """
    print(f"    Computing ({query.shape[0]}x{targets.shape[0]}) L1 distance matrix "
          f"via torch.cdist for kNN (k={k}) ...", flush=True)
    D = torch.cdist(query.float(), targets.float(), p=1)   # (Q, T) float32
    # topk returns LARGEST; we want smallest, so negate or use sort
    vals, idx = torch.topk(D, k=k, dim=1, largest=False, sorted=True)
    del D
    return idx.cpu().numpy(), vals.cpu().numpy()   # (Q, k) each


# ---------------------------------------------------------------------------
# Inverted index construction
# ---------------------------------------------------------------------------

def build_inverted_index(r_c, r_l, r_p, b_c, b_l, b_p):
    """
    Build two inverted indexes so that for any P_all point p we can instantly
    retrieve all (center_idx, level) pairs where p participates.

    p_to_red_centers[p]  = list of (red_center_idx,  level)
    p_to_blue_centers[p] = list of (blue_center_idx, level)

    P_all indexing: red points are 0..N-1, blue points are N..2N-1.
    """
    p_to_red  = {}   # p_all_idx -> {red_center_idx: level}
    p_to_blue = {}   # p_all_idx -> {blue_center_idx: level}

    r_c_l = r_c.cpu().tolist()
    r_l_l = r_l.cpu().tolist()
    r_p_l = r_p.cpu().tolist()
    for c, lv, p in zip(r_c_l, r_l_l, r_p_l):
        if p not in p_to_red:
            p_to_red[p] = {}
        # keep minimum level if a point appears under same center twice
        if c not in p_to_red[p] or lv < p_to_red[p][c]:
            p_to_red[p][c] = lv

    b_c_l = b_c.cpu().tolist()
    b_l_l = b_l.cpu().tolist()
    b_p_l = b_p.cpu().tolist()
    for c, lv, p in zip(b_c_l, b_l_l, b_p_l):
        if p not in p_to_blue:
            p_to_blue[p] = {}
        if c not in p_to_blue[p] or lv < p_to_blue[p][c]:
            p_to_blue[p][c] = lv

    return p_to_red, p_to_blue


# ---------------------------------------------------------------------------
# Proxy distance for a single pair
# ---------------------------------------------------------------------------

def proxy_distance(p_all_b, p_all_a, p_to_red, p_to_blue, epsilon):
    """
    Compute the clustering proxy distance between P_all points p_all_b and
    p_all_a.

    For every center q (red or blue) that contains BOTH points:
        proxy_q = 2 * max(level_of_b_in_q, level_of_a_in_q) * epsilon

    Return the minimum such proxy_q over all shared centers.
    If no shared center exists, return float('inf').
    """
    best = float('inf')

    # --- shared red centers ---
    rc_b = p_to_red.get(p_all_b, {})
    rc_a = p_to_red.get(p_all_a, {})
    for q in rc_b:
        if q in rc_a:
            proxy = 2 * max(rc_b[q], rc_a[q]) * epsilon
            if proxy < best:
                best = proxy

    # --- shared blue centers ---
    bc_b = p_to_blue.get(p_all_b, {})
    bc_a = p_to_blue.get(p_all_a, {})
    for q in bc_b:
        if q in bc_a:
            proxy = 2 * max(bc_b[q], bc_a[q]) * epsilon
            if proxy < best:
                best = proxy

    return best


# ---------------------------------------------------------------------------
# Compute ratio distributions
# ---------------------------------------------------------------------------

def compute_ratios(query_pall_indices, neighbor_pall_indices,
                   true_dists_np,
                   p_to_red, p_to_blue, epsilon, label):
    """
    For each query point q_i and its k nearest neighbors n_{i,j}:
      ratio_{i,j} = proxy(q_i, n_{i,j}) / true_dist(q_i, n_{i,j})

    query_pall_indices  : 1-D array, P_all indices of query points
    neighbor_pall_indices : (Q, k) array, P_all indices of each neighbor
    true_dists_np       : (Q, k) array of true L1 distances

    Returns:
      ratios      : flat list of finite ratios
      n_uncovered : count of pairs with no shared center (proxy=inf)
      n_zero_dist : count of pairs with true_dist==0 (skipped)
    """
    ratios = []
    n_uncovered = 0
    n_zero_dist = 0
    Q, k = neighbor_pall_indices.shape

    for i in range(Q):
        q_pall = int(query_pall_indices[i])
        for j in range(k):
            n_pall = int(neighbor_pall_indices[i, j])
            td = float(true_dists_np[i, j])
            if td == 0.0:
                n_zero_dist += 1
                continue
            prx = proxy_distance(q_pall, n_pall, p_to_red, p_to_blue, epsilon)
            if prx == float('inf'):
                n_uncovered += 1
            else:
                ratios.append(prx / td)

        if (i + 1) % 500 == 0:
            print(f"    [{label}] processed {i+1}/{Q} query points ...", flush=True)

    return ratios, n_uncovered, n_zero_dist


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_distributions(blue_ratios, red_ratios, epsilon, N, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Proxy/True L1 Distance Ratio Distribution  (N={N}, ε={epsilon})\n"
        f"Each query point's √N nearest cross-color neighbors",
        fontsize=13
    )

    for ax, ratios, label, color in [
        (axes[0], blue_ratios, "Blue query → Red neighbors", "steelblue"),
        (axes[1], red_ratios,  "Red query → Blue neighbors", "tomato"),
    ]:
        ratios_arr = np.array(ratios, dtype=np.float64)
        # Clip extreme outliers for display (keep 99.5th percentile)
        clip_val = np.percentile(ratios_arr, 99.5)
        ratios_clipped = ratios_arr[ratios_arr <= clip_val]

        ax.hist(ratios_clipped, bins=80, density=True, alpha=0.45,
                color=color, label="Histogram")

        if len(ratios_clipped) > 10:
            kde = gaussian_kde(ratios_clipped, bw_method="scott")
            xs = np.linspace(ratios_clipped.min(), ratios_clipped.max(), 400)
            ax.plot(xs, kde(xs), color=color, linewidth=2, label="KDE")

        ax.axvline(1.0, color="black", linestyle="--", linewidth=1, label="ratio=1")
        ax.axvline(4.0 + epsilon, color="grey", linestyle=":",
                   linewidth=1, label=f"theory bound (4+ε)")

        mean_r  = float(ratios_arr.mean())
        med_r   = float(np.median(ratios_arr))
        pct95_r = float(np.percentile(ratios_arr, 95))
        ax.set_title(
            f"{label}\n"
            f"mean={mean_r:.3f}  median={med_r:.3f}  95th pct={pct95_r:.3f}",
            fontsize=10
        )
        ax.set_xlabel("proxy / true L1", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="E4: Proxy Distance Ratio Distribution"
    )
    parser.add_argument("--n",        type=int,   default=5000)
    parser.add_argument("--epsilon",  type=float, default=0.01,
                        help="Clustering epsilon (normalized)")
    parser.add_argument("--seed",     type=int,   default=42)
    parser.add_argument("--data_dir", type=str,   default=None)
    parser.add_argument("--metric",   type=str,   default="L1", choices=["L1", "L2"])
    parser.add_argument("--out",      type=str,   default="e4_ratio_distribution.png",
                        help="Output plot filename")
    args = parser.parse_args()

    N       = args.n
    epsilon = args.epsilon
    k_nn    = max(1, int(math.floor(math.sqrt(N))))   # floor(sqrt(N))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  N={N}  ε={epsilon}  k_nn={k_nn}\n")

    # --- 1. Load data ---
    print("Loading MNIST ...", flush=True)
    red, blue = load_mnist_flat(N, seed=args.seed, data_dir=args.data_dir)
    print(f"  red: {red.shape}  blue: {blue.shape}")

    # --- 2. Clustering ---
    print("\nRunning clustering ...", flush=True)
    P_red_dev  = red.to(device)
    P_blue_dev = blue.to(device)
    P_all_dev  = torch.cat([P_red_dev, P_blue_dev], dim=0)

    # L1 diameter for normalization
    diameter = (P_all_dev.max(dim=0).values - P_all_dev.min(dim=0).values).sum().item()
    diameter = max(diameter, 1e-9)
    P_red_norm  = P_red_dev  / diameter
    P_blue_norm = P_blue_dev / diameter

    t0 = time.time()
    clustering = ColorAwareClustering(epsilon=epsilon, metric=args.metric)
    (b_c, b_l, b_p), (r_c, r_l, r_p) = clustering.run(P_red_norm, P_blue_norm)
    if device.type == "cuda":
        torch.cuda.synchronize()
    print(f"  Clustering done in {time.time()-t0:.2f}s  "
          f"({r_p.numel()} red edges, {b_p.numel()} blue edges)", flush=True)

    # Free GPU clustering data
    del P_red_norm, P_blue_norm, P_all_dev

    # --- 3. Build inverted index (CPU, small) ---
    print("\nBuilding inverted index ...", flush=True)
    t0 = time.time()
    p_to_red, p_to_blue = build_inverted_index(r_c, r_l, r_p, b_c, b_l, b_p)
    del r_c, r_l, r_p, b_c, b_l, b_p
    print(f"  Done in {time.time()-t0:.2f}s  "
          f"({len(p_to_red)} red-center entries, {len(p_to_blue)} blue-center entries)",
          flush=True)

    # --- 4a. kNN: blue queries → red targets ---
    print(f"\nComputing kNN: blue → red (k={k_nn}) ...", flush=True)
    blue_dev = blue.to(device)
    red_dev  = red.to(device)
    # True distances are in original (un-normalized) pixel space
    knn_br_idx, knn_br_dist = knn_l1(blue_dev, red_dev, k=k_nn)
    # knn_br_idx[i, j] = red index (0..N-1) of j-th nearest red neighbor of blue i
    del blue_dev, red_dev
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Convert to P_all indices for blue queries: blue i → P_all index N+i
    blue_pall = np.arange(N, 2*N, dtype=np.int64)     # (N,)
    # Red neighbor indices are already 0..N-1 (P_all indices for red)
    # knn_br_idx is already red P_all indices since red = 0..N-1

    # --- 5a. Compute ratios for blue→red ---
    print("\nComputing proxy ratios (blue → red) ...", flush=True)
    t0 = time.time()
    blue_ratios, blue_uncov, blue_zero = compute_ratios(
        query_pall_indices    = blue_pall,
        neighbor_pall_indices = knn_br_idx,
        true_dists_np         = knn_br_dist,
        p_to_red              = p_to_red,
        p_to_blue             = p_to_blue,
        epsilon               = epsilon,
        label                 = "blue→red",
    )
    print(f"  Done in {time.time()-t0:.2f}s  "
          f"covered={len(blue_ratios)}  uncovered={blue_uncov}  zero_dist={blue_zero}",
          flush=True)

    # --- 4b. kNN: red queries → blue targets ---
    print(f"\nComputing kNN: red → blue (k={k_nn}) ...", flush=True)
    red_dev2  = red.to(device)
    blue_dev2 = blue.to(device)
    knn_rb_idx, knn_rb_dist = knn_l1(red_dev2, blue_dev2, k=k_nn)
    # knn_rb_idx[i, j] = blue index (0..N-1) of j-th nearest blue neighbor of red i
    # → P_all index = N + knn_rb_idx[i, j]
    del red_dev2, blue_dev2
    if device.type == "cuda":
        torch.cuda.empty_cache()

    red_pall = np.arange(0, N, dtype=np.int64)                  # (N,) red P_all indices
    knn_rb_pall = knn_rb_idx + N                                  # (N, k) blue P_all indices

    # --- 5b. Compute ratios for red→blue ---
    print("\nComputing proxy ratios (red → blue) ...", flush=True)
    t0 = time.time()
    red_ratios, red_uncov, red_zero = compute_ratios(
        query_pall_indices    = red_pall,
        neighbor_pall_indices = knn_rb_pall,
        true_dists_np         = knn_rb_dist,
        p_to_red              = p_to_red,
        p_to_blue             = p_to_blue,
        epsilon               = epsilon,
        label                 = "red→blue",
    )
    print(f"  Done in {time.time()-t0:.2f}s  "
          f"covered={len(red_ratios)}  uncovered={red_uncov}  zero_dist={red_zero}",
          flush=True)

    # --- 6. Summary statistics ---
    print("\n" + "=" * 60)
    print(f"{'':30s}  {'blue→red':>12}  {'red→blue':>12}")
    print("-" * 60)
    for label, ratios, uncov, zero in [
        ("blue→red", blue_ratios, blue_uncov, blue_zero),
        ("red→blue", red_ratios,  red_uncov,  red_zero),
    ]:
        if ratios:
            arr = np.array(ratios)
            print(f"\n  [{label}]")
            print(f"    total pairs   : {N * k_nn}")
            print(f"    covered       : {len(ratios)}  ({100*len(ratios)/(N*k_nn):.1f}%)")
            print(f"    uncovered     : {uncov}  ({100*uncov/(N*k_nn):.1f}%)")
            print(f"    zero-dist skip: {zero}")
            print(f"    ratio mean    : {arr.mean():.4f}")
            print(f"    ratio median  : {np.median(arr):.4f}")
            print(f"    ratio 25th pct: {np.percentile(arr, 25):.4f}")
            print(f"    ratio 75th pct: {np.percentile(arr, 75):.4f}")
            print(f"    ratio 95th pct: {np.percentile(arr, 95):.4f}")
            print(f"    ratio max     : {arr.max():.4f}")
            print(f"    theory bound  : {4 + epsilon:.4f}")
    print("=" * 60)

    # --- 7. Plot ---
    print("\nGenerating plot ...", flush=True)
    if blue_ratios and red_ratios:
        plot_distributions(blue_ratios, red_ratios, epsilon, N, args.out)
    else:
        print("  Not enough covered pairs to plot.")


if __name__ == "__main__":
    main()