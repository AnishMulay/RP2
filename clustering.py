import torch
import numpy as np
import time

# ==========================================
# Low-Level Component Definitions (SOLID Principles)
# ==========================================

def default_euclidean_metric(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Computes pairwise Euclidean distances between two sets of points.
    Implements the 'metric d(.,.)' dependency.
    
    Args:
        x: Tensor of shape (N, D)
        y: Tensor of shape (M, D)
    Returns:
        Tensor of shape (N, M) containing pairwise distances.
    """
    # p=2 denotes Euclidean distance. cdist is highly optimized for GPUs.
    return torch.cdist(x, y, p=2)


class ParallelWpClusteringAlgo:
    """
    Encapsulates the core logic of the Parallel Two-Layered Clustering algorithm.
    Designed to be stateless regarding the data P, taking inputs explicitly.
    """

    def __init__(self, epsilon: float, metric_func=default_euclidean_metric):
        """
        Args:
            epsilon: approximation parameter > 0.
            metric_func: A callable implementing distance calculation between tensor batches.
                         Defaults to Euclidean distance. (Dependency Injection)
        """
        self.epsilon = epsilon
        self.dist_func = metric_func

    def _step1_parallel_sampling(self, n_points: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Implements Step 1: Parallel Initialization and Sampling.
        Selects landmarks P1 from P0 with probability n^-1/2.
        """
        # Probability threshold
        prob = n_points ** (-0.5)
        
        # Generate random values and compare against threshold in parallel
        # This creates a boolean mask of shape (N,)
        mask_P1 = torch.rand(n_points, device=device) < prob
        
        # Safety fallback: Ensure at least one landmark is chosen if n is small, 
        # otherwise subsequent steps might fail.
        if not mask_P1.any():
             mask_P1[torch.randint(0, n_points, (1,), device=device)] = True

        # Extract indices where the mask is True
        P1_indices = torch.nonzero(mask_P1).squeeze(1)
        
        # Comments map to pseudocode:
        # Lines 3-7: P0 is implicit. P1 is defined by P1_indices and mask_P1.
        return P1_indices, mask_P1

    def _step2_precompute_landmark_distances(self, P: torch.Tensor, P1_indices: torch.Tensor) -> torch.Tensor:
        """
        Implements Step 2: Precompute Nearest Landmark Distances (Parallel).
        Computes D[y] = min_{z in P1} d(y, z) for all y in P0.
        """
        # Gather landmark points using indices
        P1_points = P[P1_indices]
        
        # Compute distance matrix between all points P and landmarks P1.
        # Shape results in (N_points, N_landmarks)
        dists_to_P1 = self.dist_func(P, P1_points)
        
        # Find the minimum distance along dim 1 (across landmarks for each point y)
        # This parallelizes the minimization loop in pseudocode line 9-11.
        # D_y shape is (N_points,)
        D_y, _ = torch.min(dists_to_P1, dim=1)
        return D_y

    def _step3_define_radii_scales(self, P: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Implements Step 3: Define Radii Scales.
        Computes max distance Delta and generates the geometric sequence of radii.
        """
        # Line 12: Compute Delta via parallel reduction.
        # Note: cdist(P,P) is O(N^2) memory. For extremely large N on limited GPU memory,
        # this step might need batched processing or approximation, but fits standard HPC usage.
        all_pairwise_dists = self.dist_func(P, P)
        Delta = torch.max(all_pairwise_dists)

        # Handle edge case where all points are identical
        if Delta <= 1e-9:
             return torch.tensor([0.0], device=device)

        # Line 13: Calculate t
        base = 1.0 + self.epsilon / 4.0
        # log_base(Delta) = ln(Delta) / ln(base)
        t_float = torch.log(Delta) / torch.log(torch.tensor(base, device=device))
        t = torch.ceil(t_float).long().item()

        # Line 14: Define radii sequence
        # Generate indices 1 to t
        indices = torch.arange(1, t + 1, device=device, dtype=P.dtype)
        radii_tail = torch.pow(base, indices)
        
        # Concatenate r_0 = 0 with the rest of the sequence.
        # Radii shape is (t+1,)
        radii = torch.cat([torch.tensor([0.0], device=device, dtype=P.dtype), radii_tail])
        return radii

    def _step4_parallel_cluster_construction(self, P: torch.Tensor, mask_P1: torch.Tensor, 
                                             D_y: torch.Tensor, radii: torch.Tensor) -> list[torch.Tensor]:
        """
        Implements Step 4: Parallel Cluster Construction.
        This is the core complex logic involving nested loops and conditional checks
        based on landmark status (Case A vs Case B). We vectorize this heavily.
        """
        N = P.shape[0]
        num_radii = radii.shape[0]

        # Precompute all pairwise distances d(x,q) for the inner loops.
        # Shape: (N_x, N_q)
        d_xq = self.dist_func(P, P)

        # --- Vectorizing the conditions ---
        # We aim to create a massive boolean mask of shape (N_x, N_q, num_radii)
        # representing whether point x belongs to cluster C_q[i].

        # Condition 1: d(x, q) <= r_i (Used in both Case A and B)
        # Broadcast d_xq (N, N, 1) against radii (1, 1, T+1)
        # cond_dist_mask shape: (N_x, N_q, num_radii)
        cond_dist_mask = d_xq.unsqueeze(-1) <= radii.reshape(1, 1, -1)

        # Condition 2: d(x, q) < D[x] (Used only in Case B: q is NOT a landmark)
        # Broadcast d_xq (N, N) against D_y (N, 1)
        # cond_voronoi_mask shape: (N_x, N_q)
        cond_voronoi_mask = d_xq < D_y.unsqueeze(1)

        # --- Combining conditions based on whether q is a landmark ---
        # mask_P1 is shape (N_q,). Reshape to (1, N_q, 1) for broadcasting.
        is_landmark_q = mask_P1.reshape(1, N, 1)

        # The selection logic corresponds to pseudocode lines 18-25:
        # If q is landmark (True in is_landmark_q): membership is just cond_dist_mask.
        # If q is NOT landmark (False): membership is cond_dist_mask AND cond_voronoi_mask.
        # We must unsqueeze cond_voronoi_mask to match the radii dimension.
        final_membership_3d = torch.where(
            is_landmark_q,
            cond_dist_mask,
            cond_dist_mask & cond_voronoi_mask.unsqueeze(-1)
        )

        # --- Collecting results ("Atomic Add to C") ---
        # Pseudocode lines 26-27 imply adding non-empty clusters to a collection.
        # In PyTorch, we iterate the realized 3D mask to extract indices.
        # While the comparisons were fully parallel on GPU, extracting variable-sized
        # lists is best done via a sequential scan on the CPU or via advanced GPU kernels
        # (like prefix sums/stream compaction) if N is massive. 
        # A sequential scan over the (q, i) dimensions is usually acceptable here.
        
        clusters: list[torch.Tensor] = []
        
        # Move mask to CPU for sequential extraction loop if N is moderate size, 
        # otherwise keep on GPU and accept slower indexing. Let's keep on GPU for now.
        # final_membership_3d_cpu = final_membership_3d.cpu() 

        # Iterate over centers q
        for q_idx in range(N):
            # Iterate over radii index i
            for r_idx in range(num_radii):
                 # Extract boolean mask of members x for this specific (q, i) cluster
                 member_mask_x = final_membership_3d[:, q_idx, r_idx]
                 
                 # If cluster is not empty (line 26)
                 if member_mask_x.any():
                     # Get indices of members (line 27 implementation)
                     member_indices = torch.nonzero(member_mask_x).squeeze(1)
                     clusters.append(member_indices)

        return clusters

    def run(self, P: torch.Tensor) -> list[torch.Tensor]:
        """
        Main execution method tying all steps together.
        
        Args:
            P: Input point set tensor of shape (N, Dimensions).
        Returns:
            A list of tensors, where each tensor contains indices of points belonging to a cluster.
        """
        n_points = P.shape[0]
        device = P.device
        
        # Ensure data is floating point for distance calculations
        if not P.is_floating_point():
             P = P.to(torch.float32)

        # Step 1: Parallel Initialization and Sampling
        P1_indices, mask_P1 = self._step1_parallel_sampling(n_points, device)
        
        # Step 2: Precompute Nearest Landmark Distances
        D_y = self._step2_precompute_landmark_distances(P, P1_indices)
        
        # Step 3: Define Radii Scales
        radii = self._step3_define_radii_scales(P, device)
        
        # Step 4: Parallel Cluster Construction
        clusters = self._step4_parallel_cluster_construction(P, mask_P1, D_y, radii)
        
        return clusters


# ==========================================
# Test Infrastructure and Execution
# ==========================================

def generate_synthetic_data(n_samples=1000, n_features=2, centers=5, cluster_std=1.0):
    """Generates synthetic blobs for testing using sklearn."""
    from sklearn.datasets import make_blobs
    X, y = make_blobs(n_samples=n_samples, n_features=n_features, 
                      centers=centers, cluster_std=cluster_std, random_state=42)
    return torch.tensor(X, dtype=torch.float32)

def run_test_case(test_name, P_tensor, epsilon, device_str="cuda"):
    print(f"\n--- Running Test Case: {test_name} ---")
    
    if not torch.cuda.is_available() and device_str == "cuda":
        print("CUDA not available, reverting to CPU.")
        device_str = "cpu"
        
    device = torch.device(device_str)
    print(f"Running on device: {device}")
    
    # Move data to target device
    P_gpu = P_tensor.to(device)
    N = P_gpu.shape[0]
    print(f"Input data shape: {P_gpu.shape} (N={N})")
    print(f"Epsilon: {epsilon}")

    # Instantiate algorithm
    algo = ParallelWpClusteringAlgo(epsilon=epsilon)

    # Run and time execution
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    
    clusters = algo.run(P_gpu)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    end_time = time.time()
    
    print(f"Clustering completed in {end_time - start_time:.4f} seconds.")
    print(f"Total clusters found: {len(clusters)}")
    
    # Basic validation stats
    if len(clusters) > 0:
        avg_size = sum(len(c) for c in clusters) / len(clusters)
        max_size = max(len(c) for c in clusters)
        print(f"Average cluster size: {avg_size:.2f}")
        print(f"Max cluster size: {max_size}")
        
        # Example: print indices of the first cluster found
        # print(f"Indices of first cluster: {clusters[0].cpu().numpy()}")
    else:
        print("WARNING: No clusters found (unexpected for non-trivial data).")
    
    print("-------------------------------------\n")


if __name__ == "__main__":
    # Configuration for tests
    EPSILON = 0.1
    TARGET_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- Test Case 1: Small debugging set (Manual points) ---
    # Create a few distinct groups manually
    P_small = torch.tensor([
        [1.0, 1.0], [1.1, 1.1], [1.0, 1.2], # Group 1
        [10.0, 10.0], [10.1, 10.1],         # Group 2
        [50.0, 50.0]                        # Group 3 (isolated)
    ], dtype=torch.float32)
    run_test_case("Small Debug Set", P_small, epsilon=2.0, device_str=TARGET_DEVICE)

    # --- Test Case 2: Medium sized, well-separated blobs ---
    # This should yield a reasonable number of clusters reflecting the structure.
    try:
        P_blobs_clean = generate_synthetic_data(n_samples=2000, centers=10, cluster_std=0.5)
        run_test_case("Medium Well-Separated Blobs", P_blobs_clean, epsilon=EPSILON, device_str=TARGET_DEVICE)
    except ImportError:
        print("Skipping synthetic data tests (sklearn not installed).")

    # --- Test Case 3: Larger, noisy/overlapping blobs ---
    # A harder case for approximation algorithms.
    try:
        P_blobs_noisy = generate_synthetic_data(n_samples=5000, centers=15, cluster_std=2.5)
        run_test_case("Large Noisy Blobs", P_blobs_noisy, epsilon=EPSILON, device_str=TARGET_DEVICE)
    except ImportError:
        pass

    # --- Test Case 4: Higher dimensionality ---
    try:
        P_high_dim = generate_synthetic_data(n_samples=1000, n_features=64, centers=5, cluster_std=1.0)
        run_test_case("High Dimensional Data (D=64)", P_high_dim, epsilon=EPSILON, device_str=TARGET_DEVICE)
    except ImportError:
        pass