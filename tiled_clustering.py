import torch
import math
import time
import numpy as np

# ==========================================
# Component 1: Tiled Euclidean Kernel (The Engine)
# ==========================================

class TiledEuclideanKernel:
    """
    Handles distance computations in a memory-efficient, tiled manner.
    Leverages the expansion: ||x - y||^2 = ||x||^2 + ||y||^2 - 2 <x, y>.
    This avoids materializing (N, N, D) tensors and uses highly optimized BLAS (addmm).
    """

    def __init__(self, chunk_size: int = 1024):
        self.chunk_size = chunk_size

    def prepare_workspace(self, P: torch.Tensor) -> dict:
        """
        Precomputes values needed for fast distance calculation.
        Args:
            P: Input tensor (N, D)
        Returns:
            Dictionary containing cached norms and transposed data.
        """
        # Precompute squared norms: ||x||^2 for all points
        # shape: (N, 1)
        P_norms_sq = (P ** 2).sum(dim=1, keepdim=True)
        
        # Transpose P for matrix multiplication later: (D, N)
        P_T = P.t()
        
        return {
            "P": P,
            "P_T": P_T,
            "P_norms_sq": P_norms_sq
        }

    def compute_squared_dist_tile(self, query_indices: torch.Tensor, workspace: dict) -> torch.Tensor:
        """
        Computes squared Euclidean distances between ALL points P and a subset (query_indices).
        
        Args:
            query_indices: Indices of the 'centers' or 'landmarks' we are comparing against.
            workspace: The cached data from prepare_workspace.
            
        Returns:
            Tensor of shape (N, Batch_Size) containing squared distances.
        """
        P = workspace["P"]
        P_T = workspace["P_T"]
        P_norms_sq = workspace["P_norms_sq"]
        
        # 1. Get the subset of points Q corresponding to query_indices
        # shape: (Batch_Size, D)
        Q = P[query_indices]
        
        # 2. Get their norms
        # shape: (Batch_Size, 1) -> transpose to (1, Batch_Size) for broadcasting
        Q_norms_sq = P_norms_sq[query_indices].t()
        
        # 3. Compute term -2 <P, Q^T>
        # P is (N, D), Q is (Batch, D). We actually want P @ Q.T
        # Optimization: We use addmm for alpha*AB + beta*C
        # We want: P_norms_sq + Q_norms_sq - 2(P @ Q.T)
        
        # Initialize result with P_norms (N, 1) broadcasted against Q_norms (1, Batch)
        # Result shape: (N, Batch)
        dists_sq = P_norms_sq + Q_norms_sq
        
        # Perform matrix multiplication: result += -2.0 * (P @ Q.T)
        # P (N, D) @ Q.T (D, Batch) -> (N, Batch)
        dists_sq.addmm_(P, Q.t(), beta=1.0, alpha=-2.0)
        
        # Clamp negative values due to floating point noise to 0.0
        return torch.clamp(dists_sq, min=0.0)


# ==========================================
# Component 2: The Parallel Clustering Algorithm
# ==========================================

class ParallelWpClusteringAlgo:
    """
    Space-Efficient Implementation of Parallel Two-Layered Clustering.
    Refactored to stream the distance matrix instead of materializing it.
    """

    def __init__(self, epsilon: float, batch_size: int = 2048):
        """
        Args:
            epsilon: Approximation parameter > 0.
            batch_size: Size of chunks for processing steps 2, 3, and 4. 
                        Adjust based on GPU VRAM (1024-4096 is usually sweet spot).
        """
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.kernel = TiledEuclideanKernel(chunk_size=batch_size)

    def _step1_parallel_sampling(self, P: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Step 1: Parallel Initialization and Sampling.
        O(N) memory, O(1) ops per point.
        """
        n_points = P.shape[0]
        device = P.device
        
        # Probability n^(-1/2)
        prob = n_points ** (-0.5)
        
        # Generate mask
        mask_P1 = torch.rand(n_points, device=device) < prob
        
        # Safety: Ensure at least one landmark
        if not mask_P1.any():
             mask_P1[torch.randint(0, n_points, (1,), device=device)] = True

        P1_indices = torch.nonzero(mask_P1).squeeze(1)
        return P1_indices, mask_P1

    def _step2_precompute_landmark_distances(self, P: torch.Tensor, P1_indices: torch.Tensor, workspace: dict) -> torch.Tensor:
        """
        Step 2: Precompute Nearest Landmark Distances (Tiled).
        Computes D[y] = min_{z in P1} d(y, z).
        
        Optimized: Instead of one giant matrix, we process P1 in chunks if P1 is large.
        Since |P1| is sqrt(N), it's usually small, but for robustness we tile it.
        """
        n_points = P.shape[0]
        device = P.device
        num_landmarks = P1_indices.shape[0]
        
        # Initialize D[y] with infinity
        D_y_sq = torch.full((n_points,), float('inf'), device=device)
        
        # Iterate over landmarks in batches
        for i in range(0, num_landmarks, self.batch_size):
            end_i = min(i + self.batch_size, num_landmarks)
            batch_indices = P1_indices[i:end_i]
            
            # Compute squared distances from all points P to this batch of landmarks
            # Shape: (N, Batch_Size)
            dists_sq_batch = self.kernel.compute_squared_dist_tile(batch_indices, workspace)
            
            # Find min for this batch
            batch_min_sq, _ = torch.min(dists_sq_batch, dim=1)
            
            # Update global min
            D_y_sq = torch.min(D_y_sq, batch_min_sq)
            
        return D_y_sq

    def _step3_define_radii_scales(self, P: torch.Tensor, workspace: dict) -> torch.Tensor:
        """
        Step 3: Define Radii Scales (Tiled Reduction).
        Computes Delta = max_{p,q} d(p,q) without N^2 matrix.
        """
        n_points = P.shape[0]
        device = P.device
        max_dist_sq = 0.0
        
        # Compute max distance by streaming chunks of P
        for i in range(0, n_points, self.batch_size):
            end_i = min(i + self.batch_size, n_points)
            batch_indices = torch.arange(i, end_i, device=device)
            
            # Compute distances from all P to this batch P[i:end]
            # Shape: (N, Batch_Size)
            dists_sq_batch = self.kernel.compute_squared_dist_tile(batch_indices, workspace)
            
            # Get max of this batch
            current_batch_max = torch.max(dists_sq_batch).item()
            if current_batch_max > max_dist_sq:
                max_dist_sq = current_batch_max
        
        Delta = math.sqrt(max_dist_sq)
        
        # Handle single point case
        if Delta <= 1e-9:
             return torch.tensor([0.0], device=device)

        # Geometric sequence definition
        base = 1.0 + self.epsilon / 4.0
        t_float = math.log(Delta) / math.log(base)
        t = int(math.ceil(t_float))
        
        # Construct radii vector
        indices = torch.arange(1, t + 1, device=device, dtype=P.dtype)
        radii_tail = torch.pow(base, indices)
        radii = torch.cat([torch.tensor([0.0], device=device, dtype=P.dtype), radii_tail])
        
        return radii

    def _step4_tiled_cluster_construction(self, P: torch.Tensor, mask_P1: torch.Tensor, 
                                         D_y_sq: torch.Tensor, radii: torch.Tensor, 
                                         workspace: dict) -> list[torch.Tensor]:
        """
        Step 4: Parallel Cluster Construction (The Tiled 'Solver' Loop).
        Iterates over 'centers' q in batches to avoid OOM.
        """
        n_points = P.shape[0]
        device = P.device
        
        # Pre-square radii for fast comparison
        radii_sq = radii ** 2
        num_radii = radii.shape[0]
        
        clusters = []
        
        # Loop over potential centers 'q' in batches
        for start_q in range(0, n_points, self.batch_size):
            end_q = min(start_q + self.batch_size, n_points)
            q_batch_size = end_q - start_q
            
            # Indices for this batch of centers
            q_indices = torch.arange(start_q, end_q, device=device)
            
            # 1. Compute distances: All points x vs Batch of centers q
            # Shape: (N_x, Batch_q)
            d_xq_sq = self.kernel.compute_squared_dist_tile(q_indices, workspace)
            
            # 2. Prepare Condition Checks
            
            # Determine which centers in this batch are landmarks
            # Shape: (1, Batch_q)
            is_landmark_batch = mask_P1[q_indices].unsqueeze(0)
            
            # Voronoi Condition: d(x, q) < D[x]
            # D[x] is D_y_sq (N_x, 1)
            # This is only relevant if q is NOT a landmark.
            # Shape: (N_x, Batch_q)
            if not is_landmark_batch.all():
                cond_voronoi = d_xq_sq < D_y_sq.unsqueeze(1)
            else:
                # If all q are landmarks, we don't need this check (optimization)
                cond_voronoi = None 

            # 3. Iterate over Radii
            # Since T is small (usually < 20), we loop sequentially over radii.
            # This is safer than broadcasting T which triples memory usage.
            for r_idx, r_sq in enumerate(radii_sq):
                
                # Condition A: Distance Threshold d(x, q) <= r_i
                # Shape: (N_x, Batch_q)
                cond_dist = d_xq_sq <= r_sq
                
                # Combine Conditions
                if cond_voronoi is not None:
                    # Logic: If q is landmark -> cond_dist
                    #        Else             -> cond_dist AND cond_voronoi
                    membership_mask = torch.where(
                        is_landmark_batch,
                        cond_dist,
                        cond_dist & cond_voronoi
                    )
                else:
                    membership_mask = cond_dist

                # 4. Extract Clusters from Mask
                # We need to find which columns (centers) have any members.
                # Optimization: Check if ANY point matches first.
                if not membership_mask.any():
                    continue

                # We iterate the batch dimension to extract specific clusters
                # Moving this tiny loop to CPU is fine, or we keep on GPU.
                # Since we need to append to a list, we must process individually.
                
                # Identify which centers in this batch have members
                has_members = membership_mask.any(dim=0) # Shape (Batch_q,)
                valid_centers_local = torch.nonzero(has_members).squeeze(1) # Indices in 0..Batch_q
                
                for local_idx in valid_centers_local:
                    # Get the members for this specific center
                    members = membership_mask[:, local_idx]
                    member_indices = torch.nonzero(members).squeeze(1)
                    clusters.append(member_indices)

        return clusters

    def run(self, P: torch.Tensor) -> list[torch.Tensor]:
        """
        Main execution entry point.
        """
        # Ensure float32 for performance/memory balance
        if not P.is_floating_point():
             P = P.to(torch.float32)

        # Prepare workspace (precomputed norms for kernels)
        workspace = self.kernel.prepare_workspace(P)

        # Step 1: Sampling
        P1_indices, mask_P1 = self._step1_parallel_sampling(P)
        
        # Step 2: Landmark Distances (Squared)
        D_y_sq = self._step2_precompute_landmark_distances(P, P1_indices, workspace)
        
        # Step 3: Radii Scales
        radii = self._step3_define_radii_scales(P, workspace)
        
        # Step 4: Cluster Construction (Streaming)
        clusters = self._step4_tiled_cluster_construction(P, mask_P1, D_y_sq, radii, workspace)
        
        return clusters


# ==========================================
# Test Infrastructure
# ==========================================

def generate_blobs_torch(n_samples=1000, n_features=2, centers=5, cluster_std=1.0, device='cpu'):
    """
    Self-contained blob generation using PyTorch (avoids sklearn dependency for pure torch envs).
    """
    torch.manual_seed(42)
    
    # 1. Generate center locations
    center_locs = torch.randn(centers, n_features, device=device) * 5.0
    
    # 2. Assign each point to a center
    labels = torch.randint(0, centers, (n_samples,), device=device)
    
    # 3. Generate points around centers
    data = torch.randn(n_samples, n_features, device=device) * cluster_std
    data += center_locs[labels]
    
    return data

def run_stress_test():
    print("="*60)
    print("RUNNING STREAMING CLUSTERING STRESS TEST")
    print("="*60)
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Device: CUDA ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        print("Device: CPU (Note: Performance will be slower)")

    # Configuration
    N_POINTS = 20000     # Large enough to stress memory if not tiled
    D_DIM = 64           # Higher dimension
    EPSILON = 2.0
    BATCH_SIZE = 1024    # Tiling size
    
    print(f"Configuration: N={N_POINTS}, D={D_DIM}, Batch={BATCH_SIZE}")
    print("Generating data...")
    P = generate_blobs_torch(n_samples=N_POINTS, n_features=D_DIM, centers=20, device=device)
    
    # Initialize Algorithm
    algo = ParallelWpClusteringAlgo(epsilon=EPSILON, batch_size=BATCH_SIZE)
    
    # Execution
    print("Starting clustering...")
    if device.type == 'cuda': torch.cuda.synchronize()
    start_t = time.perf_counter()
    
    clusters = algo.run(P)
    
    if device.type == 'cuda': torch.cuda.synchronize()
    end_t = time.perf_counter()
    
    # Reporting
    print(f"Done in {end_t - start_t:.4f} seconds.")
    print(f"Clusters found: {len(clusters)}")
    
    if len(clusters) > 0:
        sizes = torch.tensor([len(c) for c in clusters], dtype=torch.float32)
        print(f"Avg Cluster Size: {sizes.mean().item():.2f}")
        print(f"Max Cluster Size: {sizes.max().item():.0f}")
        
        # Correctness sanity check: 
        # For Blobs, we expect roughly 'centers' count of major clusters, 
        # though this algo produces overlapping covers, so count might be higher.
        print("Sanity Check: Passed (Clusters created successfully)")
    else:
        print("Sanity Check: FAILED (No clusters produced)")
        
    print("="*60)

if __name__ == "__main__":
    run_stress_test()