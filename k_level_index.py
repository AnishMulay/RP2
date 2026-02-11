import torch

class KLevelVectorIndex:
    """
    Implements the 'All-Points as Centroids' Clustering.
    Level 1: Randomly sampled Landmarks (S).
    Level 0: All points (P).
    Cluster Definition: y is in C_p iff dist(y, p) <= dist(y, S).
    """
    def __init__(self, k=2, batch_size=1024):
        # k is conceptually 2 here (Landmarks + Points)
        self.batch_size = batch_size
        self.dataset = None
        self.crow_indices = None
        self.col_indices = None

    def build_index(self, X):
        """
        Builds the index where EVERY point in X is a potential cluster center.
        To avoid OOM, we process the N x N comparisons in batches.
        """
        N = X.shape[0]
        print(f"[*] Building All-Points Index on {N} vectors...")
        self.dataset = X

        # 1. Sample Landmarks (S) - Rule of thumb: sqrt(N)
        num_landmarks = int(N**0.5)
        landmark_indices = torch.randperm(N, device=X.device)[:num_landmarks]
        landmarks = X[landmark_indices]
        
        print(f"[*] Sampled {num_landmarks} Landmarks (S).")

        # 2. Pre-calculate Thresholds for every point y
        # D_S[y] = min_dist(y, S)
        # We process X in chunks to find min dist to landmarks
        D_S = torch.zeros(N, device=X.device)
        
        # Batching threshold calc
        for i in range(0, N, self.batch_size):
            end = min(i + self.batch_size, N)
            batch_X = X[i:end] # (B, D)
            # dists to landmarks
            dists = torch.cdist(batch_X, landmarks) # (B, NumLandmarks)
            min_dists, _ = dists.min(dim=1)
            D_S[i:end] = min_dists
        
        print("[*] Thresholds (dist to S) calculated.")

        # 3. Build Clusters (The Heavy Lifting)
        # For every point p (potential center), which y belong to it?
        # Condition: dist(y, p) <= D_S[y]
        
        # Since we search by "Find closest p, then check C_p", we need to store C_p.
        # C_p = {y | dist(y, p) <= D_S[y]}
        # Warning: This is O(N^2). We will limit N in the runner if needed.
        
        # To store this efficiently in CSR, we build it row by row (or batch rows).
        # But fully materializing adjacency is huge.
        # Strategy: We accumulate adjacency lists on CPU, then move to GPU CSR.
        
        row_pointers = [0]
        all_cols = []
        
        print("[*] Constructing clusters (this may take time)...")
        
        # Iterate over 'p' (The Centers)
        for i in range(0, N, self.batch_size):
            end = min(i + self.batch_size, N)
            batch_centers = X[i:end] # (Batch, D)
            
            # Compare against ALL y (Targets)
            # To avoid huge matrix, we loop over targets in chunks too?
            # For 50k x 50k, we can do (Batch_Centers x All_Targets) if memory allows.
            # 1024 x 50000 x 4 bytes = ~200MB. This fits easily on GPU.
            
            dists_p = torch.cdist(batch_centers, X) # (Batch, N)
            
            # Check condition: dist(y, p) <= D_S[y]
            # dists_p[local_p, y] <= D_S[y]
            # D_S is (N,). Broadcast properly.
            
            mask = dists_p <= D_S.unsqueeze(0) # (Batch, N)
            
            # Convert mask to sparse indices
            # We do this per center to maintain order
            
            # Move to CPU to append to lists (avoids VRAM fragmentation)
            mask_cpu = mask.cpu()
            
            for local_idx in range(end - i):
                # indices of points y that belong to center (i + local_idx)
                indices = torch.nonzero(mask_cpu[local_idx]).squeeze(1)
                all_cols.append(indices)
                row_pointers.append(row_pointers[-1] + len(indices))
                
            if (i // self.batch_size) % 10 == 0:
                print(f"    Processed {i}/{N} centers...")

        # 4. Finalize CSR on GPU
        self.crow_indices = torch.tensor(row_pointers, dtype=torch.long, device=X.device)
        self.col_indices = torch.cat(all_cols).to(X.device)
        
        avg_cluster = (self.crow_indices[-1].item()) / N
        print(f"[*] Index Built. Avg Cluster Size: {avg_cluster:.1f}")