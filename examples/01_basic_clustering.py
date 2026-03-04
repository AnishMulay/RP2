import torch
from clustered_push_relabel import k_level_cluster

def main():
    print("Running 01_basic_clustering.py...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Generate random point clouds
    N = 5000
    D = 3
    x = torch.rand(N, D, device=device)
    y = torch.rand(N, D, device=device)
    
    # Run K-level clustering
    epsilon = 0.05
    k = 4
    
    print(f"Running clustering with K={k}, Epsilon={epsilon}...")
    result = k_level_cluster(x, y, epsilon=epsilon, k=k)
    
    print("\nClustering Output:")
    print(f"Red cover (source) edges: {result['red_cover'][0].numel()}")
    print(f"Blue cover (target) edges: {result['blue_cover'][0].numel()}")
    print("Done!")

if __name__ == "__main__":
    main()
