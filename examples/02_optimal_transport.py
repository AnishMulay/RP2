import torch
from clustered_push_relabel import solve_optimal_transport

def main():
    print("Running 02_optimal_transport.py...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    N, M = 200, 200
    D = 2
    
    # Coordinates
    x = torch.rand(N, D, device=device)
    y = torch.rand(M, D, device=device)
    
    # Masses (Unbalanced to test fractional routing)
    mass_x = torch.ones(N, device=device) * 1.5
    mass_y = torch.ones(M, device=device)
    
    epsilon = 0.1
    k = 3
    
    print(f"Solving Optimal Transport with K={k}, Epsilon={epsilon}...")
    result = solve_optimal_transport(x, y, mass_x, mass_y, epsilon=epsilon, k=k)
    
    print("\nOptimal Transport Output:")
    print(f"Number of active flow edges: {result['flow'].numel()}")
    total_flow = result['flow'].float().sum()
    print(f"Total transported mass: {total_flow.item():.4f}")
    print("Done!")

if __name__ == "__main__":
    main()
