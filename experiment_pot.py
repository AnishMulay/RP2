import os, time, gzip, csv
import numpy as np
import torch

# Import POT (Python Optimal Transport) library
import ot

# Import the GPU OT solvers (assumes these modules are in the same directory or installed)
from clustered_push_relabel import GPUClusteredSolver as TwoLevelSolver
from k_level_clustered_push_relabel import GPUClusteredSolver as KLevelSolver

def load_mnist_data(n_samples, device):
    """Load MNIST images, flatten to 784-D, normalize each to sum 1 (with 1e-6 jitter), and sample two sets."""
    # Load raw MNIST training images from local .gz file
    data_path = os.path.join("data", "train-images-idx3-ubyte.gz")
    if not os.path.isfile(data_path):
        raise FileNotFoundError("MNIST file not found at data/train-images-idx3-ubyte.gz. Please download it.")
    with gzip.open(data_path, "rb") as f:
        raw = f.read()
    # Skip header and reshape to [num_images, 784]
    data_np = np.frombuffer(raw, dtype=np.uint8, offset=16).reshape(-1, 784).copy()
    data = torch.from_numpy(data_np).float()
    # Normalize each image to have a total sum of 1 (add small epsilon to avoid zero mass)
    data = data + 1e-6
    data = data / data.sum(dim=1, keepdim=True)
    # Randomly sample n_samples images for red and another n_samples for blue
    total_images = data.size(0)
    if 2 * n_samples > total_images:
        raise ValueError(f"Not enough images: need {2*n_samples}, but dataset has {total_images}.")
    torch.manual_seed(42)  # for reproducibility
    indices = torch.randperm(total_images)
    idx_red = indices[:n_samples]
    idx_blue = indices[n_samples:2*n_samples]
    P_red = data[idx_red].to(device)
    P_blue = data[idx_blue].to(device)
    return P_red, P_blue

def compute_cost_matrix_l1(P_red, P_blue):
    """Compute the n×n cost matrix of L1 distances between two sets of points (red and blue)."""
    # Ensure data on CPU for cost computation (POT expects numpy arrays)
    X = P_red.cpu().numpy()
    Y = P_blue.cpu().numpy()
    n = X.shape[0]
    # Compute Manhattan distances. Use vectorization in chunks to avoid memory blow-up.
    C = np.zeros((n, n), dtype=np.float64)
    chunk_size = 100  # adjust as needed for memory
    for i in range(0, n, chunk_size):
        i_end = min(n, i + chunk_size)
        # Broadcast subtraction to get shape (chunk, n, 784), then sum absolute differences
        diff = np.abs(X[i:i_end, None, :] - Y[None, :, :])
        C[i:i_end] = diff.sum(axis=2)
    return C

def run_experiment(n_values, epsilon=0.05, k_level=4, output_csv="compare_pot_emd_results.csv"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Prepare CSV file with header
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n", "epsilon", "k", "algo", "time_s", "cost", "abs_error", "rel_error"])
    # Run experiment for each n
    for n in n_values:
        # 1. Load data
        P_red, P_blue = load_mnist_data(n, device)
        # 2. Compute cost matrix for POT (on CPU)
        C = compute_cost_matrix_l1(P_red, P_blue)
        # Uniform weights for source and target (each of length n summing to 1)
        a = np.full(n, 1.0/n, dtype=np.float64)
        b = np.full(n, 1.0/n, dtype=np.float64)
        # 3. Solve with exact POT EMD (network simplex)
        t0 = time.time()
        P_plan = ot.emd(a, b, C)              # optimal transport plan as an n×n matrix
        exact_cost = float((P_plan * C).sum())  # total cost = <P, C>
        t1 = time.time()
        exact_time = t1 - t0
        # (Optional: verify marginals of P_plan match a and b within tolerance)
        # Compute uniqueness (should be 100% if one-to-one matching)
        row_match = P_plan.argmax(axis=1)
        unique_percent = (np.unique(row_match).size / n) * 100.0
        # 4. Solve with Two-Level Push-Relabel (approximate OT on GPU)
        t2 = time.time()
        solver2 = TwoLevelSolver(P_red, P_blue, epsilon, metric="L1")
        torch.cuda.synchronize() if device.type == "cuda" else None
        t2_clust = time.time()
        solver2.solve()
        torch.cuda.synchronize() if device.type == "cuda" else None
        t3 = time.time()
        approx_cost2 = torch.norm(P_blue - P_red[solver2.MB], p=1, dim=1).sum().item()  # L1 cost of matching
        approx_time2 = t3 - t2
        abs_err2 = abs(approx_cost2 - exact_cost)
        rel_err2 = abs_err2 / max(abs(exact_cost), 1e-9)
        # 5. Solve with k-Level Push-Relabel (k=4)
        t4 = time.time()
        solverK = KLevelSolver(P_red, P_blue, epsilon, k=k_level, metric="L1")
        torch.cuda.synchronize() if device.type == "cuda" else None
        t4_clust = time.time()
        solverK.solve()
        torch.cuda.synchronize() if device.type == "cuda" else None
        t5 = time.time()
        approx_costK = torch.norm(P_blue - P_red[solverK.MB], p=1, dim=1).sum().item()
        approx_timeK = t5 - t4
        abs_errK = abs(approx_costK - exact_cost)
        rel_errK = abs_errK / max(abs(exact_cost), 1e-9)
        # 6. Write results for this n (three lines: exact, 2-level, k-level)
        with open(output_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([n, epsilon, k_level, "POT-EMD-L1", f"{exact_time:.6f}", f"{exact_cost:.6f}", "0.0", "0.0"])
            writer.writerow([n, epsilon, k_level, "2-Level-L1", f"{approx_time2:.6f}", f"{approx_cost2:.6f}", f"{abs_err2:.6f}", f"{rel_err2:.6f}"])
            writer.writerow([n, epsilon, k_level, "k-Level-L1", f"{approx_timeK:.6f}", f"{approx_costK:.6f}", f"{abs_errK:.6f}", f"{rel_errK:.6f}"])
        print(f"n={n}: Exact cost={exact_cost:.6f}, TwoLevel cost={approx_cost2:.6f}, kLevel cost={approx_costK:.6f}")
        # Clean up solver objects to free GPU memory
        del solver2, solverK

if __name__ == "__main__":
    # Define the sample sizes to test (100, 200, ..., 1000)
    n_values = [100 * i for i in range(1, 11)]
    run_experiment(n_values, epsilon=0.05, k_level=4, output_csv="compare_pot_emd_results.csv")
