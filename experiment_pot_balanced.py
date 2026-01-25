import os, time, gzip, csv
import numpy as np
import torch
import ot

# Import your solvers
# Ensure these files are in the same directory
from clustered_push_relabel import GPUClusteredSolver as TwoLevelSolver
from k_level_clustered_push_relabel import GPUClusteredSolver as KLevelSolver

def load_balanced_mnist_data(n_samples, device):
    """
    Samples n_samples for Red and n_samples for Blue, ensuring strictly balanced classes.
    n_samples must be divisible by 10.
    Returns: P_red, P_blue, labels_red, labels_blue
    """
    if n_samples % 10 != 0:
        raise ValueError(f"n_samples ({n_samples}) must be divisible by 10 for balanced sampling.")
    
    samples_per_class = n_samples // 10
    
    # 1. Load Images
    img_path = os.path.join("data", "train-images-idx3-ubyte.gz")
    if not os.path.isfile(img_path):
        raise FileNotFoundError(f"Images not found at {img_path}")
    
    with gzip.open(img_path, "rb") as f:
        # Offset 16 for images
        data_np = np.frombuffer(f.read(), dtype=np.uint8, offset=16).reshape(-1, 784).copy()

    # 2. Load Labels
    lbl_path = os.path.join("data", "train-labels-idx1-ubyte.gz")
    if not os.path.isfile(lbl_path):
        raise FileNotFoundError(f"Labels not found at {lbl_path}. Run download_labels.py first.")
    
    with gzip.open(lbl_path, "rb") as f:
        # Offset 8 for labels
        labels_np = np.frombuffer(f.read(), dtype=np.uint8, offset=8).copy()

    # 3. Organize indices by class
    class_indices = [np.where(labels_np == i)[0] for i in range(10)]
    
    red_indices = []
    blue_indices = []
    
    # Reproducibility
    np.random.seed(42) 
    
    # 4. Sample Balanced Sets
    for i in range(10):
        available = class_indices[i]
        if len(available) < 2 * samples_per_class:
            raise ValueError(f"Not enough images for class {i}")
        
        # Shuffle and pick distinct sets for Red and Blue
        np.random.shuffle(available)
        red_indices.extend(available[:samples_per_class])
        blue_indices.extend(available[samples_per_class : 2*samples_per_class])
    
    # Convert to Tensors
    P_red = torch.from_numpy(data_np[red_indices]).float().to(device)
    P_blue = torch.from_numpy(data_np[blue_indices]).float().to(device)
    
    # Normalize (add jitter to avoid 0 mass)
    P_red = (P_red + 1e-6) / (P_red + 1e-6).sum(dim=1, keepdim=True)
    P_blue = (P_blue + 1e-6) / (P_blue + 1e-6).sum(dim=1, keepdim=True)
    
    L_red = labels_np[red_indices]
    L_blue = labels_np[blue_indices]
    
    return P_red, P_blue, L_red, L_blue

def compute_cost_matrix_l1(P_red, P_blue):
    """Compute L1 cost matrix on CPU for POT."""
    X = P_red.cpu().numpy()
    Y = P_blue.cpu().numpy()
    n = X.shape[0]
    C = np.zeros((n, n), dtype=np.float64)
    chunk_size = 100
    for i in range(0, n, chunk_size):
        end = min(n, i + chunk_size)
        diff = np.abs(X[i:end, None, :] - Y[None, :, :])
        C[i:end] = diff.sum(axis=2)
    return C

def calc_violations_pot(P_plan, labels_red, labels_blue):
    """
    Calculate violations for POT.
    P_plan is an NxN matrix. We take argmax to find the match.
    """
    n = len(labels_red)
    # P_plan[i, j] > 0 means Red i matched to Blue j
    # For 1-to-1 matching, row_match[i] gives the index j of the Blue point
    row_match = np.argmax(P_plan, axis=1)
    
    violations = 0
    for i in range(n):
        j = row_match[i]
        if labels_red[i] != labels_blue[j]:
            violations += 1
            
    return (violations / n) * 100.0

def calc_violations_solver(MB_tensor, labels_red, labels_blue):
    """
    Calculate violations for Push-Relabel Solvers.
    MB_tensor[j] = i means Blue j is matched to Red i.
    """
    n = len(labels_blue)
    MB = MB_tensor.cpu().numpy()
    
    violations = 0
    for j in range(n):
        i = MB[j]
        # MB might be -1 if unmatched (though solver should return perfect matching)
        if i != -1:
            if labels_red[i] != labels_blue[j]:
                violations += 1
        else:
            violations += 1 # Count unmatched as violation? Or ignore? Usually 100% match.
            
    return (violations / n) * 100.0

def run_experiment(n_values, epsilon=0.05, k_level=4, output_csv="balanced_results.csv"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")
    
    # CSV Header
    headers = ["n", "epsilon", "k", "algo", "time_s", "cost", "abs_error", "rel_error", "violation_pct"]
    
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
    for n in n_values:
        print(f"\n--- Running N={n} ---")
        
        # 1. Load Balanced Data
        try:
            P_red, P_blue, L_red, L_blue = load_balanced_mnist_data(n, device)
        except Exception as e:
            print(f"Skipping N={n}: {e}")
            continue

        # 2. POT (Exact)
        C = compute_cost_matrix_l1(P_red, P_blue)
        a = np.full(n, 1.0/n)
        b = np.full(n, 1.0/n)
        
        t0 = time.time()
        P_plan = ot.emd(a, b, C)
        t1 = time.time()
        
        exact_cost = float((P_plan * C).sum())
        exact_time = t1 - t0
        viol_pot = calc_violations_pot(P_plan, L_red, L_blue)
        
        # Write POT Result
        with open(output_csv, "a", newline="") as f:
            csv.writer(f).writerow([n, epsilon, k_level, "POT-Exact", f"{exact_time:.6f}", f"{exact_cost:.6f}", "0.0", "0.0", f"{viol_pot:.2f}"])

        # 3. Two-Level Solver
        t2 = time.time()
        solver2 = TwoLevelSolver(P_red, P_blue, epsilon, metric="L1")
        if device.type == 'cuda': torch.cuda.synchronize()
        solver2.solve()
        if device.type == 'cuda': torch.cuda.synchronize()
        t3 = time.time()
        
        cost2 = torch.norm(P_blue - P_red[solver2.MB], p=1, dim=1).sum().item() / n
        time2 = t3 - t2
        err2 = abs(cost2 - exact_cost)
        rel2 = err2 / max(abs(exact_cost), 1e-9)
        viol2 = calc_violations_solver(solver2.MB, L_red, L_blue)
        
        with open(output_csv, "a", newline="") as f:
            csv.writer(f).writerow([n, epsilon, k_level, "2-Level", f"{time2:.6f}", f"{cost2:.6f}", f"{err2:.6f}", f"{rel2:.6f}", f"{viol2:.2f}"])

        # 4. K-Level Solver
        t4 = time.time()
        solverK = KLevelSolver(P_red, P_blue, epsilon, k=k_level, metric="L1")
        if device.type == 'cuda': torch.cuda.synchronize()
        solverK.solve()
        if device.type == 'cuda': torch.cuda.synchronize()
        t5 = time.time()
        
        costK = torch.norm(P_blue - P_red[solverK.MB], p=1, dim=1).sum().item() / n
        timeK = t5 - t4
        errK = abs(costK - exact_cost)
        relK = errK / max(abs(exact_cost), 1e-9)
        violK = calc_violations_solver(solverK.MB, L_red, L_blue)
        
        with open(output_csv, "a", newline="") as f:
            csv.writer(f).writerow([n, epsilon, k_level, "K-Level", f"{timeK:.6f}", f"{costK:.6f}", f"{errK:.6f}", f"{relK:.6f}", f"{violK:.2f}"])
            
        print(f"Violations: POT={viol_pot:.1f}%, 2-Lvl={viol2:.1f}%, K-Lvl={violK:.1f}%")
        
        # Cleanup
        del solver2, solverK, P_red, P_blue, C, P_plan
        torch.cuda.empty_cache()

    print_summary_table(output_csv)

def print_summary_table(csv_filename):
    print("\n" + "="*85)
    print(f"{'N':<6} | {'Algo':<12} | {'Time(s)':<8} | {'Cost':<10} | {'Err %':<8} | {'Viol %':<8}")
    print("-" * 85)
    
    with open(csv_filename, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            err = float(row['rel_error']) * 100
            print(f"{row['n']:<6} | {row['algo']:<12} | {float(row['time_s']):<8.4f} | {float(row['cost']):<10.4f} | {err:<8.4f} | {row['violation_pct']:<8}")
    print("="*85 + "\n")

if __name__ == "__main__":
    # N values must be divisible by 10
    n_values = [100, 200, 500, 1000, 2000] 
    run_experiment(n_values, epsilon=0.1, k_level=4, output_csv="balanced_results.csv")