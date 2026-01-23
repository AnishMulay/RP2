import torch
import time
import csv
import fcntl
import os
import sys
import random
import math
import warnings

# ==========================================
# IMPORTS: STRICT INDUSTRY STANDARDS
# ==========================================

# 1. Custom Solvers
try:
    from clustered_push_relabel import GPUClusteredSolver as TwoLevelSolver
    from k_level_clustered_push_relabel import GPUClusteredSolver as KLevelSolver
except ImportError as e:
    print(f"[Error] Could not import your custom solvers: {e}")
    sys.exit(1)

# 2. Python Optimal Transport (POT)
try:
    import ot
except ImportError:
    print("[Error] 'POT' library not found. Please run: pip install POT")
    sys.exit(1)

# 3. GeomLoss
try:
    import geomloss
except ImportError:
    print("[Error] 'geomloss' library not found. Please run: pip install geomloss pykeops")
    sys.exit(1)

# ==========================================
# UTILITIES
# ==========================================

WORD_LIST = [
    "ant", "bat", "cat", "dog", "eel", "fox", "gem", "hat", "ice", "joy",
    "kite", "lion", "moon", "nest", "owl", "pear", "quiz", "rose", "sun", "tree",
    "apple", "bread", "chair", "dance", "eagle", "flame", "grape", "house", "image", "jelly"
]

def generate_run_id():
    w3 = random.choice([w for w in WORD_LIST if len(w) == 3])
    w4 = random.choice([w for w in WORD_LIST if len(w) == 4])
    w5 = random.choice([w for w in WORD_LIST if len(w) == 5])
    return f"{w3}-{w4}-{w5}"

def get_memory_usage():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return 0.0

def reset_memory_stats():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

# ==========================================
# SOLVER WRAPPERS (Industry Standards)
# ==========================================

def run_pot_sinkhorn(P_red, P_blue, epsilon):
    """
    Benchmarks Python Optimal Transport (POT).
    Uses log-domain stabilization to handle low epsilon values.
    """
    N = P_red.size(0)
    device = P_red.device
    
    # POT expects CPU numpy arrays usually, but modern versions support PyTorch tensors 
    # if we explicitly handle the backend or pass tensors directly.
    # To be safe and performant on GPU, we compute the Cost Matrix in PyTorch first.
    
    # Cost Matrix: Squared Euclidean
    x_norm = (P_red**2).sum(1).view(-1, 1)
    y_norm = (P_blue**2).sum(1).view(1, -1)
    C = x_norm + y_norm - 2.0 * torch.mm(P_red, P_blue.t())
    
    # Uniform weights
    a = torch.ones(N, device=device) / N
    b = torch.ones(N, device=device) / N
    
    # Run Sinkhorn
    # method='sinkhorn_log' is critical for stability at low epsilon (The "Sinkhorn Wall")
    # POT returns the Transport Plan (P) matrix directly
    P = ot.sinkhorn(a, b, C, reg=epsilon, method='sinkhorn_log', numItermax=100000)
    
    # 1. Soft Cost (The Objective Function Value)
    soft_cost = torch.sum(P * C).item()
    
    # 2. Hard Cost (Greedy Rounding)
    # This is the "Apples-to-Apples" comparison metric for Push-Relabel
    matches = torch.argmax(P, dim=1)
    hard_cost = C[torch.arange(N, device=device), matches].sum().item()
    
    return soft_cost, hard_cost

def run_geomloss_sinkhorn(P_red, P_blue, epsilon):
    """
    Benchmarks GeomLoss (Symbolic / KeOps).
    """
    # GeomLoss uses 'blur' instead of 'epsilon'.
    # Relationship: epsilon ~ blur^2
    blur = math.sqrt(epsilon)
    
    # Define Loss
    # scaling=0.9 is standard for stability in GeomLoss
    loss_fn = geomloss.SamplesLoss(loss="sinkhorn", p=2, blur=blur, scaling=0.9, potentials=True)
    
    # Compute Dual Potentials (f, g)
    # This avoids materializing the NxN matrix initially (O(N) memory)
    f, g = loss_fn(P_red, P_blue)
    
    # Reconstruct Cost Matrix C to calculate Hard Cost
    x_norm = (P_red**2).sum(1).view(-1, 1)
    y_norm = (P_blue**2).sum(1).view(1, -1)
    C = x_norm + y_norm - 2.0 * torch.mm(P_red, P_blue.t())
    
    # Reconstruct Transport Plan from Potentials
    # Log_P_ij = (f_i + g_j - C_ij) / epsilon
    # We do this to get the 'argmax' for hard matching.
    # For N=15k, this (15k x 15k) matrix fits in GPU memory (approx 900MB).
    P_log = (f.view(-1, 1) + g.view(1, -1) - C) / epsilon
    
    # 1. Hard Cost (Greedy Rounding via Argmax)
    matches = torch.argmax(P_log, dim=1)
    hard_cost = C[torch.arange(len(P_red), device=P_red.device), matches].sum().item()
    
    # 2. Soft Cost (Reconstructed)
    # We use the plan to compute soft cost to be consistent with POT reporting
    P = torch.exp(P_log)
    soft_cost = torch.sum(P * C).item()
    
    return soft_cost, hard_cost

# ==========================================
# EXPERIMENT RUNNER
# ==========================================

def run_experiment(config):
    run_id = generate_run_id()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\n" + "="*85)
    print(f"BENCHMARK START | ID: {run_id}")
    print(f"Config: N={config.n} | Dim={config.dim} | Eps={config.epsilon} | K={config.k} | Device={device}")
    print("="*85)
    
    # Generate Data (Standard Normal / Gaussian blobs)
    # This matches the 'Synthetic' setup in the research paper
    torch.manual_seed(config.seed)
    P_red = torch.randn(config.n, config.dim, device=device) * math.sqrt(config.epsilon)
    P_blue = torch.randn(config.n, config.dim, device=device) * math.sqrt(config.epsilon)
    
    results = []
    
    # -----------------------------------------------------------------
    # 1. INDUSTRY STANDARD: POT (Python Optimal Transport)
    # -----------------------------------------------------------------
    print(f">> Running POT (ot.sinkhorn_log)...")
    try:
        reset_memory_stats()
        t_start = time.time()
        
        soft_cost, hard_cost = run_pot_sinkhorn(P_red, P_blue, config.epsilon)
        torch.cuda.synchronize()
        t_end = time.time()
        
        results.append({
            "algo": "POT-Sinkhorn",
            "total_time": t_end - t_start,
            "clust_time": 0.0,
            "avg_cost": hard_cost / config.n,
            "soft_cost": soft_cost / config.n,
            "mem": get_memory_usage()
        })
    except Exception as e:
        print(f"   [FAILED] POT failed: {e}")
        # Add a placeholder for CSV consistency
        results.append({
            "algo": "POT-Sinkhorn",
            "total_time": 0.0, "clust_time": 0.0, "avg_cost": -1.0, "soft_cost": -1.0, "mem": 0.0
        })

    # -----------------------------------------------------------------
    # 2. INDUSTRY STANDARD: GeomLoss (Symbolic / KeOps)
    # -----------------------------------------------------------------
    print(f">> Running GeomLoss (KeOps)...")
    try:
        reset_memory_stats()
        t_start = time.time()
        
        soft_cost, hard_cost = run_geomloss_sinkhorn(P_red, P_blue, config.epsilon)
        torch.cuda.synchronize()
        t_end = time.time()
        
        results.append({
            "algo": "GeomLoss",
            "total_time": t_end - t_start,
            "clust_time": 0.0,
            "avg_cost": hard_cost / config.n,
            "soft_cost": soft_cost / config.n,
            "mem": get_memory_usage()
        })
    except Exception as e:
        print(f"   [FAILED] GeomLoss failed: {e}")
        results.append({
            "algo": "GeomLoss",
            "total_time": 0.0, "clust_time": 0.0, "avg_cost": -1.0, "soft_cost": -1.0, "mem": 0.0
        })

    # -----------------------------------------------------------------
    # 3. CUSTOM: 2-Level Clustered Push-Relabel
    # -----------------------------------------------------------------
    print(f">> Running 2-Level Push-Relabel...")
    try:
        reset_memory_stats()
        t_start = time.time()
        
        solver_2lvl = TwoLevelSolver(P_red, P_blue, config.epsilon)
        torch.cuda.synchronize()
        t_clust_end = time.time()
        
        solver_2lvl.solve()
        torch.cuda.synchronize()
        t_end = time.time()
        
        # Recalculate cost externally for verification
        dists = torch.norm(P_blue - P_red[solver_2lvl.MB], p=2, dim=1)
        cost_2lvl = dists.sum().item()
        
        results.append({
            "algo": "2-Level-PR",
            "total_time": t_end - t_start,
            "clust_time": t_clust_end - t_start,
            "avg_cost": cost_2lvl / config.n,
            "soft_cost": 0.0, # N/A for hard matchers
            "mem": get_memory_usage()
        })
        del solver_2lvl
    except Exception as e:
        print(f"   [FAILED] 2-Level PR failed: {e}")
        results.append({
            "algo": "2-Level-PR", "total_time": 0.0, "clust_time": 0.0, "avg_cost": -1.0, "soft_cost": 0.0, "mem": 0.0
        })

    # -----------------------------------------------------------------
    # 4. CUSTOM: k-Level Clustered Push-Relabel
    # -----------------------------------------------------------------
    print(f">> Running k-Level Push-Relabel...")
    try:
        reset_memory_stats()
        t_start = time.time()
        
        solver_klvl = KLevelSolver(P_red, P_blue, config.epsilon, k=config.k)
        torch.cuda.synchronize()
        t_clust_end = time.time()
        
        solver_klvl.solve()
        torch.cuda.synchronize()
        t_end = time.time()
        
        dists = torch.norm(P_blue - P_red[solver_klvl.MB], p=2, dim=1)
        cost_klvl = dists.sum().item()
        
        results.append({
            "algo": "k-Level-PR",
            "total_time": t_end - t_start,
            "clust_time": t_clust_end - t_start,
            "avg_cost": cost_klvl / config.n,
            "soft_cost": 0.0,
            "mem": get_memory_usage()
        })
        del solver_klvl
    except Exception as e:
        print(f"   [FAILED] k-Level PR failed: {e}")
        results.append({
            "algo": "k-Level-PR", "total_time": 0.0, "clust_time": 0.0, "avg_cost": -1.0, "soft_cost": 0.0, "mem": 0.0
        })

    # ==========================================
    # REPORTING & LOGGING
    # ==========================================
    
    # 1. Terminal Table
    print("\n" + "="*95)
    print(f"FINAL REPORT | ID: {run_id} | N={config.n} | Eps={config.epsilon}")
    print("="*95)
    header = f"{'Algorithm':<15} | {'Total Time (s)':<14} | {'Clust Time (s)':<14} | {'Avg Cost':<10} | {'Peak Mem (MB)':<13}"
    print(header)
    print("-" * 95)
    
    for r in results:
        # Highlight failed runs
        if r['avg_cost'] == -1.0:
            print(f"{r['algo']:<15} | {'FAILED':<14} | {'--':<14} | {'--':<10} | {'--':<13}")
        else:
            print(f"{r['algo']:<15} | {r['total_time']:<14.4f} | {r['clust_time']:<14.4f} | {r['avg_cost']:<10.4f} | {r['mem']:<13.1f}")
    
    print("-" * 95)
    # Print reference soft costs for scientific comparison
    print(f"Reference Soft Costs (Lower Bound):")
    if results[0]['soft_cost'] > 0: print(f"  POT:      {results[0]['soft_cost']:.4f}")
    if results[1]['soft_cost'] > 0: print(f"  GeomLoss: {results[1]['soft_cost']:.4f}")
    print("="*95 + "\n")

    # 2. CSV Logging (Concurrency Safe)
    file_exists = os.path.isfile(config.csv)
    row_base = [run_id, time.strftime("%Y-%m-%d %H:%M:%S"), config.n, config.epsilon, config.k]
    
    try:
        with open(config.csv, "a", newline="") as f:
            fcntl.flock(f, fcntl.LOCK_EX) # Lock for HPC safety
            try:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow([
                        "run_id", "timestamp", "n_points", "epsilon", "k_levels",
                        "algo_name", "total_time_s", "clust_time_s", "avg_match_cost", "soft_cost_ref", "peak_mem_mb"
                    ])
                
                for r in results:
                    writer.writerow(row_base + [
                        r["algo"], 
                        f"{r['total_time']:.4f}", 
                        f"{r['clust_time']:.4f}", 
                        f"{r['avg_cost']:.4f}", 
                        f"{r['soft_cost']:.4f}",
                        f"{r['mem']:.1f}"
                    ])
            finally:
                fcntl.flock(f, fcntl.LOCK_UN) # Unlock
        print(f"Results appended to {config.csv}")
    except Exception as e:
        print(f"[Error] Failed to write to CSV: {e}")

if __name__ == "__main__":
    # ==========================================
    # EXPERIMENT CONFIGURATION
    # Edit values here to control the benchmark
    # ==========================================
    class ExperimentConfig:
        n = 10000              # Number of points per distribution
        dim = 2                # Dimensions
        epsilon = 0.05         # Regularization (Sinkhorn) / Error (Push-Relabel)
        k = 4                  # Hierarchy levels (for k-level solver)
        csv = "results.csv"    # Output file
        seed = 42              # Random seed

    config = ExperimentConfig()
    run_experiment(config)
