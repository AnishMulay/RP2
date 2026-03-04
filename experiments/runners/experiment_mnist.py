import pathlib
import os
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
import os
import gzip
# CRITICAL: Prevent JAX from hogging all GPU memory
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["JAX_ENABLE_X64"] = "True"

import torch
import time
import csv
import fcntl
import sys
import random
import math
import numpy as np

# ==========================================
# IMPORTS: SOLVERS
# ==========================================
try:
    from clustered_push_relabel.solvers.bipartite import TwoLevelBipartiteSolver as TwoLevelSolver
    from clustered_push_relabel.solvers.bipartite import KLevelBipartiteSolver as KLevelSolver
except ImportError as e:
    print(f"[Error] Could not import custom solvers: {e}")
    sys.exit(1)

HAS_OTT = False
try:
    import jax
    import jax.numpy as jnp
    import ott
    from ott.geometry import pointcloud
    from ott.geometry import costs
    from ott.solvers.linear import sinkhorn as ott_sinkhorn
    from ott.problems.linear import linear_problem
    HAS_OTT = True
except ImportError:
    print("[Warning] OTT-JAX not found. Comparisons will be skipped.")

# ==========================================
# UTILITIES
# ==========================================

WORD_LIST = ["ant", "bat", "cat", "dog", "eel", "fox", "gem", "hat", "ice", "joy"]

def generate_run_id():
    return f"{random.choice(WORD_LIST)}-{random.choice(WORD_LIST)}-{random.choice(WORD_LIST)}"

def get_memory_usage():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return 0.0

def reset_memory_stats():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

# ==========================================
# DATA LOADING (MNIST)
# ==========================================

def load_mnist_data(n_samples, device):
    """
    Loads MNIST from local gzip file, flattens, normalizes, and samples N pairs.
    """
    print(">> Loading MNIST dataset (manual loader)...")
    data_path = os.path.join("data", "train-images-idx3-ubyte.gz")
    if not os.path.isfile(data_path):
        raise FileNotFoundError(
            "MNIST file not found at ./data/train-images-idx3-ubyte.gz. "
            "Please upload the data to ./data."
        )

    with gzip.open(data_path, "rb") as f:
        raw = f.read()

    # Skip 16-byte header, then read as uint8 and reshape.
    data_np = np.frombuffer(raw, dtype=np.uint8, offset=16).reshape(-1, 784).copy()
    data = torch.from_numpy(data_np).float()
    
    # 4. Normalize (Sum to 1) + Jitter
    # Add epsilon to avoid zero-mass issues (standard OT practice)
    data = data + 1e-6
    data = data / data.sum(dim=1, keepdim=True)
    
    # 5. Sample
    total_images = data.size(0)
    if 2 * n_samples > total_images:
        raise ValueError(f"Requesting {2*n_samples} images, but MNIST only has {total_images}")
        
    indices = torch.randperm(total_images)
    idx_red = indices[:n_samples]
    idx_blue = indices[n_samples:2*n_samples]
    
    P_red = data[idx_red].to(device)
    P_blue = data[idx_blue].to(device)
    
    print(f"   Loaded: Red {P_red.shape}, Blue {P_blue.shape}")
    return P_red, P_blue

# ==========================================
# OTT-JAX WRAPPER (L1 CONFIGURATION)
# ==========================================

def run_ott_sinkhorn_l1(P_red, P_blue, epsilon):
    """
    OTT-JAX configured for L1 (Manhattan) Distance.
    """
    # Convert to JAX
    x = jnp.array(P_red.cpu().numpy())
    y = jnp.array(P_blue.cpu().numpy())
    
    # Define L1 Cost Function
    # OTT has a native Manhattan cost, or we can define generic p=1
    # Check version: usually ott.geometry.costs.Manhattan or p-norm
    # Fallback to explicit definition if needed, but generic CostFn is safest
    
    @jax.tree_util.register_pytree_node_class
    class ManhattanCost(costs.CostFn):
        def pairwise(self, x, y):
            return jnp.sum(jnp.abs(x - y))
        def __call__(self, x, y):
            return self.pairwise(x, y)
    
    cost_fn = ManhattanCost()
    
    # Create Geometry
    geom = pointcloud.PointCloud(x, y, cost_fn=cost_fn, epsilon=epsilon)
    prob = linear_problem.LinearProblem(geom)
    
    # Solve
    solver = ott_sinkhorn.Sinkhorn(max_iterations=20000)
    out = solver(prob)
    out.converged.block_until_ready()
    
    # Metrics
    # 1. Soft Cost (from solver)
    soft_cost = float(out.reg_ot_cost)
    
    # 2. Hard Cost (L1)
    P_matrix = out.matrix
    matches = jnp.argmax(P_matrix, axis=1)
    matched_y = y[matches]
    
    # Compute L1 distance manually for verification
    dists = jnp.sum(jnp.abs(x - matched_y), axis=1)
    hard_cost = float(jnp.sum(dists))
    
    # 3. Uniqueness
    num_unique = len(np.unique(np.array(matches)))
    uniqueness = (num_unique / x.shape[0]) * 100.0
    
    return soft_cost, hard_cost, uniqueness

# ==========================================
# EXPERIMENT RUNNER
# ==========================================

def run_experiment(config):
    run_id = generate_run_id()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\n" + "="*95)
    print(f"MNIST BENCHMARK (L1) | ID: {run_id} | N={config.n} | Eps={config.epsilon} | K={config.k}")
    print("="*95)
    
    # Load Data
    P_red, P_blue = load_mnist_data(config.n, device)
    
    results = []
    
    # 1. OTT-JAX (L1)
    print(f">> Running OTT-JAX (L1)...")
    if HAS_OTT:
        try:
            reset_memory_stats()
            t_start = time.time()
            soft, hard, unique = run_ott_sinkhorn_l1(P_red, P_blue, config.epsilon)
            t_end = time.time()
            results.append({
                "algo": "OTT-JAX-L1",
                "time": t_end-t_start,
                "clust": 0.0,
                "cost": hard/config.n,
                "unique": unique,
                "mem": 0.0
            })
        except Exception as e:
            print(f"[FAIL] OTT: {e}")
    else:
        print("[SKIP] OTT not installed")

    # 2. 2-Level PR (L1)
    print(f">> Running 2-Level Push-Relabel (L1)...")
    try:
        reset_memory_stats()
        t_start = time.time()
        # NOTE: 'metric' arg must be supported by your updated class
        solver = TwoLevelSolver(P_red, P_blue, config.epsilon, metric="L1")
        torch.cuda.synchronize()
        t_clust = time.time()
        solver.solve()
        torch.cuda.synchronize()
        t_end = time.time()
        
        # Verify L1 Cost
        cost = torch.norm(P_blue - P_red[solver.MB], p=1, dim=1).sum().item()
        
        results.append({
            "algo": "2-Level-L1",
            "time": t_end-t_start,
            "clust": t_clust-t_start,
            "cost": cost/config.n,
            "unique": 100.0,
            "mem": get_memory_usage()
        })
        del solver
    except Exception as e:
        print(f"[FAIL] 2-Level: {e}")

    # 3. k-Level PR (L1)
    print(f">> Running k-Level Push-Relabel (L1)...")
    try:
        reset_memory_stats()
        t_start = time.time()
        solver = KLevelSolver(P_red, P_blue, config.epsilon, k=config.k, metric="L1")
        torch.cuda.synchronize()
        t_clust = time.time()
        solver.solve()
        torch.cuda.synchronize()
        t_end = time.time()
        
        cost = torch.norm(P_blue - P_red[solver.MB], p=1, dim=1).sum().item()
        
        results.append({
            "algo": "k-Level-L1",
            "time": t_end-t_start,
            "clust": t_clust-t_start,
            "cost": cost/config.n,
            "unique": 100.0,
            "mem": get_memory_usage()
        })
        del solver
    except Exception as e:
        print(f"[FAIL] k-Level: {e}")

    # Report
    print("\n" + "="*95)
    print(f"FINAL REPORT | ID: {run_id}")
    print(f"{'Algorithm':<15} | {'Time (s)':<10} | {'Clust (s)':<10} | {'Avg L1 Cost':<15} | {'Unique%':<10} | {'Mem (MB)':<10}")
    print("-" * 95)
    for r in results:
        print(f"{r['algo']:<15} | {r['time']:<10.4f} | {r['clust']:<10.4f} | {r['cost']:<15.6f} | {r['unique']:<10.1f} | {r['mem']:<10.1f}")
    print("-" * 95)

    # CSV
    row = [run_id, time.strftime("%Y-%m-%d"), config.n, config.epsilon, config.k]
    with open(config.csv, "a", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        writer = csv.writer(f)
        if not os.path.isfile(config.csv):
             writer.writerow(["run_id", "date", "n", "eps", "k", "algo", "time", "clust_time", "cost", "unique", "mem"])
        for r in results:
            writer.writerow(row + [r['algo'], r['time'], r['clust'], r['cost'], r['unique'], r['mem']])
        fcntl.flock(f, fcntl.LOCK_UN)
    print(f"Results appended to {config.csv}")

if __name__ == "__main__":
    class Config:
        n = 10000          # Number of images to sample
        epsilon = 0.05     # Epsilon for MNIST L1 (Start higher, L1 costs are larger than L2^2)
        k = 4
        csv = "results_mnist.csv"
        seed = 42
    run_experiment(Config())
