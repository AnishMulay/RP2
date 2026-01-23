import os
# CRITICAL: Prevent JAX from hogging all GPU memory, allowing PyTorch to run too.
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["JAX_ENABLE_X64"] = "True" # Sinkhorn needs double precision for stability

import torch
import time
import csv
import fcntl
import sys
import random
import math
import warnings

# ==========================================
# IMPORTS: SOLVERS
# ==========================================

# 1. Custom Solvers
try:
    from clustered_push_relabel import GPUClusteredSolver as TwoLevelSolver
    from k_level_clustered_push_relabel import GPUClusteredSolver as KLevelSolver
except ImportError as e:
    print(f"[Error] Could not import your custom solvers: {e}")
    sys.exit(1)

# 2. Industry Standard: OTT-JAX (The High-Performance Google Solver)
HAS_OTT = False
try:
    import jax
    import jax.numpy as jnp
    import ott
    from ott.geometry import pointcloud
    from ott.solvers.linear import sinkhorn as ott_sinkhorn
    from ott.problems.linear import linear_problem
    HAS_OTT = True
except ImportError:
    pass

# 3. Industry Standard: POT (Legacy)
HAS_POT = False
try:
    import ot
    HAS_POT = True
except ImportError:
    pass

# 4. Industry Standard: GeomLoss (Legacy/Symbolic)
HAS_GEOMLOSS = False
try:
    import geomloss
    HAS_GEOMLOSS = True
except ImportError:
    pass

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
# SOLVER WRAPPERS
# ==========================================

def run_ott_sinkhorn(P_red, P_blue, epsilon):
    """
    Benchmarks OTT-JAX (Optimal Transport Tools).
    Fast, JIT-compiled, GPU-native.
    """
    # Convert PyTorch -> JAX (via CPU to be safe with memory contexts)
    # In production, dlpack is faster, but this is safer for scripts.
    x = jnp.array(P_red.cpu().numpy())
    y = jnp.array(P_blue.cpu().numpy())
    
    # Define Geometry (Squared Euclidean)
    geom = pointcloud.PointCloud(x, y, epsilon=epsilon)
    prob = linear_problem.LinearProblem(geom)
    
    # Solve
    # We use a high max_iterations because JAX is fast enough to handle it
    solver = ott_sinkhorn.Sinkhorn(max_iterations=100000)
    out = solver(prob)
    
    # Block until ready (JAX is asynchronous)
    out.converged.block_until_ready()

    # Estimate memory: JAX float64 (8 bytes) * N * N
    mat_size_mb = (P_red.shape[0] ** 2 * 8) / (1024**2)
    
    # 1. Soft Cost
    soft_cost = float(out.reg_ot_cost)
    
    # 2. Hard Cost (Greedy Rounding)
    # We must materialize the matrix. For N=10k, 10k*10k*4bytes = 400MB. Fine.
    P_matrix = out.matrix
    # Argmax assignment (greedy)
    matches = jnp.argmax(P_matrix, axis=1)
    
    # Compute hard cost using the cost matrix from geometry
    # C_ij = ||x_i - y_j||^2
    # We pick C[i, match[i]]
    # Using JAX vmap for efficient gather
    
    # Re-calculate costs for the matches manually to be sure
    matched_y = y[matches]
    dists = jnp.sqrt(jnp.sum((x - matched_y)**2, axis=1))
    hard_cost = float(jnp.sum(dists))
    
    return soft_cost, hard_cost, mat_size_mb

def run_pot_sinkhorn(P_red, P_blue, epsilon):
    N = P_red.size(0)
    device = P_red.device
    x_norm = (P_red**2).sum(1).view(-1, 1)
    y_norm = (P_blue**2).sum(1).view(1, -1)
    C = x_norm + y_norm - 2.0 * torch.mm(P_red, P_blue.t())
    
    a = torch.ones(N, device=device) / N
    b = torch.ones(N, device=device) / N
    
    # Increased iterations as requested
    P = ot.sinkhorn(a, b, C, reg=epsilon, method='sinkhorn_log', numItermax=100000)
    
    soft_cost = torch.sum(P * C).item()
    matches = torch.argmax(P, dim=1)
    # Gather squared costs, then sqrt, then sum
    squared_costs = C[torch.arange(N, device=device), matches]
    hard_cost = torch.sqrt(torch.clamp(squared_costs, min=0.0)).sum().item()
    return soft_cost, hard_cost

def run_geomloss_sinkhorn(P_red, P_blue, epsilon):
    blur = math.sqrt(epsilon)
    loss_fn = geomloss.SamplesLoss(loss="sinkhorn", p=2, blur=blur, scaling=0.9, potentials=True)
    f, g = loss_fn(P_red, P_blue)
    
    x_norm = (P_red**2).sum(1).view(-1, 1)
    y_norm = (P_blue**2).sum(1).view(1, -1)
    C = x_norm + y_norm - 2.0 * torch.mm(P_red, P_blue.t())
    
    P_log = (f.view(-1, 1) + g.view(1, -1) - C) / epsilon
    matches = torch.argmax(P_log, dim=1)
    # Gather squared costs (C is squared), then sqrt, then sum
    squared_costs = C[torch.arange(len(P_red), device=P_red.device), matches]
    hard_cost = torch.sqrt(torch.clamp(squared_costs, min=0.0)).sum().item()
    
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
    
    # Generate Data: UNIFORM [0,1] (Fixes underflow issues)
    torch.manual_seed(config.seed)
    P_red = torch.rand(config.n, config.dim, device=device)
    P_blue = torch.rand(config.n, config.dim, device=device)
    
    results = []
    
    # -----------------------------------------------------------------
    # 1. INDUSTRY STANDARD: OTT-JAX (Priority)
    # -----------------------------------------------------------------
    print(f">> Running OTT-JAX (Google/JIT)...")
    if HAS_OTT:
        try:
            reset_memory_stats()
            t_start = time.time()
            soft_cost, hard_cost, mat_size_mb = run_ott_sinkhorn(P_red, P_blue, config.epsilon)
            # No cuda sync needed for JAX (handled in wrapper block_until_ready)
            t_end = time.time()
            
            results.append({
                "algo": "OTT-JAX",
                "total_time": t_end - t_start,
                "clust_time": 0.0,
                "avg_cost": hard_cost / config.n,
                "soft_cost": soft_cost / config.n,
                "mem": mat_size_mb
            })
        except Exception as e:
            print(f"   [FAILED] OTT-JAX failed: {e}")
            results.append({"algo": "OTT-JAX", "total_time": 0.0, "clust_time": 0.0, "avg_cost": -1.0, "soft_cost": -1.0, "mem": 0.0})
    else:
        print("   [SKIP] OTT-JAX not installed.")

    # -----------------------------------------------------------------
    # 2. INDUSTRY STANDARD: POT (Legacy)
    # -----------------------------------------------------------------
    print(f">> Running POT (ot.sinkhorn)...")
    if HAS_POT:
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
            results.append({"algo": "POT-Sinkhorn", "total_time": 0.0, "clust_time": 0.0, "avg_cost": -1.0, "soft_cost": -1.0, "mem": 0.0})
    else:
        print("   [SKIP] POT not installed.")

    # -----------------------------------------------------------------
    # 3. INDUSTRY STANDARD: GeomLoss
    # -----------------------------------------------------------------
    print(f">> Running GeomLoss...")
    if HAS_GEOMLOSS:
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
            results.append({"algo": "GeomLoss", "total_time": 0.0, "clust_time": 0.0, "avg_cost": -1.0, "soft_cost": -1.0, "mem": 0.0})
    else:
        print("   [SKIP] GeomLoss not installed.")

    # -----------------------------------------------------------------
    # 4. CUSTOM: 2-Level
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
        dists = torch.norm(P_blue - P_red[solver_2lvl.MB], p=2, dim=1)
        cost_2lvl = dists.sum().item()
        results.append({
            "algo": "2-Level-PR",
            "total_time": t_end - t_start,
            "clust_time": t_clust_end - t_start,
            "avg_cost": cost_2lvl / config.n,
            "soft_cost": 0.0,
            "mem": get_memory_usage()
        })
        del solver_2lvl
    except Exception as e:
        print(f"   [FAILED] 2-Level PR failed: {e}")

    # -----------------------------------------------------------------
    # 5. CUSTOM: k-Level
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

    # Report
    print("\n" + "="*95)
    print(f"FINAL REPORT | ID: {run_id} | N={config.n} | Eps={config.epsilon}")
    print("="*95)
    print(f"{'Algorithm':<15} | {'Total Time (s)':<14} | {'Clust Time (s)':<14} | {'Avg Cost':<10} | {'Peak Mem (MB)':<13}")
    print("-" * 95)
    for r in results:
        cost_str = f"{r['avg_cost']:.4f}" if r['avg_cost'] > 0 else "FAIL"
        print(f"{r['algo']:<15} | {r['total_time']:<14.4f} | {r['clust_time']:<14.4f} | {cost_str:<10} | {r['mem']:<13.1f}")
    print("-" * 95)

    # CSV
    row_base = [run_id, time.strftime("%Y-%m-%d %H:%M:%S"), config.n, config.epsilon, config.k]
    with open(config.csv, "a", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            writer = csv.writer(f)
            if not os.path.isfile(config.csv):
                 writer.writerow(["run_id", "timestamp", "n_points", "epsilon", "k_levels", "algo", "time", "clust_time", "cost", "soft_ref", "mem"])
            for r in results:
                writer.writerow(row_base + [r['algo'], r['total_time'], r['clust_time'], r['avg_cost'], r['soft_cost'], r['mem']])
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)

if __name__ == "__main__":
    class ExperimentConfig:
        n = 10000
        dim = 2
        epsilon = 0.01  # Low epsilon to test stability
        k = 4
        csv = "results.csv"
        seed = 42

    config = ExperimentConfig()
    run_experiment(config)
