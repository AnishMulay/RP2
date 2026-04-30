#!/usr/bin/env python3
"""
Experiment 7 — CIFAR-10 SIFT: Exact OT vs 2-Level Solver vs 3-Level Solver (Scalability)
Both approximate solvers run until OOM. Distance: Chamfer (precomputed).
"""

import gzip
import math
import pathlib
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

FINAL2_DIR = Path(__file__).resolve().parent.parent
BASE_DIR = FINAL2_DIR.parent.parent.parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(FINAL2_DIR) not in sys.path:
    sys.path.insert(0, str(FINAL2_DIR))

try:
    import ot
except ImportError:
    ot = None

from clustered_push_relabel.clustering.simple_precomputed import SimplePrecomputedClustering
from clustered_push_relabel.solvers.simple_bipartite import SimpleGPUSolver
from clustered_push_relabel.solvers.three_level_bipartite import ThreeLevelGPUSolver
from shared import (
    compute_ratio, fmt_time, fmt_cost, fmt_ratio, fmt_iters,
    run_three_level_precomputed,
)

EXP_ID = 7
EXP_NAME = "CIFAR-10 SIFT — Exact vs 2L-Solver vs 3L-Solver (Scalability)"
DATASET = "CIFAR-10 SIFT"
DATA_DIR = BASE_DIR / "data" / "cifar_sift"
TRAIN_DESC_PATH = DATA_DIR / "cifar10_sift_train.pkl.gz"

N_VALUES = [1_000, 2_000, 5_000, 10_000, 20_000]
EPSILON = 0.01
BATCH_SIZE = 512
SEED = 42
EXACT_N_LIMIT = 5_000

COL_SPECS = [
    ("N",           7),
    ("Exact Time", 14),
    ("2L Time",    12),
    ("3L Time",    12),
    ("Exact Cost", 12),
    ("2L Cost",    12),
    ("3L Cost",    12),
    ("2L Ratio",    9),
    ("3L Ratio",    9),
    ("2L Iters",   10),
    ("3L Iters",   10),
]

FMT_FNS = {
    "N":          lambda r: f"{r['n']:,}",
    "Exact Time": lambda r: fmt_time(r["exact"]["time_ms"]),
    "2L Time":    lambda r: fmt_time(r["sol2"]["time_ms"]),
    "3L Time":    lambda r: fmt_time(r["sol3"]["time_ms"]),
    "Exact Cost": lambda r: fmt_cost(r["exact"]["cost"]),
    "2L Cost":    lambda r: fmt_cost(r["sol2"]["cost"]),
    "3L Cost":    lambda r: fmt_cost(r["sol3"]["cost"]),
    "2L Ratio":   lambda r: fmt_ratio(compute_ratio(r["exact"]["cost"], r["sol2"]["cost"])),
    "3L Ratio":   lambda r: fmt_ratio(compute_ratio(r["exact"]["cost"], r["sol3"]["cost"])),
    "2L Iters":   lambda r: fmt_iters(r["sol2"].get("iters", math.nan)),
    "3L Iters":   lambda r: fmt_iters(r["sol3"].get("iters", math.nan)),
}


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def _clear(device):
    if device.type == "cuda":
        torch.cuda.empty_cache()


def load_descriptors(path):
    with gzip.open(path, "rb") as f:
        descs = pickle.load(f)
    print(f"  Loaded {len(descs):,} descriptor sets from {pathlib.Path(path).name}", flush=True)
    return descs


def sample_pair(all_descs, n, seed):
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(all_descs))
    return ([all_descs[i] for i in perm[:n]],
            [all_descs[i] for i in perm[n:2*n]])


def _pad(descs, device):
    n = len(descs)
    max_K = max(d.shape[0] for d in descs)
    dim   = descs[0].shape[1]
    padded  = torch.zeros((n, max_K, dim), dtype=torch.float32, device=device)
    lengths = torch.empty(n, dtype=torch.int64, device=device)
    for i, d in enumerate(descs):
        K = d.shape[0]
        padded[i, :K] = torch.as_tensor(d, dtype=torch.float32, device=device)
        lengths[i] = K
    return padded, lengths


def chamfer_matrix(descs_B, descs_A, device, tile_b=50, tile_a=200):
    n = len(descs_B)
    pB, lB = _pad(descs_B, device)
    pA, lA = _pad(descs_A, device)
    KB, KA = pB.shape[1], pA.shape[1]
    vB = torch.arange(KB, device=device).unsqueeze(0) < lB.unsqueeze(1)
    vA = torch.arange(KA, device=device).unsqueeze(0) < lA.unsqueeze(1)
    D = torch.zeros((n, n), dtype=torch.float32, device=device)
    for bs in range(0, n, tile_b):
        be = min(bs + tile_b, n)
        tb = be - bs
        Bt = pB[bs:be].reshape(tb * KB, pB.shape[2])
        vBt = vB[bs:be]; lBt = lB[bs:be]
        for as_ in range(0, n, tile_a):
            ae = min(as_ + tile_a, n)
            ta = ae - as_
            At = pA[as_:ae].reshape(ta * KA, pA.shape[2])
            vAt = vA[as_:ae]; lAt = lA[as_:ae]
            dists = torch.cdist(Bt, At, compute_mode="use_mm_for_euclid_dist_if_necessary")
            dists = dists.reshape(tb, KB, ta, KA)
            dists.masked_fill_(~vAt.view(1, 1, ta, KA), float("inf"))
            fwd = dists.min(3).values.masked_fill(~vBt.unsqueeze(2), 0.0)
            fwd = fwd.sum(1) / lBt.float().unsqueeze(1).clamp(min=1.0)
            dists.masked_fill_(~vBt.view(tb, KB, 1, 1), float("inf"))
            bwd = dists.min(1).values.masked_fill(~vAt.unsqueeze(0), 0.0)
            bwd = bwd.sum(2) / lAt.float().unsqueeze(0).clamp(min=1.0)
            D[bs:be, as_:ae] = fwd + bwd
    _sync(device)
    del pB, pA
    return D


def chamfer_symmetric(descs, device, tile=50):
    n = len(descs)
    p, l = _pad(descs, device)
    K = p.shape[1]
    v = torch.arange(K, device=device).unsqueeze(0) < l.unsqueeze(1)
    D = torch.zeros((n, n), dtype=torch.float32, device=device)
    for is_ in range(0, n, tile):
        ie = min(is_ + tile, n)
        ti = ie - is_
        It = p[is_:ie].reshape(ti * K, p.shape[2])
        vI = v[is_:ie]; lI = l[is_:ie]
        for js in range(is_, n, tile):
            je = min(js + tile, n)
            tj = je - js
            Jt = p[js:je].reshape(tj * K, p.shape[2])
            vJ = v[js:je]; lJ = l[js:je]
            dists = torch.cdist(It, Jt, compute_mode="use_mm_for_euclid_dist_if_necessary")
            dists = dists.reshape(ti, K, tj, K)
            dists.masked_fill_(~vJ.view(1, 1, tj, K), float("inf"))
            fwd = dists.min(3).values.masked_fill(~vI.unsqueeze(2), 0.0)
            fwd = fwd.sum(1) / lI.float().unsqueeze(1).clamp(min=1.0)
            dists.masked_fill_(~vI.view(ti, K, 1, 1), float("inf"))
            bwd = dists.min(1).values.masked_fill(~vJ.unsqueeze(0), 0.0)
            bwd = bwd.sum(2) / lJ.float().unsqueeze(0).clamp(min=1.0)
            block = fwd + bwd
            if is_ == js:
                block = block.clone()
                block[torch.arange(ti, device=device), torch.arange(ti, device=device)] = 0.0
                D[is_:ie, js:je] = block
            else:
                D[is_:ie, js:je] = block
                D[js:je, is_:ie] = block.T
    _sync(device)
    del p
    return D


def normalize_chamfer(D_br, D_rr):
    diam = max(float(D_br.max()), float(D_rr.max()), 1e-6)
    return D_br / diam, D_rr / diam, diam


def _run_exact(D_br_norm, diameter):
    n = D_br_norm.shape[0]
    if n > EXACT_N_LIMIT:
        raise RuntimeError(f"Exact skipped: N > {EXACT_N_LIMIT:,}")
    if ot is None:
        raise RuntimeError("POT not installed")
    D_cpu = D_br_norm.detach().cpu()
    a = np.full(n, 1.0 / n, np.float64)
    b = np.full(n, 1.0 / n, np.float64)
    C = D_cpu.to(torch.float64).numpy()
    t0 = time.perf_counter()
    plan = ot.emd(a, b, C, numItermax=10**6)
    elapsed = (time.perf_counter() - t0) * 1000.0
    match = torch.from_numpy(plan.argmax(axis=1).astype(np.int64))
    cost = D_cpu[torch.arange(n), match].mean().item() * diameter
    return elapsed, cost


def _run_solver2(D_br_norm, D_rr_norm, device, diameter):
    c = SimplePrecomputedClustering(epsilon=EPSILON, tile_size=BATCH_SIZE).run(D_rr_norm, D_br_norm)
    _sync(device)
    t0 = time.perf_counter()
    solver = SimpleGPUSolver(None, None, epsilon=EPSILON, batch_size=BATCH_SIZE,
                             verbose=False, diameter=1.0, precomputed_clustering=c)
    solver.solve()
    _sync(device)
    elapsed = (time.perf_counter() - t0) * 1000.0
    n = D_br_norm.shape[0]
    D_cpu = D_br_norm.detach().cpu()
    cost = D_cpu[torch.arange(n), solver.match_B.cpu()].mean().item() * diameter
    iters = solver.iterations
    del solver, c
    return elapsed, cost, iters


def _run_solver3(D_br_norm, D_rr_norm, device, diameter):
    c = run_three_level_precomputed(D_rr_norm, D_br_norm, EPSILON, BATCH_SIZE)
    _sync(device)
    t0 = time.perf_counter()
    solver = ThreeLevelGPUSolver(None, None, epsilon=EPSILON, batch_size=BATCH_SIZE,
                                 verbose=False, diameter=1.0, precomputed_clustering=c)
    solver.solve()
    _sync(device)
    elapsed = (time.perf_counter() - t0) * 1000.0
    n = D_br_norm.shape[0]
    D_cpu = D_br_norm.detach().cpu()
    cost = D_cpu[torch.arange(n), solver.match_B.cpu()].mean().item() * diameter
    iters = solver.iterations
    del solver, c
    return elapsed, cost, iters


def _safe_exact(fn, label):
    try:
        t, c = fn()
        return {"time_ms": t, "cost": c, "status": "ok"}
    except Exception as exc:
        print(f"    [{label}] skipped: {exc}", flush=True)
        return {"time_ms": math.nan, "cost": math.nan, "status": "skip"}


def _safe_solver(fn, label, device):
    try:
        t, c, it = fn()
        return {"time_ms": t, "cost": c, "iters": it, "status": "ok"}
    except Exception as exc:
        print(f"    [{label}] failed: {exc}", flush=True)
        _clear(device)
        return {"time_ms": math.nan, "cost": math.nan, "iters": math.nan, "status": "fail"}


def run(device, **kwargs):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    print(f"\n{'─'*60}", flush=True)
    print(f"  Exp {EXP_ID}: {EXP_NAME}", flush=True)
    print(f"  Device: {device}  ε={EPSILON}  exact limit: N≤{EXACT_N_LIMIT:,}", flush=True)
    print(f"{'─'*60}", flush=True)

    if not TRAIN_DESC_PATH.exists():
        print(f"  ERROR: {TRAIN_DESC_PATH} not found.", flush=True)
        return []

    print("  Loading SIFT descriptors...", flush=True)
    all_descs = load_descriptors(TRAIN_DESC_PATH)

    rows = []
    for n in N_VALUES:
        print(f"\n  N = {n:,}", flush=True)
        if 2 * n > len(all_descs):
            print(f"    Not enough images. Skipping.", flush=True)
            continue

        print(f"    Sampling ...", flush=True)
        red_descs, blue_descs = sample_pair(all_descs, n, SEED)

        print(f"    Computing Chamfer matrices ...", flush=True)
        try:
            t0 = time.perf_counter()
            D_br = chamfer_matrix(blue_descs, red_descs, device)
            D_rr = chamfer_symmetric(red_descs, device)
            print(f"    Chamfer done in {(time.perf_counter()-t0)*1000:.0f} ms", flush=True)
            D_br_norm, D_rr_norm, diameter = normalize_chamfer(D_br, D_rr)
            del D_br, D_rr
        except Exception as exc:
            print(f"    Chamfer failed: {exc}", flush=True)
            _clear(device)
            continue

        print(f"    [1/3] Exact OT ...", flush=True)
        exact = _safe_exact(lambda: _run_exact(D_br_norm, diameter), "Exact")

        print(f"    [2/3] 2-Level Solver ...", flush=True)
        sol2 = _safe_solver(lambda: _run_solver2(D_br_norm, D_rr_norm, device, diameter), "2L-Solver", device)

        print(f"    [3/3] 3-Level Solver ...", flush=True)
        sol3 = _safe_solver(lambda: _run_solver3(D_br_norm, D_rr_norm, device, diameter), "3L-Solver", device)

        rows.append({"n": n, "exact": exact, "sol2": sol2, "sol3": sol3})

        del D_br_norm, D_rr_norm
        _clear(device)

    return rows


if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = run(dev)
    for r in results:
        e, s2, s3 = r["exact"], r["sol2"], r["sol3"]
        print(f"N={r['n']:>6,} exact={fmt_time(e['time_ms'])} "
              f"| 2L={fmt_time(s2['time_ms'])} iters={fmt_iters(s2.get('iters',math.nan))} "
              f"| 3L={fmt_time(s3['time_ms'])} iters={fmt_iters(s3.get('iters',math.nan))}")
