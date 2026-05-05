#!/usr/bin/env python3
"""
Experiment 9 — 20 Newsgroups: Exact OT vs 2-Level Solver vs 3-Level Solver (Scalability)
Both approximate solvers run until OOM. Distance: Chamfer / WMD (GloVe, precomputed).
"""

import gzip
import math
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
    compute_ratio, fmt_time, fmt_cost, fmt_ratio, fmt_iters, fmt_bytes,
    run_three_level_precomputed,
)

EXP_ID = 9
EXP_NAME = "20 Newsgroups — Exact vs 2L-Solver vs 3L-Solver (Scalability)"
DATASET = "20 Newsgroups (GloVe)"
DATA_DIR = BASE_DIR / "data" / "newsgroups_glove"
EMBEDDINGS_PATH = DATA_DIR / "newsgroups_embeddings.pkl.gz"

N_VALUES = [1_000, 2_000, 5_000, 7_000, 9_000]
EPSILON = 0.01
BATCH_SIZE = 512
SEED = 42
EXACT_N_LIMIT = 5_000
MAX_WORDS = 300

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
    ("2L Peak Mem", 12),
    ("3L Peak Mem", 12),
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
    "2L Peak Mem": lambda r: fmt_bytes(r["sol2"]["peak_mem_bytes"]),
    "3L Peak Mem": lambda r: fmt_bytes(r["sol3"]["peak_mem_bytes"]),
}


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def _clear(device):
    if device.type == "cuda":
        torch.cuda.empty_cache()


def load_embeddings(path):
    with gzip.open(path, "rb") as f:
        embs = pickle.load(f)
    print(f"  Loaded {len(embs):,} document embeddings from {Path(path).name}", flush=True)
    return embs


def sample_pair(all_embs, n, seed):
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(all_embs))
    return ([all_embs[i] for i in perm[:n]],
            [all_embs[i] for i in perm[n:2*n]])


def _pad(descs, device):
    descs = [d[:MAX_WORDS] if len(d) > MAX_WORDS else d for d in descs]
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


def chamfer_matrix(descs_B, descs_A, device, tile_b=50, tile_a=50):
    N = len(descs_B)
    order_B = sorted(range(N), key=lambda i: len(descs_B[i]))
    order_A = sorted(range(N), key=lambda i: len(descs_A[i]))
    sB = [descs_B[i] for i in order_B]
    sA = [descs_A[i] for i in order_A]
    D_sorted = torch.zeros((N, N), dtype=torch.float32, device=device)
    for bs in range(0, N, tile_b):
        be = min(bs + tile_b, N)
        pB, lB = _pad(sB[bs:be], device)
        tb = be - bs; KB = pB.shape[1]
        Bt = pB.reshape(tb * KB, pB.shape[2])
        vB = torch.arange(KB, device=device).unsqueeze(0) < lB.unsqueeze(1)
        for as_ in range(0, N, tile_a):
            ae = min(as_ + tile_a, N)
            pA, lA = _pad(sA[as_:ae], device)
            ta = ae - as_; KA = pA.shape[1]
            At = pA.reshape(ta * KA, pA.shape[2])
            vA = torch.arange(KA, device=device).unsqueeze(0) < lA.unsqueeze(1)
            dists = torch.cdist(Bt, At, compute_mode="use_mm_for_euclid_dist_if_necessary")
            dists = dists.reshape(tb, KB, ta, KA)
            dists.masked_fill_(~vA.view(1, 1, ta, KA), float("inf"))
            fwd = dists.min(3).values.masked_fill(~vB.unsqueeze(2), 0.0)
            fwd = fwd.sum(1) / lB.float().unsqueeze(1).clamp(min=1.0)
            dists.masked_fill_(~vB.view(tb, KB, 1, 1), float("inf"))
            bwd = dists.min(1).values.masked_fill(~vA.unsqueeze(0), 0.0)
            bwd = bwd.sum(2) / lA.float().unsqueeze(0).clamp(min=1.0)
            D_sorted[bs:be, as_:ae] = fwd + bwd
            del pA, At, dists, fwd, bwd
        del pB, Bt
    inv_B = torch.zeros(N, dtype=torch.long, device=device)
    inv_A = torch.zeros(N, dtype=torch.long, device=device)
    for i, o in enumerate(order_B): inv_B[o] = i
    for i, o in enumerate(order_A): inv_A[o] = i
    D = D_sorted[inv_B][:, inv_A]
    del D_sorted
    _sync(device)
    return D


def chamfer_symmetric(descs, device, tile=50):
    N = len(descs)
    order = sorted(range(N), key=lambda i: len(descs[i]))
    sd = [descs[i] for i in order]
    D_sorted = torch.zeros((N, N), dtype=torch.float32, device=device)
    for is_ in range(0, N, tile):
        ie = min(is_ + tile, N)
        pI, lI = _pad(sd[is_:ie], device)
        ti = ie - is_; KI = pI.shape[1]
        It = pI.reshape(ti * KI, pI.shape[2])
        vI = torch.arange(KI, device=device).unsqueeze(0) < lI.unsqueeze(1)
        for js in range(is_, N, tile):
            je = min(js + tile, N)
            pJ, lJ = _pad(sd[js:je], device)
            tj = je - js; KJ = pJ.shape[1]
            Jt = pJ.reshape(tj * KJ, pJ.shape[2])
            vJ = torch.arange(KJ, device=device).unsqueeze(0) < lJ.unsqueeze(1)
            dists = torch.cdist(It, Jt, compute_mode="use_mm_for_euclid_dist_if_necessary")
            dists = dists.reshape(ti, KI, tj, KJ)
            dists.masked_fill_(~vJ.view(1, 1, tj, KJ), float("inf"))
            fwd = dists.min(3).values.masked_fill(~vI.unsqueeze(2), 0.0)
            fwd = fwd.sum(1) / lI.float().unsqueeze(1).clamp(min=1.0)
            dists.masked_fill_(~vI.view(ti, KI, 1, 1), float("inf"))
            bwd = dists.min(1).values.masked_fill(~vJ.unsqueeze(0), 0.0)
            bwd = bwd.sum(2) / lJ.float().unsqueeze(0).clamp(min=1.0)
            block = fwd + bwd
            if is_ == js:
                block = block.clone()
                block[torch.arange(ti, device=device), torch.arange(ti, device=device)] = 0.0
                D_sorted[is_:ie, js:je] = block
            else:
                D_sorted[is_:ie, js:je] = block
                D_sorted[js:je, is_:ie] = block.T
            del pJ, Jt, dists, fwd, bwd
        del pI, It
    inv = torch.zeros(N, dtype=torch.long, device=device)
    for i, o in enumerate(order): inv[o] = i
    D = D_sorted[inv][:, inv]
    del D_sorted
    D.fill_diagonal_(0.0)
    _sync(device)
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
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    solver = SimpleGPUSolver(None, None, epsilon=EPSILON, batch_size=BATCH_SIZE,
                             verbose=False, diameter=1.0, precomputed_clustering=c)
    solver.solve()
    _sync(device)
    peak_mem = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else math.nan
    elapsed = (time.perf_counter() - t0) * 1000.0
    n = D_br_norm.shape[0]
    D_cpu = D_br_norm.detach().cpu()
    cost = D_cpu[torch.arange(n), solver.match_B.cpu()].mean().item() * diameter
    iters = solver.iterations
    del solver, c
    return elapsed, cost, iters, peak_mem


def _run_solver3(D_br_norm, D_rr_norm, device, diameter):
    c = run_three_level_precomputed(D_rr_norm, D_br_norm, EPSILON, BATCH_SIZE)
    _sync(device)
    t0 = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    solver = ThreeLevelGPUSolver(None, None, epsilon=EPSILON, batch_size=BATCH_SIZE,
                                 verbose=False, diameter=1.0, max_iters=500000,
                                 precomputed_clustering=c)
    solver.solve()
    _sync(device)
    peak_mem = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else math.nan
    elapsed = (time.perf_counter() - t0) * 1000.0
    n = D_br_norm.shape[0]
    D_cpu = D_br_norm.detach().cpu()
    cost = D_cpu[torch.arange(n), solver.match_B.cpu()].mean().item() * diameter * c["proxy_max"]
    iters = solver.iterations
    del solver, c
    return elapsed, cost, iters, peak_mem


def _safe_exact(fn, label):
    try:
        t, c = fn()
        return {"time_ms": t, "cost": c, "status": "ok"}
    except Exception as exc:
        print(f"    [{label}] skipped: {exc}", flush=True)
        return {"time_ms": math.nan, "cost": math.nan, "status": "skip"}


def _safe_solver(fn, label, device):
    try:
        t, c, it, peak_mem = fn()
        return {"time_ms": t, "cost": c, "iters": it, "peak_mem_bytes": peak_mem, "status": "ok"}
    except Exception as exc:
        print(f"    [{label}] failed: {exc}", flush=True)
        _clear(device)
        return {"time_ms": math.nan, "cost": math.nan, "iters": math.nan, "peak_mem_bytes": math.nan, "status": "fail"}


def run(device, **kwargs):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    print(f"\n{'─'*60}", flush=True)
    print(f"  Exp {EXP_ID}: {EXP_NAME}", flush=True)
    print(f"  Device: {device}  ε={EPSILON}  exact limit: N≤{EXACT_N_LIMIT:,}", flush=True)
    print(f"{'─'*60}", flush=True)

    if not EMBEDDINGS_PATH.exists():
        print(f"  ERROR: {EMBEDDINGS_PATH} not found.", flush=True)
        return []

    print("  Loading embeddings...", flush=True)
    all_embs = load_embeddings(EMBEDDINGS_PATH)

    rows = []
    for n in N_VALUES:
        print(f"\n  N = {n:,}", flush=True)
        if 2 * n > len(all_embs):
            print(f"    Not enough documents. Skipping.", flush=True)
            continue

        print(f"    Sampling ...", flush=True)
        red_descs, blue_descs = sample_pair(all_embs, n, SEED)

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
