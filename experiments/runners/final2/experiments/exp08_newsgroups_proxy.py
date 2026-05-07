#!/usr/bin/env python3
"""
Experiment 8 — 20 Newsgroups: Exact OT vs 2-Level Proxy vs 3-Level Proxy
Distance: Chamfer / Word-Mover's Distance (GloVe 300d embeddings, precomputed).
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
from shared import (
    fmt_time, fmt_cost, fmt_ratio,
    build_two_level_proxy_matrix, build_three_level_proxy_matrix,
    run_three_level_precomputed,
)

EXP_ID = 8
EXP_NAME = "20 Newsgroups — Exact vs 2L-Proxy vs 3L-Proxy"
DATASET = "20 Newsgroups (GloVe)"
DATA_DIR = BASE_DIR / "data" / "newsgroups_glove"
EMBEDDINGS_PATH = DATA_DIR / "newsgroups_embeddings.pkl.gz"

N_VALUES = [1_000, 2_000, 3_000, 5_000, 7_000, 10_000]
EPSILON = 0.01
BATCH_SIZE = 512
SEED = 42
EXACT_N_LIMIT = 10_000
MAX_WORDS = 300

COL_SPECS = [
    ("N",            7),
    ("Exact Time",  14),
    ("2L-Prx Time", 14),
    ("3L-Prx Time", 14),
    ("Exact Cost",  12),
    ("2L-Prx Cost", 12),
    ("3L-Prx Cost", 12),
    ("2L Ratio",     9),
    ("3L Ratio",     9),
]

FMT_FNS = {
    "N":            lambda r: f"{r['n']:,}",
    "Exact Time":   lambda r: fmt_time(r["exact"]["time_ms"]),
    "2L-Prx Time":  lambda r: fmt_time(r["prx2"]["time_ms"]),
    "3L-Prx Time":  lambda r: fmt_time(r["prx3"]["time_ms"]),
    "Exact Cost":   lambda r: fmt_cost(r["exact"]["cost"]),
    "2L-Prx Cost":  lambda r: fmt_cost(r["prx2"]["cost"]),
    "3L-Prx Cost":  lambda r: fmt_cost(r["prx3"]["cost"]),
    "2L Ratio":     lambda r: fmt_ratio(_cost_ratio(r["exact"]["cost"], r["prx2"]["cost"])),
    "3L Ratio":     lambda r: fmt_ratio(_cost_ratio(r["exact"]["cost"], r["prx3"]["cost"])),
}


def _cost_ratio(exact_cost, proxy_cost):
    if math.isnan(float(exact_cost)) or math.isnan(float(proxy_cost)) or float(exact_cost) == 0.0:
        return math.nan
    return float(proxy_cost) / float(exact_cost)


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
        tb = be - bs
        KB = pB.shape[1]
        Bt = pB.reshape(tb * KB, pB.shape[2])
        vB = torch.arange(KB, device=device).unsqueeze(0) < lB.unsqueeze(1)
        for as_ in range(0, N, tile_a):
            ae = min(as_ + tile_a, N)
            pA, lA = _pad(sA[as_:ae], device)
            ta = ae - as_
            KA = pA.shape[1]
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
    inv_order = torch.zeros(N, dtype=torch.long, device=device)
    for i, o in enumerate(order): inv_order[o] = i
    D = D_sorted[inv_order][:, inv_order]
    del D_sorted
    D.fill_diagonal_(0.0)
    _sync(device)
    return D


def normalize_chamfer(D_br, D_rr):
    diam = max(float(D_br.max()), float(D_rr.max()), 1e-6)
    return D_br / diam, D_rr / diam, diam


def _run_exact(D_br):
    n = D_br.shape[0]
    if n > EXACT_N_LIMIT:
        raise RuntimeError(f"Exact skipped: N > {EXACT_N_LIMIT:,}")
    if ot is None:
        raise RuntimeError("POT not installed")
    a = b = np.full(n, 1.0 / n, dtype=np.float64)
    C_true_np = D_br.cpu().to(torch.float64).numpy()
    t0 = time.perf_counter()
    exact_cost = ot.emd2(a, b, C_true_np, numItermax=10**7)
    elapsed = (time.perf_counter() - t0) * 1000.0
    return elapsed, exact_cost


def _run_proxy2(D_br_norm, D_rr_norm, device, diameter):
    n = D_br_norm.shape[0]
    if n > EXACT_N_LIMIT:
        raise RuntimeError(f"2L-Proxy skipped: N > {EXACT_N_LIMIT:,}")
    if ot is None:
        raise RuntimeError("POT not installed")
    engine = SimplePrecomputedClustering(epsilon=EPSILON, tile_size=BATCH_SIZE)
    c = engine.run(D_rr_norm, D_br_norm)
    a = b = np.full(n, 1.0 / n, dtype=np.float64)
    C_2l_np = build_two_level_proxy_matrix(c, n, device) * diameter
    t0 = time.perf_counter()
    proxy_2l_cost = ot.emd2(a, b, C_2l_np, numItermax=10**7)
    elapsed = (time.perf_counter() - t0) * 1000.0
    return elapsed, proxy_2l_cost


def _run_proxy3(D_br_norm, D_rr_norm, device, diameter):
    n = D_br_norm.shape[0]
    if n > EXACT_N_LIMIT:
        raise RuntimeError(f"3L-Proxy skipped: N > {EXACT_N_LIMIT:,}")
    if ot is None:
        raise RuntimeError("POT not installed")
    c = run_three_level_precomputed(D_rr_norm, D_br_norm, EPSILON, BATCH_SIZE)
    a = b = np.full(n, 1.0 / n, dtype=np.float64)
    C_3l_np = build_three_level_proxy_matrix(c, n, device) * diameter
    t0 = time.perf_counter()
    proxy_3l_cost = ot.emd2(a, b, C_3l_np, numItermax=10**7)
    elapsed = (time.perf_counter() - t0) * 1000.0
    return elapsed, proxy_3l_cost


def _safe(fn, label, device):
    try:
        t, c = fn()
        return {"time_ms": t, "cost": c, "status": "ok"}
    except Exception as exc:
        print(f"    [{label}] skipped: {exc}", flush=True)
        _clear(device)
        return {"time_ms": math.nan, "cost": math.nan, "status": "skip"}


def run(device, **kwargs):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    print(f"\n{'─'*60}", flush=True)
    print(f"  Exp {EXP_ID}: {EXP_NAME}", flush=True)
    print(f"  Device: {device}  ε={EPSILON}  max_words={MAX_WORDS}", flush=True)
    print(f"{'─'*60}", flush=True)

    if not EMBEDDINGS_PATH.exists():
        print(f"  ERROR: {EMBEDDINGS_PATH} not found. Run download_newsgroups_glove.py first.", flush=True)
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
            del D_rr
        except Exception as exc:
            print(f"    Chamfer failed: {exc}", flush=True)
            _clear(device)
            continue

        print(f"    [1/3] Exact OT ...", flush=True)
        exact = _safe(lambda: _run_exact(D_br), "Exact", device)

        print(f"    [2/3] 2L-Proxy ...", flush=True)
        prx2 = _safe(lambda: _run_proxy2(D_br_norm, D_rr_norm, device, diameter), "2L-Proxy", device)

        print(f"    [3/3] 3L-Proxy ...", flush=True)
        prx3 = _safe(lambda: _run_proxy3(D_br_norm, D_rr_norm, device, diameter), "3L-Proxy", device)

        rows.append({"n": n, "exact": exact, "prx2": prx2, "prx3": prx3})

        del D_br, D_br_norm, D_rr_norm
        _clear(device)

    return rows


if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = run(dev)
    for r in results:
        e, p2, p3 = r["exact"], r["prx2"], r["prx3"]
        print(f"N={r['n']:>6,} exact={fmt_time(e['time_ms'])} cost={fmt_cost(e['cost'])} "
              f"| 2L ratio={fmt_ratio(_cost_ratio(e['cost'],p2['cost']))} "
              f"| 3L ratio={fmt_ratio(_cost_ratio(e['cost'],p3['cost']))}")
