#!/usr/bin/env python3
"""
Experiment 15: Old Paper Proxy vs New 2-Level Proxy.

This is a cost-representation accuracy experiment.  POT's exact EMD evaluator
is run three times with uniform marginals: once on the true dense cost matrix,
once on the old paper's minimum shared-cluster 2r proxy, and once on the
confirmed 2-Level proxy matrix.

Implementation plan:
  1. Build true blue-red costs for dense feasible N.
  2. Build the old paper proxy over P=A union B without touching old modules.
  3. Reuse the existing confirmed 2-Level matrix builder where available.
  4. Validate proxy domination and old-builder equality on tiny synthetic data.
  5. Write Markdown, JSON, and CSV outputs for inspection.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import gzip
import json
import math
import pathlib
import pickle
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

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

from clustered_push_relabel.clustering.simple_l1 import SimpleL1Clustering
from clustered_push_relabel.clustering.simple_precomputed import SimplePrecomputedClustering
from shared import build_two_level_proxy_matrix, compute_ratio, fmt_cost, fmt_ratio, fmt_time


EXP_ID = 15
EXP_NAME = "Old Paper Proxy vs New 2-Level Proxy"
DATASET = "Multi"

DATA_DIR = FINAL2_DIR / "data"
EMNIST_DATA_DIR = BASE_DIR / "data"
EMNIST_SPLIT = "byclass"
CIFAR_DATA_DIR = BASE_DIR / "data" / "cifar_sift"
TRAIN_DESC_PATH = CIFAR_DATA_DIR / "cifar10_sift_train.pkl.gz"
NEWSGROUPS_DATA_DIR = BASE_DIR / "data" / "newsgroups_glove"
EMBEDDINGS_PATH = NEWSGROUPS_DATA_DIR / "newsgroups_embeddings.pkl.gz"
MAX_WORDS = 300

EPSILON = 0.01
BATCH_SIZE = 512
SEED = 42
OLD_RADIUS_SCHEME = "geometric"
OLD_PAIR_CHUNK = 128
VALIDATION_TOL = 1e-5

DEFAULT_DATASETS = ["synthetic2d", "mnist_equal", "mnist_biased"]
DEFAULT_N_BY_DATASET = {
    "synthetic2d": [500, 1000, 2000, 5000],
    "mnist_equal": [500, 1000, 2000, 5000],
    "mnist_biased": [500, 1000, 2000, 5000],
    "emnist_equal": [500, 1000, 2000, 5000],
    "emnist_biased": [500, 1000, 2000, 5000],
    "cifar_sift": [500, 1000, 2000, 3000],
    "newsgroups": [500, 1000, 2000, 3000],
}
QUICK_N_VALUES = [500, 1000]

BLUE_DIGITS = list(range(5))
RED_DIGITS = list(range(5, 10))
BLUE_CLASS_END = 31
RED_CLASS_START = 31

COL_SPECS = [
    ("Dataset", 14),
    ("N", 7),
    ("Exact Cost", 12),
    ("Old Cost", 12),
    ("New 2L Cost", 12),
    ("Old Ratio", 10),
    ("New Ratio", 10),
    ("Old Gap", 10),
    ("New Gap", 10),
    ("Old Build", 11),
    ("New Build", 11),
    ("Exact OT", 10),
    ("Old OT", 10),
    ("New OT", 10),
]

FMT_FNS = {
    "Dataset": lambda r: f"{r['dataset']:<14}",
    "N": lambda r: f"{r['n']:,}",
    "Exact Cost": lambda r: fmt_cost(r["exact_cost"]),
    "Old Cost": lambda r: fmt_cost(r["old_proxy_cost"]),
    "New 2L Cost": lambda r: fmt_cost(r["new_2l_proxy_cost"]),
    "Old Ratio": lambda r: fmt_ratio(r["old_ratio"]),
    "New Ratio": lambda r: fmt_ratio(r["new_ratio"]),
    "Old Gap": lambda r: fmt_cost(r["old_gap_abs"]),
    "New Gap": lambda r: fmt_cost(r["new_gap_abs"]),
    "Old Build": lambda r: fmt_time(r["old_proxy_build_time_ms"]),
    "New Build": lambda r: fmt_time(r["new_proxy_build_time_ms"]),
    "Exact OT": lambda r: fmt_time(r["exact_ot_time_ms"]),
    "Old OT": lambda r: fmt_time(r["old_ot_time_ms"]),
    "New OT": lambda r: fmt_time(r["new_ot_time_ms"]),
}


@dataclass(frozen=True)
class LoadedCase:
    red: torch.Tensor | None
    blue: torch.Tensor | None
    metric: str
    distance_note: str
    normalized: bool = False
    red_descs: list | None = None
    blue_descs: list | None = None


@dataclass(frozen=True)
class DistanceBundle:
    C_true: torch.Tensor
    D_rr: torch.Tensor
    D_bb: torch.Tensor
    metric: str
    distance_note: str
    scale: float = 1.0


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _clear(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _time_ms(device: torch.device, fn: Callable):
    _sync(device)
    t0 = time.perf_counter()
    out = fn()
    _sync(device)
    return out, (time.perf_counter() - t0) * 1000.0


def _torch_generator(device: torch.device, seed: int) -> torch.Generator:
    gen = torch.Generator(device=device.type if device.type == "cuda" else "cpu")
    gen.manual_seed(seed)
    return gen


def _sample_mask(n: int, prob: float, device: torch.device, seed: int) -> torch.Tensor:
    gen = _torch_generator(device, seed)
    mask = torch.rand(n, device=device, generator=gen) < prob
    if not mask.any():
        idx = torch.randint(n, (1,), device=device, generator=gen)
        mask[idx] = True
    return mask


def _cdist(x: torch.Tensor, y: torch.Tensor, metric: str) -> torch.Tensor:
    if metric == "l1":
        return torch.cdist(x, y, p=1)
    if metric == "l2":
        return torch.cdist(x, y, p=2, compute_mode="use_mm_for_euclid_dist_if_necessary")
    raise ValueError(f"unknown metric: {metric}")


def _geometric_ceil(values: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Return the smallest (1+epsilon)^i radius >= each nonnegative value."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    out = torch.zeros_like(values)
    mask = values > 0
    if mask.any():
        log_base = math.log1p(epsilon)
        levels = torch.ceil(torch.log(values[mask]) / log_base)
        rounded = torch.exp(levels * log_base)
        out[mask] = torch.maximum(rounded, values[mask])
    return out


def _update_old_proxy_from_center(
    C_old: torch.Tensor,
    blue_d: torch.Tensor,
    red_d: torch.Tensor,
    blue_mask: torch.Tensor,
    red_mask: torch.Tensor,
    epsilon: float,
    pair_chunk: int,
) -> None:
    blue_idx = blue_mask.nonzero(as_tuple=True)[0]
    red_idx = red_mask.nonzero(as_tuple=True)[0]
    if blue_idx.numel() == 0 or red_idx.numel() == 0:
        return
    red_vals = red_d[red_idx]
    for start in range(0, blue_idx.numel(), pair_chunk):
        end = min(start + pair_chunk, blue_idx.numel())
        b_idx = blue_idx[start:end]
        radii = torch.maximum(blue_d[b_idx].unsqueeze(1), red_vals.unsqueeze(0))
        candidate = 2.0 * _geometric_ceil(radii, epsilon)
        current = C_old[b_idx.unsqueeze(1), red_idx.unsqueeze(0)]
        C_old[b_idx.unsqueeze(1), red_idx.unsqueeze(0)] = torch.minimum(current, candidate)


def _center_distances(
    q: int,
    n: int,
    D_br: torch.Tensor,
    D_rr: torch.Tensor,
    D_bb: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if q < n:
        red_d = D_rr[q, :]
        blue_d = D_br[:, q]
    else:
        b = q - n
        red_d = D_br[b, :]
        blue_d = D_bb[b, :]
    return red_d, blue_d


def _nearest_sampled_distances(
    D_br: torch.Tensor,
    D_rr: torch.Tensor,
    D_bb: torch.Tensor,
    sampled_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    n = D_br.shape[0]
    device = D_br.device
    d1_red = torch.full((n,), float("inf"), dtype=D_br.dtype, device=device)
    d1_blue = torch.full((n,), float("inf"), dtype=D_br.dtype, device=device)
    sampled_red = sampled_mask[:n].nonzero(as_tuple=True)[0]
    sampled_blue = sampled_mask[n:].nonzero(as_tuple=True)[0]
    if sampled_red.numel() > 0:
        d1_red = torch.minimum(d1_red, D_rr[:, sampled_red].min(dim=1).values)
        d1_blue = torch.minimum(d1_blue, D_br[:, sampled_red].min(dim=1).values)
    if sampled_blue.numel() > 0:
        d1_red = torch.minimum(d1_red, D_br[sampled_blue, :].min(dim=0).values)
        d1_blue = torch.minimum(d1_blue, D_bb[:, sampled_blue].min(dim=1).values)
    return d1_red, d1_blue


def build_old_paper_proxy_matrix(
    D_br: torch.Tensor,
    D_rr: torch.Tensor,
    D_bb: torch.Tensor,
    epsilon: float,
    seed: int,
    pair_chunk: int = OLD_PAIR_CHUNK,
) -> tuple[torch.Tensor, dict]:
    """
    Build the old paper cluster-induced proxy C_old[b,a].

    P = A union B is sampled at probability |P|^-1/2.  Sampled centers make
    ordinary balls.  Non-sampled centers make balls restricted to points closer
    to that center than to any sampled landmark.  The pair cost is the minimum
    shared cluster radius times two, with geometric ceil radii.
    """
    if D_br.shape != D_rr.shape or D_br.shape != D_bb.shape:
        raise ValueError("D_br, D_rr, and D_bb must all have shape (N, N)")
    n = D_br.shape[0]
    device = D_br.device
    p_total = 2 * n
    sampled_mask = _sample_mask(p_total, p_total ** -0.5, device, seed)
    d1_red, d1_blue = _nearest_sampled_distances(D_br, D_rr, D_bb, sampled_mask)
    C_old = torch.full((n, n), float("inf"), dtype=D_br.dtype, device=device)

    all_red = torch.ones(n, dtype=torch.bool, device=device)
    all_blue = torch.ones(n, dtype=torch.bool, device=device)
    for q in range(p_total):
        red_d, blue_d = _center_distances(q, n, D_br, D_rr, D_bb)
        if bool(sampled_mask[q].item()):
            red_mask = all_red
            blue_mask = all_blue
        else:
            red_mask = red_d < d1_red
            blue_mask = blue_d < d1_blue
        _update_old_proxy_from_center(
            C_old, blue_d, red_d, blue_mask, red_mask, epsilon, pair_chunk
        )

    meta = {
        "old_radius_scheme": OLD_RADIUS_SCHEME,
        "sampled_count": int(sampled_mask.sum().item()),
        "sampled_red_count": int(sampled_mask[:n].sum().item()),
        "sampled_blue_count": int(sampled_mask[n:].sum().item()),
    }
    return C_old, meta


def build_old_paper_proxy_matrix_bruteforce(
    D_br: torch.Tensor,
    D_rr: torch.Tensor,
    D_bb: torch.Tensor,
    epsilon: float,
    seed: int,
) -> torch.Tensor:
    """Tiny-N independent checker for the optimized old proxy builder."""
    n = D_br.shape[0]
    device = D_br.device
    sampled_mask = _sample_mask(2 * n, (2 * n) ** -0.5, device, seed)
    d1_red, d1_blue = _nearest_sampled_distances(D_br, D_rr, D_bb, sampled_mask)
    C = torch.full((n, n), float("inf"), dtype=D_br.dtype, device=device)
    for b in range(n):
        for a in range(n):
            best = torch.tensor(float("inf"), dtype=D_br.dtype, device=device)
            for q in range(2 * n):
                red_d, blue_d = _center_distances(q, n, D_br, D_rr, D_bb)
                eligible = bool(sampled_mask[q].item()) or (
                    bool(red_d[a] < d1_red[a]) and bool(blue_d[b] < d1_blue[b])
                )
                if eligible:
                    radius = _geometric_ceil(torch.maximum(blue_d[b], red_d[a]).reshape(1), epsilon)[0]
                    best = torch.minimum(best, 2.0 * radius)
            C[b, a] = best
    return C


def _build_new_two_level_from_vectors_cpu(
    red: torch.Tensor,
    blue: torch.Tensor,
    metric: str,
    seed: int,
) -> torch.Tensor:
    """Experiment-local CPU equivalent of the confirmed 2-Level proxy rule."""
    n = red.shape[0]
    device = red.device
    sampled_mask = _sample_mask(n, n ** -0.5, device, seed)
    sampled_idx = sampled_mask.nonzero(as_tuple=True)[0]
    red_s = red[sampled_idx]
    DR = _cdist(red_s, red, metric)
    DB = _cdist(blue, red_s, metric)
    d_min_b, nearest_s = DB.min(dim=1)
    C = d_min_b.unsqueeze(1) + DR[nearest_s, :]
    for start in range(0, n, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n)
        dist_tile = _cdist(blue, red[start:end], metric)
        mask = dist_tile < d_min_b.unsqueeze(1)
        b_idx, t_idx = mask.nonzero(as_tuple=True)
        if b_idx.numel() > 0:
            C[b_idx, t_idx + start] = dist_tile[b_idx, t_idx]
    return C


def build_new_two_level_proxy_matrix_for_vectors(
    red: torch.Tensor,
    blue: torch.Tensor,
    metric: str,
    epsilon: float,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, str]:
    """
    Build the confirmed 2-Level proxy for vector data.

    L1 CUDA uses the existing SimpleL1Clustering and shared builder.  L2 and
    CPU-only paths use an experiment-local adapter implementing the same rule:
    local true distances, otherwise d(b,s_b)+d(s_b,a).
    """
    torch.manual_seed(seed)
    if device.type == "cuda" and metric == "l1":
        engine = SimpleL1Clustering(epsilon=epsilon, tile_size=BATCH_SIZE)
        clustering = engine.run(red.to(device), blue.to(device))
        C = torch.from_numpy(build_two_level_proxy_matrix(clustering, red.shape[0], device))
        return C.to(torch.float32), "SimpleL1Clustering + build_two_level_proxy_matrix"
    C = _build_new_two_level_from_vectors_cpu(red.cpu(), blue.cpu(), metric, seed)
    return C.to(torch.float32), "experiment_local_equivalent_2l_adapter"


def build_new_two_level_proxy_matrix_for_precomputed(
    D_br: torch.Tensor,
    D_rr: torch.Tensor,
    epsilon: float,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, str]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        clustering = SimplePrecomputedClustering(epsilon=epsilon, tile_size=BATCH_SIZE).run(D_rr, D_br)
        C = torch.from_numpy(build_two_level_proxy_matrix(clustering, D_br.shape[0], device))
        return C.to(torch.float32), "SimplePrecomputedClustering + build_two_level_proxy_matrix"
    n = D_br.shape[0]
    sampled_mask = _sample_mask(n, n ** -0.5, D_br.device, seed)
    sampled_idx = sampled_mask.nonzero(as_tuple=True)[0]
    DR = D_rr[sampled_idx, :]
    DB = D_br[:, sampled_idx]
    d_min_b, nearest_s = DB.min(dim=1)
    C = d_min_b.unsqueeze(1) + DR[nearest_s, :]
    mask = D_br < d_min_b.unsqueeze(1)
    C[mask] = D_br[mask]
    return C.to(torch.float32), "experiment_local_equivalent_precomputed_2l_adapter"


def _sample_from_digits(images, labels, digit_set, n_total, rng):
    classes = sorted(digit_set)
    spc = n_total // len(classes)
    if spc == 0:
        raise ValueError(f"n_total={n_total} too small for {len(classes)} classes")
    parts = []
    for cls in classes:
        idx = np.flatnonzero(labels == cls).copy()
        if idx.size < spc:
            warnings.warn(f"Digit {cls}: only {idx.size} available, need {spc}. Skipping.")
            continue
        rng.shuffle(idx)
        parts.append(images[idx[:spc]])
    return np.concatenate(parts)


def _normalize_histograms(red: np.ndarray, blue: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    for arr in (red, blue):
        s = arr.sum(axis=1, keepdims=True)
        np.maximum(s, 1e-8, out=s)
        arr /= s
    return torch.from_numpy(red.astype(np.float32)), torch.from_numpy(blue.astype(np.float32))


def load_synthetic2d(n: int, seed: int, device: torch.device) -> LoadedCase:
    gen = _torch_generator(device, seed)
    red = torch.rand((n, 2), generator=gen, device=device, dtype=torch.float32)
    blue = torch.rand((n, 2), generator=gen, device=device, dtype=torch.float32)
    pts = torch.cat([red, blue], dim=0)
    diameter = torch.cdist(pts, pts, p=2).max().clamp(min=1e-8)
    red = (red / diameter).cpu()
    blue = (blue / diameter).cpu()
    return LoadedCase(
        red=red,
        blue=blue,
        metric="l2",
        distance_note="L2 on synthetic [0,1]^2 points normalized by joint diameter",
        normalized=True,
    )


def load_mnist_equal(n_samples: int, seed: int, _device: torch.device) -> LoadedCase:
    import torchvision

    train = torchvision.datasets.MNIST(root=str(DATA_DIR), train=True, download=False)
    test = torchvision.datasets.MNIST(root=str(DATA_DIR), train=False, download=False)
    images = torch.cat([train.data, test.data], dim=0).numpy().reshape(-1, 784)
    labels = torch.cat([train.targets, test.targets], dim=0).numpy()
    classes = np.unique(labels)
    spc = n_samples // len(classes)
    if spc == 0:
        raise ValueError(f"n_samples={n_samples} too small for {len(classes)} classes")
    rng = np.random.RandomState(seed)
    red_parts, blue_parts = [], []
    for cls in classes:
        idx = np.flatnonzero(labels == cls).copy()
        needed = 2 * spc
        if idx.size < needed:
            warnings.warn(f"Class {cls}: only {idx.size} samples, need {needed}. Skipping.")
            continue
        rng.shuffle(idx)
        chosen = idx[:needed]
        red_parts.append(images[chosen[:spc]])
        blue_parts.append(images[chosen[spc:needed]])
    red = np.concatenate(red_parts).astype(np.float32) / 255.0
    blue = np.concatenate(blue_parts).astype(np.float32) / 255.0
    red_t, blue_t = _normalize_histograms(red, blue)
    return LoadedCase(red_t, blue_t, "l1", "L1 on probability-normalized MNIST pixel histograms")


def load_mnist_biased(n_samples: int, seed: int, _device: torch.device) -> LoadedCase:
    import torchvision

    train = torchvision.datasets.MNIST(root=str(DATA_DIR), train=True, download=False)
    test = torchvision.datasets.MNIST(root=str(DATA_DIR), train=False, download=False)
    images = torch.cat([train.data, test.data], dim=0).numpy().reshape(-1, 784)
    labels = torch.cat([train.targets, test.targets], dim=0).numpy()
    rng_r = np.random.RandomState(seed)
    rng_b = np.random.RandomState(seed + 1)
    red = _sample_from_digits(images, labels, RED_DIGITS, n_samples, rng_r).astype(np.float32) / 255.0
    blue = _sample_from_digits(images, labels, BLUE_DIGITS, n_samples, rng_b).astype(np.float32) / 255.0
    red_t, blue_t = _normalize_histograms(red, blue)
    return LoadedCase(red_t, blue_t, "l1", "L1 on probability-normalized MNIST histograms; red digits 5-9, blue digits 0-4")


def _sample_classes(images, labels, class_indices, n_total, rng):
    classes = sorted(class_indices)
    spc = n_total // len(classes)
    if spc == 0:
        raise ValueError(f"n_total={n_total} too small for {len(classes)} classes")
    parts = []
    for cls in classes:
        idx = np.flatnonzero(labels == cls).copy()
        if idx.size < spc:
            warnings.warn(f"Class {cls}: {idx.size} available, need {spc}. Skipping.")
            continue
        rng.shuffle(idx)
        parts.append(images[idx[:spc]])
    return np.concatenate(parts)


def load_emnist_equal(n_samples: int, seed: int, _device: torch.device) -> LoadedCase:
    import torchvision

    train = torchvision.datasets.EMNIST(root=str(EMNIST_DATA_DIR), split=EMNIST_SPLIT, train=True, download=False)
    test = torchvision.datasets.EMNIST(root=str(EMNIST_DATA_DIR), split=EMNIST_SPLIT, train=False, download=False)
    images = torch.cat([train.data, test.data], dim=0).numpy()
    labels = torch.cat([train.targets, test.targets], dim=0).numpy()
    images = images.reshape(-1, 28, 28).transpose(0, 2, 1).reshape(-1, 784)
    classes = np.unique(labels)
    spc = n_samples // len(classes)
    if spc == 0:
        raise ValueError(f"n_samples={n_samples} too small for {len(classes)} EMNIST classes")
    rng = np.random.RandomState(seed)
    red_parts, blue_parts = [], []
    for cls in classes:
        idx = np.flatnonzero(labels == cls).copy()
        needed = 2 * spc
        if idx.size < needed:
            warnings.warn(f"Class {cls}: {idx.size} available, need {needed}. Skipping.")
            continue
        rng.shuffle(idx)
        chosen = idx[:needed]
        red_parts.append(images[chosen[:spc]])
        blue_parts.append(images[chosen[spc:needed]])
    red = np.concatenate(red_parts).astype(np.float32) / 255.0
    blue = np.concatenate(blue_parts).astype(np.float32) / 255.0
    red_t, blue_t = _normalize_histograms(red, blue)
    return LoadedCase(red_t, blue_t, "l1", "L1 on probability-normalized EMNIST pixel histograms")


def load_emnist_biased(n_samples: int, seed: int, _device: torch.device) -> LoadedCase:
    import torchvision

    train = torchvision.datasets.EMNIST(root=str(EMNIST_DATA_DIR), split=EMNIST_SPLIT, train=True, download=False)
    test = torchvision.datasets.EMNIST(root=str(EMNIST_DATA_DIR), split=EMNIST_SPLIT, train=False, download=False)
    images = torch.cat([train.data, test.data], dim=0).numpy()
    labels = torch.cat([train.targets, test.targets], dim=0).numpy()
    images = images.reshape(-1, 28, 28).transpose(0, 2, 1).reshape(-1, 784)
    all_classes = np.unique(labels).tolist()
    blue_classes = [c for c in all_classes if c < BLUE_CLASS_END]
    red_classes = [c for c in all_classes if c >= RED_CLASS_START]
    rng_r = np.random.RandomState(seed)
    rng_b = np.random.RandomState(seed + 1)
    red = _sample_classes(images, labels, red_classes, n_samples, rng_r).astype(np.float32) / 255.0
    blue = _sample_classes(images, labels, blue_classes, n_samples, rng_b).astype(np.float32) / 255.0
    red_t, blue_t = _normalize_histograms(red, blue)
    return LoadedCase(red_t, blue_t, "l1", "L1 on probability-normalized EMNIST histograms; biased class split")


def _load_pickle_gz(path: Path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def _sample_pair(items, n: int, seed: int):
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(items))
    return [items[i] for i in perm[:n]], [items[i] for i in perm[n : 2 * n]]


def load_cifar_sift(n: int, seed: int, _device: torch.device) -> LoadedCase:
    if not TRAIN_DESC_PATH.exists():
        raise FileNotFoundError(f"{TRAIN_DESC_PATH} not found. Run download_cifar_sift.py first.")
    descs = _load_pickle_gz(TRAIN_DESC_PATH)
    if 2 * n > len(descs):
        raise ValueError(f"not enough CIFAR SIFT descriptor sets: need {2*n:,}, have {len(descs):,}")
    red_descs, blue_descs = _sample_pair(descs, n, seed)
    return LoadedCase(
        red=None,
        blue=None,
        metric="precomputed_chamfer",
        distance_note="symmetric Chamfer over CIFAR-10 SIFT descriptor sets, scaled back by joint P diameter",
        red_descs=red_descs,
        blue_descs=blue_descs,
    )


def load_newsgroups(n: int, seed: int, _device: torch.device) -> LoadedCase:
    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(f"{EMBEDDINGS_PATH} not found. Run download_newsgroups_glove.py first.")
    embs = _load_pickle_gz(EMBEDDINGS_PATH)
    if 2 * n > len(embs):
        raise ValueError(f"not enough Newsgroups documents: need {2*n:,}, have {len(embs):,}")
    red_descs, blue_descs = _sample_pair(embs, n, seed)
    red_descs = [d[:MAX_WORDS] if len(d) > MAX_WORDS else d for d in red_descs]
    blue_descs = [d[:MAX_WORDS] if len(d) > MAX_WORDS else d for d in blue_descs]
    return LoadedCase(
        red=None,
        blue=None,
        metric="precomputed_chamfer",
        distance_note=f"symmetric Chamfer over Newsgroups GloVe embeddings, max_words={MAX_WORDS}, scaled back by joint P diameter",
        red_descs=red_descs,
        blue_descs=blue_descs,
    )


DATASET_LOADERS = {
    "synthetic2d": ("Synthetic2D", load_synthetic2d),
    "mnist_equal": ("MNIST-Equal", load_mnist_equal),
    "mnist_biased": ("MNIST-Biased", load_mnist_biased),
    "emnist_equal": ("EMNIST-Equal", load_emnist_equal),
    "emnist_biased": ("EMNIST-Biased", load_emnist_biased),
    "cifar_sift": ("CIFAR-SIFT", load_cifar_sift),
    "newsgroups": ("Newsgroups", load_newsgroups),
}


def _pad_descriptor_sets(descs: list, device: torch.device):
    n = len(descs)
    max_k = max(np.asarray(d).shape[0] for d in descs)
    dim = np.asarray(descs[0]).shape[1]
    padded = torch.zeros((n, max_k, dim), dtype=torch.float32, device=device)
    lengths = torch.empty(n, dtype=torch.int64, device=device)
    for i, d in enumerate(descs):
        arr = torch.as_tensor(np.asarray(d, dtype=np.float32), device=device)
        k = arr.shape[0]
        padded[i, :k] = arr
        lengths[i] = k
    return padded, lengths


def chamfer_matrix(descs_B: list, descs_A: list, device: torch.device, tile_b: int = 50, tile_a: int = 100) -> torch.Tensor:
    n = len(descs_B)
    pB, lB = _pad_descriptor_sets(descs_B, device)
    pA, lA = _pad_descriptor_sets(descs_A, device)
    kB, kA = pB.shape[1], pA.shape[1]
    vB = torch.arange(kB, device=device).unsqueeze(0) < lB.unsqueeze(1)
    vA = torch.arange(kA, device=device).unsqueeze(0) < lA.unsqueeze(1)
    D = torch.zeros((n, n), dtype=torch.float32, device=device)
    for bs in range(0, n, tile_b):
        be = min(bs + tile_b, n)
        tb = be - bs
        Bt = pB[bs:be].reshape(tb * kB, pB.shape[2])
        vBt = vB[bs:be]
        lBt = lB[bs:be]
        for a0 in range(0, n, tile_a):
            ae = min(a0 + tile_a, n)
            ta = ae - a0
            At = pA[a0:ae].reshape(ta * kA, pA.shape[2])
            vAt = vA[a0:ae]
            lAt = lA[a0:ae]
            dists = torch.cdist(Bt, At, p=2, compute_mode="use_mm_for_euclid_dist_if_necessary")
            dists = dists.reshape(tb, kB, ta, kA)
            dists.masked_fill_(~vAt.view(1, 1, ta, kA), float("inf"))
            fwd = dists.min(3).values.masked_fill(~vBt.unsqueeze(2), 0.0)
            fwd = fwd.sum(1) / lBt.float().unsqueeze(1).clamp(min=1.0)
            dists.masked_fill_(~vBt.view(tb, kB, 1, 1), float("inf"))
            bwd = dists.min(1).values.masked_fill(~vAt.unsqueeze(0), 0.0)
            bwd = bwd.sum(2) / lAt.float().unsqueeze(0).clamp(min=1.0)
            D[bs:be, a0:ae] = fwd + bwd
    _sync(device)
    return D


def chamfer_symmetric(descs: list, device: torch.device, tile: int = 50) -> torch.Tensor:
    n = len(descs)
    p, lengths = _pad_descriptor_sets(descs, device)
    k = p.shape[1]
    valid = torch.arange(k, device=device).unsqueeze(0) < lengths.unsqueeze(1)
    D = torch.zeros((n, n), dtype=torch.float32, device=device)
    for i0 in range(0, n, tile):
        ie = min(i0 + tile, n)
        ti = ie - i0
        It = p[i0:ie].reshape(ti * k, p.shape[2])
        vI = valid[i0:ie]
        lI = lengths[i0:ie]
        for j0 in range(i0, n, tile):
            je = min(j0 + tile, n)
            tj = je - j0
            Jt = p[j0:je].reshape(tj * k, p.shape[2])
            vJ = valid[j0:je]
            lJ = lengths[j0:je]
            dists = torch.cdist(It, Jt, p=2, compute_mode="use_mm_for_euclid_dist_if_necessary")
            dists = dists.reshape(ti, k, tj, k)
            dists.masked_fill_(~vJ.view(1, 1, tj, k), float("inf"))
            fwd = dists.min(3).values.masked_fill(~vI.unsqueeze(2), 0.0)
            fwd = fwd.sum(1) / lI.float().unsqueeze(1).clamp(min=1.0)
            dists.masked_fill_(~vI.view(ti, k, 1, 1), float("inf"))
            bwd = dists.min(1).values.masked_fill(~vJ.unsqueeze(0), 0.0)
            bwd = bwd.sum(2) / lJ.float().unsqueeze(0).clamp(min=1.0)
            block = fwd + bwd
            if i0 == j0:
                block = block.clone()
                block[torch.arange(ti, device=device), torch.arange(ti, device=device)] = 0.0
            D[i0:ie, j0:je] = block
            if i0 != j0:
                D[j0:je, i0:ie] = block.T
    _sync(device)
    return D


def build_distance_bundle(case: LoadedCase, device: torch.device) -> DistanceBundle:
    if case.metric in ("l1", "l2"):
        assert case.red is not None and case.blue is not None
        red = case.red.to(device)
        blue = case.blue.to(device)
        C_true = _cdist(blue, red, case.metric).to(torch.float32)
        D_rr = _cdist(red, red, case.metric).to(torch.float32)
        D_bb = _cdist(blue, blue, case.metric).to(torch.float32)
        return DistanceBundle(C_true, D_rr, D_bb, case.metric, case.distance_note)

    if case.metric == "precomputed_chamfer":
        assert case.red_descs is not None and case.blue_descs is not None
        D_br = chamfer_matrix(case.blue_descs, case.red_descs, device)
        D_rr = chamfer_symmetric(case.red_descs, device)
        D_bb = chamfer_symmetric(case.blue_descs, device)
        scale = max(float(D_br.max().item()), float(D_rr.max().item()), float(D_bb.max().item()), 1e-8)
        return DistanceBundle(
            C_true=D_br / scale,
            D_rr=D_rr / scale,
            D_bb=D_bb / scale,
            metric=case.metric,
            distance_note=case.distance_note,
            scale=scale,
        )
    raise ValueError(f"unsupported metric: {case.metric}")


def run_emd2(C: torch.Tensor) -> tuple[float, float]:
    if ot is None:
        raise RuntimeError("POT is not installed; install POT to run exact OT values")
    C_np = C.detach().cpu().to(torch.float64).numpy()
    n = C_np.shape[0]
    hist = np.full(n, 1.0 / n, dtype=np.float64)
    t0 = time.perf_counter()
    value = float(ot.emd2(hist, hist, C_np, numItermax=10**6))
    elapsed = (time.perf_counter() - t0) * 1000.0
    return value, elapsed


def sanity_fields(proxy: torch.Tensor, true: torch.Tensor, prefix: str) -> dict:
    diff = proxy - true
    finite = torch.isfinite(proxy)
    ratio_mask = true > 1e-12
    ratios = proxy[ratio_mask] / true[ratio_mask]
    return {
        f"{prefix}_min_proxy_minus_true": float(diff.min().item()),
        f"{prefix}_nonfinite_count": int((~finite).sum().item()),
        f"{prefix}_max_ratio_pairwise": float(ratios.max().item()) if ratios.numel() else math.nan,
    }


def ratio_summary(proxy: torch.Tensor, true: torch.Tensor) -> dict:
    ratios = (proxy[true > 1e-12] / true[true > 1e-12]).detach().cpu().numpy()
    if ratios.size == 0:
        return {"max": math.nan, "mean": math.nan, "p90": math.nan, "p99": math.nan}
    return {
        "max": float(np.max(ratios)),
        "mean": float(np.mean(ratios)),
        "p90": float(np.percentile(ratios, 90)),
        "p99": float(np.percentile(ratios, 99)),
    }


def validate_old_proxy(device: torch.device, seed: int = SEED, epsilon: float = EPSILON) -> dict:
    print("  Validation: tiny synthetic old/new proxy checks", flush=True)
    case = load_synthetic2d(50, seed, device)
    bundle = build_distance_bundle(case, device)
    C_true = bundle.C_true
    C_old, old_meta = build_old_paper_proxy_matrix(
        C_true, bundle.D_rr, bundle.D_bb, epsilon, seed, pair_chunk=32
    )
    C_brute = build_old_paper_proxy_matrix_bruteforce(
        C_true, bundle.D_rr, bundle.D_bb, epsilon, seed
    )
    old_delta = float((C_old - C_brute).abs().max().item())
    if not torch.isfinite(C_old).all():
        raise RuntimeError("old proxy validation failed: non-finite entries")
    old_min = float((C_old - C_true).min().item())
    if old_min < -VALIDATION_TOL:
        raise RuntimeError(f"old proxy validation failed: min(C_old-C_true)={old_min}")
    if old_delta > VALIDATION_TOL:
        raise RuntimeError(f"old proxy brute-force equality failed: max delta={old_delta}")

    C_new, new_builder = build_new_two_level_proxy_matrix_for_vectors(
        case.red, case.blue, case.metric, epsilon, seed, device
    )
    C_new = C_new.to(device)
    if not torch.isfinite(C_new).all():
        raise RuntimeError("new proxy validation failed: non-finite entries")
    new_min = float((C_new - C_true).min().item())
    if new_min < -VALIDATION_TOL:
        raise RuntimeError(f"new proxy validation failed: min(C_new-C_true)={new_min}")

    old_ratios = ratio_summary(C_old, C_true)
    new_ratios = ratio_summary(C_new, C_true)
    summary = {
        "validation_passed": True,
        "old_radius_scheme": OLD_RADIUS_SCHEME,
        "old_bruteforce_max_abs_delta": old_delta,
        "old_min_proxy_minus_true": old_min,
        "old_ratio_summary": old_ratios,
        "new_min_proxy_minus_true": new_min,
        "new_ratio_summary": new_ratios,
        "new_proxy_builder_validation": new_builder,
        **old_meta,
    }
    print(
        "  Validation passed: "
        f"old max/mean/p90/p99={old_ratios['max']:.4f}/{old_ratios['mean']:.4f}/"
        f"{old_ratios['p90']:.4f}/{old_ratios['p99']:.4f}; "
        f"new max/mean/p90/p99={new_ratios['max']:.4f}/{new_ratios['mean']:.4f}/"
        f"{new_ratios['p90']:.4f}/{new_ratios['p99']:.4f}",
        flush=True,
    )
    return summary


def _make_skip_row(display_name: str, n: int, exc: Exception | str) -> dict:
    row = {
        "dataset": display_name,
        "n": n,
        "status": "skip",
        "error": str(exc),
        "exact_cost": math.nan,
        "old_proxy_cost": math.nan,
        "new_2l_proxy_cost": math.nan,
        "old_ratio": math.nan,
        "new_ratio": math.nan,
        "old_gap_abs": math.nan,
        "new_gap_abs": math.nan,
        "old_proxy_build_time_ms": math.nan,
        "new_proxy_build_time_ms": math.nan,
        "exact_ot_time_ms": math.nan,
        "old_ot_time_ms": math.nan,
        "new_ot_time_ms": math.nan,
        "old_min_proxy_minus_true": math.nan,
        "old_nonfinite_count": math.nan,
        "old_max_ratio_pairwise": math.nan,
        "new_min_proxy_minus_true": math.nan,
        "new_nonfinite_count": math.nan,
        "new_max_ratio_pairwise": math.nan,
        "metric": "",
        "distance_note": "",
        "old_radius_scheme": OLD_RADIUS_SCHEME,
        "new_proxy_builder": "",
    }
    return row


def run_one_case(dataset_key: str, display_name: str, n: int, device: torch.device, seed: int, epsilon: float) -> dict:
    loader = DATASET_LOADERS[dataset_key][1]
    print(f"\n  Dataset={display_name}  N={n:,}", flush=True)
    case = loader(n, seed, device)
    print(f"    Distance: {case.distance_note}", flush=True)

    bundle, dist_time = _time_ms(device, lambda: build_distance_bundle(case, device))
    print(f"    Dense true/within-set distances built in {dist_time:.0f} ms", flush=True)
    C_true = bundle.C_true

    C_old, old_build_time = _time_ms(
        device,
        lambda: build_old_paper_proxy_matrix(
            C_true, bundle.D_rr, bundle.D_bb, epsilon, seed, OLD_PAIR_CHUNK
        ),
    )
    C_old, old_meta = C_old
    print(
        f"    Old proxy built in {old_build_time:.0f} ms "
        f"(sampled={old_meta['sampled_count']})",
        flush=True,
    )

    if case.metric == "precomputed_chamfer":
        C_new, new_build_time = _time_ms(
            device,
            lambda: build_new_two_level_proxy_matrix_for_precomputed(
                C_true, bundle.D_rr, epsilon, seed, device
            ),
        )
    else:
        assert case.red is not None and case.blue is not None
        C_new, new_build_time = _time_ms(
            device,
            lambda: build_new_two_level_proxy_matrix_for_vectors(
                case.red, case.blue, case.metric, epsilon, seed, device
            ),
        )
    C_new, new_builder = C_new
    C_new = C_new.to(device)
    print(f"    New 2L proxy built in {new_build_time:.0f} ms ({new_builder})", flush=True)

    scale = float(bundle.scale)
    C_true_eval = C_true * scale
    C_old_eval = C_old * scale
    C_new_eval = C_new * scale

    exact_cost, exact_ot_time = run_emd2(C_true_eval)
    old_cost, old_ot_time = run_emd2(C_old_eval)
    new_cost, new_ot_time = run_emd2(C_new_eval)
    print(
        f"    OT values: exact={exact_cost:.6f} old={old_cost:.6f} new={new_cost:.6f}",
        flush=True,
    )

    old_ratio = compute_ratio(exact_cost, old_cost)
    new_ratio = compute_ratio(exact_cost, new_cost)
    row = {
        "dataset": display_name,
        "dataset_key": dataset_key,
        "n": n,
        "status": "ok",
        "error": "",
        "exact_cost": exact_cost,
        "old_proxy_cost": old_cost,
        "new_2l_proxy_cost": new_cost,
        "old_ratio": old_ratio,
        "new_ratio": new_ratio,
        "old_gap_abs": old_cost - exact_cost if not math.isnan(old_ratio) else math.nan,
        "new_gap_abs": new_cost - exact_cost if not math.isnan(new_ratio) else math.nan,
        "old_proxy_build_time_ms": old_build_time,
        "new_proxy_build_time_ms": new_build_time,
        "exact_ot_time_ms": exact_ot_time,
        "old_ot_time_ms": old_ot_time,
        "new_ot_time_ms": new_ot_time,
        "metric": bundle.metric,
        "distance_note": bundle.distance_note,
        "distance_scale": scale,
        "old_radius_scheme": OLD_RADIUS_SCHEME,
        "new_proxy_builder": new_builder,
        **sanity_fields(C_old_eval, C_true_eval, "old"),
        **sanity_fields(C_new_eval, C_true_eval, "new"),
        **old_meta,
    }
    del C_true, C_old, C_new, bundle
    _clear(device)
    return row


def run_experiment(
    dataset_keys: list[str],
    n_values_by_dataset: dict[str, list[int]],
    device: torch.device,
    seed: int,
    epsilon: float,
) -> list[dict]:
    rows = []
    for key in dataset_keys:
        display_name = DATASET_LOADERS[key][0]
        for n in n_values_by_dataset[key]:
            try:
                rows.append(run_one_case(key, display_name, n, device, seed, epsilon))
            except Exception as exc:
                print(f"    Skipping {display_name} N={n:,}: {exc}", flush=True)
                rows.append(_make_skip_row(display_name, n, exc))
                _clear(device)
    return rows


def run(device: torch.device, **kwargs) -> list[dict]:
    dataset_keys = kwargs.get("dataset_keys", DEFAULT_DATASETS)
    n_values = kwargs.get("n_values", QUICK_N_VALUES)
    seed = int(kwargs.get("seed", SEED))
    epsilon = float(kwargs.get("epsilon", EPSILON))
    if not kwargs.get("skip_validation", False):
        validate_old_proxy(device, seed, epsilon)
    n_values_by_dataset = {key: list(n_values) for key in dataset_keys}
    return run_experiment(dataset_keys, n_values_by_dataset, device, seed, epsilon)


def print_table(rows: list[dict]) -> None:
    headers = [h for h, _ in COL_SPECS]
    widths = [max(w, len(h)) for h, w in COL_SPECS]
    print()
    print(" | ".join(f"{h:>{w}}" for h, w in zip(headers, widths)))
    print("-+-".join("-" * w for w in widths))
    prev = None
    for row in rows:
        if prev is not None and row["dataset"] != prev:
            print()
        print(" | ".join(f"{FMT_FNS[h](row):>{w}}" for h, w in zip(headers, widths)))
        prev = row["dataset"]


def _json_safe(obj):
    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else obj
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


def write_outputs(rows: list[dict], validation: dict, output_dir: Path, seed: int, epsilon: float, device: torch.device) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts_file = _dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    ts_human = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    stem = f"exp15_old_vs_new_proxy_{ts_file}"
    md_path = output_dir / f"{stem}.md"
    json_path = output_dir / f"{stem}.json"
    csv_path = output_dir / f"{stem}.csv"

    pot_version = getattr(ot, "__version__", "not installed") if ot is not None else "not installed"
    lines = [
        "# Experiment 15: Old Paper Proxy vs New 2-Level Proxy",
        "",
        f"Generated: {ts_human}",
        "",
        "## Validation Summary",
        "",
        f"- old_radius_scheme: {OLD_RADIUS_SCHEME}",
        f"- seed: {seed}",
        f"- epsilon: {epsilon}",
        f"- validation passed: {validation.get('validation_passed', False)}",
        f"- device: {device}",
        f"- POT version: {pot_version}",
        "",
    ]
    for key in sorted({r["dataset"] for r in rows}):
        ds_rows = [r for r in rows if r["dataset"] == key]
        if not ds_rows:
            continue
        metric = next((r.get("distance_note", "") for r in ds_rows if r.get("distance_note")), "")
        lines.extend([f"## Dataset: {key}", ""])
        if metric:
            lines.extend([f"_Distance: {metric}_", ""])
        headers = [
            "N",
            "Exact Cost",
            "Old Proxy Cost",
            "New 2L Cost",
            "Old Ratio",
            "New Ratio",
            "Old Gap",
            "New Gap",
            "Old Build Time",
            "New Build Time",
            "Exact OT Time",
            "Old OT Time",
            "New OT Time",
        ]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join("---:" for _ in headers) + " |")
        for r in ds_rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"{r['n']:,}",
                        fmt_cost(r["exact_cost"]),
                        fmt_cost(r["old_proxy_cost"]),
                        fmt_cost(r["new_2l_proxy_cost"]),
                        fmt_ratio(r["old_ratio"]),
                        fmt_ratio(r["new_ratio"]),
                        fmt_cost(r["old_gap_abs"]),
                        fmt_cost(r["new_gap_abs"]),
                        fmt_time(r["old_proxy_build_time_ms"]),
                        fmt_time(r["new_proxy_build_time_ms"]),
                        fmt_time(r["exact_ot_time_ms"]),
                        fmt_time(r["old_ot_time_ms"]),
                        fmt_time(r["new_ot_time_ms"]),
                    ]
                )
                + " |"
            )
        lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    json_path.write_text(
        json.dumps(
            _json_safe(
                {
                    "exp_id": EXP_ID,
                    "exp_name": EXP_NAME,
                    "generated": ts_human,
                    "seed": seed,
                    "epsilon": epsilon,
                    "device": str(device),
                    "pot_version": pot_version,
                    "validation": validation,
                    "rows": rows,
                }
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(_json_safe(row))
    print(f"\n  Markdown: {md_path}", flush=True)
    print(f"  JSON:     {json_path}", flush=True)
    print(f"  CSV:      {csv_path}", flush=True)
    return {"markdown": str(md_path), "json": str(json_path), "csv": str(csv_path)}


def _parse_datasets(raw: str) -> list[str]:
    if raw.strip().lower() == "all":
        return list(DATASET_LOADERS.keys())
    keys = [x.strip().lower() for x in raw.split(",") if x.strip()]
    unknown = [k for k in keys if k not in DATASET_LOADERS]
    if unknown:
        raise ValueError(f"unknown dataset key(s): {', '.join(unknown)}")
    return keys


def _parse_n_values(raw: str | None, dataset_keys: list[str]) -> dict[str, list[int]]:
    if raw:
        values = [int(x.strip()) for x in raw.split(",") if x.strip()]
        return {key: values for key in dataset_keys}
    return {key: QUICK_N_VALUES if key in DEFAULT_DATASETS else DEFAULT_N_BY_DATASET[key] for key in dataset_keys}


def main() -> None:
    parser = argparse.ArgumentParser(description=EXP_NAME)
    parser.add_argument("--validate-old-proxy", action="store_true", help="Run only tiny synthetic validation checks.")
    parser.add_argument("--skip-validation", action="store_true", help="Skip validation before full experiment.")
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS), help="Comma-separated dataset keys or 'all'.")
    parser.add_argument("--n-values", default=None, help="Comma-separated N values applied to all selected datasets.")
    parser.add_argument("--output-dir", default=str(FINAL2_DIR / "results"), help="Directory for Markdown/JSON/CSV outputs.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epsilon", type=float, default=EPSILON)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    print(f"\nExperiment {EXP_ID}: {EXP_NAME}", flush=True)
    print(f"  Device={device}  epsilon={args.epsilon}  seed={args.seed}", flush=True)
    print(f"  old_radius_scheme={OLD_RADIUS_SCHEME}", flush=True)

    validation = {"validation_passed": False}
    if not args.skip_validation:
        validation = validate_old_proxy(device, args.seed, args.epsilon)
    if args.validate_old_proxy:
        return

    dataset_keys = _parse_datasets(args.datasets)
    n_values_by_dataset = _parse_n_values(args.n_values, dataset_keys)
    rows = run_experiment(dataset_keys, n_values_by_dataset, device, args.seed, args.epsilon)
    print_table(rows)
    write_outputs(rows, validation, Path(args.output_dir), args.seed, args.epsilon, device)


if __name__ == "__main__":
    main()
