#!/usr/bin/env python3
"""
Unified proxy accuracy experiment for Figure 1 and Figure 6.

This runner uses one metric throughout:

    distortion_ratio = ot.emd2(a, b, C_proxy) / ot.emd2(a, b, C_true)

where a and b are uniform length-N histograms. It does not extract a transport
plan and does not evaluate the true cost of a proxy-induced matching.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import math
import pickle
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torchvision

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
from clustered_push_relabel.clustering.simple_three_level_l1 import ThreeLevelL1Clustering
from shared import (
    build_three_level_proxy_matrix,
    build_two_level_proxy_matrix,
    run_three_level_precomputed,
)


EXP_NAME = "Unified Proxy Accuracy"
DATA_DIR = FINAL2_DIR / "data"
EMNIST_DATA_DIR = BASE_DIR / "data"
EMNIST_SPLIT = "byclass"
CIFAR_DATA_DIR = BASE_DIR / "data" / "cifar_sift"
TRAIN_DESC_PATH = CIFAR_DATA_DIR / "cifar10_sift_train.pkl.gz"
NEWSGROUPS_DATA_DIR = BASE_DIR / "data" / "newsgroups_glove"
EMBEDDINGS_PATH = NEWSGROUPS_DATA_DIR / "newsgroups_embeddings.pkl.gz"

EPSILON = 0.01
BATCH_SIZE = 512
OLD_RADIUS_SCHEME = "geometric"
OLD_PAIR_CHUNK = 128
BATCH_SIZE_OLD = 32
MAX_WORDS = 300

IMAGE_N_VALUES = [5_000, 10_000, 15_000, 20_000, 25_000]
CHAMFER_N_VALUES = [1_000, 2_000, 3_000, 5_000, 7_000, 10_000]
IMAGE_SEEDS = [42]
SINGLE_SEED = [42]

BLUE_DIGITS = list(range(5))
RED_DIGITS = list(range(5, 10))
BLUE_CLASS_END = 31
RED_CLASS_START = 31

CSV_COLUMNS = [
    "dataset",
    "sampling",
    "n",
    "seed",
    "exact_cost",
    "proxy_2L_cost",
    "ratio_2L",
    "proxy_3L_cost",
    "ratio_3L",
    "old_proxy_cost",
    "ratio_old",
    "gap_2L_pct",
    "gap_3L_pct",
    "gap_old_pct",
]


@dataclass(frozen=True)
class DatasetConfig:
    key: str
    dataset: str
    sampling: str
    n_values: list[int]
    seeds: list[int]
    kind: str
    loader: Callable[[int, int], "LoadedData"]


@dataclass(frozen=True)
class LoadedData:
    red: torch.Tensor | None = None
    blue: torch.Tensor | None = None
    red_descs: list | None = None
    blue_descs: list | None = None


@dataclass(frozen=True)
class DistanceBundle:
    D_br: torch.Tensor
    D_rr: torch.Tensor
    D_bb: torch.Tensor
    scale: float = 1.0


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _clear(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()


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
    b_blue_d: torch.Tensor,
    b_red_d: torch.Tensor,
    b_blue_m: torch.Tensor,
    b_red_m: torch.Tensor,
    epsilon: float,
    start: int,
    end: int,
    p_total: int,
) -> torch.Tensor:
    prev_start = max(0, start - BATCH_SIZE_OLD)
    if start == 0 or start // 200 != prev_start // 200:
        print(f"  [old proxy] processing centers {start}-{end-1}/{p_total}", flush=True)
    radii = torch.maximum(
        b_blue_d.unsqueeze(2),
        b_red_d.unsqueeze(1),
    )
    candidates = 2.0 * _geometric_ceil(radii, epsilon)
    active = b_blue_m.unsqueeze(2) & b_red_m.unsqueeze(1)
    candidates[~active] = float("inf")
    batch_min = candidates.min(dim=0).values
    C_old = torch.minimum(C_old, batch_min)
    del radii, candidates, active, batch_min
    return C_old


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

    store_on_cpu = p_total * n * 4 > 2 * 1024**3
    storage_device = torch.device("cpu") if store_on_cpu else device
    update_device = torch.device("cuda") if store_on_cpu and torch.cuda.is_available() else storage_device

    all_red_d = torch.cat([D_rr.to(storage_device), D_br.to(storage_device)], dim=0)
    all_blue_d = torch.cat([D_br.T.to(storage_device), D_bb.to(storage_device)], dim=0)
    if all_red_d.shape != (p_total, n) or all_blue_d.shape != (p_total, n):
        raise RuntimeError(
            f"old proxy distance stack shape mismatch: "
            f"all_red_d={tuple(all_red_d.shape)}, all_blue_d={tuple(all_blue_d.shape)}"
        )

    sampled_mask_store = sampled_mask.to(storage_device)
    d1_red_store = d1_red.to(storage_device)
    d1_blue_store = d1_blue.to(storage_device)
    red_mask_all = all_red_d < d1_red_store.unsqueeze(0)
    blue_mask_all = all_blue_d < d1_blue_store.unsqueeze(0)
    red_mask_all[sampled_mask_store] = True
    blue_mask_all[sampled_mask_store] = True

    C_old = torch.full((n, n), float("inf"), dtype=D_br.dtype, device=update_device)

    # Each batch allocates (batch, n, n) float32.
    # Target ~256MB per batch to stay safe on 8GB GPU.
    target_bytes = 256 * 1024 * 1024
    dyn_batch = max(1, int(target_bytes / (n * n * 4)))
    dyn_batch = min(dyn_batch, BATCH_SIZE_OLD)
    print(
        f"  [old proxy] n={n}, dynamic batch size={dyn_batch} "
        f"({dyn_batch*n*n*4/1e6:.0f}MB per batch, "
        f"{math.ceil(p_total/dyn_batch)} batches)",
        flush=True,
    )

    for start in range(0, p_total, dyn_batch):
        end = min(start + dyn_batch, p_total)
        b_blue_d = all_blue_d[start:end].to(update_device)
        b_red_d = all_red_d[start:end].to(update_device)
        b_blue_m = blue_mask_all[start:end].to(update_device)
        b_red_m = red_mask_all[start:end].to(update_device)
        C_old = _update_old_proxy_from_center(
            C_old, b_blue_d, b_red_d, b_blue_m, b_red_m, epsilon, start, end, p_total
        )
        del b_blue_d, b_red_d, b_blue_m, b_red_m

    meta = {
        "old_radius_scheme": OLD_RADIUS_SCHEME,
        "sampled_count": int(sampled_mask.sum().item()),
        "sampled_red_count": int(sampled_mask[:n].sum().item()),
        "sampled_blue_count": int(sampled_mask[n:].sum().item()),
    }
    return C_old, meta


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


def _normalize_histograms(red: np.ndarray, blue: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    for arr in (red, blue):
        s = arr.sum(axis=1, keepdims=True)
        np.maximum(s, 1e-8, out=s)
        arr /= s
    return torch.from_numpy(red.astype(np.float32)), torch.from_numpy(blue.astype(np.float32))


def load_mnist_equal(n_samples: int, seed: int) -> LoadedData:
    train = torchvision.datasets.MNIST(root=str(DATA_DIR), train=True, download=False)
    test = torchvision.datasets.MNIST(root=str(DATA_DIR), train=False, download=False)
    images = torch.cat([train.data, test.data], dim=0).numpy().reshape(-1, 784)
    labels = torch.cat([train.targets, test.targets], dim=0).numpy()

    classes = np.unique(labels)
    spc = n_samples // len(classes)
    if spc == 0:
        raise ValueError(f"n_samples={n_samples} too small for {len(classes)} classes")

    rng_r = np.random.RandomState(seed)
    rng_b = np.random.RandomState(seed + 1)
    red_parts, blue_parts = [], []
    for cls in classes:
        idx = np.flatnonzero(labels == cls).copy()
        needed = 2 * spc
        if idx.size < needed:
            warnings.warn(f"Class {cls}: only {idx.size} samples, need {needed}. Skipping.")
            continue
        rng_r.shuffle(idx)
        chosen = idx[:needed]
        red_parts.append(images[chosen[:spc]])
        blue_parts.append(images[chosen[spc:needed]])

    red = np.concatenate(red_parts).astype(np.float32) / 255.0
    blue = np.concatenate(blue_parts).astype(np.float32) / 255.0
    red_t, blue_t = _normalize_histograms(red, blue)
    return LoadedData(red=red_t, blue=blue_t)


def load_mnist_biased(n_samples: int, seed: int) -> LoadedData:
    train = torchvision.datasets.MNIST(root=str(DATA_DIR), train=True, download=False)
    test = torchvision.datasets.MNIST(root=str(DATA_DIR), train=False, download=False)
    images = torch.cat([train.data, test.data], dim=0).numpy().reshape(-1, 784)
    labels = torch.cat([train.targets, test.targets], dim=0).numpy()

    rng_r = np.random.RandomState(seed)
    rng_b = np.random.RandomState(seed + 1)
    red = _sample_from_digits(images, labels, RED_DIGITS, n_samples, rng_r).astype(np.float32) / 255.0
    blue = _sample_from_digits(images, labels, BLUE_DIGITS, n_samples, rng_b).astype(np.float32) / 255.0
    red_t, blue_t = _normalize_histograms(red, blue)
    return LoadedData(red=red_t, blue=blue_t)


def load_emnist_equal(n_samples: int, seed: int) -> LoadedData:
    train = torchvision.datasets.EMNIST(
        root=str(EMNIST_DATA_DIR), split=EMNIST_SPLIT, train=True, download=False
    )
    test = torchvision.datasets.EMNIST(
        root=str(EMNIST_DATA_DIR), split=EMNIST_SPLIT, train=False, download=False
    )
    images = torch.cat([train.data, test.data], dim=0).numpy()
    labels = torch.cat([train.targets, test.targets], dim=0).numpy()
    images = images.reshape(-1, 28, 28).transpose(0, 2, 1).reshape(-1, 784)

    classes = np.unique(labels)
    spc = n_samples // len(classes)
    if spc == 0:
        raise ValueError(f"n_samples={n_samples} too small for {len(classes)} EMNIST classes")

    rng_r = np.random.RandomState(seed)
    rng_b = np.random.RandomState(seed + 1)
    red_parts, blue_parts = [], []
    for cls in classes:
        idx = np.flatnonzero(labels == cls).copy()
        needed = 2 * spc
        if idx.size < needed:
            warnings.warn(f"Class {cls}: {idx.size} available, need {needed}. Skipping.")
            continue
        rng_r.shuffle(idx)
        chosen = idx[:needed]
        red_parts.append(images[chosen[:spc]])
        blue_parts.append(images[chosen[spc:needed]])

    red = np.concatenate(red_parts).astype(np.float32) / 255.0
    blue = np.concatenate(blue_parts).astype(np.float32) / 255.0
    red_t, blue_t = _normalize_histograms(red, blue)
    return LoadedData(red=red_t, blue=blue_t)


def load_emnist_biased(n_samples: int, seed: int) -> LoadedData:
    train = torchvision.datasets.EMNIST(
        root=str(EMNIST_DATA_DIR), split=EMNIST_SPLIT, train=True, download=False
    )
    test = torchvision.datasets.EMNIST(
        root=str(EMNIST_DATA_DIR), split=EMNIST_SPLIT, train=False, download=False
    )
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
    return LoadedData(red=red_t, blue=blue_t)


def _load_pickle_gz(path: Path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def _sample_pair(items, n: int, seed: int):
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(items))
    return [items[i] for i in perm[:n]], [items[i] for i in perm[n : 2 * n]]


def load_cifar_sift(n: int, seed: int) -> LoadedData:
    if not TRAIN_DESC_PATH.exists():
        raise FileNotFoundError(f"{TRAIN_DESC_PATH} not found. Run download_cifar_sift.py first.")
    descs = _load_pickle_gz(TRAIN_DESC_PATH)
    if 2 * n > len(descs):
        raise ValueError(f"not enough CIFAR SIFT descriptor sets: need {2*n:,}, have {len(descs):,}")
    red_descs, blue_descs = _sample_pair(descs, n, seed)
    return LoadedData(red_descs=red_descs, blue_descs=blue_descs)


def load_newsgroups(n: int, seed: int) -> LoadedData:
    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(f"{EMBEDDINGS_PATH} not found. Run download_newsgroups_glove.py first.")
    embs = _load_pickle_gz(EMBEDDINGS_PATH)
    if 2 * n > len(embs):
        raise ValueError(f"not enough Newsgroups documents: need {2*n:,}, have {len(embs):,}")
    red_descs, blue_descs = _sample_pair(embs, n, seed)
    red_descs = [d[:MAX_WORDS] if len(d) > MAX_WORDS else d for d in red_descs]
    blue_descs = [d[:MAX_WORDS] if len(d) > MAX_WORDS else d for d in blue_descs]
    return LoadedData(red_descs=red_descs, blue_descs=blue_descs)


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


def chamfer_matrix(
    descs_B: list,
    descs_A: list,
    device: torch.device,
    tile_b: int = 50,
    tile_a: int = 100,
) -> torch.Tensor:
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


def build_l1_bundle(red: torch.Tensor, blue: torch.Tensor, n: int, device: torch.device) -> DistanceBundle:
    if n > 15_000:
        print(
            f"Building dense {n}x{n} cost matrix, estimated {n*n*8/1e9:.2f} GB on CPU",
            flush=True,
        )
        red64 = red.cpu().to(torch.float64)
        blue64 = blue.cpu().to(torch.float64)
    else:
        red64 = red.to(device).to(torch.float64)
        blue64 = blue.to(device).to(torch.float64)

    D_br = torch.cdist(blue64, red64, p=1).to(torch.float32)
    D_rr = torch.cdist(red64, red64, p=1).to(torch.float32)
    D_bb = torch.cdist(blue64, blue64, p=1).to(torch.float32)
    return DistanceBundle(D_br=D_br, D_rr=D_rr, D_bb=D_bb)


def build_chamfer_bundle(data: LoadedData, device: torch.device) -> DistanceBundle:
    assert data.red_descs is not None and data.blue_descs is not None
    D_br = chamfer_matrix(data.blue_descs, data.red_descs, device)
    D_rr = chamfer_symmetric(data.red_descs, device)
    D_bb = chamfer_symmetric(data.blue_descs, device)
    scale = max(float(D_br.max().item()), float(D_rr.max().item()), float(D_bb.max().item()), 1e-8)
    return DistanceBundle(
        D_br=D_br / scale,
        D_rr=D_rr / scale,
        D_bb=D_bb / scale,
        scale=scale,
    )


def to_numpy64(C) -> np.ndarray:
    if isinstance(C, np.ndarray):
        return C.astype(np.float64, copy=False)
    return C.detach().cpu().to(torch.float64).numpy()


def emd2_cost(C) -> float:
    if ot is None:
        raise RuntimeError("POT is not installed")
    C_np = to_numpy64(C)
    n = C_np.shape[0]
    hist = np.full(n, 1.0 / n, dtype=np.float64)
    return float(ot.emd2(hist, hist, C_np, numItermax=10**6))


def ratio(proxy_cost: float, exact_cost: float) -> float:
    if math.isnan(proxy_cost) or math.isnan(exact_cost) or exact_cost == 0.0:
        return math.nan
    return proxy_cost / exact_cost


def gap_pct(proxy_cost: float, exact_cost: float) -> float:
    if math.isnan(proxy_cost) or math.isnan(exact_cost) or exact_cost == 0.0:
        return math.nan
    return (proxy_cost - exact_cost) / exact_cost * 100.0


def fmt_float(v: float) -> str:
    return "NaN" if math.isnan(float(v)) else f"{float(v):.6f}"


def build_proxy_2l(config: DatasetConfig, data: LoadedData, bundle: DistanceBundle, device: torch.device):
    n = bundle.D_br.shape[0]
    if config.kind == "l1":
        assert data.red is not None and data.blue is not None
        engine = SimpleL1Clustering(epsilon=EPSILON, tile_size=BATCH_SIZE)
        clustering = engine.run(data.red.to(device), data.blue.to(device))
        return build_two_level_proxy_matrix(clustering, n, device)
    clustering = SimplePrecomputedClustering(epsilon=EPSILON, tile_size=BATCH_SIZE).run(
        bundle.D_rr, bundle.D_br
    )
    return build_two_level_proxy_matrix(clustering, n, bundle.D_br.device)


def build_proxy_3l(config: DatasetConfig, data: LoadedData, bundle: DistanceBundle, device: torch.device):
    n = bundle.D_br.shape[0]
    if config.kind == "l1":
        assert data.red is not None and data.blue is not None
        engine = ThreeLevelL1Clustering(epsilon=EPSILON, tile_size=BATCH_SIZE)
        clustering = engine.run(data.red.to(device), data.blue.to(device))
        return build_three_level_proxy_matrix(clustering, n, device)
    clustering = run_three_level_precomputed(bundle.D_rr, bundle.D_br, EPSILON, BATCH_SIZE)
    return build_three_level_proxy_matrix(clustering, n, bundle.D_br.device)


def run_one_seed(config: DatasetConfig, n: int, seed: int, device: torch.device) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    row = {
        "dataset": config.dataset,
        "sampling": config.sampling,
        "n": n,
        "seed": seed,
        "exact_cost": math.nan,
        "proxy_2L_cost": math.nan,
        "ratio_2L": math.nan,
        "proxy_3L_cost": math.nan,
        "ratio_3L": math.nan,
        "old_proxy_cost": math.nan,
        "ratio_old": math.nan,
        "gap_2L_pct": math.nan,
        "gap_3L_pct": math.nan,
        "gap_old_pct": math.nan,
    }

    try:
        data = config.loader(n, seed)
        if config.kind == "l1":
            assert data.red is not None and data.blue is not None
            bundle = build_l1_bundle(data.red, data.blue, n, device)
        else:
            bundle = build_chamfer_bundle(data, device)

        try:
            exact_cost = emd2_cost(bundle.D_br)
        except (MemoryError, RuntimeError, Exception) as exc:
            print(
                f"WARNING: exact ot.emd2 failed for {config.dataset} {config.sampling} "
                f"n={n} seed={seed}: {exc}",
                flush=True,
            )
            return row

        row["exact_cost"] = exact_cost

        try:
            proxy_2l = build_proxy_2l(config, data, bundle, device)
            proxy_2l_cost = emd2_cost(proxy_2l)
            row["proxy_2L_cost"] = proxy_2l_cost
            row["ratio_2L"] = proxy_2l_cost / exact_cost
            row["gap_2L_pct"] = (proxy_2l_cost - exact_cost) / exact_cost * 100.0
        except (MemoryError, RuntimeError, Exception) as exc:
            print(
                f"WARNING: 2L proxy failed for {config.dataset} {config.sampling} "
                f"n={n} seed={seed}: {exc}",
                flush=True,
            )

        try:
            proxy_3l = build_proxy_3l(config, data, bundle, device)
            proxy_3l_cost = emd2_cost(proxy_3l)
            row["proxy_3L_cost"] = proxy_3l_cost
            row["ratio_3L"] = proxy_3l_cost / exact_cost
            row["gap_3L_pct"] = (proxy_3l_cost - exact_cost) / exact_cost * 100.0
        except (MemoryError, RuntimeError, Exception) as exc:
            print(
                f"WARNING: 3L proxy failed for {config.dataset} {config.sampling} "
                f"n={n} seed={seed}: {exc}",
                flush=True,
            )

        try:
            C_old, _old_meta = build_old_paper_proxy_matrix(
                bundle.D_br, bundle.D_rr, bundle.D_bb, EPSILON, seed, OLD_PAIR_CHUNK
            )
            old_proxy_cost = emd2_cost(C_old)
            row["old_proxy_cost"] = old_proxy_cost
            row["ratio_old"] = old_proxy_cost / exact_cost
            row["gap_old_pct"] = (old_proxy_cost - exact_cost) / exact_cost * 100.0
        except (MemoryError, RuntimeError, Exception) as exc:
            print(
                f"WARNING: old proxy failed for {config.dataset} {config.sampling} "
                f"n={n} seed={seed}: {exc}",
                flush=True,
            )
    except (MemoryError, RuntimeError, Exception) as exc:
        print(
            f"WARNING: failed {config.dataset} {config.sampling} n={n} seed={seed}: {exc}",
            flush=True,
        )
    finally:
        _clear(device)

    return row


def mean_of(rows: list[dict], key: str) -> float:
    vals = [float(r[key]) for r in rows if not math.isnan(float(r[key]))]
    return math.nan if not vals else float(np.mean(vals))


def print_incremental_summary(config: DatasetConfig, n: int, rows: list[dict]) -> None:
    print()
    print("dataset | n | mean_ratio_2L | mean_ratio_3L | mean_gap_2L_pct | mean_gap_old_pct")
    print("--- | ---: | ---: | ---: | ---: | ---:")
    print(
        f"{config.dataset} {config.sampling} | {n:,} | "
        f"{fmt_float(mean_of(rows, 'ratio_2L'))} | "
        f"{fmt_float(mean_of(rows, 'ratio_3L'))} | "
        f"{fmt_float(mean_of(rows, 'gap_2L_pct'))} | "
        f"{fmt_float(mean_of(rows, 'gap_old_pct'))}",
        flush=True,
    )
    print()


def dataset_configs() -> list[DatasetConfig]:
    return [
        DatasetConfig("mnist_equal", "MNIST", "Equal", IMAGE_N_VALUES, IMAGE_SEEDS, "l1", load_mnist_equal),
        DatasetConfig("mnist_biased", "MNIST", "Biased", IMAGE_N_VALUES, IMAGE_SEEDS, "l1", load_mnist_biased),
        DatasetConfig("emnist_equal", "EMNIST", "Equal", IMAGE_N_VALUES, IMAGE_SEEDS, "l1", load_emnist_equal),
        DatasetConfig("emnist_biased", "EMNIST", "Biased", IMAGE_N_VALUES, IMAGE_SEEDS, "l1", load_emnist_biased),
        DatasetConfig("newsgroups", "20 Newsgroups", "Default", CHAMFER_N_VALUES, SINGLE_SEED, "chamfer", load_newsgroups),
        DatasetConfig("cifar_sift", "CIFAR-10 SIFT", "Default", CHAMFER_N_VALUES, SINGLE_SEED, "chamfer", load_cifar_sift),
    ]


def parse_dataset_keys(raw: str) -> list[str]:
    configs = dataset_configs()
    valid = {c.key for c in configs}
    if raw.strip().lower() == "all":
        return [c.key for c in configs]
    keys = [x.strip().lower() for x in raw.split(",") if x.strip()]
    unknown = [k for k in keys if k not in valid]
    if unknown:
        raise ValueError(f"unknown dataset key(s): {', '.join(unknown)}. Supported: all,{','.join(sorted(valid))}")
    return keys


def make_output_path(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return output_dir / f"exp_unified_proxy_accuracy_{stamp}.csv"


def run(device: torch.device, output_dir: Path, dataset_keys: list[str]) -> Path:
    if ot is None:
        raise RuntimeError("POT is not installed")

    configs = [c for c in dataset_configs() if c.key in set(dataset_keys)]
    csv_path = make_output_path(output_dir)
    print(f"Experiment: {EXP_NAME}", flush=True)
    print(f"Device: {device}  epsilon={EPSILON}  tile_size={BATCH_SIZE}", flush=True)
    print(f"Writing CSV incrementally to {csv_path}", flush=True)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        f.flush()

        for config in configs:
            for n in config.n_values:
                completed_rows = []
                for seed in config.seeds:
                    print(
                        f"Running dataset={config.dataset} sampling={config.sampling} "
                        f"n={n:,} seed={seed}",
                        flush=True,
                    )
                    row = run_one_seed(config, n, seed, device)
                    writer.writerow(row)
                    f.flush()
                    completed_rows.append(row)
                    print(
                        f"Seed {seed}: exact={fmt_float(row['exact_cost'])} "
                        f"ratio_2L={fmt_float(row['ratio_2L'])} "
                        f"ratio_3L={fmt_float(row['ratio_3L'])} "
                        f"ratio_old={fmt_float(row['ratio_old'])}",
                        flush=True,
                    )
                print_incremental_summary(config, n, completed_rows)

    print(f"CSV written to {csv_path}", flush=True)
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description=EXP_NAME)
    parser.add_argument(
        "--datasets",
        default="all",
        help="Comma-separated dataset keys or 'all'.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(FINAL2_DIR / "results"),
        help="Directory for the timestamped CSV.",
    )
    args = parser.parse_args()

    try:
        dataset_keys = parse_dataset_keys(args.datasets)
    except ValueError as exc:
        parser.error(str(exc))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run(device, Path(args.output_dir), dataset_keys)


if __name__ == "__main__":
    main()
