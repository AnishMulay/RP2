#!/usr/bin/env python3

import gzip
import pathlib
import pickle
import sys
import time

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

try:
    import cv2
except ImportError:
    print("ERROR: opencv-python not installed.", flush=True)
    print("Install with: pip install opencv-python-headless", flush=True)
    sys.exit(1)


BASE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "cifar_sift"
DATA_DIR.mkdir(parents=True, exist_ok=True)
CIFAR_DIR = BASE_DIR / "data"


def _to_uint8_hwc(image):
    if torch.is_tensor(image):
        image = (
            image.mul(255.0)
            .clamp(0.0, 255.0)
            .to(torch.uint8)
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )
    else:
        image = np.asarray(image, dtype=np.uint8)
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    return image


def extract_sift_descriptors(dataset, split_name):
    total = len(dataset)
    sift = cv2.SIFT_create(nfeatures=128)
    descriptors = []
    keypoint_counts = []
    zero_keypoint_images = 0

    t0 = time.perf_counter()
    for i in range(total):
        image, _label = dataset[i]
        image_uint8 = _to_uint8_hwc(image)
        gray = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2GRAY)
        _kps, descs = sift.detectAndCompute(gray, None)

        if descs is None:
            zero_keypoint_images += 1
            keypoint_counts.append(0)
            descs = np.zeros((1, 128), dtype=np.float32)
        else:
            keypoint_counts.append(int(descs.shape[0]))
            descs = descs.astype(np.float32) / 255.0

        descriptors.append(descs)

        if (i + 1) % 5000 == 0 or (i + 1) == total:
            print(f"  Processed {i + 1}/{total} images...", flush=True)

    elapsed_s = time.perf_counter() - t0
    keypoint_counts = np.asarray(keypoint_counts, dtype=np.int64)
    print(
        f"  {split_name}: images with zero keypoints = {zero_keypoint_images}",
        flush=True,
    )
    print(
        f"  {split_name}: average keypoints/image = "
        f"{keypoint_counts.mean():.2f}",
        flush=True,
    )
    print(
        f"  {split_name}: min keypoints = {keypoint_counts.min()}  "
        f"max keypoints = {keypoint_counts.max()}",
        flush=True,
    )
    print(
        f"  {split_name}: extraction time = {elapsed_s:.1f} s",
        flush=True,
    )
    return descriptors


def _file_size_mb(path):
    return path.stat().st_size / (1024 * 1024)


def main():
    print("=" * 60, flush=True)
    print("CIFAR-10 SIFT Descriptor Precomputation", flush=True)
    print("=" * 60, flush=True)

    print("\n[Step 1/3] Downloading CIFAR-10...", flush=True)
    transform = transforms.ToTensor()
    train_set = torchvision.datasets.CIFAR10(
        root=str(CIFAR_DIR),
        train=True,
        download=True,
        transform=transform,
    )
    test_set = torchvision.datasets.CIFAR10(
        root=str(CIFAR_DIR),
        train=False,
        download=True,
        transform=transform,
    )
    print(f"  Train images: {len(train_set):,}", flush=True)
    print(f"  Test images : {len(test_set):,}", flush=True)

    print("\n[Step 2/3] Extracting SIFT descriptors...", flush=True)
    train_descs = extract_sift_descriptors(train_set, "Train")
    test_descs = extract_sift_descriptors(test_set, "Test")
    train_labels = np.asarray(train_set.targets, dtype=np.int64)
    test_labels = np.asarray(test_set.targets, dtype=np.int64)

    print("\n[Step 3/3] Saving descriptor files...", flush=True)
    train_desc_path = DATA_DIR / "cifar10_sift_train.pkl.gz"
    train_label_path = DATA_DIR / "cifar10_sift_train_labels.npy"
    test_desc_path = DATA_DIR / "cifar10_sift_test.pkl.gz"
    test_label_path = DATA_DIR / "cifar10_sift_test_labels.npy"

    with gzip.open(train_desc_path, "wb") as f:
        pickle.dump(train_descs, f, protocol=4)
    with gzip.open(test_desc_path, "wb") as f:
        pickle.dump(test_descs, f, protocol=4)
    np.save(train_label_path, train_labels)
    np.save(test_label_path, test_labels)

    print(
        f"  Saved {train_desc_path.name} ({_file_size_mb(train_desc_path):.2f} MB)",
        flush=True,
    )
    print(
        f"  Saved {train_label_path.name} "
        f"({_file_size_mb(train_label_path):.2f} MB)",
        flush=True,
    )
    print(
        f"  Saved {test_desc_path.name} ({_file_size_mb(test_desc_path):.2f} MB)",
        flush=True,
    )
    print(
        f"  Saved {test_label_path.name} ({_file_size_mb(test_label_path):.2f} MB)",
        flush=True,
    )
    print(f"Done! Files saved to {DATA_DIR}", flush=True)

    with gzip.open(train_desc_path, "rb") as f:
        sanity_descs = pickle.load(f)
    print(
        f"  Sanity check descriptor shape for image 0: "
        f"{sanity_descs[0].shape}",
        flush=True,
    )
    print("Sanity check passed.", flush=True)


if __name__ == "__main__":
    main()
