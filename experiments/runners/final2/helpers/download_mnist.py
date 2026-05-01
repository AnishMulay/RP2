#!/usr/bin/env python3

import pathlib

import torchvision


DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print(f"DATA_DIR: {DATA_DIR}", flush=True)

    print("Downloading MNIST train ...", flush=True)
    torchvision.datasets.MNIST(
        root=str(DATA_DIR),
        train=True,
        download=True,
    )
    print("  Done: train", flush=True)

    print("Downloading MNIST test ...", flush=True)
    torchvision.datasets.MNIST(
        root=str(DATA_DIR),
        train=False,
        download=True,
    )
    print("  Done: test", flush=True)

    raw_dir = DATA_DIR / "MNIST" / "raw"
    print(f"Files in {raw_dir}:", flush=True)
    for path in sorted(raw_dir.iterdir()):
        print(f"  {path.name}", flush=True)

    print("Download complete.", flush=True)


if __name__ == "__main__":
    main()
