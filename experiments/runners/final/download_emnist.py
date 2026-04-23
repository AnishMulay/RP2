#!/usr/bin/env python3

import pathlib

import torchvision


BASE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print(f"DATA_DIR: {DATA_DIR}", flush=True)

    for split in ["byclass", "letters"]:
        print(f"Downloading EMNIST split: {split} ...", flush=True)
        torchvision.datasets.EMNIST(
            root=str(DATA_DIR),
            split=split,
            train=True,
            download=True,
        )
        torchvision.datasets.EMNIST(
            root=str(DATA_DIR),
            split=split,
            train=False,
            download=True,
        )
        print(f"  Done: {split}", flush=True)

    raw_dir = DATA_DIR / "EMNIST" / "raw"
    print(f"Files in {raw_dir}:", flush=True)
    for path in sorted(raw_dir.iterdir()):
        print(f"  {path.name}", flush=True)

    print("Download complete.", flush=True)


if __name__ == "__main__":
    main()
