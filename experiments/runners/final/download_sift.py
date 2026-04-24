#!/usr/bin/env python3

import pathlib
import tarfile
import urllib.request


BASE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent.parent
# final/ -> runners/ -> experiments/ -> project root
DATA_DIR = BASE_DIR / "data" / "sift"
DATA_DIR.mkdir(parents=True, exist_ok=True)

URL = "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"


def main():
    print(f"DATA_DIR: {DATA_DIR}", flush=True)

    tar_path = DATA_DIR / "sift.tar.gz"
    print("Downloading SIFT1M (~163 MB)...", flush=True)
    urllib.request.urlretrieve(URL, tar_path)
    size_mb = tar_path.stat().st_size / (1024 * 1024)
    print(f"Downloaded file: {tar_path.name} ({size_mb:.2f} MB)", flush=True)

    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=DATA_DIR)

    print(f"Files under {DATA_DIR}:", flush=True)
    for path in sorted(DATA_DIR.rglob("*")):
        if path.is_file():
            print(f"  {path.relative_to(DATA_DIR)}", flush=True)

    base_path = DATA_DIR / "sift" / "sift_base.bvecs"
    if not base_path.exists():
        base_path = DATA_DIR / "sift_base.bvecs"

    with open(base_path, "rb") as f:
        d = int.from_bytes(f.read(4), byteorder="little", signed=True)
    print(f"Sanity check: first descriptor dimension = {d} (expected 128)", flush=True)
    if d != 128:
        print("WARNING: unexpected descriptor dimension in sift_base.bvecs", flush=True)

    print("Download complete.", flush=True)


if __name__ == "__main__":
    main()
