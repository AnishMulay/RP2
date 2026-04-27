#!/usr/bin/env python3

import pathlib
import sys
import urllib.request
import zipfile

import numpy as np


BASE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent.parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DATA_DIR = BASE_DIR / "data" / "glove"
GLOVE_URL = "https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip"
ZIP_PATH = DATA_DIR / "glove.6B.zip"
TXT_PATH = DATA_DIR / "glove.6B.300d.txt"
NPY_PATH = DATA_DIR / "glove.6B.300d.npy"

EXPECTED_ROWS = 400_000
EXPECTED_DIM = 300
TARGET_WORD = "king"


def make_progress_callback():
    state = {"last_printed": 0}

    def report(block_num, block_size, total_size):
        if total_size <= 0:
            return
        downloaded = min(block_num * block_size, total_size)
        percent = min(int(downloaded * 100 / total_size), 100)
        bucket = (percent // 5) * 5
        while state["last_printed"] + 5 <= bucket:
            state["last_printed"] += 5
            print(
                f"  Download progress: {state['last_printed']}%",
                flush=True,
            )

    return report


def parse_glove_text_to_npy(txt_path, npy_path):
    print(f"Parsing {txt_path.name} into a binary cache...", flush=True)
    matrix = np.empty((EXPECTED_ROWS, EXPECTED_DIM), dtype=np.float32)
    king_index = None
    rows_read = 0

    with txt_path.open("r", encoding="utf-8") as f:
        for row_idx, line in enumerate(f):
            if row_idx >= EXPECTED_ROWS:
                raise ValueError(
                    f"Found more than {EXPECTED_ROWS:,} rows in {txt_path.name}."
                )

            word, sep, values = line.partition(" ")
            if not sep:
                raise ValueError(f"Malformed GloVe row at line {row_idx + 1:,}.")

            vector = np.fromstring(values, sep=" ", dtype=np.float32)
            if vector.size != EXPECTED_DIM:
                raise ValueError(
                    f"Expected {EXPECTED_DIM} values at line {row_idx + 1:,}, "
                    f"got {vector.size}."
                )

            matrix[row_idx] = vector
            if word == TARGET_WORD:
                king_index = row_idx
            rows_read += 1

    if rows_read != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS:,} rows, found {rows_read:,} rows."
        )
    if king_index is None:
        raise ValueError(f"Word '{TARGET_WORD}' not found in {txt_path.name}.")

    np.save(npy_path, matrix)
    print(f"Saved binary cache: shape={matrix.shape} dtype={matrix.dtype}", flush=True)
    return king_index


def find_word_index(txt_path, target_word):
    with txt_path.open("r", encoding="utf-8") as f:
        for row_idx, line in enumerate(f):
            word, _, _ = line.partition(" ")
            if word == target_word:
                return row_idx
    raise ValueError(f"Word '{target_word}' not found in {txt_path.name}.")


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"DATA_DIR: {DATA_DIR}", flush=True)

    if ZIP_PATH.exists():
        print(f"Zip already exists: {ZIP_PATH.name}", flush=True)
    else:
        print(f"Downloading GloVe 6B archive from {GLOVE_URL}", flush=True)
        urllib.request.urlretrieve(GLOVE_URL, ZIP_PATH, reporthook=make_progress_callback())
        size_mb = ZIP_PATH.stat().st_size / (1024 * 1024)
        print(f"Downloaded file: {ZIP_PATH.name} ({size_mb:.2f} MB)", flush=True)

    if TXT_PATH.exists():
        print(f"Text file already extracted: {TXT_PATH.name}", flush=True)
    else:
        print(f"Extracting {TXT_PATH.name} from {ZIP_PATH.name}...", flush=True)
        with zipfile.ZipFile(ZIP_PATH, "r") as zf:
            members = set(zf.namelist())
            if TXT_PATH.name not in members:
                raise FileNotFoundError(
                    f"{TXT_PATH.name} not found inside {ZIP_PATH.name}."
                )
            zf.extract(TXT_PATH.name, path=DATA_DIR)
        print(f"Extracted: {TXT_PATH}", flush=True)

    king_index = None
    if NPY_PATH.exists():
        print(f"Binary cache already exists: {NPY_PATH.name}", flush=True)
    else:
        king_index = parse_glove_text_to_npy(TXT_PATH, NPY_PATH)

    if king_index is None:
        king_index = find_word_index(TXT_PATH, TARGET_WORD)

    matrix = np.load(NPY_PATH)
    print(f"Sanity check: loaded cache shape={matrix.shape}", flush=True)
    print(
        f"Sanity check: vector for '{TARGET_WORD}' (line {king_index:,}) = "
        f"{matrix[king_index]}",
        flush=True,
    )

    print("Download complete.", flush=True)


if __name__ == "__main__":
    main()
