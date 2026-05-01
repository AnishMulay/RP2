#!/usr/bin/env python3
"""
Experiment control file — Final2
=================================
Presents an interactive checklist of all 10 experiments, runs selected ones
with live terminal progress, prints a final summary table, and writes every
result set to a timestamped Markdown file under results/.

Usage:
  python run_experiments.py                     # interactive menu
  python run_experiments.py --run 1,3,5         # run specific experiments
  python run_experiments.py --run all           # run everything
  python run_experiments.py --nyc-data /path/to/yellow_tripdata.parquet
"""

import argparse
import datetime
import math
import sys
import time
from pathlib import Path

import torch

FINAL2_DIR = Path(__file__).resolve().parent
if str(FINAL2_DIR) not in sys.path:
    sys.path.insert(0, str(FINAL2_DIR))

from shared import get_results_path, write_markdown

# ── Import all experiment modules ────────────────────────────────────────────
from experiments import (
    exp01_mnist_proxy_equal   as exp01,
    exp02_mnist_proxy_biased  as exp02,
    exp03_emnist_proxy_equal  as exp03,
    exp04_emnist_proxy_biased as exp04,
    exp05_nyc_scalability     as exp05,
    exp06_cifar_sift_proxy    as exp06,
    exp07_cifar_sift_scalability as exp07,
    exp08_newsgroups_proxy    as exp08,
    exp09_newsgroups_scalability as exp09,
    exp10_mnist_proxy_dissimilar as exp10,
)

ALL_MODULES = [exp01, exp02, exp03, exp04, exp05, exp06, exp07, exp08, exp09, exp10]

RESULTS_DIR = FINAL2_DIR / "results"

# ── Terminal helpers ──────────────────────────────────────────────────────────

WIDTH = 70

def _bar(char="─"):
    return char * WIDTH


def _header(text, char="═"):
    pad = max(0, WIDTH - len(text) - 4)
    left = pad // 2
    right = pad - left
    return f"╔{'═' * (WIDTH - 2)}╗\n║ {' ' * left}{text}{' ' * right} ║\n╚{'═' * (WIDTH - 2)}╝"


def _section(text):
    return f"\n{_bar()}\n  {text}\n{_bar()}"


def _check(done):
    return "✓" if done else " "


def print_banner(device):
    print(_header("OT Experiment Runner — Final2"))
    if device.type == "cuda":
        name = torch.cuda.get_device_name(0)
        mem  = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  Device : cuda — {name}  ({mem:.1f} GiB)")
    else:
        print(f"  Device : cpu")
    print(f"  Results: {RESULTS_DIR}")
    print()


def print_menu(completed):
    print("  Available experiments:")
    for mod in ALL_MODULES:
        mark = _check(completed.get(mod.EXP_ID, False))
        print(f"    [{mark}] {mod.EXP_ID:>2}.  {mod.EXP_NAME}")
    print()


def ask_selection(completed):
    while True:
        raw = input("  Enter experiment numbers to run  (e.g. 1,2,3  or  all): ").strip()
        if raw.lower() == "all":
            return list(range(1, 11))
        try:
            ids = [int(x.strip()) for x in raw.split(",") if x.strip()]
            valid = [i for i in ids if 1 <= i <= 10]
            if not valid:
                raise ValueError
            return valid
        except ValueError:
            print("  Invalid input. Please enter comma-separated numbers 1–10 or 'all'.")


# ── Table printer for final summary ──────────────────────────────────────────

def _fmt_row(mod, row):
    return [fn(row) for fn in mod.FMT_FNS.values()]


def _print_table(col_specs, fmt_fns_by_header, rows):
    headers = [h for h, _ in col_specs]
    widths  = [max(len(h), w) for h, w in col_specs]
    fmt_fns = [fmt_fns_by_header[h] for h in headers]

    top = "┌" + "┬".join("─" * (w + 2) for w in widths) + "┐"
    hdr = "│" + "│".join(f" {h:^{w}} " for h, w in zip(headers, widths)) + "│"
    mid = "├" + "┼".join("─" * (w + 2) for w in widths) + "┤"
    bot = "└" + "┴".join("─" * (w + 2) for w in widths) + "┘"

    print(top)
    print(hdr)
    print(mid)
    for row in rows:
        cells = [fn(row) for fn in fmt_fns]
        line  = "│" + "│".join(f" {c:>{w}} " for c, w in zip(cells, widths)) + "│"
        print(line)
    print(bot)


def print_results_table(mod, rows):
    if not rows:
        print("  (no results)")
        return
    _print_table(mod.COL_SPECS, mod.FMT_FNS, rows)
    if hasattr(mod, "DIAG_COL_SPECS") and hasattr(mod, "DIAG_FMT_FNS"):
        print()
        _print_table(mod.DIAG_COL_SPECS, mod.DIAG_FMT_FNS, rows)


# ── Markdown section builder ──────────────────────────────────────────────────

def _md_sections(mod, rows):
    sections = [{
        "title":     f"Experiment {mod.EXP_ID}: {mod.EXP_NAME}",
        "subtitle":  f"Dataset: {mod.DATASET}",
        "col_specs": mod.COL_SPECS,
        "rows":      rows,
        "fmt_fns":   mod.FMT_FNS,
    }]
    if hasattr(mod, "DIAG_COL_SPECS") and hasattr(mod, "DIAG_FMT_FNS"):
        sections.append({
            "title":     f"Experiment {mod.EXP_ID}: {mod.EXP_NAME} Diagnostics",
            "subtitle":  f"Dataset: {mod.DATASET}",
            "col_specs": mod.DIAG_COL_SPECS,
            "rows":      rows,
            "fmt_fns":   mod.DIAG_FMT_FNS,
        })
    return sections


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run Final2 OT experiments and save results to Markdown."
    )
    parser.add_argument("--run",      default=None,
                        help="Comma-separated experiment IDs or 'all'.")
    parser.add_argument("--nyc-data", default=None,
                        help="Path to NYC taxi parquet file (for Exp 5).")
    parser.add_argument("--nyc-day",  default=None,
                        help="Date filter for NYC taxi data (YYYY-MM-DD).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print_banner(device)
    completed = {}

    # ── Select experiments ────────────────────────────────────────────────────
    if args.run is not None:
        if args.run.lower() == "all":
            selected_ids = list(range(1, 11))
        else:
            selected_ids = [int(x.strip()) for x in args.run.split(",") if x.strip()]
    else:
        print_menu(completed)
        selected_ids = ask_selection(completed)

    selected_mods = [m for m in ALL_MODULES if m.EXP_ID in selected_ids]
    print(f"\n  Will run {len(selected_mods)} experiment(s): "
          f"{', '.join(str(m.EXP_ID) for m in selected_mods)}\n")

    # ── Run each experiment ───────────────────────────────────────────────────
    all_results = {}
    md_sections = []
    overall_start = time.perf_counter()

    for idx, mod in enumerate(selected_mods, 1):
        print(_section(
            f"[{idx}/{len(selected_mods)}]  Exp {mod.EXP_ID}: {mod.EXP_NAME}"
        ), flush=True)

        kwargs = {}
        if mod.EXP_ID == 5:
            if args.nyc_data:
                kwargs["nyc_data_path"] = args.nyc_data
            if args.nyc_day:
                kwargs["nyc_day"] = args.nyc_day

        t0 = time.perf_counter()
        try:
            rows = mod.run(device, **kwargs)
        except KeyboardInterrupt:
            print("\n  Interrupted by user. Saving results collected so far.", flush=True)
            rows = []
        elapsed_s = time.perf_counter() - t0

        completed[mod.EXP_ID] = True
        all_results[mod.EXP_ID] = rows
        md_sections.extend(_md_sections(mod, rows))

        print(f"\n  ── Results: Exp {mod.EXP_ID} ({elapsed_s:.1f}s total) ──", flush=True)
        print_results_table(mod, rows)

    # ── Grand total timing ────────────────────────────────────────────────────
    total_s = time.perf_counter() - overall_start
    print(f"\n{_bar()}", flush=True)
    print(f"  All selected experiments completed in {total_s:.1f}s", flush=True)
    print(_bar(), flush=True)

    # ── Final consolidated summary ────────────────────────────────────────────
    if len(selected_mods) > 1:
        print("\n\n" + _header("FINAL RESULTS SUMMARY"))
        for mod in selected_mods:
            rows = all_results.get(mod.EXP_ID, [])
            if rows:
                print(f"\n  Exp {mod.EXP_ID}: {mod.EXP_NAME}")
                print_results_table(mod, rows)

    # ── Write Markdown ────────────────────────────────────────────────────────
    if md_sections:
        md_path = get_results_path(RESULTS_DIR)
        write_markdown(md_path, md_sections)
    else:
        print("\n  No results to write.", flush=True)


if __name__ == "__main__":
    main()
