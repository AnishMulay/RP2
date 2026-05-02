#!/usr/bin/env python3
"""
Aggregate Final2 per-experiment JSON files into one Markdown report.

Usage:
  python experiments/runners/final2/aggregate_results.py --results-dir /path/to/run_dir
"""

import argparse
import json
import sys
from pathlib import Path


FINAL2_DIR = Path(__file__).resolve().parent
if str(FINAL2_DIR) not in sys.path:
    sys.path.insert(0, str(FINAL2_DIR))

from shared import write_markdown

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
    exp11_landmark_density as exp11,
)

ALL_MODULES = [exp01, exp02, exp03, exp04, exp05, exp06, exp07, exp08, exp09, exp10, exp11]
MODULES_BY_EXP_ID = {mod.EXP_ID: mod for mod in ALL_MODULES}


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


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate Final2 experiment JSON files into Markdown."
    )
    parser.add_argument("--results-dir", required=True,
                        help="Absolute path to the timestamped run directory.")
    args = parser.parse_args()

    results_dir = Path(args.results_dir).expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    present_paths = sorted(results_dir.glob("exp*.json"))
    present_names = {path.name for path in present_paths}
    for exp_id in range(1, 12):
        expected_name = f"exp{exp_id:02d}.json"
        if expected_name not in present_names:
            print(f"Warning: missing {results_dir / expected_name}; skipping.", flush=True)

    md_sections = []
    for json_path in present_paths:
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            exp_id = int(payload["exp_id"])
            rows = payload.get("rows", [])
        except Exception as exc:
            print(f"Warning: could not read {json_path}: {exc}; skipping.", flush=True)
            continue

        mod = MODULES_BY_EXP_ID.get(exp_id)
        if mod is None:
            print(f"Warning: no module found for exp_id={exp_id}; skipping {json_path}.", flush=True)
            continue
        md_sections.extend(_md_sections(mod, rows))

    output_path = results_dir / "final_results.md"
    try:
        write_markdown(output_path, md_sections)
        print(f"Final markdown written to: {output_path.resolve()}", flush=True)
    except Exception as exc:
        print(f"Warning: could not write final markdown: {exc}", flush=True)


if __name__ == "__main__":
    main()
