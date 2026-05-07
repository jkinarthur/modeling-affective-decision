"""Aggregate evaluation artifacts into paper-ready tables.

Usage:
  python scripts/reviewer/aggregate_results.py \
    --input_dir checkpoints/artifacts \
    --output_dir checkpoints/artifacts/paper_ready
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate reviewer diagnostics")
    p.add_argument("--input_dir", required=True, type=str)
    p.add_argument("--output_dir", required=True, type=str)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_rows = []
    confusion_rows = []
    adi_rows = []
    distress_rows = []

    for metrics_file in sorted(in_dir.glob("*_metrics.json")):
        model_name = metrics_file.stem.replace("_metrics", "")
        metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
        metrics_rows.append({"model": model_name, **metrics})

        c_path = in_dir / f"{model_name}_confusion.csv"
        if c_path.exists():
            c_df = pd.read_csv(c_path)
            c_df.insert(0, "model", model_name)
            confusion_rows.append(c_df)

        a_path = in_dir / f"{model_name}_adi_stats.csv"
        if a_path.exists():
            a_df = pd.read_csv(a_path)
            a_df.insert(0, "model", model_name)
            adi_rows.append(a_df)

        d_path = in_dir / f"{model_name}_distress_prior.json"
        if d_path.exists():
            d = json.loads(d_path.read_text(encoding="utf-8"))
            distress_rows.append({"model": model_name, **d})

    if metrics_rows:
        pd.DataFrame(metrics_rows).to_csv(out_dir / "table_main_metrics.csv", index=False)
    if confusion_rows:
        pd.concat(confusion_rows, ignore_index=True).to_csv(
            out_dir / "table_confusion_matrices.csv", index=False
        )
    if adi_rows:
        pd.concat(adi_rows, ignore_index=True).to_csv(
            out_dir / "table_adi_distribution_stats.csv", index=False
        )
    if distress_rows:
        pd.DataFrame(distress_rows).to_csv(
            out_dir / "table_distress_prior.csv", index=False
        )

    print(f"Saved aggregated tables to: {out_dir}")


if __name__ == "__main__":
    main()
