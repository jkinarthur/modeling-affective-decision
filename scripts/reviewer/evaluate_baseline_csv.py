"""Evaluate baseline prediction CSVs with the same ADI metrics as evaluate.py.

Expected CSV columns:
  sample_id, S_aff, S_dec, dec_true
Optional columns:
  dec_pred, emotion_true

Usage:
python scripts/reviewer/evaluate_baseline_csv.py \
  --input_csv results/gpt41_predictions.csv \
  --model_name GPT-4.1 \
  --tau 0.5 \
  --artifacts_dir checkpoints/artifacts
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate baseline csv metrics")
    p.add_argument("--input_csv", required=True, type=str)
    p.add_argument("--model_name", required=True, type=str)
    p.add_argument("--tau", default=0.5, type=float)
    p.add_argument("--artifacts_dir", required=True, type=str)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_csv)

    needed = {"S_aff", "S_dec", "dec_true"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    s_aff = df["S_aff"].to_numpy(dtype=float)
    s_dec = df["S_dec"].to_numpy(dtype=float)
    adi = np.maximum(0.0, s_aff - s_dec)

    if "dec_pred" in df.columns:
        dec_pred = df["dec_pred"].to_numpy(dtype=int)
    else:
        dec_pred = (s_dec > 0.5).astype(int)
    dec_true = df["dec_true"].to_numpy(dtype=int)

    adir_val = float((adi > args.tau).mean() * 100.0)
    adis_val = float(adi.mean())
    decacc = float((dec_pred == dec_true).mean() * 100.0)

    tn = int(((dec_true == 0) & (dec_pred == 0)).sum())
    fp = int(((dec_true == 0) & (dec_pred == 1)).sum())
    fn = int(((dec_true == 1) & (dec_pred == 0)).sum())
    tp = int(((dec_true == 1) & (dec_pred == 1)).sum())

    stats = {
        "mean": float(np.mean(adi)),
        "median": float(np.median(adi)),
        "std": float(np.std(adi)),
        "min": float(np.min(adi)),
        "max": float(np.max(adi)),
        "p90": float(np.percentile(adi, 90)),
        "p95": float(np.percentile(adi, 95)),
    }

    out_dir = Path(args.artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_key = args.model_name.replace(" ", "_").replace("/", "_")

    (out_dir / f"{model_key}_metrics.json").write_text(
        json.dumps(
            {
                "ADIR (%)": adir_val,
                "ADIS": adis_val,
                "DecAcc (%)": decacc,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    pd.DataFrame([
        {
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp,
        }
    ]).to_csv(out_dir / f"{model_key}_confusion.csv", index=False)

    pd.DataFrame([stats]).to_csv(out_dir / f"{model_key}_adi_stats.csv", index=False)

    bins = np.linspace(0.0, 1.0, 11)
    counts, edges = np.histogram(adi, bins=bins)
    pd.DataFrame(
        {
            "bin_left": edges[:-1],
            "bin_right": edges[1:],
            "count": counts,
        }
    ).to_csv(out_dir / f"{model_key}_adi_hist.csv", index=False)

    print(f"Saved baseline artifacts for {args.model_name} to {out_dir}")


if __name__ == "__main__":
    main()
