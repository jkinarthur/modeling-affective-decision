"""Full evaluation script for AD-DAN.

This script evaluates a checkpoint and can export reviewer-facing diagnostics:
- ADIR/ADIS/DecAcc with confidence intervals
- confusion matrices (counts + normalized percentages)
- ADI summary statistics and histogram bins
- S_aff/S_dec score distributions
- empirical estimate of P(poor decision | distress)
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, BartTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from src.models.ad_dan import ADDAN, ADDANConfig
from src.data.dataset import (
    make_synthetic_samples, build_dataloaders,
    load_tad_splits,
)
from src.utils.metrics import (
    adir, adis, bootstrap_ci, corpus_bleu4, corpus_rouge_l,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate AD-DAN")
    p.add_argument("--ckpt",          type=str, required=True,
                   help="Path to checkpoint directory (contains model.pt or actor.pt)")
    p.add_argument("--encoder_model", type=str, default="microsoft/deberta-v3-base")
    p.add_argument("--emotion_model", type=str,
                   default="j-hartmann/emotion-english-distilroberta-base")
    p.add_argument("--bart_model",    type=str, default="facebook/bart-base")
    p.add_argument("--num_emotions",  type=int, default=32)
    p.add_argument("--tau",           type=float, default=0.5)
    p.add_argument("--batch_size",    type=int, default=32)
    p.add_argument("--data_dir",      type=str, default=None,
                   help="Directory with tad_bench_{train,val,test}.csv")
    p.add_argument("--synthetic",     action="store_true")
    p.add_argument("--n_synthetic",   type=int, default=300)
    p.add_argument("--output_csv",    type=str, default=None)
    p.add_argument("--artifacts_dir", type=str, default=None,
                   help="Directory for reviewer diagnostics (json/csv artifacts)")
    p.add_argument("--model_name",    type=str, default="model",
                   help="Model tag used in artifact filenames")
    p.add_argument("--hist_bins",     type=int, default=10,
                   help="Number of bins for histogram exports")
    p.add_argument("--generate",      action="store_true",
                   help="Also generate response text for BLEU/ROUGE evaluation")
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--num_workers",   type=int, default=2)
    return p.parse_args()


DISTRESS_EMOTION_IDS = {
    # anxious, terrified, devastated, apprehensive
    4, 30, 10, 5,
}


def _decision_confusion(dec_true: np.ndarray, dec_pred: np.ndarray) -> dict:
    dec_true = dec_true.astype(int)
    dec_pred = dec_pred.astype(int)
    tn = int(((dec_true == 0) & (dec_pred == 0)).sum())
    fp = int(((dec_true == 0) & (dec_pred == 1)).sum())
    fn = int(((dec_true == 1) & (dec_pred == 0)).sum())
    tp = int(((dec_true == 1) & (dec_pred == 1)).sum())
    total = max(1, tn + fp + fn + tp)
    return {
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "TN_pct": 100.0 * tn / total,
        "FP_pct": 100.0 * fp / total,
        "FN_pct": 100.0 * fn / total,
        "TP_pct": 100.0 * tp / total,
    }


def _summary_stats(values: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "p05": float(np.percentile(values, 5)),
        "p10": float(np.percentile(values, 10)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
    }


def _histogram(values: np.ndarray, n_bins: int) -> pd.DataFrame:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    counts, _ = np.histogram(values, bins=edges)
    return pd.DataFrame({
        "bin_left": edges[:-1],
        "bin_right": edges[1:],
        "count": counts,
        "density": counts / max(1, counts.sum()),
    })


def _distress_prior(dec_true: np.ndarray, emotion_true: np.ndarray) -> dict:
    mask = np.isin(emotion_true.astype(int), list(DISTRESS_EMOTION_IDS))
    if mask.sum() == 0:
        return {
            "n_distress": 0,
            "p_poor_decision_given_distress": None,
            "note": "No distress emotion samples in evaluated split",
        }
    poor = (dec_true[mask] == 0).astype(int)
    p_hat = float(np.mean(poor))
    # Wilson 95% CI
    n = int(mask.sum())
    z = 1.96
    denom = 1 + (z ** 2) / n
    center = (p_hat + (z ** 2) / (2 * n)) / denom
    radius = (z / denom) * np.sqrt((p_hat * (1 - p_hat) / n) + (z ** 2) / (4 * n ** 2))
    return {
        "n_distress": n,
        "p_poor_decision_given_distress": p_hat,
        "ci95_low": float(max(0.0, center - radius)),
        "ci95_high": float(min(1.0, center + radius)),
    }


@torch.no_grad()
def run_evaluation(
    model: ADDAN,
    dl,
    tokenizer: BartTokenizer,
    device: torch.device,
    tau: float = 0.5,
    do_generate: bool = False,
    max_new_tokens: int = 64,
    hist_bins: int = 10,
) -> tuple[dict, pd.DataFrame, dict[str, pd.DataFrame | dict]]:
    """
    Full evaluation pass.

    Returns (metrics, predictions_df, diagnostics).
    """
    model.eval()

    rows = []
    all_s_aff, all_s_dec = [], []
    all_e_preds, all_e_lbls = [], []
    all_dec_preds, all_dec_lbls = [], []
    all_adi = []
    hypotheses, references = [], []

    for batch in dl:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        out = model(
            context_input_ids=batch["context_input_ids"],
            context_attention_mask=batch["context_attention_mask"],
            user_input_ids=batch["user_input_ids"],
            user_attention_mask=batch["user_attention_mask"],
            response_input_ids=batch["response_input_ids"],
            response_attention_mask=batch["response_attention_mask"],
            emotion_input_ids=batch.get("emotion_input_ids"),
            emotion_attention_mask=batch.get("emotion_attention_mask"),
        )

        s_aff  = out["S_aff"].cpu().numpy()
        s_dec  = out["S_dec"].cpu().numpy()
        e_pred = out["e_hat"].cpu().numpy().argmax(axis=-1)
        adi_h  = out["adi_hat"].cpu().numpy()

        e_lbl  = batch["emotion_labels"].cpu().numpy()
        d_lbl  = batch["dec_labels"].cpu().numpy()

        all_s_aff.extend(s_aff.tolist())
        all_s_dec.extend(s_dec.tolist())
        all_adi.extend(np.maximum(0.0, s_aff - s_dec).tolist())
        all_e_preds.extend(e_pred.tolist())
        all_e_lbls.extend(e_lbl.tolist())
        all_dec_preds.extend((s_dec > 0.5).astype(int).tolist())
        all_dec_lbls.extend(d_lbl.tolist())

        # Optional generation for BLEU/ROUGE (only when BART is loaded)
        if do_generate and model.seq2seq is not None:
            bart_ctx_ids  = batch.get("bart_context_input_ids",  batch["context_input_ids"])
            bart_ctx_mask = batch.get("bart_context_attention_mask", batch["context_attention_mask"])
            gen_ids = model.generate(
                context_input_ids=bart_ctx_ids,
                context_attention_mask=bart_ctx_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=4,
            )
            gen_texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            ref_texts = tokenizer.batch_decode(
                batch["response_input_ids"], skip_special_tokens=True
            )
            hypotheses.extend(gen_texts)
            references.extend(ref_texts)

        for i, sid in enumerate(batch["sample_ids"]):
            rows.append({
                "sample_id": sid,
                "S_aff":     float(s_aff[i]),
                "S_dec":     float(s_dec[i]),
                "ADI":       float(max(0.0, s_aff[i] - s_dec[i])),
                "ADI_hat":   float(adi_h[i]),
                "ADI_flag":  int(max(0.0, s_aff[i] - s_dec[i]) > tau),
                "emotion_pred": int(e_pred[i]),
                "emotion_true": int(e_lbl[i]),
                "dec_pred":  int(s_dec[i] > 0.5),
                "dec_true":  int(d_lbl[i]),
            })

    s_aff_arr = np.array(all_s_aff)
    s_dec_arr = np.array(all_s_dec)
    e_arr     = np.array(all_e_preds)
    e_lbl_arr = np.array(all_e_lbls)
    d_arr     = np.array(all_dec_preds)
    d_lbl_arr = np.array(all_dec_lbls)
    adi_arr   = np.array(all_adi)

    # Core metrics
    metrics = {
        "ADIR (%)":  adir(s_aff_arr, s_dec_arr, tau=tau),
        "ADIS":      adis(s_aff_arr, s_dec_arr),
        "EmotAcc (%)": float(np.mean(e_arr == e_lbl_arr) * 100.0),
        "DecAcc (%)":  float(np.mean(d_arr == d_lbl_arr) * 100.0),
    }

    # Bootstrap 95% CI for ADIR
    def _adir_fn(preds, lbls):
        return adir(preds, lbls, tau=tau)

    ci_lo, ci_hi = bootstrap_ci(
        _adir_fn,
        s_aff_arr,
        s_dec_arr,
        n_bootstrap=10_000,
    )
    metrics["ADIR_CI_95"] = f"[{ci_lo:.2f}%, {ci_hi:.2f}%]"

    def _acc_fn(preds, labels):
        return float(np.mean(preds == labels) * 100.0)

    dec_ci_lo, dec_ci_hi = bootstrap_ci(
        _acc_fn,
        d_arr.astype(float),
        d_lbl_arr.astype(float),
        n_bootstrap=10_000,
    )
    metrics["DecAcc_CI_95"] = f"[{dec_ci_lo:.2f}%, {dec_ci_hi:.2f}%]"

    if do_generate and hypotheses:
        metrics["BLEU-4"]   = corpus_bleu4(hypotheses, references)
        metrics["ROUGE-L"]  = corpus_rouge_l(hypotheses, references)

    diagnostics: dict[str, pd.DataFrame | dict] = {
        "confusion": pd.DataFrame([_decision_confusion(d_lbl_arr, d_arr)]),
        "adi_stats": pd.DataFrame([_summary_stats(adi_arr)]),
        "saff_stats": pd.DataFrame([_summary_stats(s_aff_arr)]),
        "sdec_stats": pd.DataFrame([_summary_stats(s_dec_arr)]),
        "adi_hist": _histogram(adi_arr, n_bins=hist_bins),
        "saff_hist": _histogram(s_aff_arr, n_bins=hist_bins),
        "sdec_hist": _histogram(s_dec_arr, n_bins=hist_bins),
        "distress_prior": _distress_prior(d_lbl_arr, e_lbl_arr),
        "metric_consistency": pd.DataFrame([
            {
                "n_total": int(len(adi_arr)),
                "n_critical_adi": int((adi_arr > tau).sum()),
                "pct_critical_adi": float(100.0 * (adi_arr > tau).mean()),
                "n_decision_correct": int((d_arr == d_lbl_arr).sum()),
                "pct_decision_correct": float(100.0 * (d_arr == d_lbl_arr).mean()),
                "n_near_threshold_adi_0.4_0.5": int(((adi_arr >= 0.4) & (adi_arr <= 0.5)).sum()),
            }
        ]),
    }

    return metrics, pd.DataFrame(rows), diagnostics


def _save_artifacts(
    artifacts_dir: str,
    model_name: str,
    metrics: dict,
    preds_df: pd.DataFrame,
    diagnostics: dict[str, pd.DataFrame | dict],
) -> None:
    out_dir = Path(artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = model_name.replace(" ", "_").replace("/", "_")

    (out_dir / f"{safe_name}_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    preds_df.to_csv(out_dir / f"{safe_name}_predictions.csv", index=False)

    for key, value in diagnostics.items():
        if isinstance(value, pd.DataFrame):
            value.to_csv(out_dir / f"{safe_name}_{key}.csv", index=False)
        else:
            (out_dir / f"{safe_name}_{key}.json").write_text(
                json.dumps(value, indent=2), encoding="utf-8"
            )


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.encoder_model)
    emotion_tokenizer = AutoTokenizer.from_pretrained(args.emotion_model)
    bart_tokenizer = BartTokenizer.from_pretrained(args.bart_model)

    # ---- Data --------------------------------------------------------
    if args.synthetic:
        samples = make_synthetic_samples(n=args.n_synthetic, seed=99)
        n = len(samples)
        train_s = samples[:int(0.8 * n)]
        val_s   = samples[int(0.8 * n):int(0.9 * n)]
        test_s  = samples[int(0.9 * n):]
        _, _, test_dl = build_dataloaders(
            tokenizer,
            train_samples=train_s,
            val_samples=val_s,
            test_samples=test_s,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            emotion_tokenizer=emotion_tokenizer,
            bart_tokenizer=bart_tokenizer,
        )
    elif args.data_dir:
        train_s, val_s, test_s = load_tad_splits(args.data_dir)
        _, _, test_dl = build_dataloaders(
            tokenizer,
            train_samples=train_s,
            val_samples=val_s,
            test_samples=test_s,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            emotion_tokenizer=emotion_tokenizer,
            bart_tokenizer=bart_tokenizer,
        )
    else:
        raise ValueError("Provide --data_dir or --synthetic")

    # ---- Model -------------------------------------------------------
    ckpt_dir = Path(args.ckpt)
    is_rl_ckpt = (ckpt_dir / "actor.pt").exists()
    cfg   = ADDANConfig(
        encoder_model_name=args.encoder_model,
        emotion_model_name=args.emotion_model,
        bart_model_name=args.bart_model,
        num_emotions=args.num_emotions,
        gradient_checkpointing=False,          # no training — save load time
        load_generation_model=is_rl_ckpt,      # only load BART if evaluating RL ckpt
    )
    model = ADDAN(cfg).to(device)

    # Try actor.pt (RL stage) then model.pt (SFT)
    ckpt_dir = Path(args.ckpt)
    ckpt_file = ckpt_dir / "actor.pt" if (ckpt_dir / "actor.pt").exists() \
                else ckpt_dir / "model.pt"
    model.load_state_dict(torch.load(ckpt_file, map_location=device), strict=False)
    print(f"Loaded checkpoint: {ckpt_file}")

    # ---- Evaluate ----------------------------------------------------
    metrics, df, diagnostics = run_evaluation(
        model, test_dl, tokenizer, device,
        tau=args.tau,
        do_generate=args.generate,
        max_new_tokens=args.max_new_tokens,
        hist_bins=args.hist_bins,
    )

    print("\n=== Evaluation Results ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:20s}: {v:.4f}")
        else:
            print(f"  {k:20s}: {v}")

    if args.output_csv:
        Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output_csv, index=False)
        print(f"\nPer-sample predictions saved to {args.output_csv}")

    if args.artifacts_dir:
        _save_artifacts(args.artifacts_dir, args.model_name, metrics, df, diagnostics)
        print(f"Reviewer diagnostics saved to {args.artifacts_dir}")


if __name__ == "__main__":
    main()
