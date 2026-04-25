"""
evaluate.py  –  Full evaluation script for AD-DAN
===================================================
Loads a checkpoint (SFT or RL), runs inference on the test set,
reports all paper metrics with 95% bootstrap confidence intervals,
and optionally saves per-sample predictions to a CSV.

Usage
-----
python evaluate.py --ckpt checkpoints/sft/best_model --synthetic
python evaluate.py --ckpt checkpoints/rl/rl_epoch_03 \
                   --output_csv results/predictions.csv
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import BartTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from src.models.ad_dan import ADDAN, ADDANConfig
from src.data.dataset import (
    make_synthetic_samples, build_dataloaders, TADBenchDataset, TADCollator,
    load_tad_splits,
)
from src.utils.metrics import (
    evaluate_all, adir, adis, bootstrap_ci, corpus_bleu4, corpus_rouge_l,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate AD-DAN")
    p.add_argument("--ckpt",          type=str, required=True,
                   help="Path to checkpoint directory (contains model.pt or actor.pt)")
    p.add_argument("--bart_model",    type=str, default="facebook/bart-base")
    p.add_argument("--num_emotions",  type=int, default=32)
    p.add_argument("--tau",           type=float, default=0.5)
    p.add_argument("--batch_size",    type=int, default=32)
    p.add_argument("--data_dir",      type=str, default=None,
                   help="Directory with tad_bench_{train,val,test}.csv")
    p.add_argument("--synthetic",     action="store_true")
    p.add_argument("--n_synthetic",   type=int, default=300)
    p.add_argument("--output_csv",    type=str, default=None)
    p.add_argument("--generate",      action="store_true",
                   help="Also generate response text for BLEU/ROUGE evaluation")
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--num_workers",   type=int, default=2)
    return p.parse_args()


@torch.no_grad()
def run_evaluation(
    model: ADDAN,
    dl,
    tokenizer: BartTokenizer,
    device: torch.device,
    tau: float = 0.5,
    do_generate: bool = False,
    max_new_tokens: int = 64,
) -> tuple[dict, pd.DataFrame]:
    """
    Full evaluation pass.

    Returns
    -------
    metrics : dict of all paper metrics
    df      : per-sample predictions DataFrame
    """
    model.eval()

    rows = []
    all_s_aff, all_s_dec = [], []
    all_e_preds, all_e_lbls = [], []
    all_dec_preds, all_dec_lbls = [], []
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
        )

        s_aff  = out["S_aff"].cpu().numpy()
        s_dec  = out["S_dec"].cpu().numpy()
        e_pred = out["e_hat"].cpu().numpy().argmax(axis=-1)
        adi_h  = out["adi_hat"].cpu().numpy()

        e_lbl  = batch["emotion_labels"].cpu().numpy()
        d_lbl  = batch["dec_labels"].cpu().numpy()

        all_s_aff.extend(s_aff.tolist())
        all_s_dec.extend(s_dec.tolist())
        all_e_preds.extend(e_pred.tolist())
        all_e_lbls.extend(e_lbl.tolist())
        all_dec_preds.extend((s_dec > 0.5).astype(int).tolist())
        all_dec_lbls.extend(d_lbl.tolist())

        # Optional generation for BLEU/ROUGE
        if do_generate:
            gen_ids = model.generate(
                context_input_ids=batch["context_input_ids"],
                context_attention_mask=batch["context_attention_mask"],
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
                "ADI_hat":   float(adi_h[i]),
                "ADI_flag":  int(max(0.0, s_aff[i] - s_dec[i]) > 0.5),
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

    # Core metrics
    metrics = {
        "ADIR (%)":  adir(s_aff_arr, s_dec_arr, tau=0.5),
        "ADIS":      adis(s_aff_arr, s_dec_arr),
        "EmotAcc (%)": float(np.mean(e_arr == e_lbl_arr) * 100.0),
        "DecAcc (%)":  float(np.mean(d_arr == d_lbl_arr) * 100.0),
    }

    # Bootstrap 95% CI for ADIR
    def _adir_fn(preds, lbls):
        return adir(preds, lbls, tau=0.5)

    ci_lo, ci_hi = bootstrap_ci(
        _adir_fn,
        s_aff_arr,
        s_dec_arr,
        n_bootstrap=10_000,
    )
    metrics["ADIR_CI_95"] = f"[{ci_lo:.2f}%, {ci_hi:.2f}%]"

    if do_generate and hypotheses:
        metrics["BLEU-4"]   = corpus_bleu4(hypotheses, references)
        metrics["ROUGE-L"]  = corpus_rouge_l(hypotheses, references)

    return metrics, pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = BartTokenizer.from_pretrained(args.bart_model)

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
        )
    else:
        raise ValueError("Provide --data_dir or --synthetic")

    # ---- Model -------------------------------------------------------
    cfg   = ADDANConfig(bart_model_name=args.bart_model, num_emotions=args.num_emotions)
    model = ADDAN(cfg).to(device)

    # Try actor.pt (RL stage) then model.pt (SFT)
    ckpt_dir = Path(args.ckpt)
    ckpt_file = ckpt_dir / "actor.pt" if (ckpt_dir / "actor.pt").exists() \
                else ckpt_dir / "model.pt"
    model.load_state_dict(torch.load(ckpt_file, map_location=device))
    print(f"Loaded checkpoint: {ckpt_file}")

    # ---- Evaluate ----------------------------------------------------
    metrics, df = run_evaluation(
        model, test_dl, tokenizer, device,
        tau=args.tau,
        do_generate=args.generate,
        max_new_tokens=args.max_new_tokens,
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


if __name__ == "__main__":
    main()
