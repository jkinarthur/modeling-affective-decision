"""
train.py  –  End-to-end training script for AD-DAN
====================================================
Usage
-----
# Stage 1 – Supervised fine-tuning:
python train.py --stage sft --output_dir checkpoints/sft

# Stage 2 – PPO RL fine-tuning (loads best SFT checkpoint):
python train.py --stage rl  --sft_ckpt checkpoints/sft/best_model \
                             --output_dir checkpoints/rl

# Quick smoke-test with synthetic data (no GPU required):
python train.py --stage sft --synthetic --max_epochs 2 --batch_size 4
"""

import argparse
import os
import sys

import torch
from transformers import BartTokenizer

# ---- project imports -------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))
from src.models.ad_dan import ADDAN, ADDANConfig
from src.data.dataset import (
    make_synthetic_samples, load_tad_samples_from_csv, build_dataloaders,
    load_tad_splits,
)
from src.training.trainer import SFTTrainer
from src.training.rl_trainer import PPOTrainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train AD-DAN")

    # Stage
    p.add_argument("--stage", choices=["sft", "rl"], default="sft",
                   help="Training stage: sft (supervised) or rl (PPO)")

    # Data
    p.add_argument("--data_dir",      type=str, default=None,
                   help="Directory with tad_bench_{train,val,test}.csv (built by build_tad_bench.py)")
    p.add_argument("--train_csv",     type=str, default=None,
                   help="Path to TAD-Bench training annotation CSV (legacy)")
    p.add_argument("--val_csv",       type=str, default=None,
                   help="Path to TAD-Bench validation annotation CSV (legacy)")
    p.add_argument("--synthetic",     action="store_true",
                   help="Use synthetic data (for smoke-testing)")
    p.add_argument("--n_synthetic",   type=int, default=500)

    # Model
    p.add_argument("--bart_model",    type=str, default="facebook/bart-base")
    p.add_argument("--num_emotions",  type=int, default=32)

    # SFT hyper-parameters
    p.add_argument("--max_epochs",    type=int, default=20)
    p.add_argument("--batch_size",    type=int, default=32)
    p.add_argument("--lr",            type=float, default=3e-5)
    p.add_argument("--warmup_steps",  type=int, default=1_000)
    p.add_argument("--lambda_adi",    type=float, default=0.5)
    p.add_argument("--grad_clip",     type=float, default=1.0)
    p.add_argument("--fp16",          action="store_true", default=True)
    p.add_argument("--no_fp16",       action="store_true")

    # RL hyper-parameters
    p.add_argument("--rl_epochs",     type=int, default=3)
    p.add_argument("--rl_lr",         type=float, default=1e-5)
    p.add_argument("--ppo_epochs",    type=int, default=5)
    p.add_argument("--ppo_clip",      type=float, default=0.1)
    p.add_argument("--sft_ckpt",      type=str, default=None,
                   help="Path to SFT model checkpoint for RL init")

    # Checkpointing
    p.add_argument("--output_dir",    type=str, default="checkpoints/sft")
    p.add_argument("--log_every",     type=int, default=50)
    p.add_argument("--num_workers",   type=int, default=4)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    fp16 = args.fp16 and not args.no_fp16

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  FP16: {fp16}")

    # ---- Tokenizer -------------------------------------------------------
    tokenizer = BartTokenizer.from_pretrained(args.bart_model)

    # ---- Data ------------------------------------------------------------
    if args.synthetic:
        print(f"Using synthetic data ({args.n_synthetic} samples).")
        samples = make_synthetic_samples(n=args.n_synthetic)
        # 80 / 10 / 10 split
        n = len(samples)
        n_train = int(0.8 * n)
        n_val   = int(0.1 * n)
        train_s = samples[:n_train]
        val_s   = samples[n_train : n_train + n_val]
        test_s  = samples[n_train + n_val :]

    elif args.data_dir:
        # Load CSVs built by build_tad_bench.py
        train_s, val_s, test_s = load_tad_splits(args.data_dir)

    else:
        assert args.train_csv and args.val_csv, (
            "Provide --data_dir, --train_csv/--val_csv, or --synthetic"
        )
        raise NotImplementedError(
            "Legacy CSV loading: implement load_tad_samples_from_csv() "
            "with your HuggingFace dataset."
        )

    train_dl, val_dl, test_dl = build_dataloaders(
        tokenizer, train_s, val_s, test_s,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # ---- Model -----------------------------------------------------------
    cfg = ADDANConfig(
        bart_model_name=args.bart_model,
        num_emotions=args.num_emotions,
        lambda_adi=args.lambda_adi,
    )
    model = ADDAN(cfg)

    if args.sft_ckpt:
        ckpt_path = os.path.join(args.sft_ckpt, "model.pt")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"Loaded SFT checkpoint from {ckpt_path}")

    # ---- Training --------------------------------------------------------
    if args.stage == "sft":
        trainer = SFTTrainer(
            model=model,
            train_dl=train_dl,
            val_dl=val_dl,
            output_dir=args.output_dir,
            config=cfg,
            lr=args.lr,
            warmup_steps=args.warmup_steps,
            max_epochs=args.max_epochs,
            grad_clip=args.grad_clip,
            fp16=fp16,
            log_every=args.log_every,
            save_best=True,
            device=device,
        )
        history = trainer.train()
        print("\nTraining complete. Final history:")
        for k, v in history.items():
            print(f"  {k}: {[f'{x:.4f}' for x in v]}")

    elif args.stage == "rl":
        rl_trainer = PPOTrainer(
            actor=model,
            train_dl=train_dl,
            output_dir=args.output_dir,
            config=cfg,
            ppo_epochs=args.ppo_epochs,
            ppo_clip=args.ppo_clip,
            lr=args.rl_lr,
            fp16=fp16,
            device=device,
        )
        history = rl_trainer.train(num_epochs=args.rl_epochs)
        print("\nRL training complete. Final history:")
        for k, v in history.items():
            print(f"  {k}: {[f'{x:.4f}' for x in v]}")


if __name__ == "__main__":
    main()
