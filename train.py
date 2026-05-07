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
from transformers import AutoTokenizer, BartTokenizer

# ---- project imports -------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))
from src.models.ad_dan import ADDAN, ADDANConfig
from src.data.dataset import (
    make_synthetic_samples, load_tad_samples_from_csv, build_dataloaders,
    load_tad_splits, compute_class_weights,
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
    p.add_argument("--encoder_model",   type=str, default="microsoft/deberta-v3-base",
                   help="DeBERTa encoder model name")
    p.add_argument("--emotion_model",   type=str,
                   default="j-hartmann/emotion-english-distilroberta-base",
                   help="Pre-trained emotion backbone model name")
    p.add_argument("--bart_model",      type=str, default="facebook/bart-base",
                   help="BART model for RL-stage generation")
    p.add_argument("--num_emotions",    type=int, default=32)

    # SFT hyper-parameters
    p.add_argument("--max_epochs",        type=int,   default=20)
    p.add_argument("--batch_size",        type=int,   default=2,
                   help="Per-step batch size (effective batch = batch_size * grad_accum_steps)")
    p.add_argument("--grad_accum_steps",  type=int,   default=32,
                   help="Gradient accumulation steps (effective batch = batch_size * this)")
    p.add_argument("--paa_pretrain_epochs", type=int, default=5,
                   help="Epochs to pre-train PAA+encoder only before joint training")
    p.add_argument("--dem_pretrain_epochs", type=int, default=5,
                   help="Epochs to pre-train DEM+encoder only before joint training")
    p.add_argument("--freeze_emotion_epochs", type=int, default=5,
                   help="Epochs to keep emotion encoder frozen before unfreezing")
    p.add_argument("--lr",            type=float, default=3e-5)
    p.add_argument("--weight_decay",  type=float, default=0.01,
                   help="Weight decay for AdamW")
    p.add_argument("--warmup_steps",  type=int,   default=1_000)
    p.add_argument("--lambda_adi",    type=float, default=0.5)
    p.add_argument("--dropout",       type=float, default=0.1,
                   help="Dropout used in PAA/DEM/CMIA modules")
    p.add_argument("--emotion_label_smoothing", type=float, default=0.1,
                   help="Label smoothing for emotion CE loss")
    p.add_argument("--grad_clip",     type=float, default=1.0)
    p.add_argument("--fp16",          action="store_true", default=True)
    p.add_argument("--no_fp16",       action="store_true")

    # RL hyper-parameters
    p.add_argument("--rl_epochs",     type=int, default=3)
    p.add_argument("--rl_lr",         type=float, default=5e-6)
    p.add_argument("--rl_batch_size", type=int, default=8,
                   help="Batch size for RL rollouts (smaller to fit T4 memory)")
    p.add_argument("--rl_max_new_tokens", type=int, default=64,
                   help="Max tokens to generate per rollout")
    p.add_argument("--ppo_epochs",    type=int, default=5)
    p.add_argument("--ppo_clip",      type=float, default=0.1)
    p.add_argument("--reward_margin_spread", type=float, default=0.05,
                   help="Reward bonus coefficient to discourage score-collapse")
    p.add_argument("--reward_alpha", type=float, default=0.35,
                   help="Reward weight for S_aff term")
    p.add_argument("--reward_beta", type=float, default=0.85,
                   help="Reward weight for S_dec term")
    p.add_argument("--reward_gamma_adi", type=float, default=0.80,
                   help="Penalty weight for ADI=max(0,S_aff-S_dec) term")
    p.add_argument("--reward_adir_tau_bonus", type=float, default=0.80,
                   help="Penalty weight for crossing ADIR threshold tau")
    p.add_argument("--reward_tau_sharpness", type=float, default=16.0,
                   help="Sharpness of smooth ADIR threshold indicator")
    p.add_argument("--reward_direct_alpha", type=float, default=1.5,
                   help="Weight for eval-aligned reward on original dataset response")
    p.add_argument("--sft_ckpt",      type=str, default=None,
                   help="Path to SFT model checkpoint for RL init")

    # Data augmentation (train split only)
    p.add_argument("--augment_train", action="store_true",
                   help="Enable train-time context augmentation")
    p.add_argument("--augment_prob", type=float, default=0.5,
                   help="Probability of applying augmentation to a sample")
    p.add_argument("--augment_word_dropout_prob", type=float, default=0.12,
                   help="Per-token dropout probability for context augmentation")

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

    # ---- Tokenizer (DeBERTa tokenizer for context/response encoding) ------
    # BART tokenizer still used for generation in RL stage
    tokenizer = AutoTokenizer.from_pretrained(args.encoder_model)
    # Separate RoBERTa tokenizer for the emotion encoder (different vocab ~50K)
    emotion_tokenizer = AutoTokenizer.from_pretrained(args.emotion_model)

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
        use_weighted_sampler=True,
        emotion_tokenizer=emotion_tokenizer,
        augment_train=args.augment_train,
        augment_prob=args.augment_prob,
        augment_word_dropout_prob=args.augment_word_dropout_prob,
    )
    # Separate smaller-batch dataloader for RL rollouts (no weighted sampler needed)
    # bart_tokenizer ensures context is tokenized with BART's 50K vocab for generate()
    bart_tokenizer = BartTokenizer.from_pretrained(args.bart_model)
    rl_train_dl, _, _ = build_dataloaders(
        tokenizer, train_s, val_s, test_s,
        batch_size=args.rl_batch_size,
        num_workers=args.num_workers,
        use_weighted_sampler=False,
        emotion_tokenizer=emotion_tokenizer,
        bart_tokenizer=bart_tokenizer,
        augment_train=args.augment_train,
        augment_prob=args.augment_prob,
        augment_word_dropout_prob=args.augment_word_dropout_prob,
    )

    # ---- Compute class weights from training data -------------------------
    emotion_weights, dec_pos_weight = compute_class_weights(train_s, args.num_emotions)
    print(f"dec_pos_weight={dec_pos_weight:.3f}  "
          f"emotion weight range=[{emotion_weights.min():.3f}, {emotion_weights.max():.3f}]")

    # ---- Model -----------------------------------------------------------
    cfg = ADDANConfig(
        encoder_model_name=args.encoder_model,
        emotion_model_name=args.emotion_model,
        bart_model_name=args.bart_model,
        num_emotions=args.num_emotions,
        lambda_adi=args.lambda_adi,
        paa_dropout=args.dropout,
        dem_dropout=args.dropout,
        cmia_dropout=args.dropout,
        weight_decay=args.weight_decay,
        emotion_label_smoothing=args.emotion_label_smoothing,
        dec_pos_weight=dec_pos_weight,
        emotion_class_weights=emotion_weights,
        reward_alpha=args.reward_alpha,
        reward_beta=args.reward_beta,
        reward_gamma_adi=args.reward_gamma_adi,
        reward_adir_tau_bonus=args.reward_adir_tau_bonus,
        reward_tau_sharpness=args.reward_tau_sharpness,
        reward_margin_spread=args.reward_margin_spread,
        reward_direct_alpha=args.reward_direct_alpha,
        grad_accum_steps=args.grad_accum_steps,
        freeze_emotion_encoder_epochs=args.freeze_emotion_epochs,
        gradient_checkpointing=True,
        # Don't load BART during SFT — saves ~530MB on the T4
        load_generation_model=(args.stage == "rl"),
    )
    model = ADDAN(cfg)

    if args.sft_ckpt:
        ckpt_path = os.path.join(args.sft_ckpt, "model.pt")
        # strict=False: SFT checkpoint has no seq2seq keys (BART not loaded
        # during SFT); RL model has BART freshly initialized from HuggingFace.
        model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=False)
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
            paa_pretrain_epochs=args.paa_pretrain_epochs,
            dem_pretrain_epochs=args.dem_pretrain_epochs,
            grad_accum_steps=args.grad_accum_steps,
            device=device,
        )
        history = trainer.train()
        print("\nTraining complete. Final history:")
        for k, v in history.items():
            print(f"  {k}: {[f'{x:.4f}' for x in v]}")

    elif args.stage == "rl":
        rl_trainer = PPOTrainer(
            actor=model,
            train_dl=rl_train_dl,
            output_dir=args.output_dir,
            config=cfg,
            ppo_epochs=args.ppo_epochs,
            ppo_clip=args.ppo_clip,
            lr=args.rl_lr,
            max_new_tokens=args.rl_max_new_tokens,
            fp16=fp16,
            device=device,
        )
        history = rl_trainer.train(num_epochs=args.rl_epochs)
        print("\nRL training complete. Final history:")
        for k, v in history.items():
            print(f"  {k}: {[f'{x:.4f}' for x in v]}")


if __name__ == "__main__":
    main()
