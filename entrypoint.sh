#!/usr/bin/env bash
set -euo pipefail

CMD="${1:-help}"
shift || true

DATA_DIR="${DATA_DIR:-/app/data}"
CKPT_DIR="${CKPT_DIR:-/app/checkpoints}"
ARTIFACT_DIR="${ARTIFACT_DIR:-/app/checkpoints/artifacts}"

case "$CMD" in
  build)
    python src/data/build_tad_bench.py --output_dir "$DATA_DIR"
    ;;

  train-sft)
    python train.py \
      --stage sft \
      --data_dir "$DATA_DIR" \
      --output_dir "$CKPT_DIR/sft" \
      --max_epochs "${MAX_EPOCHS:-20}" \
      --batch_size "${BATCH_SIZE:-8}" \
      --lr "${LR:-3e-5}" \
      --augment_train \
      --dropout "${DROPOUT:-0.1}" \
      --weight_decay "${WEIGHT_DECAY:-0.01}" \
      --emotion_label_smoothing "${EMOTION_LABEL_SMOOTHING:-0.1}"
    ;;

  train-rl)
    python train.py \
      --stage rl \
      --data_dir "$DATA_DIR" \
      --sft_ckpt "$CKPT_DIR/sft" \
      --output_dir "$CKPT_DIR/rl" \
      --rl_epochs "${RL_EPOCHS:-6}" \
      --ppo_epochs "${PPO_EPOCHS:-5}" \
      --rl_batch_size "${RL_BATCH_SIZE:-8}" \
      --reward_alpha "${REWARD_ALPHA:-0.35}" \
      --reward_beta "${REWARD_BETA:-0.85}" \
      --reward_gamma_adi "${REWARD_GAMMA_ADI:-0.80}" \
      --reward_adir_tau_bonus "${REWARD_ADIR_TAU_BONUS:-0.80}" \
      --reward_direct_alpha "${REWARD_DIRECT_ALPHA:-1.5}"
    ;;

  evaluate)
    python evaluate.py \
      --ckpt "$CKPT_DIR/rl/rl_epoch_06" \
      --data_dir "$DATA_DIR" \
      --tau "${TAU:-0.50}" \
      --output_csv "$ARTIFACT_DIR/eval_predictions.csv" \
      --artifacts_dir "$ARTIFACT_DIR" \
      --model_name "AD-DAN_RL"
    ;;

  reviewer-suite)
    python scripts/reviewer/aggregate_results.py \
      --input_dir "$ARTIFACT_DIR" \
      --output_dir "$ARTIFACT_DIR/paper_ready"
    ;;

  full-experiment)
    # End-to-end pipeline for single-job cloud execution.
    python src/data/build_tad_bench.py --output_dir "$DATA_DIR"

    python train.py \
      --stage sft \
      --data_dir "$DATA_DIR" \
      --output_dir "$CKPT_DIR/sft" \
      --max_epochs "${MAX_EPOCHS:-20}" \
      --batch_size "${BATCH_SIZE:-8}" \
      --lr "${LR:-3e-5}" \
      --augment_train \
      --dropout "${DROPOUT:-0.1}" \
      --weight_decay "${WEIGHT_DECAY:-0.01}" \
      --emotion_label_smoothing "${EMOTION_LABEL_SMOOTHING:-0.1}"

    python train.py \
      --stage rl \
      --data_dir "$DATA_DIR" \
      --sft_ckpt "$CKPT_DIR/sft" \
      --output_dir "$CKPT_DIR/rl" \
      --rl_epochs "${RL_EPOCHS:-6}" \
      --ppo_epochs "${PPO_EPOCHS:-5}" \
      --rl_batch_size "${RL_BATCH_SIZE:-8}" \
      --reward_alpha "${REWARD_ALPHA:-0.35}" \
      --reward_beta "${REWARD_BETA:-0.85}" \
      --reward_gamma_adi "${REWARD_GAMMA_ADI:-0.80}" \
      --reward_adir_tau_bonus "${REWARD_ADIR_TAU_BONUS:-0.80}" \
      --reward_direct_alpha "${REWARD_DIRECT_ALPHA:-1.5}"

    python evaluate.py \
      --ckpt "$CKPT_DIR/rl/rl_epoch_06" \
      --data_dir "$DATA_DIR" \
      --tau "${TAU:-0.50}" \
      --output_csv "$ARTIFACT_DIR/eval_predictions.csv" \
      --artifacts_dir "$ARTIFACT_DIR" \
      --model_name "AD-DAN_RL"

    python scripts/reviewer/aggregate_results.py \
      --input_dir "$ARTIFACT_DIR" \
      --output_dir "$ARTIFACT_DIR/paper_ready"
    ;;

  smoke)
    python train.py --stage sft --synthetic --n_synthetic 128 --max_epochs 2 --batch_size 4
    python evaluate.py --ckpt "$CKPT_DIR/sft" --synthetic --n_synthetic 128 --model_name smoke --artifacts_dir "$ARTIFACT_DIR"
    ;;

  shell)
    exec bash "$@"
    ;;

  help|*)
    cat <<'EOF'
AD-DAN container commands:
  build          Build TAD-Bench csv splits
  train-sft      Run supervised training
  train-rl       Run PPO fine-tuning
  evaluate       Evaluate RL checkpoint + export reviewer diagnostics
  reviewer-suite Aggregate metrics artifacts into paper-ready tables
      full-experiment Run build + SFT + RL + evaluate + aggregation
  smoke          Synthetic end-to-end smoke test
  shell          Open shell inside container
EOF
    ;;
esac
