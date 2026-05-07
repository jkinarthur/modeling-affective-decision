# Affect–Decision Dual Alignment Network (AD-DAN)

**An architecture for detecting and mitigating affective–decision inconsistency in conversational AI systems.**

## Overview

Affective computing systems excel at generating empathetic responses but harbor a critical overlooked failure: emotionally aligned responses can actively undermine the quality of recommended decisions. We call this **Affective–Decision Inconsistency (ADI)**.

This repository contains the official implementation of **AD-DAN**, a unified neural architecture that jointly optimizes empathetic alignment and decision correctness, with a provable upper bound on expected ADI. The model is trained with a Proximal Policy Optimization (PPO) stage using a distribution-aligned composite reward.

**Paper**: [Modeling Affective–Decision Inconsistency in High-Stakes Dialogue Systems (IEEE TAC 2026)](https://arxiv.org/abs/2026.xxxxx)

## Key Results

| Model | ADIR (%) | DecAcc (%) | ADIS | 
|-------|----------|-----------|------|
| **AD-DAN_SFT** | 0.00 | 53.2 | — |
| **AD-DAN_RL** | **6.60** | **94.3** | **0.431** |
| GLHG (baseline) | 49.8 | 61.4 | 0.847 |

- **86.7% ADIR reduction** vs. strongest baseline (GLHG)
- **94.3% decision accuracy** (vs. 53.2% SFT-only)
- Statistically significant ($p < 0.01$, paired bootstrap, Bonferroni-corrected)

## Features

- **Cross-Modal Inconsistency Attention (CMIA)**: Explicit bidirectional gradient coupling between affective and decision modules
- **Inconsistency-Aware Dual-Objective Loss**: Closed-form upper bound on expected ADI (Theorem in paper)
- **Distribution-Aligned RL Reward**: Scores both generated exploration responses and original dataset responses to close train/evaluation gap
- **TAD-Bench Benchmark**: First annotation layer pairing affective and decision quality labels for standard dialogue datasets
- **Reproducible Setup**: Complete Docker-based environment with GPU support

## Quick Start

### Prerequisites
- Docker & Docker Compose
- NVIDIA GPU with CUDA 12.1+
- 16GB VRAM recommended (T4 GPU tested)

### Installation

```bash
git clone https://github.com/jkinarthur/modeling-affective-decision.git
cd modeling-affective-decision
chmod +x setup.sh
./setup.sh
```

### Data Preparation

Download TAD-Bench (EmpatheticDialogues + ESConv with decision quality annotations):
```bash
python src/data/build_tad_bench.py --output_dir ./data
```

### Training (SFT Stage)

```bash
docker compose run --rm addan python train.py \
  --stage sft \
  --data_dir ./data \
  --output_dir ./checkpoints/sft \
  --num_epochs 20 \
  --batch_size 8
```

### Training (RL Stage)

```bash
docker compose run --rm addan python train.py \
  --stage rl \
  --data_dir ./data \
  --sft_ckpt ./checkpoints/sft/model.pt \
  --output_dir ./checkpoints/rl \
  --rl_epochs 6 \
  --ppo_epochs 5 \
  --reward_alpha 0.35 \
  --reward_beta 0.85 \
  --reward_gamma_adi 0.80 \
  --reward_direct_alpha 1.5 \
  --augment_train \
  --dropout 0.2 \
  --weight_decay 0.02 \
  --emotion_label_smoothing 0.1
```

### Evaluation

```bash
docker compose run --rm addan python evaluate.py \
  --ckpt ./checkpoints/rl \
  --data_dir ./data \
  --output_csv ./results/eval.csv \
  --tau 0.50
```

### Reviewer Diagnostics Export

To directly populate the revised paper sections (confusion matrix, ADI
distribution statistics, histogram bins, distress prior estimate):

```bash
docker compose run --rm addan python evaluate.py \
  --ckpt ./checkpoints/rl/rl_epoch_06 \
  --data_dir ./data \
  --tau 0.50 \
  --model_name AD-DAN_RL \
  --artifacts_dir ./checkpoints/artifacts
```

This writes `*_metrics.json`, `*_confusion.csv`, `*_adi_stats.csv`,
`*_adi_hist.csv`, `*_saff_hist.csv`, `*_sdec_hist.csv`, and
`*_distress_prior.json`.

To aggregate multiple model outputs into paper-ready tables:

```bash
python scripts/reviewer/aggregate_results.py \
  --input_dir ./checkpoints/artifacts \
  --output_dir ./checkpoints/artifacts/paper_ready
```

For external baseline model outputs (e.g., GPT-4.1, Llama-3.1), evaluate a
prediction CSV with the same ADI metrics:

```bash
python scripts/reviewer/evaluate_baseline_csv.py \
  --input_csv ./results/gpt41_predictions.csv \
  --model_name GPT-4.1 \
  --tau 0.50 \
  --artifacts_dir ./checkpoints/artifacts
```

For open-source modern baselines, generate responses and score them with the
same AD-DAN scorer:

```bash
python scripts/reviewer/run_open_llm_baseline.py \
  --llm_model meta-llama/Llama-3.1-8B-Instruct \
  --scorer_ckpt ./checkpoints/rl/rl_epoch_06 \
  --data_dir ./data \
  --output_csv ./checkpoints/artifacts/llama31_preds.csv \
  --max_samples 300
```

## AWS Migration (SageMaker)

You can run the same containerized pipeline on AWS and pull artifacts back for
paper updates.

1. Build and push the Docker image to ECR.
2. Submit stage jobs (`build`, `train-sft`, `train-rl`, `evaluate`,
   `reviewer-suite`) with:

```bash
python aws/submit_sagemaker_job.py \
  --role-arn <SAGEMAKER_EXECUTION_ROLE_ARN> \
  --image-uri <ACCOUNT>.dkr.ecr.<REGION>.amazonaws.com/addan:latest \
  --s3-output s3://<BUCKET>/addan/jobs \
  --instance-type ml.g5.2xlarge \
  --entrypoint train-rl
```

Set `--entrypoint evaluate` or `--entrypoint reviewer-suite` for diagnostic and
aggregation runs.

For a full end-to-end run in one SageMaker job (build + SFT + RL + evaluate +
aggregation), use:

```bash
python aws/submit_sagemaker_job.py \
  --role-arn <SAGEMAKER_EXECUTION_ROLE_ARN> \
  --image-uri <ACCOUNT>.dkr.ecr.<REGION>.amazonaws.com/addan:latest \
  --s3-output s3://<BUCKET>/addan/jobs \
  --instance-type ml.g5.2xlarge \
  --entrypoint full-experiment
```

To re-create AWS resources, push image, and launch in one command from Windows:

```powershell
pwsh aws/redeploy_and_launch.ps1 \
  -AccountId <ACCOUNT_ID> \
  -Region us-east-1 \
  -RoleArn <SAGEMAKER_EXECUTION_ROLE_ARN> \
  -Bucket <S3_BUCKET_NAME>
```

## Architecture

### Core Components

1. **Shared Encoder**: DeBERTa-v3-base context encoder
2. **Affective Alignment Module (PAA)**: Scores empathetic alignment using fine-tuned emotion model
3. **Decision Evaluation Module (DEM)**: Scores decision quality via supervised regression
4. **Cross-Modal Inconsistency Attention (CMIA)**: Creates bidirectional gradient pathways
5. **RL Value Head**: Estimates expected return for PPO optimization

### Model Configuration

```python
ADDANConfig(
    encoder_model="microsoft/deberta-v3-base",
    emotion_model="j-hartmann/emotion-english-distilroberta-base",
    bart_model="facebook/bart-base",  # RL stage only
    
    # Dropout & Regularization
    paa_dropout=0.1,
    dem_dropout=0.1,
    cmia_dropout=0.1,
    weight_decay=0.01,
    emotion_label_smoothing=0.1,
    
    # Loss weights
    lambda_aff=1.0,
    lambda_dec=1.0,
    lambda_adi=0.5,
    
    # RL reward parameters
    reward_alpha=0.35,
    reward_beta=0.85,
    reward_gamma_adi=0.80,
    reward_adir_tau_bonus=0.80,
    reward_tau_sharpness=16.0,
    reward_margin_spread=0.05,
    reward_direct_alpha=1.5,
    tau=0.50
)
```

## Dataset: TAD-Bench

TAD-Bench is built on:
- **EmpatheticDialogues** (9.4K training dialogues)
- **ESConv** (10.7K training dialogues)

Each sample is annotated with:
- **Affective alignment score** (0–1): how well the response validates the user's emotional state
- **Decision quality label** (0–1): whether the suggested action is appropriate/safe
- Emotion category for breakdown analysis

## Metrics

| Metric | Definition |
|--------|-----------|
| **ADIR** | Affective–Decision Inconsistency Rate: % samples where S_aff > S_dec + τ |
| **ADIS** | Inconsistency Severity: mean(max(0, S_aff - S_dec)) across test set |
| **DecAcc** | Decision Accuracy: correctness of binary decision quality predictions |
| **EmotAcc** | Emotion Recognition Accuracy: % correct emotion category predictions |

## CLI Flags

### Common Training Flags
- `--stage {sft|rl}`: Training stage
- `--data_dir PATH`: Path to TAD-Bench directory
- `--output_dir PATH`: Checkpoint output directory
- `--batch_size N`: Training batch size (default: 8)
- `--lr FLOAT`: Learning rate (default: 1e-4 SFT, 5e-6 RL)

### Augmentation & Regularization
- `--augment_train`: Enable data augmentation on training set
- `--augment_prob FLOAT`: Probability of augmentation per sample (default: 0.5)
- `--augment_word_dropout_prob FLOAT`: Token dropout rate in augmentation (default: 0.12)
- `--dropout FLOAT`: Dropout rate in encoder/modules (default: 0.1)
- `--weight_decay FLOAT`: L2 regularization (default: 0.01)
- `--emotion_label_smoothing FLOAT`: Label smoothing for emotion CE loss (default: 0.1)

### RL Reward Parameters
- `--reward_alpha FLOAT`: Weight on empathetic quality (default: 0.35)
- `--reward_beta FLOAT`: Weight on decision correctness (default: 0.85)
- `--reward_gamma_adi FLOAT`: Weight on ADI penalty (default: 0.80)
- `--reward_adir_tau_bonus FLOAT`: Bonus for low ADIR (default: 0.80)
- `--reward_direct_alpha FLOAT`: Weight on distribution-aligned direct reward term (default: 1.5)

## Reproducibility

**Single T4 GPU (16GB VRAM)**:
- SFT: ~2 hours (20 epochs)
- RL: ~5.5 hours (6 epochs, 5 PPO updates each)
- Total: ~7.5 hours

**Multi-GPU scaling** (A100s): ~40% wall-clock reduction expected

All random seeds are fixed in `setup.sh`. See paper appendix for full experimental protocol.

## Publication & Citation

```bibtex
@article{arthur2026adi,
  title={Modeling Affective--Decision Inconsistency in High-Stakes Dialogue Systems},
  author={Arthur, John K. and Others},
  journal={IEEE Transactions on Affective Computing},
  year={2026},
  volume={XX},
  pages={XXX--XXX}
}
```

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

## Contributing

We welcome community contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Troubleshooting

### CUDA Out of Memory
- Reduce `--rl_batch_size` or `--batch_size` to 4
- Use `--fp16` for mixed-precision training (already default in templates)
- Consider single-GPU training if multi-GPU setup fails

### Slow Training
- Ensure GPU is being used: `nvidia-smi` should show container process
- Verify Docker image built successfully: `docker compose build`
- Check data I/O: `--num_workers 4` (adjust based on CPU count)

### Evaluation Crashes
- Ensure checkpoint path matches saved model format
- Use `--output_csv PATH` to save results to CSV for inspection

## Contact

**Maintainer**: John K. Arthur ([jkinarthur@example.com](mailto:jkinarthur@example.com))

For questions, issues, or collaboration: please open an issue on GitHub or contact the authors.

---

**Last Updated**: April 28, 2026  
**Repository**: https://github.com/jkinarthur/modeling-affective-decision
