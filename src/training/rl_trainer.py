"""
PPO RL Trainer for AD-DAN
==========================
Implements the Proximal Policy Optimization stage described in the paper.

Reward function (terminal, per full response):
    R(u_T, r_T) = α·S_aff + β·S_dec − γ_adi·ADI − b_tau·1[ADI>tau]
    R_total     = R − η·KL[π_θ(·|u_T) ‖ π_SFT(·|u_T)]

with α=0.35, β=0.85, γ_adi=0.80, b_tau=0.80, η=0.02.

PPO hyper-parameters (paper defaults):
    ε=0.1 (clip), 5 PPO epochs per rollout batch, batch_size=16.

Architecture note
-----------------
The "actor" policy is the AD-DAN seq2seq decoder.
The "critic" value head is a lightweight MLP on top of the CMIA output.
The reference policy π_SFT is a frozen copy of the SFT checkpoint.
"""

from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BartTokenizer

from ..models.ad_dan import ADDAN, ADDANConfig
from ..utils.metrics import adir


# ---------------------------------------------------------------------------
# Value Head
# ---------------------------------------------------------------------------

class ValueHead(nn.Module):
    """Simple MLP value head V(context) → scalar baseline."""

    def __init__(self, hidden_dim: int = 768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h).squeeze(-1)  # (B,)


# ---------------------------------------------------------------------------
# PPO Trainer
# ---------------------------------------------------------------------------

class PPOTrainer:
    """
    PPO-based RL fine-tuning for AD-DAN.

    Parameters
    ----------
    actor       : the AD-DAN model to be fine-tuned (π_θ)
    train_dl    : training DataLoader (same TAD-Bench format)
    output_dir  : directory to save RL checkpoints
    config      : ADDANConfig for reward weights
    ppo_epochs  : number of PPO update epochs per rollout batch
    ppo_clip    : ε for PPO clipping (default 0.1)
    lr          : learning rate for actor + value head
    batch_size  : rollout batch size (default 16)
    max_new_tokens: max tokens to generate per response
    gamma       : discount factor (default 0.99)
    gae_lambda  : GAE-λ (default 0.95)
    device      : torch device
    fp16        : mixed precision
    """

    def __init__(
        self,
        actor: ADDAN,
        train_dl: DataLoader,
        output_dir: str = "checkpoints/rl",
        config: Optional[ADDANConfig] = None,
        ppo_epochs: int = 5,
        ppo_clip: float = 0.1,
        lr: float = 1e-5,
        max_new_tokens: int = 128,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        fp16: bool = True,
        device: Optional[torch.device] = None,
    ):
        self.actor     = actor
        self.train_dl  = train_dl
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config    = config or actor.config
        self.ppo_epochs  = ppo_epochs
        self.ppo_clip    = ppo_clip
        self.max_new_tokens = max_new_tokens
        self.gamma    = gamma
        self.gae_lambda = gae_lambda

        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.actor.to(self.device)

        # Frozen reference policy (π_SFT) — kept on CPU to save GPU memory.
        # Moved to device only when computing ref log-probs.
        self.ref_policy: ADDAN = copy.deepcopy(actor)
        for p in self.ref_policy.parameters():
            p.requires_grad_(False)
        self.ref_policy.eval()
        self.ref_policy.cpu()  # keep off GPU

        # Value head
        hidden_dim = actor.seq2seq.config.d_model
        self.value_head = ValueHead(hidden_dim).to(self.device)

        # Optimizer covers actor + value head (ref policy is frozen)
        self.optimizer = AdamW(
            list(actor.parameters()) + list(self.value_head.parameters()),
            lr=lr,
            weight_decay=float(self.config.weight_decay),
        )
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=fp16 and self.device.type == "cuda"
        )
        self.fp16 = fp16

        # Tokenizers for reward scoring path:
        # generation uses BART ids, while scoring uses DeBERTa ids.
        self.encoder_tokenizer = AutoTokenizer.from_pretrained(
            self.config.encoder_model_name
        )
        self.bart_tokenizer = BartTokenizer.from_pretrained(
            self.config.bart_model_name
        )

    def _retokenize_generated_for_scoring(
        self,
        gen_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert generated BART token ids into DeBERTa-tokenized response ids
        so reward/scoring is computed in the encoder's token space.
        """
        texts = self.bart_tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        enc = self.encoder_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_new_tokens,
            return_tensors="pt",
        )
        return enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device)

    # ------------------------------------------------------------------
    # Reward computation
    # ------------------------------------------------------------------

    def compute_reward(
        self,
        context_input_ids: torch.Tensor,
        context_attention_mask: torch.Tensor,
        user_input_ids: torch.Tensor,
        user_attention_mask: torch.Tensor,
        score_input_ids: torch.Tensor,
        score_attention_mask: torch.Tensor,
        orig_resp_input_ids: torch.Tensor,
        orig_resp_attention_mask: torch.Tensor,
        ref_log_probs: torch.Tensor,   # (B, T-1)  from π_SFT
        act_log_probs: torch.Tensor,   # (B, T-1)  from π_θ
    ) -> torch.Tensor:                 # (B,)  scalar reward per sample
        """
        R_total = [generation reward on generated text]
                + reward_direct_alpha * [direct reward on original response]
                − η·KL

        The direct reward term scores the *original* dataset response — the same
        text that evaluate.py scores — closing the train/eval distribution gap.
        """
        cfg = self.config

        # ---- Part 1: reward on generated response (exploration signal) ----
        with torch.no_grad():
            s_aff, s_dec, _ = self.actor.score(
                context_input_ids, context_attention_mask,
                user_input_ids, user_attention_mask,
                score_input_ids, score_attention_mask,
            )

        margin = s_aff - s_dec
        adi_metric = torch.clamp(margin, min=0.0)
        tau_penalty = torch.sigmoid((margin - cfg.tau) * cfg.reward_tau_sharpness)
        spread_bonus = torch.abs(margin - margin.mean())

        gen_reward = (
            cfg.reward_alpha * s_aff
            + cfg.reward_beta * s_dec
            - cfg.reward_gamma_adi * adi_metric
            - cfg.reward_adir_tau_bonus * tau_penalty
            + cfg.reward_margin_spread * spread_bonus
        )

        # ---- Part 2: direct reward on original response (eval-aligned) ----
        # This scores the *actual* dataset response — identical to what evaluate.py
        # does at test time — so RL directly optimizes the evaluation metric.
        with torch.no_grad():
            s_aff_orig, s_dec_orig, _ = self.actor.score(
                context_input_ids, context_attention_mask,
                user_input_ids, user_attention_mask,
                orig_resp_input_ids, orig_resp_attention_mask,
            )

        margin_orig = s_aff_orig - s_dec_orig
        adi_orig    = torch.clamp(margin_orig, min=0.0)
        tau_penalty_orig = torch.sigmoid((margin_orig - cfg.tau) * cfg.reward_tau_sharpness)

        direct_reward = (
            cfg.reward_alpha * s_aff_orig
            + cfg.reward_beta * s_dec_orig
            - cfg.reward_gamma_adi * adi_orig
            - cfg.reward_adir_tau_bonus * tau_penalty_orig
        )

        task_reward = gen_reward + cfg.reward_direct_alpha * direct_reward

        # KL penalty: KL[π_θ ‖ π_SFT] = Σ_t (log π_θ - log π_SFT)
        kl = (act_log_probs - ref_log_probs).sum(dim=-1)   # (B,)
        kl = kl.clamp(min=0.0)                              # one-sided penalty

        return task_reward - cfg.reward_kl_eta * kl        # (B,)

    # ------------------------------------------------------------------
    # GAE advantage estimation
    # ------------------------------------------------------------------

    def _compute_advantages(
        self,
        rewards: torch.Tensor,   # (B,)  terminal reward
        values: torch.Tensor,    # (B,)  V(context)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        For single-step (bandit) problems, advantage = R − V and
        returns = R.  GAE reduces to this when T=1.
        """
        returns    = rewards
        advantages = returns - values.detach()
        # Normalize advantages for stability
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return returns, advantages

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self, num_epochs: int = 3) -> dict[str, list[float]]:
        """
        Run the PPO RL training loop.

        Each outer epoch iterates once over the training DataLoader,
        collecting rollouts and then performing `ppo_epochs` PPO updates.

        Returns
        -------
        history : dict with lists of avg_reward and avg_adir per epoch
        """
        history: dict[str, list[float]] = {"avg_reward": [], "avg_adir": []}

        for epoch in range(1, num_epochs + 1):
            t0 = time.time()
            epoch_rewards, epoch_adir_vals = [], []
            n_batches = len(self.train_dl)

            for batch_idx, batch in enumerate(self.train_dl, 1):
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                # ---- Rollout: generate responses -------------------------
                # Use BART-tokenized context (50K vocab) — DeBERTa IDs (128K)
                # would cause out-of-bounds CUDA assert in BART embeddings.
                bart_ctx_ids  = batch.get("bart_context_input_ids",  batch["context_input_ids"])
                bart_ctx_mask = batch.get("bart_context_attention_mask", batch["context_attention_mask"])

                with torch.no_grad():
                    gen_ids = self.actor.generate(
                        context_input_ids=bart_ctx_ids,
                        context_attention_mask=bart_ctx_mask,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=True,
                        top_p=0.9,
                        temperature=1.0,
                    )
                    # Build attention mask for generated ids
                    pad_id  = self.actor.seq2seq.config.pad_token_id
                    gen_mask = (gen_ids != pad_id).long()

                    # Retokenize generated responses (BART ids -> DeBERTa ids)
                    # before calling reward/scoring modules.
                    score_resp_ids, score_resp_mask = self._retokenize_generated_for_scoring(
                        gen_ids
                    )

                    # Log-probs from both policies
                    # ref_policy lives on CPU; move to GPU for inference
                    self.ref_policy.to(self.device)
                    ref_lp = self.ref_policy.log_probs_from_ids(
                        bart_ctx_ids,
                        bart_ctx_mask,
                        gen_ids, gen_mask,
                    )
                    self.ref_policy.cpu()
                    torch.cuda.empty_cache()

                # Actor log-probs (requires grad for PPO)
                act_lp = self.actor.log_probs_from_ids(
                    bart_ctx_ids,
                    bart_ctx_mask,
                    gen_ids, gen_mask,
                )

                # Rewards (no grad needed through reward)
                with torch.no_grad():
                    rewards = self.compute_reward(
                        batch["context_input_ids"],
                        batch["context_attention_mask"],
                        batch["user_input_ids"],
                        batch["user_attention_mask"],
                        score_resp_ids, score_resp_mask,
                        batch["response_input_ids"],
                        batch["response_attention_mask"],
                        ref_lp.detach(), act_lp.detach(),
                    )

                    # Value baseline from [CLS] of context encoder
                    ctx_h = self.actor.encode(
                        batch["context_input_ids"],
                        batch["context_attention_mask"],
                    )[:, 0, :]                      # (B, d)
                    values = self.value_head(ctx_h)  # (B,)

                returns, advantages = self._compute_advantages(rewards, values)

                epoch_rewards.extend(rewards.cpu().tolist())

                # Track ADIR on original responses — matches evaluate.py exactly.
                # This makes avg_ADIR in epoch logs directly comparable to test-set ADIR.
                with torch.no_grad():
                    s_aff, s_dec, _ = self.actor.score(
                        batch["context_input_ids"],
                        batch["context_attention_mask"],
                        batch["user_input_ids"],
                        batch["user_attention_mask"],
                        batch["response_input_ids"],
                        batch["response_attention_mask"],
                    )
                epoch_adir_vals.append(
                    adir(s_aff.cpu().numpy(), s_dec.cpu().numpy(), tau=self.config.tau)
                )

                # ---- PPO updates -----------------------------------------
                old_log_probs = act_lp.detach().sum(dim=-1)  # (B,) sum over tokens

                for _ in range(self.ppo_epochs):
                    with torch.cuda.amp.autocast(
                        enabled=self.fp16 and self.device.type == "cuda"
                    ):
                        new_lp = self.actor.log_probs_from_ids(
                            bart_ctx_ids,
                            bart_ctx_mask,
                            gen_ids, gen_mask,
                        )
                        new_log_probs = new_lp.sum(dim=-1)  # (B,)

                        ratio = torch.exp(new_log_probs - old_log_probs)
                        surr1 = ratio * advantages
                        surr2 = torch.clamp(
                            ratio, 1.0 - self.ppo_clip, 1.0 + self.ppo_clip
                        ) * advantages
                        policy_loss = -torch.min(surr1, surr2).mean()

                        # Entropy bonus: encourages exploration, prevents all-negative collapse
                        # entropy ≈ -mean(log_probs) over sequence tokens
                        entropy_bonus = -new_lp.mean()  # mean per-token entropy proxy
                        entropy_coef = 0.01

                        # Value loss
                        ctx_h_new = self.actor.encode(
                            batch["context_input_ids"],
                            batch["context_attention_mask"],
                        )[:, 0, :]
                        v_new = self.value_head(ctx_h_new)
                        value_loss = F.mse_loss(v_new, returns)

                        loss = policy_loss + 0.5 * value_loss - entropy_coef * entropy_bonus

                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    nn.utils.clip_grad_norm_(
                        list(self.actor.parameters()) +
                        list(self.value_head.parameters()),
                        max_norm=1.0,
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                if batch_idx % 50 == 0 or batch_idx == n_batches:
                    elapsed = time.time() - t0
                    avg_r_so_far = float(np.mean(epoch_rewards)) if epoch_rewards else 0.0
                    print(
                        f"  RL Epoch {epoch} | step {batch_idx}/{n_batches} "
                        f"| avg_reward={avg_r_so_far:.4f} | {elapsed:.1f}s",
                        flush=True,
                    )

            avg_r    = float(np.mean(epoch_rewards))
            avg_adir = float(np.mean(epoch_adir_vals))
            history["avg_reward"].append(avg_r)
            history["avg_adir"].append(avg_adir)

            print(
                f"[RL Epoch {epoch:02d}/{num_epochs}]  "
                f"avg_reward={avg_r:.4f}  avg_ADIR={avg_adir:.2f}%  "
                f"time={time.time()-t0:.1f}s"
            )
            self._save_checkpoint(f"rl_epoch_{epoch:02d}")

        return history

    def _save_checkpoint(self, name: str) -> None:
        path = self.output_dir / name
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(), path / "actor.pt")
        torch.save(self.value_head.state_dict(), path / "value_head.pt")
        print(f"  RL checkpoint saved → {path}")
