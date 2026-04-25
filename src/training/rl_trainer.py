"""
PPO RL Trainer for AD-DAN
==========================
Implements the Proximal Policy Optimization stage described in the paper.

Reward function (terminal, per full response):
    R(u_T, r_T) = α·S_aff + β·S_dec − γ_adi·ADI
    R_total     = R − η·KL[π_θ(·|u_T) ‖ π_SFT(·|u_T)]

with α=0.35, β=0.45, γ_adi=0.20, η=0.02.

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

from ..models.ad_dan import ADDAN, ADDANConfig
from ..utils.metrics import adir, adis


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

        # Frozen reference policy (π_SFT)
        self.ref_policy: ADDAN = copy.deepcopy(actor)
        self.ref_policy.to(self.device)
        for p in self.ref_policy.parameters():
            p.requires_grad_(False)
        self.ref_policy.eval()

        # Value head
        hidden_dim = actor.seq2seq.config.d_model
        self.value_head = ValueHead(hidden_dim).to(self.device)

        # Optimizer covers actor + value head (ref policy is frozen)
        self.optimizer = AdamW(
            list(actor.parameters()) + list(self.value_head.parameters()),
            lr=lr,
        )
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=fp16 and self.device.type == "cuda"
        )
        self.fp16 = fp16

    # ------------------------------------------------------------------
    # Reward computation
    # ------------------------------------------------------------------

    def compute_reward(
        self,
        context_input_ids: torch.Tensor,
        context_attention_mask: torch.Tensor,
        user_input_ids: torch.Tensor,
        user_attention_mask: torch.Tensor,
        gen_input_ids: torch.Tensor,
        gen_attention_mask: torch.Tensor,
        ref_log_probs: torch.Tensor,   # (B, T-1)  from π_SFT
        act_log_probs: torch.Tensor,   # (B, T-1)  from π_θ
    ) -> torch.Tensor:                 # (B,)  scalar reward per sample
        """
        R_total = α·S_aff + β·S_dec − γ_adi·ADI − η·KL
        """
        cfg = self.config

        # Score generated responses
        with torch.no_grad():
            s_aff, s_dec, adi_hat = self.actor.score(
                context_input_ids, context_attention_mask,
                user_input_ids, user_attention_mask,
                gen_input_ids, gen_attention_mask,
            )

        # Task reward
        task_reward = (
            cfg.reward_alpha * s_aff
            + cfg.reward_beta * s_dec
            - cfg.reward_gamma_adi * adi_hat
        )

        # KL penalty: KL[π_θ ‖ π_SFT] = Σ_t (log π_θ - log π_SFT)
        # (token-level mask already applied in log_probs_from_ids)
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

            for batch in self.train_dl:
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                # ---- Rollout: generate responses -------------------------
                with torch.no_grad():
                    gen_ids = self.actor.generate(
                        context_input_ids=batch["context_input_ids"],
                        context_attention_mask=batch["context_attention_mask"],
                        max_new_tokens=self.max_new_tokens,
                        do_sample=True,
                        top_p=0.9,
                        temperature=1.0,
                    )
                    # Build attention mask for generated ids
                    pad_id  = self.actor.seq2seq.config.pad_token_id
                    gen_mask = (gen_ids != pad_id).long()

                    # Log-probs from both policies
                    ref_lp = self.ref_policy.log_probs_from_ids(
                        batch["context_input_ids"],
                        batch["context_attention_mask"],
                        gen_ids, gen_mask,
                    )

                # Actor log-probs (requires grad for PPO)
                act_lp = self.actor.log_probs_from_ids(
                    batch["context_input_ids"],
                    batch["context_attention_mask"],
                    gen_ids, gen_mask,
                )

                # Rewards (no grad needed through reward)
                with torch.no_grad():
                    rewards = self.compute_reward(
                        batch["context_input_ids"],
                        batch["context_attention_mask"],
                        batch["user_input_ids"],
                        batch["user_attention_mask"],
                        gen_ids, gen_mask,
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

                # Track ADIR for logging
                with torch.no_grad():
                    s_aff, s_dec, _ = self.actor.score(
                        batch["context_input_ids"],
                        batch["context_attention_mask"],
                        batch["user_input_ids"],
                        batch["user_attention_mask"],
                        gen_ids, gen_mask,
                    )
                epoch_adir_vals.append(
                    adir(s_aff.cpu().numpy(), s_dec.cpu().numpy())
                )

                # ---- PPO updates -----------------------------------------
                old_log_probs = act_lp.detach().sum(dim=-1)  # (B,) sum over tokens

                for _ in range(self.ppo_epochs):
                    with torch.cuda.amp.autocast(
                        enabled=self.fp16 and self.device.type == "cuda"
                    ):
                        new_lp = self.actor.log_probs_from_ids(
                            batch["context_input_ids"],
                            batch["context_attention_mask"],
                            gen_ids, gen_mask,
                        )
                        new_log_probs = new_lp.sum(dim=-1)  # (B,)

                        ratio = torch.exp(new_log_probs - old_log_probs)
                        surr1 = ratio * advantages
                        surr2 = torch.clamp(
                            ratio, 1.0 - self.ppo_clip, 1.0 + self.ppo_clip
                        ) * advantages
                        policy_loss = -torch.min(surr1, surr2).mean()

                        # Value loss
                        ctx_h_new = self.actor.encode(
                            batch["context_input_ids"],
                            batch["context_attention_mask"],
                        )[:, 0, :]
                        v_new = self.value_head(ctx_h_new)
                        value_loss = F.mse_loss(v_new, returns)

                        loss = policy_loss + 0.5 * value_loss

                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    nn.utils.clip_grad_norm_(
                        list(self.actor.parameters()) +
                        list(self.value_head.parameters()),
                        max_norm=1.0,
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

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
