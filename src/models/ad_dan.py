"""
AD-DAN: Affect--Decision Dual Alignment Network
================================================
Full model wrapping the BART-base shared encoder with PAA, DEM, CMIA,
and the InconsistencyEstimator.

The model can be used in two modes:
  1. Supervised fine-tuning (SFT): forward() returns all scores and losses.
  2. RL stage: generate() produces a response token-by-token using the
     BART decoder, and score() computes scalar rewards for a completed
     response string.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import torch
import torch.nn as nn
from transformers import BartModel, BartForConditionalGeneration, BartTokenizer

from .modules import PAA, DEM, CMIA, InconsistencyEstimator


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ADDANConfig:
    """Hyper-parameters for AD-DAN (defaults match the paper)."""

    # Encoder
    bart_model_name: str = "facebook/bart-base"
    max_seq_len: int = 512

    # PAA
    num_emotions: int = 32          # EmpatheticDialogues emotion categories
    paa_num_heads: int = 8
    paa_dropout: float = 0.1

    # DEM
    dem_num_heads: int = 8
    dem_dropout: float = 0.1

    # CMIA
    cmia_dropout: float = 0.1

    # Loss weights
    lambda_adi: float = 0.5         # weight on L_adi
    mu1: float = 0.1                # empathy score regression weight
    mu2: float = 0.1                # calibration weight
    mu3: float = 0.1                # ADI regression weight

    # Reward weights (RL stage)
    reward_alpha: float = 0.35      # S_aff weight
    reward_beta: float = 0.45       # S_dec weight
    reward_gamma_adi: float = 0.20  # ADI penalty weight
    reward_kl_eta: float = 0.02     # KL divergence penalty

    # ADI threshold for ADIR computation
    tau: float = 0.5


# ---------------------------------------------------------------------------
# AD-DAN
# ---------------------------------------------------------------------------

class ADDAN(nn.Module):
    """
    Affect--Decision Dual Alignment Network.

    Architecture
    ------------
    Input context (dialogue history + current user turn) →
      BART encoder → H ∈ ℝ^{L×768}
        ├── PAA  → h_aff, e_hat, S_aff
        ├── DEM  → h_dec, S_dec
        └── CMIA(h_aff, h_dec) → h_cmia → ADI_hat

    During SFT training the full model is optimized end-to-end with
    L = L_aff + L_dec + λ·L_adi.

    For generation (RL stage) the underlying BartForConditionalGeneration
    decoder is used.
    """

    def __init__(self, config: ADDANConfig | None = None):
        super().__init__()
        self.config = config or ADDANConfig()
        cfg = self.config

        # ---- Shared encoder (BART-base encoder only for scoring) ----------
        self.encoder_model = BartModel.from_pretrained(cfg.bart_model_name)
        hidden_dim: int = self.encoder_model.config.d_model  # 768

        # ---- Full seq2seq model for generation ----------------------------
        self.seq2seq = BartForConditionalGeneration.from_pretrained(
            cfg.bart_model_name
        )

        # Share the encoder weights between scorer and generator
        self.encoder_model.encoder = self.seq2seq.model.encoder

        # ---- Task-specific modules ----------------------------------------
        self.paa = PAA(
            hidden_dim=hidden_dim,
            num_emotions=cfg.num_emotions,
            num_heads=cfg.paa_num_heads,
            dropout=cfg.paa_dropout,
        )
        self.dem = DEM(
            hidden_dim=hidden_dim,
            num_heads=cfg.dem_num_heads,
            dropout=cfg.dem_dropout,
        )
        self.cmia = CMIA(hidden_dim=hidden_dim, dropout=cfg.cmia_dropout)
        self.ie = InconsistencyEstimator(hidden_dim=hidden_dim)

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    def encode(
        self,
        input_ids: torch.Tensor,        # (B, L)
        attention_mask: torch.Tensor,   # (B, L)
    ) -> torch.Tensor:
        """Run BART encoder and return hidden states (B, L, d)."""
        outputs = self.seq2seq.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return outputs.last_hidden_state  # (B, L, d)

    # ------------------------------------------------------------------
    # Forward – SFT scoring pass
    # ------------------------------------------------------------------

    def forward(
        self,
        # Context (dialogue history + user turn), tokenized together
        context_input_ids: torch.Tensor,        # (B, L_ctx)
        context_attention_mask: torch.Tensor,   # (B, L_ctx)
        # User utterance alone (for DEM cross-attention)
        user_input_ids: torch.Tensor,           # (B, L_u)
        user_attention_mask: torch.Tensor,      # (B, L_u)
        # Response to be scored, tokenized separately
        response_input_ids: torch.Tensor,       # (B, L_r)
        response_attention_mask: torch.Tensor,  # (B, L_r)
        # Ground-truth labels (optional, for training)
        emotion_labels: Optional[torch.Tensor] = None,   # (B,) LongTensor
        aff_score_labels: Optional[torch.Tensor] = None, # (B,) float ∈ [0,1]
        dec_labels: Optional[torch.Tensor] = None,       # (B,) binary float
        adi_labels: Optional[torch.Tensor] = None,       # (B,) float ≥ 0
    ) -> dict:
        """
        Returns a dict with:
          h_aff, e_hat, S_aff, h_dec, S_dec, h_cmia, adi_hat
          (and total_loss + individual losses when ground-truth labels provided)
        """
        # ---- Encode context and responses --------------------------------
        H_ctx  = self.encode(context_input_ids,  context_attention_mask)  # (B,L,d)
        H_resp = self.encode(response_input_ids, response_attention_mask)  # (B,L_r,d)
        H_user = self.encode(user_input_ids,     user_attention_mask)      # (B,L_u,d)

        h_global = H_ctx[:, 0, :]  # [CLS] global context vector (B, d)

        # ---- PAA ----------------------------------------------------------
        h_aff, e_hat, S_aff = self.paa(H_ctx, context_attention_mask)

        # ---- DEM ----------------------------------------------------------
        h_dec, S_dec = self.dem(H_resp, H_user, h_global, user_attention_mask)

        # ---- CMIA + IE ----------------------------------------------------
        h_cmia  = self.cmia(h_aff, h_dec)
        adi_hat = self.ie(h_cmia)

        output = dict(
            h_aff=h_aff, e_hat=e_hat, S_aff=S_aff,
            h_dec=h_dec, S_dec=S_dec,
            h_cmia=h_cmia, adi_hat=adi_hat,
        )

        # ---- Loss computation (training only) ----------------------------
        if emotion_labels is not None:
            cfg = self.config
            L_aff = self._loss_aff(e_hat, S_aff, emotion_labels, aff_score_labels)
            L_dec = self._loss_dec(S_dec, dec_labels)
            L_adi = self._loss_adi(adi_hat, adi_labels)
            total_loss = L_aff + L_dec + cfg.lambda_adi * L_adi
            output.update(
                total_loss=total_loss,
                loss_aff=L_aff,
                loss_dec=L_dec,
                loss_adi=L_adi,
            )

        return output

    # ------------------------------------------------------------------
    # Individual loss terms
    # ------------------------------------------------------------------

    def _loss_aff(
        self,
        e_hat: torch.Tensor,                          # (B, 32)
        S_aff: torch.Tensor,                          # (B,)
        emotion_labels: torch.Tensor,                 # (B,)  LongTensor
        aff_score_labels: Optional[torch.Tensor],     # (B,) or None
    ) -> torch.Tensor:
        """L_aff = CE(emotion) + μ1 * MSE(S_aff)"""
        L_ce = nn.functional.cross_entropy(e_hat, emotion_labels)
        L_reg = torch.tensor(0.0, device=e_hat.device)
        if aff_score_labels is not None:
            L_reg = nn.functional.mse_loss(S_aff, aff_score_labels)
        return L_ce + self.config.mu1 * L_reg

    def _loss_dec(
        self,
        S_dec: torch.Tensor,           # (B,)
        dec_labels: Optional[torch.Tensor],  # (B,) binary float
    ) -> torch.Tensor:
        """L_dec = BCE(S_dec) + μ2 * ECE_proxy"""
        if dec_labels is None:
            return torch.tensor(0.0, device=S_dec.device)
        L_bce = nn.functional.binary_cross_entropy(S_dec, dec_labels)
        # ECE proxy: penalise overconfident predictions away from 0.5
        # (true ECE requires binning; this differentiable proxy approximates it)
        ece_proxy = ((S_dec - dec_labels).abs() * (S_dec - 0.5).abs()).mean()
        return L_bce + self.config.mu2 * ece_proxy

    def _loss_adi(
        self,
        adi_hat: torch.Tensor,                       # (B,)
        adi_labels: Optional[torch.Tensor],          # (B,) or None
    ) -> torch.Tensor:
        """L_adi = E[adi_hat] + μ3 * MSE(adi_hat - adi*)"""
        L_exp = adi_hat.mean()
        L_reg = torch.tensor(0.0, device=adi_hat.device)
        if adi_labels is not None:
            L_reg = nn.functional.mse_loss(adi_hat, adi_labels)
        return L_exp + self.config.mu3 * L_reg

    # ------------------------------------------------------------------
    # Scoring a (context, response) pair → (S_aff, S_dec, ADI_hat)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def score(
        self,
        context_input_ids: torch.Tensor,
        context_attention_mask: torch.Tensor,
        user_input_ids: torch.Tensor,
        user_attention_mask: torch.Tensor,
        response_input_ids: torch.Tensor,
        response_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns S_aff (B,), S_dec (B,), adi_hat (B,)  – no gradients.
        Used by the RL reward function.
        """
        out = self.forward(
            context_input_ids, context_attention_mask,
            user_input_ids, user_attention_mask,
            response_input_ids, response_attention_mask,
        )
        return out["S_aff"], out["S_dec"], out["adi_hat"]

    # ------------------------------------------------------------------
    # Text generation (RL stage)
    # ------------------------------------------------------------------

    def generate(
        self,
        context_input_ids: torch.Tensor,
        context_attention_mask: torch.Tensor,
        max_new_tokens: int = 128,
        num_beams: int = 1,
        do_sample: bool = True,
        top_p: float = 0.9,
        temperature: float = 1.0,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate a response using the BART decoder.

        Returns
        -------
        generated_ids : (B, T)  token ids
        """
        return self.seq2seq.generate(
            input_ids=context_input_ids,
            attention_mask=context_attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Token-level log-probabilities (used for PPO ratio computation)
    # ------------------------------------------------------------------

    def log_probs_from_ids(
        self,
        context_input_ids: torch.Tensor,
        context_attention_mask: torch.Tensor,
        response_input_ids: torch.Tensor,
        response_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute per-token log-probabilities of response_input_ids under
        the current policy (BartForConditionalGeneration).

        Returns
        -------
        log_probs : (B, T-1)  log P(token_t | token_{<t}, context)
        """
        decoder_input_ids = response_input_ids[:, :-1].contiguous()
        labels            = response_input_ids[:, 1:].contiguous()

        outputs = self.seq2seq(
            input_ids=context_input_ids,
            attention_mask=context_attention_mask,
            decoder_input_ids=decoder_input_ids,
        )
        logits = outputs.logits  # (B, T-1, vocab)
        log_probs = nn.functional.log_softmax(logits, dim=-1)

        # Gather log prob of each gold token
        token_log_probs = log_probs.gather(
            dim=-1, index=labels.unsqueeze(-1)
        ).squeeze(-1)  # (B, T-1)

        # Mask padding
        pad_id = self.seq2seq.config.pad_token_id
        token_mask = labels != pad_id
        return token_log_probs * token_mask
