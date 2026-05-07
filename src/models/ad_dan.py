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
from transformers import (
    AutoModel,
    BartForConditionalGeneration,
    BartTokenizer,
    DebertaV2Model,
)

from .modules import PAA, DEM, CMIA, InconsistencyEstimator


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ADDANConfig:
    """Hyper-parameters for AD-DAN."""

    # Shared encoder: DeBERTa-v3-base
    encoder_model_name: str = "microsoft/deberta-v3-base"
    # Pre-trained emotion backbone for PAA
    emotion_model_name: str = "j-hartmann/emotion-english-distilroberta-base"
    # Generation model (BART, kept for RL decoding only)
    bart_model_name: str = "facebook/bart-base"
    max_seq_len: int = 512
    freeze_emotion_encoder_epochs: int = 5  # unfreeze after this many epochs

    # Memory optimisation flags
    gradient_checkpointing: bool = True   # saves ~60% activation memory at ~30% speed cost
    load_generation_model: bool = True    # set False during SFT to skip loading BART (~530MB)

    # PAA
    num_emotions: int = 32          # EmpatheticDialogues emotion categories
    paa_num_heads: int = 8
    paa_dropout: float = 0.1

    # DEM
    dem_num_heads: int = 8
    dem_dropout: float = 0.1

    # CMIA
    cmia_dropout: float = 0.1

    # Regularization
    weight_decay: float = 0.01
    emotion_label_smoothing: float = 0.1

    # Loss weights
    lambda_adi: float = 0.5         # weight on L_adi
    mu1: float = 0.1                # empathy score regression weight
    mu2: float = 0.1                # calibration weight
    mu3: float = 0.1                # ADI regression weight
    dec_pos_weight: float = 1.0     # positive class weight for BCE (set from data)
    emotion_class_weights: Optional[object] = None  # torch.Tensor of shape (num_emotions,)

    # Gradient accumulation (effective batch = batch_size * accum_steps)
    grad_accum_steps: int = 16

    # Reward weights (RL stage)
    reward_alpha: float = 0.35      # S_aff reward weight
    reward_beta: float = 0.85       # S_dec reward weight
    reward_gamma_adi: float = 0.80  # ADI penalty weight
    reward_adir_tau_bonus: float = 0.80  # ADIR-threshold penalty weight
    reward_tau_sharpness: float = 16.0   # sharpness of smooth threshold penalty
    reward_kl_eta: float = 0.02     # KL divergence penalty
    reward_margin_spread: float = 0.05  # discourages score-collapse in RL
    reward_direct_alpha: float = 1.5    # weight for eval-aligned reward on original response

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

        # ---- Shared encoder: DeBERTa-v3-base (768-dim) --------------------
        self.encoder_model = DebertaV2Model.from_pretrained(cfg.encoder_model_name)
        hidden_dim: int = self.encoder_model.config.hidden_size  # 768

        # ---- Pre-trained emotion backbone for PAA -------------------------
        # j-hartmann/emotion-english-distilroberta-base: DistilRoBERTa
        # fine-tuned on GoEmotions → strong emotion representations out-of-box.
        # We use its [CLS] hidden state (768-dim) as input to the PAA head.
        # Frozen for the first `freeze_emotion_encoder_epochs` epochs.
        self.emotion_encoder = AutoModel.from_pretrained(cfg.emotion_model_name)
        emotion_hidden_dim: int = self.emotion_encoder.config.hidden_size  # 768
        self._emotion_frozen: bool = True
        for p in self.emotion_encoder.parameters():
            p.requires_grad_(False)

        # ---- Full seq2seq model for generation (RL stage only) ------------
        # Skip loading during SFT to save ~530MB on the T4.
        if cfg.load_generation_model:
            self.seq2seq = BartForConditionalGeneration.from_pretrained(
                cfg.bart_model_name
            )
        else:
            self.seq2seq = None

        # ---- Gradient checkpointing (trade compute for activation memory) -
        if cfg.gradient_checkpointing:
            self.encoder_model.gradient_checkpointing_enable()
            # emotion_encoder is frozen but still forward-passes; enable too
            self.emotion_encoder.gradient_checkpointing_enable()

        # ---- Task-specific modules ----------------------------------------
        # PAA now receives emotion_hidden_dim from emotion_encoder, not hidden_dim
        self.paa = PAA(
            hidden_dim=hidden_dim,
            emotion_hidden_dim=emotion_hidden_dim,
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

    def unfreeze_emotion_encoder(self) -> None:
        """Unfreeze the emotion backbone (called by trainer after N epochs)."""
        if self._emotion_frozen:
            for p in self.emotion_encoder.parameters():
                p.requires_grad_(True)
            self._emotion_frozen = False
            print("  [AD-DAN] Emotion encoder unfrozen.")

    def encode(
        self,
        input_ids: torch.Tensor,        # (B, L)
        attention_mask: torch.Tensor,   # (B, L)
    ) -> torch.Tensor:
        """Run DeBERTa encoder and return hidden states (B, L, d)."""
        outputs = self.encoder_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return outputs.last_hidden_state  # (B, L, d)

    def encode_emotion(
        self,
        input_ids: torch.Tensor,        # (B, L)
        attention_mask: torch.Tensor,   # (B, L)
    ) -> torch.Tensor:
        """Run pre-trained emotion encoder and return [CLS] vector (B, d)."""
        outputs = self.emotion_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return outputs.last_hidden_state[:, 0, :]  # (B, d)

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
        # Context re-tokenized with the emotion encoder's own tokenizer
        # (RoBERTa vocab ~50K ≠ DeBERTa vocab ~128K — must be separate)
        emotion_input_ids: Optional[torch.Tensor] = None,       # (B, L_emo)
        emotion_attention_mask: Optional[torch.Tensor] = None,  # (B, L_emo)
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

        # ---- PAA (uses pre-trained emotion encoder for [CLS] features) ----
        # Use emotion_input_ids (RoBERTa-tokenized) if provided; skip if not.
        h_emo_cls = None
        if emotion_input_ids is not None and emotion_attention_mask is not None:
            h_emo_cls = self.encode_emotion(emotion_input_ids, emotion_attention_mask)
        h_aff, e_hat, S_aff = self.paa(H_ctx, context_attention_mask, h_emo_cls)

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
        e_hat: torch.Tensor,                          # (B, 32) raw logits
        S_aff: torch.Tensor,                          # (B,)
        emotion_labels: torch.Tensor,                 # (B,)  LongTensor
        aff_score_labels: Optional[torch.Tensor],     # (B,) or None
    ) -> torch.Tensor:
        """L_aff = CE(emotion) + μ1 * MSE(S_aff)"""
        # Class-weighted cross-entropy to handle 32-class imbalance
        weight = None
        if self.config.emotion_class_weights is not None:
            weight = self.config.emotion_class_weights.to(e_hat.device)
        L_ce = nn.functional.cross_entropy(
            e_hat,
            emotion_labels,
            weight=weight,
            label_smoothing=float(self.config.emotion_label_smoothing),
        )
        L_reg = torch.tensor(0.0, device=e_hat.device)
        if aff_score_labels is not None:
            with torch.amp.autocast(device_type="cuda", enabled=False):
                L_reg = nn.functional.mse_loss(S_aff.float(), aff_score_labels.float())
        return L_ce + self.config.mu1 * L_reg

    def _loss_dec(
        self,
        S_dec: torch.Tensor,           # (B,)
        dec_labels: Optional[torch.Tensor],  # (B,) binary float
    ) -> torch.Tensor:
        """L_dec = BCE(S_dec) + μ2 * ECE_proxy"""
        if dec_labels is None:
            return torch.tensor(0.0, device=S_dec.device)
        # Use pos_weight to handle class imbalance (set from training data)
        with torch.amp.autocast(device_type="cuda", enabled=False):
            S_dec_f = S_dec.float()
            dec_labels_f = dec_labels.float()
            eps = 1e-6
            logit = torch.logit(S_dec_f.clamp(eps, 1.0 - eps))
            pos_weight = float(self.config.dec_pos_weight)
            # BCE with logits (handles numerical stability internally)
            L_bce = nn.functional.binary_cross_entropy_with_logits(
                logit, dec_labels_f,
                pos_weight=torch.tensor([pos_weight], device=S_dec.device),
                reduction='mean'
            )
            ece_proxy = ((S_dec_f - dec_labels_f).abs() * (S_dec_f - 0.5).abs()).mean()
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
