"""
Sub-modules of AD-DAN:
  - PAA  : Parallel Affective Alignment module
  - DEM  : Decision Evaluation Module
  - CMIA : Cross-Modal Inconsistency Attention
  - InconsistencyEstimator : predicts ADI score from CMIA output
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """Two-layer MLP with LayerNorm and GELU activation."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.net(x))


# ---------------------------------------------------------------------------
# PAA – Parallel Affective Alignment Module
# ---------------------------------------------------------------------------

class PAA(nn.Module):
    """
    Parallel Affective Alignment module.

    Steps
    -----
    1. Emotion-detection MHA over DeBERTa encoder hidden states H ∈ ℝ^{L×d}
    2. Fuse with pre-trained emotion encoder [CLS] (h_emo_cls) via projection
    3. 32-way softmax emotion classifier over fused representation
    4. Two-layer MLP fuses emotion logits with context → h_aff
    5. Linear sigmoid head → S_aff ∈ [0, 1]

    Parameters
    ----------
    hidden_dim       : DeBERTa encoder hidden size (768)
    emotion_hidden_dim: pre-trained emotion encoder hidden size (768)
    num_emotions     : number of emotion categories (32 for EmpatheticDialogues)
    num_heads        : heads for emotion MHA  (paper: 8, d_k = 96)
    dropout          : dropout probability
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        emotion_hidden_dim: int = 768,
        num_emotions: int = 32,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Multi-head self-attention for emotion detection on DeBERTa states
        self.emotion_mha = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.emotion_norm = nn.LayerNorm(hidden_dim)

        # Project pre-trained emotion encoder output into shared hidden_dim
        self.emo_proj = nn.Linear(emotion_hidden_dim, hidden_dim)
        self.emo_fusion_norm = nn.LayerNorm(hidden_dim)

        # 32-way emotion classifier (operates on fused [CLS] token)
        self.emotion_head = nn.Linear(hidden_dim, num_emotions)

        # Affective alignment estimator MLP
        # Input: concat(fused [CLS], emotion logits)
        self.alignment_mlp = MLP(
            in_dim=hidden_dim + num_emotions,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            dropout=dropout,
        )

        # Scalar affective score head
        self.score_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        H: torch.Tensor,                        # (B, L, d)  DeBERTa hidden states
        attention_mask: torch.Tensor | None = None,  # (B, L)
        h_emo_cls: torch.Tensor | None = None,  # (B, emotion_hidden_dim) pre-trained [CLS]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        h_aff : (B, d)   affective hidden state
        e_hat : (B, 32)  emotion probability distribution
        S_aff : (B,)     affective alignment score ∈ [0, 1]
        """
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0  # (B, L)

        H_emo, _ = self.emotion_mha(H, H, H, key_padding_mask=key_padding_mask)
        H_emo = self.emotion_norm(H_emo + H)          # residual

        cls_emo = H_emo[:, 0, :]                      # (B, d) [CLS] from DeBERTa

        # Fuse with pre-trained emotion encoder output if provided
        if h_emo_cls is not None:
            cls_emo = self.emo_fusion_norm(cls_emo + self.emo_proj(h_emo_cls))

        # Keep raw logits for CE loss; use probabilities only for feature fusion.
        e_logits = self.emotion_head(cls_emo)                    # (B, 32)
        e_probs = F.softmax(e_logits, dim=-1)                    # (B, 32)

        # Fuse emotion logits with fused context
        h_aff_in = torch.cat([cls_emo, e_probs], dim=-1)        # (B, d+32)
        h_aff = self.alignment_mlp(h_aff_in)                    # (B, d)

        S_aff = torch.sigmoid(self.score_head(h_aff)).squeeze(-1)  # (B,)
        return h_aff, e_logits, S_aff


# ---------------------------------------------------------------------------
# DEM – Decision Evaluation Module
# ---------------------------------------------------------------------------

class DEM(nn.Module):
    """
    Decision Evaluation Module.

    Steps
    -----
    1. Cross-attention: H_response attends to H_user → H_sit
    2. Two-layer MLP fuses H_sit[CLS] with global context → h_dec
    3. Linear sigmoid head → S_dec ∈ [0, 1]

    Parameters
    ----------
    hidden_dim : encoder hidden size
    num_heads  : heads for cross-attention
    dropout    : dropout probability
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Cross-attention: query = response tokens, key/value = user tokens
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(hidden_dim)

        # Decision quality MLP
        # Input: concat(H_sit[CLS], global context h_0)
        self.decision_mlp = MLP(
            in_dim=hidden_dim * 2,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            dropout=dropout,
        )

        # Scalar decision score head
        self.score_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        H_response: torch.Tensor,              # (B, L_r, d)
        H_user: torch.Tensor,                  # (B, L_u, d)
        h_global: torch.Tensor,                # (B, d)  [CLS] global context
        user_mask: torch.Tensor | None = None, # (B, L_u) HF mask
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        h_dec : (B, d)  decision hidden state
        S_dec : (B,)    decision correctness score ∈ [0, 1]
        """
        key_padding_mask = None
        if user_mask is not None:
            key_padding_mask = user_mask == 0  # (B, L_u)

        H_sit, _ = self.cross_attn(
            H_response, H_user, H_user,
            key_padding_mask=key_padding_mask,
        )
        H_sit = self.cross_norm(H_sit + H_response)   # residual (B, L_r, d)

        sit_cls = H_sit[:, 0, :]                       # (B, d)
        h_dec_in = torch.cat([sit_cls, h_global], dim=-1)  # (B, 2d)
        h_dec = self.decision_mlp(h_dec_in)            # (B, d)

        S_dec = torch.sigmoid(self.score_head(h_dec)).squeeze(-1)  # (B,)
        return h_dec, S_dec


# ---------------------------------------------------------------------------
# CMIA – Cross-Modal Inconsistency Attention
# ---------------------------------------------------------------------------

class CMIA(nn.Module):
    """
    Cross-Modal Inconsistency Attention.

    Creates bidirectional attention pathways between h_aff and h_dec:
      c_{a→d} = Attn(h_aff, h_dec, h_dec)
      c_{d→a} = Attn(h_dec, h_aff, h_aff)
    Then fuses all four vectors with LayerNorm → h_cmia.

    The bidirectional gradient flow is the architectural key:
    decision gradients regularise the affective module and vice versa.

    Parameters
    ----------
    hidden_dim : hidden size (must match PAA and DEM output)
    dropout    : dropout probability
    """

    def __init__(self, hidden_dim: int = 768, dropout: float = 0.1):
        super().__init__()
        self.scale = math.sqrt(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Projection matrices for both attention directions
        self.W_q_atd = nn.Linear(hidden_dim, hidden_dim)
        self.W_k_atd = nn.Linear(hidden_dim, hidden_dim)
        self.W_v_atd = nn.Linear(hidden_dim, hidden_dim)

        self.W_q_dta = nn.Linear(hidden_dim, hidden_dim)
        self.W_k_dta = nn.Linear(hidden_dim, hidden_dim)
        self.W_v_dta = nn.Linear(hidden_dim, hidden_dim)

        self.fusion_norm = nn.LayerNorm(hidden_dim)

    def _scaled_dot_attn(
        self,
        Q: torch.Tensor,  # (B, d)
        K: torch.Tensor,  # (B, d)
        V: torch.Tensor,  # (B, d)
    ) -> torch.Tensor:    # (B, d)
        """Single-vector (token-level) scaled dot-product attention."""
        # Treat each vector as a single-token sequence for Attn(Q, K, V)
        score = (Q * K).sum(dim=-1, keepdim=True) / self.scale  # (B, 1)
        weight = torch.sigmoid(score)                            # soft gate
        return self.dropout(weight * V)

    def forward(
        self,
        h_aff: torch.Tensor,  # (B, d)
        h_dec: torch.Tensor,  # (B, d)
    ) -> torch.Tensor:        # (B, d)  h_cmia
        """
        Returns
        -------
        h_cmia : (B, d)  inconsistency-aware fused representation
        """
        # c_{a→d}: affective queries decision
        c_atd = self._scaled_dot_attn(
            self.W_q_atd(h_aff),
            self.W_k_atd(h_dec),
            self.W_v_atd(h_dec),
        )

        # c_{d→a}: decision queries affective
        c_dta = self._scaled_dot_attn(
            self.W_q_dta(h_dec),
            self.W_k_dta(h_aff),
            self.W_v_dta(h_aff),
        )

        # Inconsistency-aware fusion with residual
        h_cmia = self.fusion_norm(h_aff + h_dec + c_atd + c_dta)  # (B, d)
        return h_cmia


# ---------------------------------------------------------------------------
# InconsistencyEstimator
# ---------------------------------------------------------------------------

class InconsistencyEstimator(nn.Module):
    """
    Predicts the ADI score from h_cmia:
      ADI_hat = ReLU(w_adi^T h_cmia + b_adi)

    Also has a small calibration head to ensure predicted ADI
    tracks ground-truth ADI (used by the regression term in L_adi).
    """

    def __init__(self, hidden_dim: int = 768):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, 1)

    def forward(self, h_cmia: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        h_cmia : (B, d)

        Returns
        -------
        adi_hat : (B,)  predicted ADI score ≥ 0
        """
        return F.relu(self.proj(h_cmia)).squeeze(-1)  # (B,)
