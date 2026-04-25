"""
Evaluation metrics for AD-DAN:
  - ADIR  : ADI Rate (%) – fraction of samples with ADI > τ
  - ADIS  : mean ADI severity
  - compute_adi : element-wise ADI = max(0, S_aff - S_dec)
  - emotion_accuracy
  - empathy_pearson
  - decision_accuracy / precision
  - bleu4 / rouge_l
  - bootstrap_ci : 95% bootstrap confidence interval for a scalar metric
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, precision_score


# ---------------------------------------------------------------------------
# Core ADI metrics
# ---------------------------------------------------------------------------

def compute_adi(
    s_aff: np.ndarray | Sequence[float],
    s_dec: np.ndarray | Sequence[float],
) -> np.ndarray:
    """Element-wise ADI = max(0, S_aff - S_dec)."""
    s_aff = np.asarray(s_aff, dtype=float)
    s_dec = np.asarray(s_dec, dtype=float)
    return np.maximum(0.0, s_aff - s_dec)


def adir(
    s_aff: np.ndarray,
    s_dec: np.ndarray,
    tau: float = 0.5,
) -> float:
    """ADI Rate (%) — fraction of samples where ADI > τ."""
    adi = compute_adi(s_aff, s_dec)
    return float((adi > tau).mean() * 100.0)


def adis(
    s_aff: np.ndarray,
    s_dec: np.ndarray,
) -> float:
    """Mean ADI Severity."""
    return float(compute_adi(s_aff, s_dec).mean())


# ---------------------------------------------------------------------------
# Affective metrics
# ---------------------------------------------------------------------------

def emotion_accuracy(
    preds: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Top-1 emotion classification accuracy (%)."""
    return float(accuracy_score(labels, preds) * 100.0)


def empathy_pearson(
    pred_scores: np.ndarray,
    gold_scores: np.ndarray,
) -> float:
    """Pearson correlation between predicted and gold empathy scores."""
    r, _ = pearsonr(pred_scores, gold_scores)
    return float(r)


# ---------------------------------------------------------------------------
# Decision metrics
# ---------------------------------------------------------------------------

def decision_accuracy(
    preds: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Binary decision accuracy (%)."""
    return float(accuracy_score(labels, preds) * 100.0)


def decision_precision(
    preds: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Binary decision precision."""
    return float(precision_score(labels, preds, zero_division=0))


# ---------------------------------------------------------------------------
# Generation quality
# ---------------------------------------------------------------------------

def _ngrams(tokens: list[str], n: int) -> dict[tuple[str, ...], int]:
    ngrams: dict[tuple[str, ...], int] = {}
    for i in range(len(tokens) - n + 1):
        key = tuple(tokens[i : i + n])
        ngrams[key] = ngrams.get(key, 0) + 1
    return ngrams


def sentence_bleu4(hypothesis: str, reference: str) -> float:
    """Corpus-level BLEU-4 approximation for a single sentence pair."""
    hyp_toks = hypothesis.lower().split()
    ref_toks = reference.lower().split()
    if len(hyp_toks) == 0:
        return 0.0

    # Brevity penalty
    bp = math.exp(min(0.0, 1.0 - len(ref_toks) / max(len(hyp_toks), 1)))

    precisions = []
    for n in range(1, 5):
        hyp_ng = _ngrams(hyp_toks, n)
        ref_ng = _ngrams(ref_toks, n)
        clipped = sum(min(c, ref_ng.get(ng, 0)) for ng, c in hyp_ng.items())
        total   = max(sum(hyp_ng.values()), 1)
        precisions.append(clipped / total if total > 0 else 0.0)

    if any(p == 0 for p in precisions):
        return 0.0
    log_avg = sum(math.log(p) for p in precisions) / 4.0
    return float(bp * math.exp(log_avg))


def rouge_l(hypothesis: str, reference: str) -> float:
    """Sentence-level ROUGE-L (F1)."""
    hyp = hypothesis.lower().split()
    ref = reference.lower().split()
    if not hyp or not ref:
        return 0.0

    # LCS length via DP
    m, n = len(ref), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i-1][j-1] + 1 if ref[i-1] == hyp[j-1] else max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]

    precision = lcs / n if n > 0 else 0.0
    recall    = lcs / m if m > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def corpus_bleu4(hypotheses: list[str], references: list[str]) -> float:
    """Macro-averaged BLEU-4 over the corpus."""
    scores = [sentence_bleu4(h, r) for h, r in zip(hypotheses, references)]
    return float(np.mean(scores))


def corpus_rouge_l(hypotheses: list[str], references: list[str]) -> float:
    """Macro-averaged ROUGE-L over the corpus."""
    scores = [rouge_l(h, r) for h, r in zip(hypotheses, references)]
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_ci(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    preds: np.ndarray,
    labels: np.ndarray,
    n_bootstrap: int = 10_000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Bootstrap 95% CI for a scalar metric.

    Parameters
    ----------
    metric_fn : function(preds, labels) -> float
    preds     : model predictions
    labels    : ground truth
    n_bootstrap : number of resamples
    ci        : confidence level (default 0.95)

    Returns
    -------
    (lower, upper) confidence interval bounds
    """
    rng = np.random.default_rng(seed)
    n = len(preds)
    boot_scores = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_scores[i] = metric_fn(preds[idx], labels[idx])

    alpha = (1.0 - ci) / 2.0
    lower = float(np.quantile(boot_scores, alpha))
    upper = float(np.quantile(boot_scores, 1.0 - alpha))
    return lower, upper


# ---------------------------------------------------------------------------
# All-in-one evaluation summary
# ---------------------------------------------------------------------------

def evaluate_all(
    s_aff_pred: np.ndarray,
    s_dec_pred: np.ndarray,
    emotion_preds: np.ndarray,
    emotion_labels: np.ndarray,
    s_aff_gold: np.ndarray,
    dec_preds: np.ndarray,
    dec_labels: np.ndarray,
    hypotheses: list[str],
    references: list[str],
    tau: float = 0.5,
) -> dict[str, float]:
    """
    Compute all paper metrics in one call.

    Returns
    -------
    dict with keys: ADIR, ADIS, EmotionAcc, EmpS, DecAcc, DecPrec,
                    BLEU4, ROUGE_L
    """
    return {
        "ADIR":       adir(s_aff_pred, s_dec_pred, tau=tau),
        "ADIS":       adis(s_aff_pred, s_dec_pred),
        "EmotionAcc": emotion_accuracy(emotion_preds, emotion_labels),
        "EmpS":       empathy_pearson(s_aff_pred, s_aff_gold),
        "DecAcc":     decision_accuracy(dec_preds, dec_labels),
        "DecPrec":    decision_precision(dec_preds, dec_labels),
        "BLEU4":      corpus_bleu4(hypotheses, references),
        "ROUGE_L":    corpus_rouge_l(hypotheses, references),
    }
