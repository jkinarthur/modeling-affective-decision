"""
TAD-Bench 10K Builder
=====================
Builds a ~10,000-instance TAD-Bench dataset by:

  1. Loading EmpatheticDialogues (Rashkin et al., 2019) from HuggingFace
     — ~8,000 conversation turns across 32 emotion categories
  2. Loading ESConv (Liu et al., 2021) from HuggingFace
     — ~2,000 conversation turns from mental-health support dialogues
  3. Applying rule-based pseudo-labeling for aff_score, dec_label, adi_score
  4. Splitting 70/15/15 and saving CSVs to --output_dir

Pseudo-label strategy
---------------------
  aff_score : Affective alignment.  Estimated from emotion distress level
              (mapped from the ED 32-class label) plus response empathy
              density.  Quantised to {0.0, 0.25, 0.5, 0.75, 1.0}.
  dec_label : Decision appropriateness.  Binary 1 if the response contains
              actionable / constructive guidance markers without harmful
              language, else 0.
  adi_score : max(0, aff_score - dec_label) — affective–decision
              inconsistency (the paper's target signal).

Usage
-----
    python -m src.data.build_tad_bench --output_dir data/

Output
------
    data/tad_bench_train.csv
    data/tad_bench_val.csv
    data/tad_bench_test.csv
    data/tad_bench_all.csv
"""

from __future__ import annotations

import argparse
import os
import random
import re
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Emotion taxonomy
# ---------------------------------------------------------------------------

ED_EMOTIONS = [
    "afraid", "angry", "annoyed", "anticipating", "anxious",
    "apprehensive", "ashamed", "caring", "confident", "content",
    "devastated", "disappointed", "disgusted", "embarrassed", "excited",
    "faithful", "furious", "grateful", "guilty", "hopeful",
    "impressed", "jealous", "joyful", "lonely", "nostalgic",
    "prepared", "proud", "sad", "sentimental", "surprised",
    "terrified", "trusting",
]
EMOTION2ID = {e: i for i, e in enumerate(ED_EMOTIONS)}

# Emotion → base affective-alignment score (how much empathy is warranted)
_HIGH_DISTRESS = {
    "afraid", "angry", "anxious", "apprehensive", "devastated",
    "disgusted", "furious", "guilty", "jealous", "lonely", "sad",
    "terrified", "embarrassed",
}
_MED_DISTRESS = {"annoyed", "ashamed", "disappointed", "sentimental"}
_LOW_DISTRESS = {
    "anticipating", "caring", "confident", "content", "excited", "faithful",
    "grateful", "hopeful", "impressed", "joyful", "nostalgic", "prepared",
    "proud", "surprised", "trusting",
}

def _emotion_base_aff(emotion: str) -> float:
    emotion = emotion.lower().strip()
    if emotion in _HIGH_DISTRESS:
        return 0.75
    if emotion in _MED_DISTRESS:
        return 0.50
    return 0.25

# ---------------------------------------------------------------------------
# Lexical markers for pseudo-labeling
# ---------------------------------------------------------------------------

_EMPATHY_RE = re.compile(
    r"\b(i understand|that must|i can imagine|i hear you|sounds like|"
    r"that('?s| is) (difficult|hard|tough|overwhelming)|must be hard|"
    r"i('?m| am) sorry|i feel for you|i can see why|that sounds|"
    r"you('?re| are) not alone|it makes sense)\b",
    re.IGNORECASE,
)

_ACTION_RE = re.compile(
    r"\b(you might|you could|consider|try to|i suggest|i recommend|"
    r"speak to|talk to|consult|important to|it would help|have you tried|"
    r"you may want|one option|another option|it('?s| is) worth|"
    r"perhaps you|why not|what if you)\b",
    re.IGNORECASE,
)

_HARMFUL_RE = re.compile(
    r"\b(just do it|don'?t worry about it|it'?s nothing|that'?s silly|"
    r"you'?re (wrong|overreacting|too sensitive)|stop being|"
    r"just get over|it'?s not a big deal|who cares)\b",
    re.IGNORECASE,
)


def _quantise(v: float) -> float:
    """Snap a [0,1] float to the nearest element of {0, 0.25, 0.5, 0.75, 1.0}."""
    grid = [0.0, 0.25, 0.5, 0.75, 1.0]
    return min(grid, key=lambda x: abs(x - v))


def pseudo_label(
    emotion: str,
    response: str,
    rng: random.Random,
    noise: float = 0.08,
) -> tuple[float, float, float]:
    """
    Compute (aff_score, dec_label, adi_score) heuristically.

    Parameters
    ----------
    emotion  : EmpatheticDialogues emotion string
    response : listener / system response text
    rng      : seeded RNG for controlled noise
    noise    : std-dev of Gaussian noise added before quantisation
    """
    resp = response.strip()
    resp_len = len(resp)

    # ---- affective alignment ------------------------------------------------
    base_aff = _emotion_base_aff(emotion)

    if _EMPATHY_RE.search(resp):
        base_aff += 0.25
    if _HARMFUL_RE.search(resp):
        base_aff -= 0.50
    if resp_len < 40:
        base_aff -= 0.25
    elif resp_len > 200:
        base_aff += 0.10

    # controlled noise so labels aren't deterministically identical
    base_aff += rng.gauss(0, noise)
    aff_score = _quantise(float(np.clip(base_aff, 0.0, 1.0)))

    # ---- decision appropriateness -------------------------------------------
    has_action  = bool(_ACTION_RE.search(resp))
    has_empathy = bool(_EMPATHY_RE.search(resp))
    has_harmful = bool(_HARMFUL_RE.search(resp))

    if has_harmful or resp_len < 30:
        dec_label = 0.0
    elif has_action:
        dec_label = 1.0
    elif has_empathy and resp_len >= 60:
        dec_label = 1.0
    else:
        # borderline — conservatively label as not decision-appropriate
        dec_label = 0.0

    # ---- ADI ----------------------------------------------------------------
    adi_score = float(max(0.0, aff_score - dec_label))

    return aff_score, dec_label, adi_score


# ---------------------------------------------------------------------------
# EmpatheticDialogues builder
# ---------------------------------------------------------------------------

def _build_from_empathetic_dialogues(
    target_n: int,
    rng: random.Random,
) -> list[dict]:
    """
    Extract (context, user_utterance, response) triples from
    EmpatheticDialogues and attach pseudo-labels.
    """
    from datasets import load_dataset  # lazy import

    print("Loading EmpatheticDialogues from HuggingFace…")
    raw = load_dataset("empathetic_dialogues", trust_remote_code=True)

    # Merge all splits so we can sample freely
    all_rows: list[dict] = []
    for split in ("train", "validation", "test"):
        all_rows.extend(raw[split])

    # Group by conv_id
    convs: dict[str, list[dict]] = {}
    for row in all_rows:
        cid = row["conv_id"]
        convs.setdefault(cid, []).append(row)

    # Sort each conversation by utterance_idx
    for cid in convs:
        convs[cid].sort(key=lambda r: int(r["utterance_idx"]))

    samples: list[dict] = []
    conv_ids = list(convs.keys())
    rng.shuffle(conv_ids)

    for cid in conv_ids:
        turns = convs[cid]
        emotion_label = turns[0]["context"].lower().strip()
        emotion_id = EMOTION2ID.get(emotion_label, 0)

        # Extract (context, speaker_turn, listener_turn) triples
        # turns alternate speaker(0) → listener(1) → speaker(0) → …
        for i in range(0, len(turns) - 1, 2):
            user_turn  = turns[i]
            resp_turn  = turns[i + 1] if i + 1 < len(turns) else None
            if resp_turn is None:
                continue

            # Context = prompt (situation) + all prior turns
            prior = [turns[j]["utterance"] for j in range(i)]
            context_parts = [turns[0].get("prompt", "")]
            context_parts.extend(prior)
            context_parts.append(user_turn["utterance"])
            context = " [SEP] ".join(p for p in context_parts if p.strip())

            user_utt = user_turn["utterance"].strip()
            response = resp_turn["utterance"].strip()

            if not user_utt or not response:
                continue

            aff, dec, adi = pseudo_label(emotion_label, response, rng)

            samples.append({
                "sample_id":      f"ed_{cid}_{i}",
                "source":         "empathetic_dialogues",
                "context":        context,
                "user_utterance": user_utt,
                "response":       response,
                "emotion":        emotion_id,
                "emotion_str":    emotion_label,
                "aff_score":      aff,
                "dec_label":      dec,
                "adi_score":      adi,
            })

        if len(samples) >= target_n:
            break

    print(f"  → {len(samples)} samples extracted from EmpatheticDialogues")
    return samples[:target_n]


# ---------------------------------------------------------------------------
# ESConv builder
# ---------------------------------------------------------------------------

# Map ESConv emotion types to nearest ED emotion
_ESCONV_EMOTION_MAP = {
    "depression":   "sad",
    "anxiety":      "anxious",
    "anger":        "angry",
    "fear":         "afraid",
    "shame":        "ashamed",
    "disgust":      "disgusted",
    "sadness":      "sad",
    "nervousness":  "apprehensive",
    "guilt":        "guilty",
    "loneliness":   "lonely",
    "pain":         "devastated",
    "confusion":    "apprehensive",
}

# Map ESConv strategy labels to dec_label signals
_ESCONV_GOOD_STRATEGIES = {
    "Providing Suggestions",
    "Information",
    "Affirmation and Reassurance",
    "Restatement or Paraphrasing",
    "Reflection of Feelings",
    "Self-disclosure",
}


def _build_from_esconv(
    target_n: int,
    rng: random.Random,
) -> list[dict]:
    """
    Extract (context, user_utterance, response) triples from ESConv.
    Uses strategy annotations to improve dec_label quality.
    """
    try:
        from datasets import load_dataset
        print("Loading ESConv from HuggingFace…")
        raw = load_dataset("thu-coai/esconv", trust_remote_code=True)
    except Exception as e:
        print(f"  ! Could not load thu-coai/esconv: {e}")
        print("  Falling back to EmpatheticDialogues top-up…")
        return []

    all_convs: list[dict] = []
    for split in raw:
        all_convs.extend(raw[split])

    samples: list[dict] = []
    rng.shuffle(all_convs)

    for conv in all_convs:
        # Normalise schema — ESConv uses 'dialog' with 'role'/'content'/'strategy'
        dialog = conv.get("dialog", conv.get("dialogue", []))
        emotion_raw = str(conv.get("emotion_type", "sadness")).lower()
        emotion_str = _ESCONV_EMOTION_MAP.get(emotion_raw, "sad")
        emotion_id  = EMOTION2ID.get(emotion_str, 27)  # default: sad

        if len(dialog) < 2:
            continue

        prior_turns: list[str] = []
        situation = str(conv.get("situation", ""))

        for i, turn in enumerate(dialog):
            role     = str(turn.get("role", turn.get("speaker", ""))).lower()
            content  = str(turn.get("content", turn.get("text", ""))).strip()
            strategy = str(turn.get("strategy", "")).strip()

            if role in ("usr", "user", "seeker") and i + 1 < len(dialog):
                next_turn    = dialog[i + 1]
                next_role    = str(next_turn.get("role", next_turn.get("speaker", ""))).lower()
                next_content = str(next_turn.get("content", next_turn.get("text", ""))).strip()
                next_strat   = str(next_turn.get("strategy", "")).strip()

                if next_role not in ("sys", "system", "supporter"):
                    prior_turns.append(content)
                    continue

                context_parts = ([situation] if situation else []) + prior_turns + [content]
                context = " [SEP] ".join(p for p in context_parts if p.strip())

                aff, dec, adi = pseudo_label(emotion_str, next_content, rng)

                # Override dec_label with strategy signal when available
                if next_strat in _ESCONV_GOOD_STRATEGIES:
                    dec = 1.0
                    adi = float(max(0.0, aff - dec))

                samples.append({
                    "sample_id":      f"esconv_{id(conv)}_{i}",
                    "source":         "esconv",
                    "context":        context,
                    "user_utterance": content,
                    "response":       next_content,
                    "emotion":        emotion_id,
                    "emotion_str":    emotion_str,
                    "aff_score":      aff,
                    "dec_label":      dec,
                    "adi_score":      adi,
                })

            prior_turns.append(content)

        if len(samples) >= target_n:
            break

    print(f"  → {len(samples)} samples extracted from ESConv")
    return samples[:target_n]


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build(
    total_n:    int = 10_000,
    ed_ratio:   float = 0.80,
    output_dir: str = "data",
    seed:       int = 42,
) -> pd.DataFrame:
    """
    Build the TAD-Bench 10K dataset and save CSVs.

    Parameters
    ----------
    total_n    : total number of samples (default 10,000)
    ed_ratio   : fraction from EmpatheticDialogues (rest from ESConv)
    output_dir : directory to save CSV files
    seed       : random seed for reproducibility
    """
    rng = random.Random(seed)
    np.random.seed(seed)

    ed_n    = int(total_n * ed_ratio)
    esconv_n = total_n - ed_n

    ed_samples     = _build_from_empathetic_dialogues(ed_n,     rng)
    esconv_samples = _build_from_esconv(esconv_n, rng)

    # If ESConv failed, top-up from ED
    deficit = esconv_n - len(esconv_samples)
    if deficit > 0:
        print(f"  Topping up {deficit} samples from EmpatheticDialogues…")
        extra = _build_from_empathetic_dialogues(len(ed_samples) + deficit, rng)
        ed_samples = extra

    all_samples = ed_samples + esconv_samples
    rng.shuffle(all_samples)
    all_samples = all_samples[:total_n]

    df = pd.DataFrame(all_samples)

    # ---------------------------------------------------------------------------
    # Print label distribution summary
    # ---------------------------------------------------------------------------
    print("\nLabel distribution summary")
    print(f"  Total samples : {len(df)}")
    print(f"  aff_score mean: {df['aff_score'].mean():.3f}  std: {df['aff_score'].std():.3f}")
    print(f"  dec_label mean: {df['dec_label'].mean():.3f}")
    print(f"  adi_score mean: {df['adi_score'].mean():.3f}  (>0: {(df['adi_score'] > 0).mean()*100:.1f}%)")
    print(f"  Source breakdown:\n{df['source'].value_counts().to_string()}")

    # ---------------------------------------------------------------------------
    # Train / val / test split  (70 / 15 / 15)
    # ---------------------------------------------------------------------------
    n = len(df)
    idx = list(range(n))
    rng.shuffle(idx)

    train_end = int(n * 0.70)
    val_end   = int(n * 0.85)

    train_idx = idx[:train_end]
    val_idx   = idx[train_end:val_end]
    test_idx  = idx[val_end:]

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val   = df.iloc[val_idx].reset_index(drop=True)
    df_test  = df.iloc[test_idx].reset_index(drop=True)

    os.makedirs(output_dir, exist_ok=True)

    df.to_csv(       os.path.join(output_dir, "tad_bench_all.csv"),   index=False)
    df_train.to_csv( os.path.join(output_dir, "tad_bench_train.csv"), index=False)
    df_val.to_csv(   os.path.join(output_dir, "tad_bench_val.csv"),   index=False)
    df_test.to_csv(  os.path.join(output_dir, "tad_bench_test.csv"),  index=False)

    print(f"\nSplit sizes — train: {len(df_train)}, val: {len(df_val)}, test: {len(df_test)}")
    print(f"Saved to: {os.path.abspath(output_dir)}/")

    return df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build TAD-Bench 10K dataset")
    p.add_argument("--output_dir", default="data",     help="Directory to save CSV files")
    p.add_argument("--total_n",    type=int, default=10_000, help="Total number of samples")
    p.add_argument("--ed_ratio",   type=float, default=0.80, help="Fraction from EmpatheticDialogues")
    p.add_argument("--seed",       type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    build(
        total_n=args.total_n,
        ed_ratio=args.ed_ratio,
        output_dir=args.output_dir,
        seed=args.seed,
    )
