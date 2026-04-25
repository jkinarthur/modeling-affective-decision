"""
TAD-Bench Dataset
=================
Wraps the EmpatheticDialogues and ESConv datasets from HuggingFace,
augmented with TAD-Bench annotation columns:
  - aff_score    : ordinal affective alignment score ∈ {0, 0.25, 0.5, 0.75, 1.0}
  - dec_label    : binary decision correctness ∈ {0, 1}
  - adi_score    : ground-truth ADI = max(0, aff_score - dec_label)

In real use, load the released TAD-Bench annotation CSV and join it onto
the standard dataset splits.  This file provides the Dataset classes,
collator, and a factory function that creates DataLoaders.

The annotation CSV is expected to have columns:
  id, split, aff_score, dec_label
and an `id` column that matches the dataset's built-in id field.

For prototyping without the CSV, synthetic annotations are generated.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BartTokenizer


# ---------------------------------------------------------------------------
# Emotion label map (EmpatheticDialogues – 32 categories)
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


# ---------------------------------------------------------------------------
# TAD-Bench Dataset
# ---------------------------------------------------------------------------

@dataclass
class TADSample:
    """Single annotated dialogue sample."""
    sample_id:      str
    context:        str   # full dialogue history + current user utterance
    user_utterance: str   # current user turn only
    response:       str   # system response to score
    emotion:        int   # emotion label index (0–31)
    aff_score:      float # ∈ {0, 0.25, 0.5, 0.75, 1.0}
    dec_label:      float # binary {0.0, 1.0}
    adi_score:      float # = max(0, aff_score - dec_label)


class TADBenchDataset(Dataset):
    """
    TAD-Bench PyTorch Dataset.

    Parameters
    ----------
    samples     : list of TADSample
    tokenizer   : BartTokenizer
    max_ctx_len : max tokens for context encoding
    max_resp_len: max tokens for response encoding
    """

    def __init__(
        self,
        samples: list[TADSample],
        tokenizer: BartTokenizer,
        max_ctx_len: int = 512,
        max_resp_len: int = 128,
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_ctx_len = max_ctx_len
        self.max_resp_len = max_resp_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]

        ctx_enc = self.tokenizer(
            s.context,
            max_length=self.max_ctx_len,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        user_enc = self.tokenizer(
            s.user_utterance,
            max_length=self.max_resp_len,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        resp_enc = self.tokenizer(
            s.response,
            max_length=self.max_resp_len,
            truncation=True,
            padding=False,
            return_tensors=None,
        )

        return {
            "sample_id":              s.sample_id,
            "context_input_ids":      ctx_enc["input_ids"],
            "context_attention_mask": ctx_enc["attention_mask"],
            "user_input_ids":         user_enc["input_ids"],
            "user_attention_mask":    user_enc["attention_mask"],
            "response_input_ids":     resp_enc["input_ids"],
            "response_attention_mask":resp_enc["attention_mask"],
            "emotion_label":          s.emotion,
            "aff_score":              s.aff_score,
            "dec_label":              s.dec_label,
            "adi_score":              s.adi_score,
        }


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------

class TADCollator:
    """
    Pads variable-length sequences in a batch.
    Works with BartTokenizer (pad_token_id = 1).
    """

    def __init__(self, pad_token_id: int = 1):
        self.pad = pad_token_id

    def _pad(
        self,
        sequences: list[list[int]],
        pad_value: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        max_len = max(len(s) for s in sequences)
        padded  = torch.full((len(sequences), max_len), pad_value, dtype=torch.long)
        mask    = torch.zeros(len(sequences), max_len, dtype=torch.long)
        for i, seq in enumerate(sequences):
            length = len(seq)
            padded[i, :length] = torch.tensor(seq, dtype=torch.long)
            mask[i, :length]   = 1
        return padded, mask

    def __call__(self, batch: list[dict]) -> dict:
        ctx_ids,  ctx_mask  = self._pad([b["context_input_ids"]       for b in batch], self.pad)
        user_ids, user_mask = self._pad([b["user_input_ids"]           for b in batch], self.pad)
        resp_ids, resp_mask = self._pad([b["response_input_ids"]       for b in batch], self.pad)

        return {
            "sample_ids":               [b["sample_id"] for b in batch],
            "context_input_ids":        ctx_ids,
            "context_attention_mask":   ctx_mask,
            "user_input_ids":           user_ids,
            "user_attention_mask":      user_mask,
            "response_input_ids":       resp_ids,
            "response_attention_mask":  resp_mask,
            "emotion_labels":   torch.tensor([b["emotion_label"] for b in batch], dtype=torch.long),
            "aff_score_labels": torch.tensor([b["aff_score"]     for b in batch], dtype=torch.float),
            "dec_labels":       torch.tensor([b["dec_label"]     for b in batch], dtype=torch.float),
            "adi_labels":       torch.tensor([b["adi_score"]     for b in batch], dtype=torch.float),
        }


# ---------------------------------------------------------------------------
# Loader from HuggingFace datasets + annotation CSV
# ---------------------------------------------------------------------------

def load_tad_samples_from_csv(
    annotation_csv: str,
    hf_dataset,           # HuggingFace dataset split (e.g. datasets.load_dataset("empathetic_dialogues")["test"])
    id_field: str = "conv_id",
) -> list[TADSample]:
    """
    Join TAD-Bench annotation CSV onto a HuggingFace dataset split.

    The annotation CSV must have columns:
      id, aff_score, dec_label

    Returns a list of TADSample instances.
    """
    ann = pd.read_csv(annotation_csv).set_index("id")

    samples = []
    for row in hf_dataset:
        sid = str(row[id_field])
        if sid not in ann.index:
            continue

        aff = float(ann.loc[sid, "aff_score"])
        dec = float(ann.loc[sid, "dec_label"])
        adi = float(max(0.0, aff - dec))

        # Build context string: concatenate utterances in recency order
        utterances = row.get("utterances", [row.get("utterance", "")])
        context = " [SEP] ".join(str(u) for u in utterances[-10:])  # last 10 turns
        user_utt = str(utterances[-1]) if utterances else ""
        response = str(row.get("response", row.get("utterances", [""])[-1]))

        emotion_str = str(row.get("context", "neutral")).lower()
        emotion_id  = EMOTION2ID.get(emotion_str, 0)

        samples.append(TADSample(
            sample_id=sid,
            context=context,
            user_utterance=user_utt,
            response=response,
            emotion=emotion_id,
            aff_score=aff,
            dec_label=dec,
            adi_score=adi,
        ))
    return samples


def make_synthetic_samples(n: int = 200, seed: int = 42) -> list[TADSample]:
    """
    Generate synthetic TADSample instances for unit-testing and
    smoke-testing the training pipeline without real data.
    """
    rng = random.Random(seed)
    contexts = [
        "I'm so anxious about money. I think I'll just take out another loan.",
        "I aced my exam! I'm thinking of celebrating with my friends.",
        "I feel really hopeless about my situation. Nothing is working.",
        "My doctor said I need to rest, but I have a big presentation tomorrow.",
        "I want to quit my job and start a business. My partner thinks it's risky.",
    ]
    responses = [
        "That sounds incredibly stressful. If a loan feels right to you, trust your instincts.",
        "Congratulations! You absolutely deserve to celebrate!",
        "I hear how overwhelmed you are feeling. Please consider speaking to a professional.",
        "That sounds really difficult. Your health is very important.",
        "I can understand the tension. Would you like to talk through the risks together?",
    ]

    samples = []
    for i in range(n):
        ctx_idx  = rng.randint(0, len(contexts) - 1)
        emotion  = rng.randint(0, 31)
        aff      = rng.choice([0.0, 0.25, 0.5, 0.75, 1.0])
        dec      = float(rng.randint(0, 1))
        adi      = max(0.0, aff - dec)
        samples.append(TADSample(
            sample_id=f"syn_{i:05d}",
            context=contexts[ctx_idx],
            user_utterance=contexts[ctx_idx],
            response=responses[ctx_idx],
            emotion=emotion,
            aff_score=aff,
            dec_label=dec,
            adi_score=adi,
        ))
    return samples


def build_dataloaders(
    tokenizer: BartTokenizer,
    train_samples: list[TADSample],
    val_samples: list[TADSample],
    test_samples: list[TADSample],
    batch_size: int = 32,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train / val / test DataLoaders."""
    collator = TADCollator(pad_token_id=tokenizer.pad_token_id)

    train_ds = TADBenchDataset(train_samples, tokenizer)
    val_ds   = TADBenchDataset(val_samples,   tokenizer)
    test_ds  = TADBenchDataset(test_samples,  tokenizer)

    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collator, num_workers=num_workers, pin_memory=True,
    )
    val_dl = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collator, num_workers=num_workers, pin_memory=True,
    )
    test_dl = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collator, num_workers=num_workers, pin_memory=True,
    )
    return train_dl, val_dl, test_dl


# ---------------------------------------------------------------------------
# Loader for TAD-Bench CSVs produced by build_tad_bench.py
# ---------------------------------------------------------------------------

def load_tad_from_csv(csv_path: str) -> list[TADSample]:
    """
    Load a TAD-Bench CSV (produced by build_tad_bench.py) into a list of
    TADSample instances.

    Expected columns
    ----------------
    sample_id, context, user_utterance, response,
    emotion, aff_score, dec_label, adi_score

    The `emotion` column may be an integer id or a string emotion name.
    """
    df = pd.read_csv(csv_path)

    required = {"sample_id", "context", "user_utterance", "response",
                "emotion", "aff_score", "dec_label", "adi_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV {csv_path!r} is missing columns: {missing}")

    samples: list[TADSample] = []
    for _, row in df.iterrows():
        # Emotion can be stored as int id or string name
        emo_raw = row["emotion"]
        if isinstance(emo_raw, str):
            emotion_id = EMOTION2ID.get(emo_raw.lower().strip(), 0)
        else:
            emotion_id = int(emo_raw)

        aff = float(row["aff_score"])
        dec = float(row["dec_label"])
        adi = float(row["adi_score"])

        samples.append(TADSample(
            sample_id=str(row["sample_id"]),
            context=str(row["context"]),
            user_utterance=str(row["user_utterance"]),
            response=str(row["response"]),
            emotion=emotion_id,
            aff_score=aff,
            dec_label=dec,
            adi_score=adi,
        ))
    return samples


def load_tad_splits(data_dir: str) -> tuple[list[TADSample], list[TADSample], list[TADSample]]:
    """
    Load train / val / test TADSample lists from a directory containing
    tad_bench_train.csv, tad_bench_val.csv, tad_bench_test.csv.
    """
    train = load_tad_from_csv(os.path.join(data_dir, "tad_bench_train.csv"))
    val   = load_tad_from_csv(os.path.join(data_dir, "tad_bench_val.csv"))
    test  = load_tad_from_csv(os.path.join(data_dir, "tad_bench_test.csv"))
    print(f"Loaded TAD-Bench — train: {len(train)}, val: {len(val)}, test: {len(test)}")
    return train, val, test
