"""
SFT Trainer for AD-DAN
=======================
Trains the full model (encoder + PAA + DEM + CMIA + IE) end-to-end with
the composite loss:
    L = L_aff + L_dec + λ * L_adi

Training configuration matches the paper:
  - AdamW, lr = 3e-5, linear warmup 1 000 steps
  - Batch size 32, 20 epochs
  - Mixed precision FP16 on GPU
  - λ = 0.5  (best ablation result)
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from ..models.ad_dan import ADDAN, ADDANConfig
from ..utils.metrics import adir, adis, emotion_accuracy, empathy_pearson


class SFTTrainer:
    """
    Supervised fine-tuning trainer for AD-DAN.

    Parameters
    ----------
    model       : ADDAN instance
    train_dl    : training DataLoader
    val_dl      : validation DataLoader
    output_dir  : directory to save checkpoints
    config      : ADDANConfig (used for loss weights)
    lr          : learning rate (default 3e-5)
    warmup_steps: linear warmup steps (default 1 000)
    max_epochs  : number of training epochs (default 20)
    grad_clip   : gradient clipping norm (default 1.0)
    fp16        : whether to use mixed-precision training
    log_every   : log training loss every N steps
    eval_every  : run validation every N steps (0 = epoch-level only)
    save_best   : save checkpoint only when validation ADIR improves
    device      : torch device (auto-detected if None)
    """

    def __init__(
        self,
        model: ADDAN,
        train_dl: DataLoader,
        val_dl: DataLoader,
        output_dir: str = "checkpoints/sft",
        config: Optional[ADDANConfig] = None,
        lr: float = 3e-5,
        warmup_steps: int = 1_000,
        max_epochs: int = 20,
        grad_clip: float = 1.0,
        fp16: bool = True,
        log_every: int = 50,
        eval_every: int = 0,
        save_best: bool = True,
        device: Optional[torch.device] = None,
    ):
        self.model     = model
        self.train_dl  = train_dl
        self.val_dl    = val_dl
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config    = config or model.config
        self.lr        = lr
        self.warmup_steps = warmup_steps
        self.max_epochs   = max_epochs
        self.grad_clip    = grad_clip
        self.fp16         = fp16
        self.log_every    = log_every
        self.eval_every   = eval_every
        self.save_best    = save_best

        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model.to(self.device)

        # Optimizer
        no_decay = {"bias", "LayerNorm.weight"}
        params = [
            {
                "params": [p for n, p in model.named_parameters()
                           if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in model.named_parameters()
                           if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = AdamW(params, lr=lr)

        total_steps = max_epochs * len(train_dl)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        self.scaler = torch.cuda.amp.GradScaler(enabled=fp16 and self.device.type == "cuda")

        self.best_val_adir = float("inf")
        self.global_step   = 0

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self) -> dict[str, list[float]]:
        """
        Run the full training loop.

        Returns
        -------
        history : dict with lists of train_loss, val_adir per epoch
        """
        history: dict[str, list[float]] = {"train_loss": [], "val_adir": []}

        for epoch in range(1, self.max_epochs + 1):
            epoch_loss = self._train_epoch(epoch)
            val_metrics = self.evaluate(self.val_dl)

            history["train_loss"].append(epoch_loss)
            history["val_adir"].append(val_metrics["ADIR"])

            print(
                f"[Epoch {epoch:02d}/{self.max_epochs}]  "
                f"train_loss={epoch_loss:.4f}  "
                f"val_ADIR={val_metrics['ADIR']:.2f}%  "
                f"val_EmpS={val_metrics['EmpS']:.4f}  "
                f"val_DecAcc={val_metrics['DecAcc']:.2f}%"
            )

            if self.save_best and val_metrics["ADIR"] < self.best_val_adir:
                self.best_val_adir = val_metrics["ADIR"]
                self._save_checkpoint("best_model")
                print(f"  → New best ADIR={self.best_val_adir:.2f}%  checkpoint saved.")

        self._save_checkpoint("final_model")
        return history

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        t0 = time.time()

        for step, batch in enumerate(self.train_dl, start=1):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            with torch.cuda.amp.autocast(
                enabled=self.fp16 and self.device.type == "cuda"
            ):
                out = self.model(
                    context_input_ids=batch["context_input_ids"],
                    context_attention_mask=batch["context_attention_mask"],
                    user_input_ids=batch["user_input_ids"],
                    user_attention_mask=batch["user_attention_mask"],
                    response_input_ids=batch["response_input_ids"],
                    response_attention_mask=batch["response_attention_mask"],
                    emotion_labels=batch["emotion_labels"],
                    aff_score_labels=batch["aff_score_labels"],
                    dec_labels=batch["dec_labels"],
                    adi_labels=batch["adi_labels"],
                )
                loss = out["total_loss"]

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            total_loss += loss.item()
            self.global_step += 1

            if step % self.log_every == 0:
                avg = total_loss / step
                elapsed = time.time() - t0
                print(
                    f"  Epoch {epoch} | step {step}/{len(self.train_dl)} | "
                    f"loss={avg:.4f} | {elapsed:.1f}s"
                )

            if self.eval_every > 0 and self.global_step % self.eval_every == 0:
                val_metrics = self.evaluate(self.val_dl)
                print(f"  [step {self.global_step}] val_ADIR={val_metrics['ADIR']:.2f}%")
                self.model.train()

        return total_loss / len(self.train_dl)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, dl: DataLoader) -> dict[str, float]:
        """Run evaluation on a DataLoader and return metric dict."""
        self.model.eval()

        all_s_aff, all_s_dec  = [], []
        all_e_preds, all_e_lbls = [], []
        all_dec_preds, all_dec_lbls = [], []

        for batch in dl:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            out = self.model(
                context_input_ids=batch["context_input_ids"],
                context_attention_mask=batch["context_attention_mask"],
                user_input_ids=batch["user_input_ids"],
                user_attention_mask=batch["user_attention_mask"],
                response_input_ids=batch["response_input_ids"],
                response_attention_mask=batch["response_attention_mask"],
            )
            s_aff = out["S_aff"].cpu().numpy()
            s_dec = out["S_dec"].cpu().numpy()
            e_hat = out["e_hat"].cpu().numpy().argmax(axis=-1)

            all_s_aff.extend(s_aff.tolist())
            all_s_dec.extend(s_dec.tolist())
            all_e_preds.extend(e_hat.tolist())
            all_e_lbls.extend(batch["emotion_labels"].cpu().numpy().tolist())
            all_dec_preds.extend((s_dec > 0.5).astype(int).tolist())
            all_dec_lbls.extend(batch["dec_labels"].cpu().numpy().tolist())

        s_aff_arr = np.array(all_s_aff)
        s_dec_arr = np.array(all_s_dec)

        return {
            "ADIR":     adir(s_aff_arr, s_dec_arr),
            "ADIS":     adis(s_aff_arr, s_dec_arr),
            "EmpS":     empathy_pearson(s_aff_arr, np.array(all_s_aff)),
            "EmotAcc":  emotion_accuracy(np.array(all_e_preds), np.array(all_e_lbls)),
            "DecAcc":   float(
                np.mean(np.array(all_dec_preds) == np.array(all_dec_lbls)) * 100.0
            ),
        }

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def _save_checkpoint(self, name: str) -> None:
        path = self.output_dir / name
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path / "model.pt")
        torch.save(self.optimizer.state_dict(), path / "optimizer.pt")
        print(f"  Checkpoint saved → {path}")

    def load_checkpoint(self, name: str = "best_model") -> None:
        path = self.output_dir / name / "model.pt"
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Loaded checkpoint from {path}")
