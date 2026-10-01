# engine/trainer.py
# ==================
# Two-stage training loop.
# Stage 1: self-supervised (NT-Xent) - learns color invariance.
# Stage 2: supervised (SupCon) - adds class discrimination.

import os
from typing import Optional, Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.embedding import SareeEmbeddingNet
from losses.ntxent import NTXentLoss
from losses.supcon import SupConLoss
from config import Config


class Trainer:
    """
    Owns the training loop, optimizer, scheduler, and checkpointing.
    Does NOT own the data or the model construction.
    """

    def __init__(self, model: SareeEmbeddingNet, cfg: Config):
        self.model = model.to(cfg.device)
        self.cfg = cfg
        self.history: Dict[str, List[float]] = {
            "stage1_loss": [], "stage2_loss": [], "val_loss": [],
        }

    # ------------------------------------------------------------------
    # Stage 1
    # ------------------------------------------------------------------
    def train_stage1(
        self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None
    ) -> None:
        """Self-supervised contrastive pretraining."""
        print("\n" + "=" * 60)
        print("STAGE 1: Self-Supervised Contrastive Pretraining")
        print("=" * 60)

        criterion = NTXentLoss(self.cfg.stage1_temperature)
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.cfg.stage1_lr,
            weight_decay=self.cfg.weight_decay,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.cfg.stage1_epochs, eta_min=1e-6,
        )

        best_val = float("inf")
        for epoch in range(1, self.cfg.stage1_epochs + 1):
            avg = self._train_epoch(train_loader, criterion, optimizer,
                                    supervised=False)
            self.history["stage1_loss"].append(avg)

            val_str = ""
            if val_loader is not None:
                val = self._validate(val_loader, criterion, supervised=False)
                self.history["val_loss"].append(val)
                val_str = f" | Val: {val:.4f}"
                if val < best_val:
                    best_val = val
                    self._save("best_stage1.pth")

            scheduler.step()
            if epoch % 5 == 0 or epoch == 1:
                lr = scheduler.get_last_lr()[0]
                print(f"  Epoch {epoch:3d}/{self.cfg.stage1_epochs}"
                      f" | Loss: {avg:.4f}{val_str} | LR: {lr:.6f}")

        self._save("final_stage1.pth")
        print(f"  Stage 1 done. Best val: {best_val:.4f}")

    # ------------------------------------------------------------------
    # Stage 2
    # ------------------------------------------------------------------
    def train_stage2(
        self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None
    ) -> None:
        """
        Supervised contrastive fine-tuning.

        Differential LR: backbone gets 10x lower LR to preserve
        color-invariant features learned in Stage 1.
        """
        print("\n" + "=" * 60)
        print("STAGE 2: Supervised Contrastive Fine-Tuning")
        print("=" * 60)

        criterion = SupConLoss(self.cfg.stage2_temperature)
        optimizer = optim.Adam([
            {"params": self.model.backbone.parameters(),
             "lr": self.cfg.stage2_lr * 0.1},
            {"params": self.model.projection.parameters(),
             "lr": self.cfg.stage2_lr},
        ], weight_decay=self.cfg.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.cfg.stage2_epochs, eta_min=1e-6,
        )

        best_val = float("inf")
        for epoch in range(1, self.cfg.stage2_epochs + 1):
            avg = self._train_epoch(train_loader, criterion, optimizer,
                                    supervised=True)
            self.history["stage2_loss"].append(avg)

            val_str = ""
            if val_loader is not None:
                val = self._validate(val_loader, criterion, supervised=True)
                val_str = f" | Val: {val:.4f}"
                if val < best_val:
                    best_val = val
                    self._save("best_stage2.pth")

            scheduler.step()
            if epoch % 5 == 0 or epoch == 1:
                print(f"  Epoch {epoch:3d}/{self.cfg.stage2_epochs}"
                      f" | Loss: {avg:.4f}{val_str}")

        self._save("final_stage2.pth")
        print(f"  Stage 2 done. Best val: {best_val:.4f}")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _train_epoch(
        self, loader: DataLoader, criterion: nn.Module,
        optimizer: optim.Optimizer, supervised: bool,
    ) -> float:
        self.model.train()
        total, count = 0.0, 0
        for v1, v2, labels in loader:
            v1 = v1.to(self.cfg.device)
            v2 = v2.to(self.cfg.device)
            z1, z2 = self.model(v1), self.model(v2)

            if supervised:
                loss = criterion(z1, z2, labels.to(self.cfg.device))
            else:
                loss = criterion(z1, z2)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()

            total += loss.item()
            count += 1
        return total / count

    @torch.no_grad()
    def _validate(
        self, loader: DataLoader, criterion: nn.Module, supervised: bool,
    ) -> float:
        self.model.eval()
        total, count = 0.0, 0
        for v1, v2, labels in loader:
            v1 = v1.to(self.cfg.device)
            v2 = v2.to(self.cfg.device)
            z1, z2 = self.model(v1), self.model(v2)
            if supervised:
                loss = criterion(z1, z2, labels.to(self.cfg.device))
            else:
                loss = criterion(z1, z2)
            total += loss.item()
            count += 1
        return total / count

    def _save(self, filename: str) -> None:
        path = os.path.join(self.cfg.output_dir, filename)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": vars(self.cfg),
        }, path)

    def load_checkpoint(self, filename: str) -> None:
        path = os.path.join(self.cfg.output_dir, filename)
        ckpt = torch.load(path, map_location=self.cfg.device, weights_only=True)
        self.model.load_state_dict(ckpt["model_state_dict"])
        print(f"  Loaded: {filename}")
