# Configuration
# ==============
# Central configuration for the entire project.
# Every hyperparameter lives here. Nothing is hard-coded in modules.

from dataclasses import dataclass, field
from typing import List
import os
import torch


@dataclass
class Config:
    """
    Single source of truth for all hyperparameters.

    Grouped by concern:
        - Paths: where data lives, where outputs go
        - Model: architecture choices
        - Stage 1: self-supervised contrastive pretraining
        - Stage 2: supervised contrastive fine-tuning
        - Training: shared training settings
        - Evaluation: metric configuration
    """

    # --- Paths ---
    data_root: str = "./archive"
    output_dir: str = "./outputs"

    # --- Model ---
    backbone: str = "resnet18"
    embedding_dim: int = 128
    pretrained: bool = True

    # --- Stage 1: Self-supervised contrastive ---
    stage1_epochs: int = 30
    stage1_lr: float = 1e-3
    stage1_temperature: float = 0.07  # NT-Xent tau. Lower = harder negatives.

    # --- Stage 2: Supervised contrastive ---
    stage2_epochs: int = 20
    stage2_lr: float = 1e-4            # 10x lower than stage 1
    stage2_temperature: float = 0.1

    # --- Training (shared) ---
    batch_size: int = 32
    weight_decay: float = 1e-4
    num_workers: int = 0               # 0 on Windows to avoid multiprocessing issues
    image_size: int = 224

    # --- Evaluation ---
    recall_k_values: List[int] = field(default_factory=lambda: [1, 3, 5])
    verification_num_pairs: int = 2000

    # --- Device ---
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
