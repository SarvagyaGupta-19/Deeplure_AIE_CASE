# engine/extractor.py
# ====================
# Batch embedding extraction from a trained model.

from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.embedding import SareeEmbeddingNet
from data.dataset import SareeDataset
from config import Config


class EmbeddingExtractor:
    """Extracts L2-normalized embeddings for an entire dataset."""

    def __init__(self, model: SareeEmbeddingNet, cfg: Config):
        self.model = model.to(cfg.device)
        self.model.eval()
        self.cfg = cfg

    @torch.no_grad()
    def extract(
        self, dataset: SareeDataset,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Returns:
            embeddings: (N, D) float32
            labels:     (N,) int
            paths:      list of file paths
        """
        loader = DataLoader(
            dataset, batch_size=self.cfg.batch_size,
            shuffle=False, num_workers=self.cfg.num_workers,
        )
        embs, labs, paths = [], [], []
        for images, labels, fpaths in loader:
            images = images.to(self.cfg.device)
            embs.append(self.model(images).cpu().numpy())
            labs.append(labels.numpy())
            paths.extend(fpaths)

        return (np.concatenate(embs), np.concatenate(labs), paths)
