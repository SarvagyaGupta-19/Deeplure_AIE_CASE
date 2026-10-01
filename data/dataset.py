# data/dataset.py
# ================
# PyTorch Dataset for saree images.
# Handles two modes: contrastive (two views) and eval (single image).

from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class SareeDataset(Dataset):
    """
    Loads saree images and returns them in one of two modes.

    Modes:
        'contrastive' - returns (view1, view2, label)
            Used during training. The transform should be a TwoViewTransform.

        'eval' - returns (image_tensor, label, path)
            Used during embedding extraction. Single deterministic view.

    Args:
        file_list: from explorer.build_file_list(). Each entry is
                   (file_path, class_idx, source_id).
        transform: augmentation pipeline. TwoViewTransform for contrastive,
                   standard Compose for eval.
        mode:      'contrastive' or 'eval'.
    """

    def __init__(
        self,
        file_list: List[Tuple[str, int, str]],
        transform,
        mode: str = "contrastive",
    ):
        assert mode in ("contrastive", "eval")
        self.file_list = file_list
        self.transform = transform
        self.mode = mode

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int):
        path, cls_idx, _ = self.file_list[idx]
        img = Image.open(path).convert("RGB")

        if self.mode == "contrastive":
            view1, view2 = self.transform(img)
            return view1, view2, cls_idx
        else:
            return self.transform(img), cls_idx, path

    def get_labels(self) -> np.ndarray:
        """All labels as a numpy array (useful for stratified splitting)."""
        return np.array([e[1] for e in self.file_list])
