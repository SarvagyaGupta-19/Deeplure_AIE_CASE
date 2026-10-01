# inference.py
# =============
# Production-ready inference API.
# Usage:
#     matcher = SareeDesignMatcher.from_checkpoint("outputs/best_stage2.pth")
#     matcher.build_gallery(gallery_paths, gallery_labels)
#     results = matcher.identify("query.jpg", top_k=5)
#     is_match, sim = matcher.verify("a.jpg", "b.jpg")

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from models.embedding import SareeEmbeddingNet
from data.augmentations import eval_transform
from data.explorer import CLASS_NAMES


class SareeDesignMatcher:
    """
    Stateful inference wrapper.

    Build the gallery once, then run identify/verify queries cheaply.
    """

    def __init__(self, model: SareeEmbeddingNet, transform, device: str):
        self.model = model.to(device)
        self.model.eval()
        self.transform = transform
        self.device = device
        self.gallery_embs: Optional[np.ndarray] = None
        self.gallery_paths: Optional[List[str]] = None
        self.gallery_labels: Optional[np.ndarray] = None

    @classmethod
    def from_checkpoint(cls, path: str, device: str = "cpu"):
        ckpt = torch.load(path, map_location=device, weights_only=True)
        c = ckpt.get("config", {})
        model = SareeEmbeddingNet(
            c.get("backbone", "resnet18"),
            c.get("embedding_dim", 128),
            pretrained=False,
        )
        model.load_state_dict(ckpt["model_state_dict"])
        t = eval_transform(c.get("image_size", 224))
        return cls(model, t, device)

    @torch.no_grad()
    def _embed(self, path: str) -> np.ndarray:
        img = Image.open(path).convert("RGB")
        x = self.transform(img).unsqueeze(0).to(self.device)
        return self.model(x).cpu().numpy()[0]

    def build_gallery(
        self, paths: List[str], labels: Optional[List[int]] = None
    ) -> None:
        self.gallery_embs = np.array([self._embed(p) for p in paths])
        self.gallery_paths = paths
        self.gallery_labels = np.array(labels) if labels else None
        print(f"Gallery: {len(paths)} images, "
              f"{self.gallery_embs.shape[1]}-d embeddings")

    def identify(self, query_path: str, top_k: int = 5) -> List[Dict]:
        assert self.gallery_embs is not None, "Call build_gallery first"
        q = self._embed(query_path)
        sims = q @ self.gallery_embs.T
        top = np.argsort(sims)[::-1][:top_k]
        results = []
        for rank, idx in enumerate(top):
            r = {"rank": rank + 1, "path": self.gallery_paths[idx],
                 "similarity": float(sims[idx])}
            if self.gallery_labels is not None:
                r["label"] = int(self.gallery_labels[idx])
                r["class"] = CLASS_NAMES[r["label"]]
            results.append(r)
        return results

    def verify(
        self, path1: str, path2: str, threshold: float = 0.5
    ) -> Tuple[bool, float]:
        sim = float(self._embed(path1) @ self._embed(path2))
        return sim >= threshold, sim
