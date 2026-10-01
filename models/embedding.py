# models/embedding.py
# ====================
# The embedding network: backbone + projection head -> L2-normalized vector.
#
# Architecture decisions documented inline. Every layer is justified.

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class SareeEmbeddingNet(nn.Module):
    """
    Backbone + projection head -> L2-normalized embedding.

    Architecture:
        ResNet-18 (ImageNet) -> AdaptiveAvgPool -> flatten
        -> FC(512, 256) -> BN -> ReLU -> FC(256, 128) -> L2-Norm

    Why ResNet-18:
        - 11.2M params: large enough for texture features, small enough
          to avoid overfitting on ~430 training images.
        - Conv inductive bias (translation equivariance) matches the
          repeating nature of textile patterns.
        - ImageNet pretraining provides strong low-level texture/edge
          features out of the box.

    Why two-layer projection head:
        - SimCLR showed a non-linear projection head significantly
          improves contrastive representation quality.
        - BatchNorm stabilizes training with contrastive losses
          that are sensitive to embedding magnitude.

    Why 128-d:
        - Face recognition uses 128-d for 10000+ identities.
        - We have 4 categories with ~100 designs each. 128-d is plenty.
        - 128 x float32 = 512 bytes per image: trivial storage.
    """

    def __init__(
        self,
        backbone_name: str = "resnet18",
        embedding_dim: int = 128,
        pretrained: bool = True,
    ):
        super().__init__()

        if backbone_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            backbone = models.resnet18(weights=weights)
            feat_dim = 512
        elif backbone_name == "resnet34":
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            backbone = models.resnet34(weights=weights)
            feat_dim = 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        # Strip the classification head
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        self.projection = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, embedding_dim),
        )

        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) -> (B, embedding_dim), L2-normalized."""
        h = self.backbone(x).flatten(1)          # (B, 512)
        z = self.projection(h)                   # (B, 128)
        return F.normalize(z, p=2, dim=1)

    def param_count(self) -> Dict[str, int]:
        """Parameter breakdown for efficiency reporting."""
        bb = sum(p.numel() for p in self.backbone.parameters())
        pj = sum(p.numel() for p in self.projection.parameters())
        tr = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"backbone": bb, "projection": pj,
                "total": bb + pj, "trainable": tr}
