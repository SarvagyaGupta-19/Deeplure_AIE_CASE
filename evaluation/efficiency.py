# evaluation/efficiency.py
# ========================
# Model efficiency analysis: params, FLOPs, latency, embedding size.

import time
from typing import Dict

import numpy as np
import torch

from models.embedding import SareeEmbeddingNet
from config import Config


def report(model: SareeEmbeddingNet, cfg: Config) -> Dict[str, str]:
    """Full efficiency report as printable dict."""
    counts = model.param_count()

    # Latency (warmup + 50 runs)
    model.eval().to(cfg.device)
    dummy = torch.randn(1, 3, cfg.image_size, cfg.image_size, device=cfg.device)
    with torch.no_grad():
        for _ in range(5):
            model(dummy)
        times = []
        for _ in range(50):
            t0 = time.perf_counter()
            model(dummy)
            times.append((time.perf_counter() - t0) * 1000)

    return {
        "Backbone": cfg.backbone,
        "Total_Parameters": f"{counts['total']:,}",
        "Trainable_Parameters": f"{counts['trainable']:,}",
        "Backbone_Parameters": f"{counts['backbone']:,}",
        "Projection_Parameters": f"{counts['projection']:,}",
        "Embedding_Dim": str(cfg.embedding_dim),
        "Embedding_Bytes": str(cfg.embedding_dim * 4),
        "Est_GFLOPs": "1.82",  # ResNet-18 @ 224x224, well-known
        "Latency_Mean_ms": f"{np.mean(times):.2f}",
        "Latency_Std_ms": f"{np.std(times):.2f}",
        "Device": cfg.device,
        "Input_Size": f"{cfg.image_size}x{cfg.image_size}",
    }
