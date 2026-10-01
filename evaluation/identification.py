# evaluation/identification.py
# =============================
# Retrieval evaluation: given a query, rank gallery by similarity.
# Reports Recall@K and mAP.

from typing import Dict, List

import numpy as np


def evaluate(
    query_embs: np.ndarray,
    query_labels: np.ndarray,
    gallery_embs: np.ndarray,
    gallery_labels: np.ndarray,
    k_values: List[int] = (1, 3, 5),
) -> Dict[str, float]:
    """
    Compute retrieval metrics.

    Args:
        query_embs:    (Q, D) L2-normalized embeddings
        query_labels:  (Q,) ground-truth class indices
        gallery_embs:  (G, D)
        gallery_labels:(G,)
        k_values:      list of K for Recall@K

    Returns:
        dict with Recall@K and mAP.
    """
    sim = query_embs @ gallery_embs.T  # (Q, G) cosine similarity
    Q = len(query_labels)
    results = {}

    # ---- Recall@K ----
    for k in k_values:
        correct = 0
        for i in range(Q):
            top_k = np.argsort(sim[i])[::-1][:k]
            if query_labels[i] in gallery_labels[top_k]:
                correct += 1
        results[f"Recall@{k}"] = correct / Q

    # ---- mAP ----
    aps = []
    for i in range(Q):
        order = np.argsort(sim[i])[::-1]
        relevant = gallery_labels[order] == query_labels[i]
        if relevant.sum() == 0:
            continue
        cum = np.cumsum(relevant)
        prec = cum / (np.arange(len(relevant)) + 1)
        ap = (prec * relevant).sum() / relevant.sum()
        aps.append(ap)
    results["mAP"] = np.mean(aps) if aps else 0.0

    return results
