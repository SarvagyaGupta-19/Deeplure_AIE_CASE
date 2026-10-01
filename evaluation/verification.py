# evaluation/verification.py
# ===========================
# Pairwise verification: given two images, are they the same design?
# Reports AUC-ROC, EER, optimal-threshold accuracy.

import random
from typing import Dict, List
from collections import defaultdict

import numpy as np


def evaluate(
    embeddings: np.ndarray,
    labels: np.ndarray,
    num_pairs: int = 2000,
) -> Dict[str, float]:
    """
    Generate positive and negative pairs, compute cosine similarity,
    evaluate binary classification via AUC-ROC and EER.

    Returns:
        dict with AUC-ROC, EER, best accuracy, threshold, similarity stats.
    """
    by_class: Dict[int, List[int]] = defaultdict(list)
    for i, lab in enumerate(labels):
        by_class[lab].append(i)

    classes = list(by_class.keys())
    half = num_pairs // 2

    # Positive pairs (same class)
    pos = []
    for _ in range(half):
        c = random.choice(classes)
        if len(by_class[c]) < 2:
            continue
        i, j = random.sample(by_class[c], 2)
        pos.append(float(embeddings[i] @ embeddings[j]))

    # Negative pairs (different class)
    neg = []
    for _ in range(num_pairs - half):
        c1, c2 = random.sample(classes, 2)
        i = random.choice(by_class[c1])
        j = random.choice(by_class[c2])
        neg.append(float(embeddings[i] @ embeddings[j]))

    sims = np.array(pos + neg)
    truth = np.array([1] * len(pos) + [0] * len(neg))

    # ROC curve
    order = np.argsort(-sims)
    sorted_truth = truth[order]
    sorted_sims = sims[order]

    tp, fp = 0, 0
    total_p = sorted_truth.sum()
    total_n = len(sorted_truth) - total_p
    tprs, fprs = [], []
    for lab in sorted_truth:
        if lab == 1:
            tp += 1
        else:
            fp += 1
        tprs.append(tp / total_p if total_p else 0)
        fprs.append(fp / total_n if total_n else 0)

    tprs = np.array(tprs)
    fprs = np.array(fprs)

    auc = float(np.trapezoid(tprs, x=fprs))

    # EER
    fnrs = 1 - tprs
    eer_idx = int(np.argmin(np.abs(fprs - fnrs)))
    eer = float((fprs[eer_idx] + fnrs[eer_idx]) / 2)

    # Best threshold
    best_acc, best_t = 0.0, 0.0
    for t in np.unique(sorted_sims):
        acc = float(((sims >= t) == truth).mean())
        if acc > best_acc:
            best_acc, best_t = acc, float(t)

    return {
        "AUC-ROC": auc,
        "EER": eer,
        "Best_Accuracy": best_acc,
        "Best_Threshold": best_t,
        "Num_Pos_Pairs": len(pos),
        "Num_Neg_Pairs": len(neg),
        "Mean_Pos_Sim": float(np.mean(pos)),
        "Mean_Neg_Sim": float(np.mean(neg)),
        "Similarity_Gap": float(np.mean(pos) - np.mean(neg)),
    }
