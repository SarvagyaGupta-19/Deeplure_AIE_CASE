# visualization/plots.py
# ======================
# All plot generation. Each function takes data in, writes a PNG out,
# returns the file path. No side effects.

import os
import random
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from collections import defaultdict

from data.explorer import CLASS_NAMES

COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]


def training_history(history: Dict, output_dir: str) -> str:
    """Loss curves for both stages."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if history["stage1_loss"]:
        axes[0].plot(history["stage1_loss"], "b-", lw=2, label="Train")
        n1 = len(history["stage1_loss"])
        if history["val_loss"]:
            axes[0].plot(history["val_loss"][:n1], "r--", lw=2, label="Val")
        axes[0].set_title("Stage 1: Self-Supervised (NT-Xent)", fontsize=13)
        axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
        axes[0].legend(); axes[0].grid(alpha=0.3)

    if history["stage2_loss"]:
        axes[1].plot(history["stage2_loss"], "b-", lw=2, label="Train")
        axes[1].set_title("Stage 2: Supervised (SupCon)", fontsize=13)
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
        axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "training_history.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def embedding_space(
    embeddings: np.ndarray, labels: np.ndarray, output_dir: str,
    title: str = "Embedding Space",
) -> str:
    """PCA-projected 2D scatter plot, colored by class."""
    centered = embeddings - embeddings.mean(axis=0)
    U, S, _ = np.linalg.svd(centered, full_matrices=False)
    xy = U[:, :2] * S[:2]

    fig, ax = plt.subplots(figsize=(10, 8))
    for idx, name in enumerate(CLASS_NAMES):
        mask = labels == idx
        ax.scatter(xy[mask, 0], xy[mask, 1], c=COLORS[idx], label=name,
                   alpha=0.7, s=50, edgecolors="white", linewidths=0.5)

    ax.set_title(f"{title} (PCA)", fontsize=14)
    ax.legend(fontsize=12, markerscale=1.5)
    ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
    ax.grid(alpha=0.2)

    path = os.path.join(output_dir, "embedding_space.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def similarity_distributions(
    pos_sims: List[float], neg_sims: List[float], output_dir: str,
) -> str:
    """Histogram of cosine similarities for same-class vs diff-class pairs."""
    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.linspace(-0.5, 1.0, 60)
    ax.hist(pos_sims, bins, alpha=0.6, color="#2ecc71", density=True,
            label=f"Same class (mean={np.mean(pos_sims):.3f})")
    ax.hist(neg_sims, bins, alpha=0.6, color="#e74c3c", density=True,
            label=f"Diff class (mean={np.mean(neg_sims):.3f})")
    ax.axvline(np.mean(pos_sims), color="#27ae60", ls="--", lw=2)
    ax.axvline(np.mean(neg_sims), color="#c0392b", ls="--", lw=2)
    ax.set_title("Similarity Distribution", fontsize=14)
    ax.set_xlabel("Cosine Similarity"); ax.set_ylabel("Density")
    ax.legend(fontsize=12); ax.grid(alpha=0.2)

    path = os.path.join(output_dir, "similarity_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def retrieval_examples(
    q_embs: np.ndarray, q_labels: np.ndarray, q_paths: List[str],
    g_embs: np.ndarray, g_labels: np.ndarray, g_paths: List[str],
    output_dir: str, num_queries: int = 4, top_k: int = 5,
) -> str:
    """Query image + top-K gallery matches with correct/incorrect borders."""
    sim = q_embs @ g_embs.T

    # Pick one query per class
    indices = []
    for ci in range(len(CLASS_NAMES)):
        where = np.where(q_labels == ci)[0]
        if len(where):
            indices.append(where[0])
        if len(indices) >= num_queries:
            break
    while len(indices) < num_queries and len(indices) < len(q_labels):
        idx = random.randint(0, len(q_labels) - 1)
        if idx not in indices:
            indices.append(idx)

    nrows = len(indices)
    fig, axes = plt.subplots(nrows, top_k + 1,
                             figsize=(3 * (top_k + 1), 3 * nrows))
    if nrows == 1:
        axes = [axes]

    for row, qi in enumerate(indices):
        # Query
        ax = axes[row][0]
        ax.imshow(Image.open(q_paths[qi]).convert("RGB"))
        ax.set_title(f"Query\n{CLASS_NAMES[q_labels[qi]]}", fontsize=10,
                     fontweight="bold")
        ax.axis("off")

        # Top-K
        topk = np.argsort(sim[qi])[::-1][:top_k]
        for col, gi in enumerate(topk):
            ax = axes[row][col + 1]
            ax.imshow(Image.open(g_paths[gi]).convert("RGB"))
            ok = q_labels[qi] == g_labels[gi]
            tag = "OK" if ok else "X"
            ax.set_title(f"{tag} {CLASS_NAMES[g_labels[gi]]}\n"
                         f"sim={sim[qi, gi]:.3f}", fontsize=9)
            ax.axis("off")
            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_color("#2ecc71" if ok else "#e74c3c")
                sp.set_linewidth(3)

    plt.suptitle("Retrieval: Query -> Top-5 Gallery", fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, "retrieval_examples.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def confusion_matrix(
    q_labels: np.ndarray, g_labels: np.ndarray,
    sim_matrix: np.ndarray, output_dir: str, k: int = 1,
) -> str:
    """Confusion matrix from top-K majority vote retrieval."""
    nc = len(CLASS_NAMES)
    cm = np.zeros((nc, nc), dtype=int)
    for i in range(len(q_labels)):
        topk = np.argsort(sim_matrix[i])[::-1][:k]
        pred = np.bincount(g_labels[topk], minlength=nc).argmax()
        cm[q_labels[i], pred] += 1

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(cm, cmap="Blues")
    for i in range(nc):
        for j in range(nc):
            clr = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color=clr, fontsize=14, fontweight="bold")
    ax.set_xticks(range(nc)); ax.set_yticks(range(nc))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted (Top-1)"); ax.set_ylabel("True")
    ax.set_title(f"Retrieval Confusion Matrix (Top-{k})", fontsize=14)

    plt.tight_layout()
    path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path
