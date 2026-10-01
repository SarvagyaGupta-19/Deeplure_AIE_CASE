# Saree Design Recognition: Color-Invariant Metric Learning

[![PyTorch](https://img.shields.io/badge/PyTorch-2.14-EE4C2C.svg?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](https://opensource.org/licenses/MIT)

A scalable metric learning system that identifies saree designs independent of their color palette. Functioning similarly to facial recognition systems, this model processes a query image and ranks a reference gallery by structural design similarity, explicitly ignoring color variations.

## Approach Note

The system utilizes a ResNet-18 backbone projecting to a 128-dimensional embedding space, trained via a two-stage contrastive learning pipeline. **Stage 1** applies self-supervised NT-Xent loss with aggressive color augmentation (hue rotation, channel permutation, and grayscale conversion) to learn color-invariant structural features. **Stage 2** applies Supervised Contrastive loss using structural categories to enforce inter-class discrimination. The resulting model successfully identifies structural patterns across distinct colorways, achieving 88.3% Recall@1 on the validation gallery.

## Performance & Results

The model was evaluated on a dedicated test split utilizing both retrieval (gallery search) and pairwise verification metrics.

![Retrieval Examples](outputs/retrieval_examples.png)

### 1. Identification (Retrieval)
Evaluated by querying test images against a gallery of 115 known reference designs.

| Metric | Score | Description |
|--------|-------|-------------|
| **Recall@1** | **88.33%** | The exact correct design was the top match. |
| **Recall@3** | **95.00%** | The correct design appeared in the top 3 results. |
| **Recall@5** | **96.67%** | The correct design appeared in the top 5 results. |
| **mAP** | **0.7531** | Mean Average Precision across all queries. |

### 2. Verification (Pairwise)
Evaluated on 2,000 random image pairs to verify whether two given images contain the same design.

![Similarity Distribution](outputs/similarity_distribution.png)

| Metric | Score | Description |
|--------|-------|-------------|
| **AUC-ROC** | **0.8960** | Area under the Receiver Operating Characteristic curve. |
| **EER** | **0.1790** | Equal Error Rate. |
| **Threshold** | **0.7444** | Optimal cosine similarity threshold for a positive match. |

### 3. Color Invariance Stress Test
Tests the model's ability to maintain high similarity for identical designs under extreme, synthetic colorway alterations.

* **Same Design (Extreme Colorways) Similarity:** 0.9912 ± 0.0158
* **Different Design Similarity:** 0.6018 ± 0.1005
* **Color Invariance Gap:** **+0.3894** *(indicates strong structural discrimination independent of color)*

### 4. Embedding Space & Clustering
The color-invariant features naturally cluster designs from the same category together.

![Embedding Space (PCA)](outputs/embedding_space.png)
![Confusion Matrix](outputs/confusion_matrix.png)

---

## System Architecture

The pipeline is structured as a modular, production-ready Python package with strict separation of concerns.

```text
Deeplure/
├── config.py                  # Global hyperparameters and path configurations
├── train.py                   # Training orchestrator
├── inference.py               # Production API for inference (Identification/Verification)
├── data/
│   ├── explorer.py            # Dataset statistics and deduplication logic
│   ├── augmentations.py       # Color and geometric augmentation pipelines
│   └── dataset.py             # PyTorch Dataset implementation
├── models/
│   └── embedding.py           # ResNet-18 + projection head -> L2 normalized embedding
├── losses/
│   ├── ntxent.py              # NT-Xent loss (Stage 1)
│   └── supcon.py              # Supervised Contrastive loss (Stage 2)
├── engine/
│   ├── trainer.py             # Training loop, LR scheduling, checkpointing
│   └── extractor.py           # Batch embedding extraction
├── evaluation/                # Independent evaluation modules
│   ├── identification.py      
│   ├── verification.py        
│   ├── color_invariance.py    
│   └── efficiency.py          
└── visualization/             # Matplotlib generation functions
    └── plots.py               
```

## Model Pipeline

### 1. Network Architecture
* **Backbone**: `ResNet-18` (ImageNet pre-trained). Chosen for its translation-equivariant convolutions which effectively capture repeating textile patterns without unnecessary parameter bloat (11.2M parameters).
* **Projection Head**: Linear(512, 256) -> BatchNorm1d -> ReLU -> Linear(256, 128) -> L2 Normalization.
* **Embedding**: 128-d float32 vector (512 bytes per image).

### 2. Training Strategy
Color invariance is learned directly through contrastive augmentations rather than static preprocessing (e.g., standard grayscale conversion), which would destroy chromatic texture contrast.

* **Stage 1 (Self-Supervised)**: Applies NT-Xent loss treating each image as its own class. The model matches aggressively color-augmented views of the same image, learning to discard color reliance.
* **Stage 2 (Supervised)**: Applies Supervised Contrastive (SupCon) loss using dataset category labels to pull distinct designs within the same family closer together in the embedding space.

## Installation & Usage

### Dependencies
```bash
pip install torch torchvision numpy Pillow matplotlib scikit-learn
```

### Production Inference
The `SareeDesignMatcher` class provides a clean API for production deployments.

```python
from inference import SareeDesignMatcher

# 1. Initialize and load weights
matcher = SareeDesignMatcher.from_checkpoint('outputs/best_stage2.pth', device='cpu')

# 2. Build the reference gallery
matcher.build_gallery(['gallery/design_A.jpg', 'gallery/design_B.jpg'])

# 3. Identify a query image
results = matcher.identify('query.jpg', top_k=5)
for res in results:
    print(f"Rank {res['rank']}: {res['path']} (Similarity: {res['similarity']:.3f})")

# 4. Pairwise verification
is_match, similarity_score = matcher.verify('image1.jpg', 'image2.jpg')
```

### Training
To reproduce the training pipeline and evaluation metrics:
```bash
python train.py
```
Outputs, including checkpoints (`best_stage2.pth`), metrics (`results.json`), and visualizations will be saved to the `./outputs` directory.

## Dataset Considerations
The system utilizes the **Indian Saree Patterns** dataset (1,468 images across 4 categories). 

*Note: The raw training set includes 3x augmented copies using simple salt-and-pepper noise. The `data.explorer` module automatically deduplicates these (1,293 -> 431 images) prior to training to prevent data leakage and allow our targeted color-invariance augmentations to govern the learning process.*
