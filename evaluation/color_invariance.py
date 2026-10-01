# evaluation/color_invariance.py
# ===============================
# The unique evaluation for this problem: does the model actually
# ignore color and match on pattern structure alone?
#
# Test: take an image, apply extreme color augmentation (simulating
# a completely different colorway), and check if the embeddings
# still match. Compare against cross-image similarity (negatives).

import random
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

from models.embedding import SareeEmbeddingNet
from data.augmentations import stress_color_transform, eval_transform


def evaluate(
    model: SareeEmbeddingNet,
    file_list: List[Tuple[str, int, str]],
    image_size: int = 224,
    device: str = "cpu",
    num_samples: int = 50,
    num_colorways: int = 5,
) -> Dict[str, float]:
    """
    For each sampled image:
      1. Embed the original.
      2. Apply extreme color augmentation num_colorways times.
      3. Embed each augmented version.
      4. Measure cosine similarity original <-> augmented.
    Also compute cross-image (negative) similarity for comparison.

    A good model: same_design_sim >> diff_design_sim.
    """
    model.eval()
    det_transform = eval_transform(image_size)
    color_aug = stress_color_transform()

    # Normalize for augmented images (no random crop, just resize+center)
    import torchvision.transforms as T
    aug_post = T.Compose([
        T.Resize(int(image_size * 1.14)),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
    ])

    samples = random.sample(file_list, min(num_samples, len(file_list)))

    same_sims = []   # same image, different color
    diff_sims = []   # different image

    with torch.no_grad():
        # Cache original embeddings
        orig_embs = []
        for path, _, _ in samples:
            img = Image.open(path).convert("RGB")
            t = det_transform(img).unsqueeze(0).to(device)
            orig_embs.append(model(t).cpu().numpy()[0])

            # Color-augmented views
            for _ in range(num_colorways):
                aug = color_aug(img)
                at = aug_post(aug).unsqueeze(0).to(device)
                ae = model(at).cpu().numpy()[0]
                same_sims.append(float(orig_embs[-1] @ ae))

        # Cross-image negatives
        orig_embs_arr = np.array(orig_embs)
        for i in range(len(samples)):
            for j in range(i + 1, min(i + 3, len(samples))):
                if samples[i][1] != samples[j][1]:  # different class
                    diff_sims.append(
                        float(orig_embs_arr[i] @ orig_embs_arr[j])
                    )

    same_arr = np.array(same_sims)
    diff_arr = np.array(diff_sims) if diff_sims else np.array([0.0])

    return {
        "Same_Design_Diff_Color_Mean": float(same_arr.mean()),
        "Same_Design_Diff_Color_Std": float(same_arr.std()),
        "Same_Design_Diff_Color_Min": float(same_arr.min()),
        "Diff_Design_Mean": float(diff_arr.mean()),
        "Diff_Design_Std": float(diff_arr.std()),
        "Color_Invariance_Gap": float(same_arr.mean() - diff_arr.mean()),
        "Num_Same_Pairs": len(same_sims),
        "Num_Diff_Pairs": len(diff_sims),
    }
