# data/augmentations.py
# =====================
# All augmentation pipelines in one place.
#
# Design principle: color augmentations are AGGRESSIVE (they simulate
# different colorways), geometric augmentations are MILD (spatial
# pattern structure IS the design identity).

from typing import Tuple

import torch
import torchvision.transforms as T
from PIL import Image


class ChannelShuffle:
    """
    Randomly permute RGB channels.

    A red-and-gold saree becomes blue-and-gold, green-and-gold, etc.
    This forces the model to not rely on absolute color assignments.
    """

    def __call__(self, img: Image.Image) -> Image.Image:
        channels = list(img.split())
        if len(channels) == 3:
            perm = torch.randperm(3).tolist()
            channels = [channels[i] for i in perm]
            return Image.merge("RGB", channels)
        return img


class TwoViewTransform:
    """
    Produces two independently augmented views of a single image.

    Contrastive learning needs pairs: the model learns that two views
    of the same image should embed nearby, while views of different
    images should embed far apart.
    """

    def __init__(self, base_transform: T.Compose):
        self.base = base_transform

    def __call__(self, img: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.base(img), self.base(img)


# ---------------------------------------------------------------------------
# Public factory functions
# ---------------------------------------------------------------------------

def _imagenet_normalize() -> T.Normalize:
    """ImageNet channel statistics (required for pretrained backbone)."""
    return T.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])


def _color_augmentation() -> T.Compose:
    """
    Extreme color augmentation simulating different colorways.

    - hue=0.5: full hue rotation (red -> blue -> green -> red).
    - saturation 0.2..2.0: muted to vivid.
    - RandomGrayscale(0.2): 20% chance of stripping color entirely.
    - ChannelShuffle(0.3): 30% chance of swapping RGB channels.
    - GaussianBlur: slight softening for robustness.
    """
    return T.Compose([
        T.ColorJitter(brightness=0.4, contrast=0.4,
                      saturation=0.8, hue=0.5),
        T.RandomGrayscale(p=0.2),
        T.RandomApply([ChannelShuffle()], p=0.3),
        T.RandomApply(
            [T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.2
        ),
    ])


def _geometric_augmentation(size: int) -> T.Compose:
    """
    Mild geometric augmentation preserving spatial pattern structure.

    - Crop scale >= 0.5: keeps enough pattern context for matching.
    - Rotation <= 10 degrees: slight tilt from photography.
    - No random erasing: would destroy pattern structure.
    """
    return T.Compose([
        T.RandomResizedCrop(size, scale=(0.5, 1.0), ratio=(0.8, 1.2)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=10),
    ])


def contrastive_transform(image_size: int) -> TwoViewTransform:
    """Training transform: returns two augmented views per image."""
    base = T.Compose([
        _geometric_augmentation(image_size),
        _color_augmentation(),
        T.ToTensor(),
        _imagenet_normalize(),
    ])
    return TwoViewTransform(base)


def eval_transform(image_size: int) -> T.Compose:
    """Deterministic transform for inference. No augmentation."""
    return T.Compose([
        T.Resize(int(image_size * 1.14)),   # 256 for 224 input
        T.CenterCrop(image_size),
        T.ToTensor(),
        _imagenet_normalize(),
    ])


def stress_color_transform() -> T.Compose:
    """
    Extra-aggressive color augmentation for the color-invariance test.
    More extreme than training augmentation to stress-test the model.
    """
    return T.Compose([
        T.ColorJitter(brightness=0.5, contrast=0.5,
                      saturation=1.0, hue=0.5),
        T.RandomGrayscale(p=0.3),
        T.RandomApply([ChannelShuffle()], p=0.5),
    ])
