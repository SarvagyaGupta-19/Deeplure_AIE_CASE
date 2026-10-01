# data/explorer.py
# =================
# Understands the dataset structure, computes statistics,
# builds file lists, and handles de-duplication.
#
# This module ONLY reads the filesystem. It does not load images
# into memory or apply transforms. That's the dataset's job.

from pathlib import Path
from typing import Dict, List, Tuple

CLASS_NAMES = ["Banarasi", "Bandhani", "Ikat", "Pichwai"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)


def get_split_stats(data_root: str) -> Dict:
    """Per-split, per-class image counts and unique source-image counts."""
    root = Path(data_root)
    stats = {}
    for split in ["train", "valid", "test"]:
        split_stats = {}
        for cls_name in CLASS_NAMES:
            cls_dir = root / split / cls_name
            if not cls_dir.exists():
                split_stats[cls_name] = {"images": 0, "sources": 0}
                continue
            files = list(cls_dir.iterdir())
            source_ids = {
                f.stem.split(".rf.")[0] if ".rf." in f.stem else f.stem
                for f in files
            }
            split_stats[cls_name] = {
                "images": len(files),
                "sources": len(source_ids),
            }
        stats[split] = split_stats
    return stats


def print_summary(data_root: str) -> None:
    """Formatted dataset summary to stdout."""
    stats = get_split_stats(data_root)
    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)
    header = f"{'Split':<8} | "
    header += "  ".join(f"{c:>12s}" for c in CLASS_NAMES)
    header += f" | {'Total':>6s}"
    print(header)
    print("-" * 70)

    grand_total = 0
    for split, ss in stats.items():
        total = sum(v["images"] for v in ss.values())
        grand_total += total
        row = f"{split:<8} | "
        for c in CLASS_NAMES:
            s = ss[c]
            aug = s["images"] / s["sources"] if s["sources"] else 0
            row += f"{s['images']:>4d}({aug:.0f}x)   "
        row += f"|  {total:>4d}"
        print(row)

    print("-" * 70)
    print(f"Grand total: {grand_total} images")
    print(f"Source images (train): "
          f"{sum(stats['train'][c]['sources'] for c in CLASS_NAMES)}")
    print("=" * 70)


def build_file_list(
    data_root: str, split: str
) -> List[Tuple[str, int, str]]:
    """
    Build (file_path, class_idx, source_id) tuples for a given split.

    source_id groups Roboflow's augmented copies of the same original
    image. Everything before '.rf.' in the filename is the source ID.
    """
    root = Path(data_root)
    entries = []
    for cls_name in CLASS_NAMES:
        cls_dir = root / split / cls_name
        if not cls_dir.exists():
            continue
        cls_idx = CLASS_TO_IDX[cls_name]
        for f in sorted(cls_dir.iterdir()):
            if f.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
                continue
            source_id = (
                f.stem.split(".rf.")[0] if ".rf." in f.stem else f.stem
            )
            entries.append((str(f), cls_idx, source_id))
    return entries


def deduplicate(
    entries: List[Tuple[str, int, str]],
) -> List[Tuple[str, int, str]]:
    """
    Keep one image per (class, source_id) pair.

    Roboflow created 3x copies with salt-and-pepper noise. They are
    near-identical and inflate apparent performance without real learning.
    We apply our own meaningful augmentations during training instead.
    """
    seen = {}
    for path, cls_idx, source_id in entries:
        key = (cls_idx, source_id)
        if key not in seen:
            seen[key] = (path, cls_idx, source_id)
    deduped = list(seen.values())
    print(f"  De-duplicated: {len(entries)} -> {len(deduped)} images")
    return deduped
