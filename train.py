# train.py
# ========
# Entry point: data prep -> train Stage 1 -> train Stage 2 -> evaluate.
# Run:  py train.py

import random
import json
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import Config
from data import explorer
from data.augmentations import contrastive_transform, eval_transform
from data.dataset import SareeDataset
from models.embedding import SareeEmbeddingNet
from engine.trainer import Trainer
from engine.extractor import EmbeddingExtractor
from evaluation import identification, verification, color_invariance, efficiency
from visualization import plots
from collections import defaultdict

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def main() -> None:
    cfg = Config()
    print(f"Device: {cfg.device} | PyTorch: {torch.__version__}")

    # ---- 1. Data ----
    print("\n[1/8] Dataset")
    explorer.print_summary(cfg.data_root)

    train_files = explorer.deduplicate(
        explorer.build_file_list(cfg.data_root, "train")
    )
    valid_files = explorer.build_file_list(cfg.data_root, "valid")
    test_files  = explorer.build_file_list(cfg.data_root, "test")
    print(f"  Train: {len(train_files)} | Valid: {len(valid_files)}"
          f" | Test: {len(test_files)}")

    # ---- 2. Loaders ----
    print("\n[2/8] DataLoaders")
    ct = contrastive_transform(cfg.image_size)
    et = eval_transform(cfg.image_size)

    train_ds = SareeDataset(train_files, ct, mode="contrastive")
    val_ds   = SareeDataset(valid_files, ct, mode="contrastive")

    train_ld = DataLoader(train_ds, cfg.batch_size, shuffle=True,
                          num_workers=cfg.num_workers, drop_last=True)
    val_ld   = DataLoader(val_ds, cfg.batch_size, shuffle=False,
                          num_workers=cfg.num_workers, drop_last=True)
    print(f"  Train: {len(train_ld)} batches | Val: {len(val_ld)} batches")

    # ---- 3. Model ----
    print("\n[3/8] Model")
    model = SareeEmbeddingNet(cfg.backbone, cfg.embedding_dim, cfg.pretrained)
    pc = model.param_count()
    print(f"  {cfg.backbone} | {pc['total']:,} params | {cfg.embedding_dim}-d embedding")

    # ---- 4. Train ----
    print("\n[4/8] Training")
    trainer = Trainer(model, cfg)
    trainer.train_stage1(train_ld, val_ld)
    trainer.train_stage2(train_ld, val_ld)

    # Load best checkpoint
    try:
        trainer.load_checkpoint("best_stage2.pth")
    except FileNotFoundError:
        trainer.load_checkpoint("final_stage2.pth")

    # ---- 5. Embeddings ----
    print("\n[5/8] Extracting embeddings")
    ext = EmbeddingExtractor(model, cfg)

    gallery_embs, gallery_labs, gallery_paths = ext.extract(
        SareeDataset(valid_files, et, mode="eval")
    )
    query_embs, query_labs, query_paths = ext.extract(
        SareeDataset(test_files, et, mode="eval")
    )
    train_embs, train_labs, _ = ext.extract(
        SareeDataset(train_files, et, mode="eval")
    )
    print(f"  Gallery: {gallery_embs.shape} | Query: {query_embs.shape}")

    # ---- 6. Evaluate ----
    print("\n[6/8] Evaluation")

    id_res = identification.evaluate(
        query_embs, query_labs, gallery_embs, gallery_labs, cfg.recall_k_values
    )
    print("  -- Identification --")
    for k, v in id_res.items():
        print(f"     {k}: {v:.4f}")

    all_embs = np.concatenate([gallery_embs, query_embs])
    all_labs = np.concatenate([gallery_labs, query_labs])
    ver_res = verification.evaluate(all_embs, all_labs, cfg.verification_num_pairs)
    print("  -- Verification --")
    for k, v in ver_res.items():
        print(f"     {k}: {v:.4f}" if isinstance(v, float) else f"     {k}: {v}")

    ci_res = color_invariance.evaluate(
        model, test_files, cfg.image_size, cfg.device,
        num_samples=min(50, len(test_files)),
    )
    print("  -- Color Invariance --")
    for k, v in ci_res.items():
        print(f"     {k}: {v:.4f}" if isinstance(v, float) else f"     {k}: {v}")

    # ---- 7. Visualizations ----
    print("\n[7/8] Plots")
    p = plots.training_history(trainer.history, cfg.output_dir)
    print(f"  {p}")
    p = plots.embedding_space(train_embs, train_labs, cfg.output_dir)
    print(f"  {p}")

    # Similarity distribution data
    by_cls = defaultdict(list)
    for i, lab in enumerate(all_labs):
        by_cls[lab].append(i)
    pos_s, neg_s = [], []
    classes = list(by_cls.keys())
    for _ in range(500):
        c = random.choice(classes)
        if len(by_cls[c]) >= 2:
            a, b = random.sample(by_cls[c], 2)
            pos_s.append(float(all_embs[a] @ all_embs[b]))
    for _ in range(500):
        c1, c2 = random.sample(classes, 2)
        a, b = random.choice(by_cls[c1]), random.choice(by_cls[c2])
        neg_s.append(float(all_embs[a] @ all_embs[b]))
    p = plots.similarity_distributions(pos_s, neg_s, cfg.output_dir)
    print(f"  {p}")

    p = plots.retrieval_examples(
        query_embs, query_labs, query_paths,
        gallery_embs, gallery_labs, gallery_paths, cfg.output_dir,
    )
    print(f"  {p}")

    sim_mat = query_embs @ gallery_embs.T
    p = plots.confusion_matrix(query_labs, gallery_labs, sim_mat, cfg.output_dir)
    print(f"  {p}")

    # ---- 8. Efficiency + Save ----
    print("\n[8/8] Efficiency & Save")
    eff = efficiency.report(model, cfg)
    for k, v in eff.items():
        print(f"     {k}: {v}")

    results = {
        "identification": id_res,
        "verification": {k: (float(v) if isinstance(v, (float, np.floating)) else v)
                         for k, v in ver_res.items()},
        "color_invariance": {k: (float(v) if isinstance(v, (float, np.floating)) else v)
                             for k, v in ci_res.items()},
        "efficiency": eff,
    }
    out = f"{cfg.output_dir}/results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results -> {out}")

    # ---- Summary ----
    print("\n" + "=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50)
    print(f"  Recall@1:           {id_res['Recall@1']:.4f}")
    print(f"  Recall@5:           {id_res.get('Recall@5', 'N/A')}")
    print(f"  mAP:                {id_res['mAP']:.4f}")
    print(f"  AUC-ROC:            {ver_res['AUC-ROC']:.4f}")
    print(f"  EER:                {ver_res['EER']:.4f}")
    print(f"  Color Inv. Gap:     {ci_res['Color_Invariance_Gap']:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
