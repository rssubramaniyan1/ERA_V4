
import argparse
import os
import random
import numpy as np
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from resnet50 import resnet50
from utilities import (
    mixup_cutmix,
    soft_cross_entropy,
    ModelEMA,
    TrainConfig,
    build_dataloaders,
)
from semantic_distill import (
    ProjectionHead,
    SemanticDistiller,
    DistillWeights,
)

def load_prototypes(path: str, expected_dim: int) -> torch.Tensor:
    """
    Loads a prototypes npz/npy/pt file containing [num_classes, D] vectors.
    """
    ext = os.path.splitext(path)[-1].lower()
    if ext in [".npy", ".npz"]:
        arr = np.load(path, allow_pickle=True)
        if isinstance(arr, np.lib.npyio.NpzFile):
            # try common keys
            for k in ["prototypes", "T", "text_embs", "text_prototypes"]:
                if k in arr.files:
                    arr = arr[k]
                    break
            else:
                raise ValueError(f"No known prototype key found in {list(arr.files)}")
        arr = torch.tensor(arr)
    elif ext == ".pt":
        arr = torch.load(path, map_location="cpu")
    else:
        raise ValueError(f"Unsupported prototype file extension: {ext}")
    if arr.ndim != 2 or arr.shape[1] != expected_dim:
        raise ValueError(f"Prototype shape must be [C,{expected_dim}], got {arr.shape}")
    return arr.float()

def load_image_embeddings(path: Optional[str]) -> Optional[torch.Tensor]:
    if path is None:
        return None
    ext = os.path.splitext(path)[-1].lower()
    if ext in [".npy", ".npz"]:
        arr = np.load(path, allow_pickle=True)
        if isinstance(arr, np.lib.npyio.NpzFile):
            for k in ["img_embs", "E", "image_embeddings"]:
                if k in arr.files:
                    arr = arr[k]
                    break
            else:
                raise ValueError(f"No known image-embedding key found in {list(arr.files)}")
        arr = torch.tensor(arr)
    elif ext == ".pt":
        arr = torch.load(path, map_location="cpu")
    else:
        raise ValueError(f"Unsupported image-embedding file extension: {ext}")
    return arr.float()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="./tiny-imagenet-200")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--ema-decay", type=float, default=0.9998)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--num-classes", type=int, default=200)
    parser.add_argument("--embed-dim", type=int, default=512, help="Teacher/VLM embedding dim")
    parser.add_argument("--prototypes", type=str, required=True, help="Path to text-prototype matrix [C,D]")
    parser.add_argument("--image-embeddings", type=str, default=None, help="Optional path to per-image embeddings [N,D]")
    parser.add_argument("--lam-ce", type=float, default=1.0)
    parser.add_argument("--lam-sd", type=float, default=0.5)
    parser.add_argument("--lam-fa", type=float, default=0.25)
    parser.add_argument("--tau-s", type=float, default=0.07)
    parser.add_argument("--tau-t", type=float, default=0.07)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # --- Seed ---
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    # --- Dataloaders ---
    cfg = TrainConfig(
        data_root=args.data_root,
        num_classes=args.num_classes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        amp=args.amp,
        ema_decay=args.ema_decay,
    )
    train_loader, val_loader = build_dataloaders(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Model & heads ---
    model = resnet50(num_classes=cfg.num_classes).to(device)
    proj = ProjectionHead(in_dim=2048, out_dim=args.embed_dim).to(device)

    # --- Distiller ---
    prototypes = load_prototypes(args.prototypes, expected_dim=args.embed_dim).to(device)
    distiller = SemanticDistiller(
        prototypes=prototypes,
        distill_weights=DistillWeights(
            ce=args.lam_ce,
            semantic_kl=args.lam_sd,
            feat_align=args.lam_fa,
            tau_student=args.tau_s,
            tau_teacher=args.tau_t,
        ),
        use_feature_alignment=True,
    ).to(device)

    # Optional per-image embeddings (aligned with dataset order)
    img_embs = load_image_embeddings(args.image_embeddings)
    if img_embs is not None:
        img_embs = img_embs.to(device)

    # --- Optimizer/Scheduler/EMA ---
    params = list(model.parameters()) + list(proj.parameters())
    optimizer = optim.SGD(
        params, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler = GradScaler(enabled=cfg.amp)
    ema = ModelEMA(model, decay=cfg.ema_decay, device=device)

    # Helper: forward hook to grab pooled features
    pooled = {"feat": None}
    def _hook(m, i, o):
        pooled["feat"] = torch.flatten(o, 1)
    handle = model.avgpool.register_forward_hook(_hook)

    # --- Training ---
    for epoch in range(cfg.epochs):
        model.train(); proj.train()
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs}")

        for step, (x, y) in enumerate(progress):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            # Mixup/CutMix from utilities
            x_aug, y_soft = mixup_cutmix(
                x, y, num_classes=cfg.num_classes,
                mixup_alpha=cfg.mixup_alpha,
                cutmix_alpha=cfg.cutmix_alpha,
                mix_prob=cfg.mix_prob,
            )

            with autocast(enabled=cfg.amp):
                logits = model(x_aug)  # [B, C]
                if pooled["feat"] is None:
                    raise RuntimeError("Failed to capture pooled features. Consider exposing features in resnet.")
                f_hat = proj(pooled["feat"])  # [B, D]

                # Teacher image embeddings for this batch (if provided offline)
                e_img = None
                if img_embs is not None:
                    # NOTE: For a robust mapping, modify your dataset to also return 'index'
                    # and use that to slice img_embs[index]. Here we fallback to 'range' slice.
                    e_img = img_embs[: f_hat.shape[0]]

                loss_ce = soft_cross_entropy(logits, y_soft, label_smoothing=cfg.label_smoothing)
                loss_sem, stats = distiller(f_hat, labels=y, e_img=e_img)
                loss = args.lam_ce * loss_ce + stats["weighted_sem"]

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            ema.update(model)
            pooled["feat"] = None  # reset for next batch

            progress.set_postfix(loss=float(loss.detach().cpu()), ce=float(loss_ce.detach().cpu()), **{k: round(v,4) for k,v in stats.items()})

        # --- Validation (EMA model) ---
        model.eval(); proj.eval()
        val_loss = 0.0; val_acc = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                with autocast(enabled=cfg.amp):
                    logits = ema.ema(x)
                    val_loss += nn.CrossEntropyLoss()(logits, y).item()
                val_acc += (logits.argmax(dim=1) == y).float().mean().item()

        val_loss /= len(val_loader)
        val_acc  /= len(val_loader)
        print(f"Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}")
        scheduler.step()

    handle.remove()


if __name__ == "__main__":
    main()
