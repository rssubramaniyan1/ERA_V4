import math
import random
import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# -------------------------------
# Utilities: Mixup / CutMix / Targets
# -------------------------------

def rand_bbox(W, H, lam):
    # CutMix helper: returns bbox coords
    cut_w = int(W * math.sqrt(1 - lam))
    cut_h = int(H * math.sqrt(1 - lam))
    cx = random.randint(0, W)
    cy = random.randint(0, H)
    x1 = max(cx - cut_w // 2, 0)
    y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, W)
    y2 = min(cy + cut_h // 2, H)
    return x1, y1, x2, y2

def one_hot(labels: torch.Tensor, num_classes: int, device=None):
    device = device or labels.device
    y = torch.zeros((labels.size(0), num_classes), device=device, dtype=torch.float32)
    y.scatter_(1, labels.view(-1, 1), 1.0)
    return y

def mixup_cutmix(
    x: torch.Tensor,
    y: torch.Tensor,
    num_classes: int,
    mixup_alpha: float = 0.2,
    cutmix_alpha: float = 1.0,
    mix_prob: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns possibly-augmented (x, y_soft).
    y_soft is soft-labels suitable for label-smoothing / soft CE.
    With prob 0..mix_prob, applies either Mixup or CutMix (50/50 if both enabled).
    """
    B, C, H, W = x.shape
    y1 = one_hot(y, num_classes, device=x.device)

    if random.random() > mix_prob:
        return x, y1  # no mix

    use_cutmix = (cutmix_alpha is not None and cutmix_alpha > 0.0) and \
                 (mixup_alpha is None or mixup_alpha <= 0.0 or random.random() < 0.5)

    if use_cutmix:
        lam = torch.distributions.Beta(cutmix_alpha, cutmix_alpha).sample().item()
        idx = torch.randperm(B, device=x.device)
        x2, y2 = x[idx], y1[idx]
        x1b, y1b, x2b, y2b = rand_bbox(W, H, lam)
        x[:, :, y1b:y2b, x1b:x2b] = x2[:, :, y1b:y2b, x1b:x2b]
        # Adjust lam based on exact area
        lam = 1.0 - ((x2b - x1b) * (y2b - y1b) / float(W * H))
        y_soft = lam * y1 + (1 - lam) * y2
        return x, y_soft
    else:
        # Mixup
        lam = torch.distributions.Beta(mixup_alpha, mixup_alpha).sample().item()
        idx = torch.randperm(B, device=x.device)
        x = lam * x + (1 - lam) * x[idx]
        y2 = y1[idx]
        y_soft = lam * y1 + (1 - lam) * y2
        return x, y_soft

def soft_cross_entropy(logits, targets, label_smoothing=0.0):
    """
    Handles both hard (class indices) and soft targets (prob distributions).
    If targets are soft, we ignore label_smoothing (already soft).
    """
    if targets.dtype in (torch.long, torch.int64):
        # use built-in label smoothing
        return F.cross_entropy(logits, targets, label_smoothing=label_smoothing)
    # soft targets: CE = -sum q * log p
    log_p = F.log_softmax(logits, dim=1)
    loss = -(targets * log_p).sum(dim=1).mean()
    return loss

# -------------------------------
# EMA of weights
# -------------------------------

class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.9998, device: Optional[torch.device] = None):
        self.ema = deepcopy(model).eval()
        self.decay = decay
        self.device = device
        if device is not None:
            self.ema.to(device=device)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1.0 - d)
@dataclass
class TrainConfig:
    data_root: str
    num_classes: int = 200
    epochs: int = 200
    batch_size: int = 256
    lr: float = 0.3       # for SGD w/ large bs; scale by bs/256
    momentum: float = 0.9
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0
    mix_prob: float = 1.0
    label_smoothing: float = 0.1
    amp: bool = True
    ema_decay: float = 0.9998
    workers: int = 8
    out_dir: str = "./ckpts"
    seed: int = 42

# -------------------------------
# Datasets / Transforms (ImageNet-style)
# -------------------------------

def build_dataloaders(cfg: TrainConfig):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        normalize,
        transforms.RandomErasing(p=0.2),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    train_ds = datasets.ImageFolder(os.path.join(cfg.data_root, "train"), transform=train_tf)
    val_ds   = datasets.ImageFolder(os.path.join(cfg.data_root, "val"),   transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.workers, pin_memory=True)
    return train_loader, val_loader
