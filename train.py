import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import random
import numpy as np

from resnet50 import resnet50
from utilities import (
    mixup_cutmix,
    soft_cross_entropy,
    ModelEMA,
    TrainConfig,
    build_dataloaders,
)


def main():
    cfg = TrainConfig(data_root="./tiny-imagenet-200", num_classes=200)

    # Seed everything
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    # Dataloaders
    train_loader, val_loader = build_dataloaders(cfg)

    # Model, optimizer, scheduler, loss
    model = resnet50(num_classes=cfg.num_classes).cuda()
    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler = GradScaler(enabled=cfg.amp)
    ema = ModelEMA(model, decay=cfg.ema_decay)

    # Training loop
    for epoch in range(cfg.epochs):
        model.train()
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs}")
        for x, y in progress:
            x, y = x.cuda(), y.cuda()

            # Mixup/Cutmix
            x, y_soft = mixup_cutmix(
                x,
                y,
                num_classes=cfg.num_classes,
                mixup_alpha=cfg.mixup_alpha,
                cutmix_alpha=cfg.cutmix_alpha,
            )

            with autocast(enabled=cfg.amp):
                logits = model(x)
                loss = soft_cross_entropy(logits, y_soft, label_smoothing=cfg.label_smoothing)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)

            progress.set_postfix(loss=loss.item())

        # Validation
        model.eval()
        val_loss = 0
        val_acc = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.cuda(), y.cuda()
                with autocast(enabled=cfg.amp):
                    logits = ema.ema(x)  # Use EMA model for validation
                    val_loss += nn.CrossEntropyLoss()(logits, y).item()
                val_acc += (logits.argmax(dim=1) == y).float().mean().item()

        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        print(f"Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}")

        scheduler.step()


if __name__ == "__main__":
    main()
