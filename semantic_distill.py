
import math
from dataclasses import dataclass
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------
# Projection head (training-only)
# -------------------------------

class ProjectionHead(nn.Module):
    """
    Maps pooled ResNet-50 features (2048-d) into teacher/VLM embedding dim (e.g., 512).
    A simple BN -> Linear works well; add another BN if desired.
    """
    def __init__(self, in_dim: int = 2048, out_dim: int = 512, with_bn: bool = True):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_dim) if with_bn else nn.Identity()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.normal_(self.fc.weight, std=0.02)

        # avoid BN on tiny batch by setting eval mode dynamically if needed
        self._use_bn = with_bn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._use_bn:
            x = self.bn(x)
        x = self.fc(x)
        x = F.normalize(x, dim=1)  # L2 norm
        return x


# -------------------------------
# Losses for semantic distillation
# -------------------------------

def kl_divergence(p_t: torch.Tensor, p_s: torch.Tensor) -> torch.Tensor:
    """
    KL(p_t || p_s) with safe-guards; both are probs (softmaxed).
    Returns mean over batch.
    """
    eps = 1e-7
    p_t = p_t.clamp_min(eps)
    p_s = p_s.clamp_min(eps)
    return torch.sum(p_t * (torch.log(p_t) - torch.log(p_s)), dim=1).mean()


def cosine_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    1 - cosine similarity (mean over batch).
    Inputs should be L2-normalized, but we still protect for numeric stability.
    """
    a = F.normalize(a, dim=1)
    b = F.normalize(b, dim=1)
    cos = torch.sum(a * b, dim=1)
    return (1.0 - cos).mean()


# -------------------------------
# Semantic distillation module
# -------------------------------

@dataclass
class DistillWeights:
    ce: float = 1.0
    semantic_kl: float = 0.5
    feat_align: float = 0.25
    tau_student: float = 0.07
    tau_teacher: float = 0.07


class SemanticDistiller(nn.Module):
    """
    Wraps the semantic distillation computation:
      - Builds student semantic scores s = (f_hat @ T^T)/tau_s
      - Uses teacher scores t = (e_img @ T^T)/tau_t if image embeddings are given
        otherwise falls back to a class-prototype push/pull loss
      - Adds an optional feature alignment loss between f_hat and e_img
    """
    def __init__(
        self,
        prototypes: torch.Tensor,   # [C, D], L2-normalized
        distill_weights: DistillWeights,
        use_feature_alignment: bool = True,
    ):
        super().__init__()
        assert prototypes.ndim == 2, "prototypes must be [num_classes, embed_dim]"
        self.register_buffer("prototypes", F.normalize(prototypes.float(), dim=1), persistent=False)
        self.cfg = distill_weights
        self.use_fa = use_feature_alignment

    def forward(
        self,
        f_hat: torch.Tensor,        # [B, D] student projected features (L2-normalized)
        labels: torch.Tensor,       # [B] int64
        e_img: Optional[torch.Tensor] = None,  # [B, D] teacher image embeddings (L2-normalized) or None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Returns (loss_semantic, stats_dict). Does NOT include CE; add externally.
        """
        B, D = f_hat.shape
        C, Dp = self.prototypes.shape
        assert D == Dp, "f_hat dim must match prototypes dim"

        # Student semantic scores -> probs
        s = (f_hat @ self.prototypes.t()) / self.cfg.tau_student  # [B, C]
        p_s = F.softmax(s, dim=1)

        stats = {}

        if e_img is not None:
            # Teacher per-image probs via prototypes
            t = (e_img @ self.prototypes.t()) / self.cfg.tau_teacher  # [B, C]
            p_t = F.softmax(t, dim=1)
            loss_sem = kl_divergence(p_t, p_s)
            stats["kl"] = float(loss_sem.detach().cpu())
        else:
            # Fallback: supervised contrast / margin against negatives
            # Pull to true class prototype; push from hardest negatives
            pos = torch.sum(f_hat * self.prototypes[labels], dim=1)              # [B]
            scores = f_hat @ self.prototypes.t()  # [B, C]
            scores[torch.arange(B), labels] = -1e9
            neg, _ = torch.max(scores, dim=1)  # [B]
            margin = 0.2
            loss_sem = torch.clamp(margin - pos + neg, min=0.0).mean()
            stats["contrastive"] = float(loss_sem.detach().cpu())

        # Optional feature alignment
        loss_fa = torch.tensor(0.0, device=f_hat.device)
        if self.use_fa and (e_img is not None):
            loss_fa = cosine_loss(f_hat, e_img)
            stats["fa"] = float(loss_fa.detach().cpu())

        # Weighted sum (CE is added outside)
        weighted = self.cfg.semantic_kl * loss_sem + self.cfg.feat_align * loss_fa
        stats["weighted_sem"] = float(weighted.detach().cpu())
        return weighted, stats
