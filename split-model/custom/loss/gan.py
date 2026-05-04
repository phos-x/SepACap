from __future__ import annotations

from torch import Tensor
import torch.nn.functional as F


def discriminator_hinge_loss(
    real_scores: Tensor,
    fake_scores: Tensor,
) -> Tensor:
    """
    Hinge loss for discriminator.
    real_scores, fake_scores: (B, 1, H, W) or (B, 1)
    """
    loss_real = F.relu(1.0 - real_scores).mean()
    loss_fake = F.relu(1.0 + fake_scores).mean()
    return loss_real + loss_fake


def generator_hinge_loss(fake_scores: Tensor) -> Tensor:
    """
    Hinge loss for generator.
    fake_scores: (B, 1, H, W) or (B, 1)
    """
    return -fake_scores.mean()
