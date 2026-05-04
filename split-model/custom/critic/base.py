from __future__ import annotations

from abc import ABC, abstractmethod
from torch import Tensor, nn


class CriticBase(nn.Module, ABC):
    """
    Base interface for all discriminators.
    Ensures pluggability and consistent signatures.
    """

    @abstractmethod
    def forward(self, mixture_spec: Tensor, stem_spec: Tensor) -> Tensor:
        """
        Args:
            mixture_spec: (B, F, T)
            stem_spec: (B, F, T)

        Returns:
            score: (B, 1, H, W) or (B, 1)
        """
        raise NotImplementedError
