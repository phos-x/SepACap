from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch
from torch import Tensor, nn

from .base import CriticBase


@dataclass
class PatchGANConfig:
    in_channels: int
    base_channels: int = 64
    num_layers: int = 4
    spectral_norm: bool = True


def _maybe_sn(layer: nn.Module, use_sn: bool) -> nn.Module:
    return nn.utils.spectral_norm(layer) if use_sn else layer


class PatchGANDiscriminator(CriticBase):
    """
    Standard PatchGAN discriminator for spectrograms.
    Configurable and pluggable.
    """

    def __init__(self, cfg: PatchGANConfig) -> None:
        super().__init__()
        self._cfg = cfg

        layers = []
        in_ch = cfg.in_channels
        out_ch = cfg.base_channels

        for _ in range(cfg.num_layers):
            conv = nn.Conv2d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=4,
                stride=2,
                padding=1,
            )
            conv = _maybe_sn(conv, cfg.spectral_norm)

            layers.append(conv)
            layers.append(nn.LeakyReLU(0.2, inplace=True))

            in_ch = out_ch
            out_ch *= 2

        # Final prediction layer
        final_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=1,
            kernel_size=4,
            stride=1,
            padding=1,
        )
        final_conv = _maybe_sn(final_conv, cfg.spectral_norm)

        layers.append(final_conv)

        self.model = nn.Sequential(*layers)

    @classmethod
    def from_dict(cls, cfg_dict: Dict[str, Any]) -> PatchGANDiscriminator:
        cfg = PatchGANConfig(**cfg_dict)
        return cls(cfg)

    def forward(self, mixture_spec: Tensor, stem_spec: Tensor) -> Tensor:
        """
        Args:
            mixture_spec: (B, F, T)
            stem_spec: (B, F, T)

        Returns:
            PatchGAN score map: (B, 1, H, W)
        """
        if mixture_spec.shape != stem_spec.shape:
            raise ValueError(
                f"Shape mismatch: mixture {mixture_spec.shape}, stem {stem_spec.shape}"
            )

        # Concatenate along channel dimension
        x = torch.stack([mixture_spec, stem_spec], dim=1)  # (B, 2, F, T)

        return self.model(x)
