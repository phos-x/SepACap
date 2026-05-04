from __future__ import annotations

from typing import Any, Dict, Optional

from torch import nn

from .patchgan import PatchGANDiscriminator


def build_critic(cfg: Dict[str, Any]) -> Optional[nn.Module]:
    if not cfg.get("enabled", False):
        return None

    name = cfg.get("name")
    params = cfg.get("params", {})

    if name == "PatchGANDiscriminator":
        return PatchGANDiscriminator.from_dict(params)

    raise ValueError(f"Unknown critic name: {name}")
