from __future__ import annotations

from typing import Any, Dict, Optional

from torch import nn

from .ssl_encoder import SSLChoralEncoder


def build_ssl_encoder(cfg: Dict[str, Any]) -> Optional[nn.Module]:
    """
    Build SSL encoder from config dict.

    Expected cfg structure:
    {
      "enabled": bool,
      "name": "SSLChoralEncoder",
      "params": { ... }
    }
    """
    if not cfg.get("enabled", False):
        return None

    name = cfg.get("name")
    params = cfg.get("params", {})

    if name == "SSLChoralEncoder":
        return SSLChoralEncoder.from_dict(params)

    raise ValueError(f"Unknown SSL encoder name: {name}")
