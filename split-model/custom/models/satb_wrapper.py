import logging
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

from custom.encoder.factory import build_ssl_encoder
from models.bs_roformer.bs_roformer import BSRoformer

logger = logging.getLogger(__name__)


class SSLCrossAttention(nn.Module):
    """
    A purely modular Cross-Attention injector.
    Dimensions are strictly passed in; no hardcoded assumptions about the context model.
    """

    def __init__(self, roformer_dim: int, context_dim: int, heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        
        # Guardrail: MultiheadAttention requires embed_dim to be divisible by num_heads
        if roformer_dim % heads != 0:
            raise ValueError(f"SSLCrossAttention: roformer_dim ({roformer_dim}) must be divisible by heads ({heads})")
        
        self.k_proj = nn.Linear(context_dim, roformer_dim)
        self.v_proj = nn.Linear(context_dim, roformer_dim)
        
        self.mha = nn.MultiheadAttention(
            embed_dim=roformer_dim, 
            num_heads=heads, 
            dropout=dropout,
            batch_first=True
        )
        
        # ZERO-INIT: Acts as an Identity function at step 0 to protect pre-trained weights.
        # This MUST be normal_ with a tiny std, not zeros_, to prevent Adam division-by-zero crashes.
        nn.init.normal_(self.mha.out_proj.weight, mean=0.0, std=1e-6)
        if self.mha.out_proj.bias is not None:
            nn.init.zeros_(self.mha.out_proj.bias)

    def forward(self, x_roformer: Tensor, context_embeddings: Tensor) -> Tensor:
        # Determine the correct device for the autocast shield
        device_type = x_roformer.device.type if x_roformer.device.type in ['cuda', 'cpu'] else 'cuda'
        
        # SHIELD ACTIVATED: Force this specific mathematical operation into FP32
        with torch.autocast(device_type=device_type, enabled=False):
            # Safely cast inputs up to 32-bit floats
            q_fp32 = x_roformer.float()
            context_fp32 = context_embeddings.float()
            
            # Project Context into Key/Value space
            k_fp32 = self.k_proj(context_fp32)
            v_fp32 = self.v_proj(context_fp32)
            
            # The heavy dot-product happens safely in 32-bit space.
            # Max capacity jumps from 65,504 (FP16) to 3.4e38 (FP32), preventing NaN overflows.
            attn_out, _ = self.mha(query=q_fp32, key=k_fp32, value=v_fp32)
            
        return attn_out


class SATBSeparatorWrapper(nn.Module):
    """
    A fully portable, config-driven 'Troakes Horse' Wrapper.
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__()
        
        logger.info("Initializing modular SATBSeparatorWrapper...")
        
        # 1. Build the Context Encoder
        ssl_cfg = cfg.get("ssl_encoder", {})
        self.context_encoder = build_ssl_encoder(ssl_cfg)
        
        context_dim = None
        if self.context_encoder is not None:
            # Handle gradients & memory
            is_trainable = ssl_cfg.get("params", {}).get("trainable", False)
            if not is_trainable:
                logger.info("Context Encoder is frozen. Locking gradients...")
                for param in self.context_encoder.parameters():
                    param.requires_grad = False
                self.context_encoder.eval()
            
            # --- DYNAMIC DIMENSION INFERENCE ---
            # Do not assume 768. Try to read from config, fallback to model introspection.
            context_dim = ssl_cfg.get("context_dim") 
            if context_dim is None:
                try:
                    # Introspect the HuggingFace model config
                    context_dim = self.context_encoder._ssl.config.hidden_size
                except AttributeError:
                    raise ValueError(
                        "Could not automatically infer 'context_dim' from the SSL model. "
                        "Please specify it explicitly in your YAML under 'ssl_encoder: context_dim: X'."
                    )
            logger.info(f"Resolved context_dim: {context_dim}")

        # 2. Build the Base Separator
        model_cfg = cfg.get("model")
        if not model_cfg:
            raise ValueError("Missing 'model' configuration block in YAML.")
            
        roformer_dim = model_cfg.get("dim")
        if roformer_dim is None:
            raise ValueError("Missing 'dim' in 'model' YAML config. Cannot wire cross-attention.")

        self.separator = BSRoformer(**model_cfg)
        
        # 3. Patch the Separator
        if self.context_encoder is not None:
            cross_attn_heads = cfg.get("cross_attn_heads", 8)
            cross_attn_dropout = cfg.get("cross_attn_dropout", 0.1)
            
            logger.info(f"Patching BSRoformer: SSLCrossAttention(roformer_dim={roformer_dim}, context_dim={context_dim}, heads={cross_attn_heads})")
            
            self.separator.context_injector = SSLCrossAttention(
                roformer_dim=roformer_dim, 
                context_dim=context_dim,
                heads=cross_attn_heads,
                dropout=cross_attn_dropout
            )
        else:
            self.separator.context_injector = None

    def forward(self, mix: Tensor, target: Optional[Tensor] = None, **kwargs) -> Union[Tensor, tuple]:
        context_embeddings = None
        
        if self.context_encoder is not None:
            # Safely check if the encoder was set to trainable
            is_trainable = False
            if hasattr(self.context_encoder, '_cfg'):
                is_trainable = getattr(self.context_encoder._cfg, 'trainable', False)
                
            context_manager = torch.enable_grad() if is_trainable else torch.no_grad()
            
            with context_manager:
                context_embeddings = self.context_encoder(mix)  # (B, T_ssl, D_ssl)
        
        # Pass to separator
        out = self.separator(
            mix, 
            context_embeddings=context_embeddings, 
            target=target,
            **kwargs
        )
        
        return out