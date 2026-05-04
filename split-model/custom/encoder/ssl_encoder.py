from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torchaudio
from torch import Tensor, nn
from transformers import Wav2Vec2Model

# Setup module-level logger
logger = logging.getLogger(__name__)


@dataclass
class SSLEncoderConfig:
    model_name: str
    trainable: bool = False
    input_sr: int = 44100     # The sample rate coming from the ZFTurbo dataloader
    target_sr: int = 16000    # The sample rate expected by Wav2Vec2/HuBERT


class SSLChoralEncoder(nn.Module):
    """
    Configurable SSL encoder using a pretrained Wav2Vec2 model.

    Engineering Features:
    - Automatically handles Stereo -> Mono conversion.
    - Automatically handles dynamic resampling (e.g., 44.1kHz -> 16kHz).
    - Device-aware transformations.
    - Security: Prevents untrusted remote code execution.
    """

    def __init__(self, cfg: SSLEncoderConfig) -> None:
        super().__init__()
        self._cfg = cfg

        logger.info(f"Initializing SSLChoralEncoder with model: {cfg.model_name}")

        # SECURITY: Explicitly set local_files_only=False but prevent trust_remote_code=True
        # This ensures we don't accidentally execute arbitrary python files downloaded from the HuggingFace Hub.
        self._ssl = Wav2Vec2Model.from_pretrained(
            cfg.model_name,
            local_files_only=False,
            # trust_remote_code=False is the default, but enforcing it ensures security best practices.
        )

        # Optimization: Freeze or unfreeze weights based on config
        for param in self._ssl.parameters():
            param.requires_grad = cfg.trainable

        if not cfg.trainable:
            self._ssl.eval() # Force eval mode if frozen to disable dropout

        # Engineering: Setup conditional resampler
        self.resample: Optional[nn.Module] = None
        if cfg.input_sr != cfg.target_sr:
            logger.info(f"Setting up Resampler: {cfg.input_sr}Hz -> {cfg.target_sr}Hz")
            self.resample = torchaudio.transforms.Resample(
                orig_freq=cfg.input_sr,
                new_freq=cfg.target_sr
            )

    @classmethod
    def from_dict(cls, cfg_dict: Dict[str, Any]) -> SSLChoralEncoder:
        """Safely parse config dict into dataclass."""
        # Filter out unexpected keys to prevent initialization crashes
        valid_keys = SSLEncoderConfig.__dataclass_fields__.keys()
        filtered_cfg = {k: v for k, v in cfg_dict.items() if k in valid_keys}
        
        cfg = SSLEncoderConfig(**filtered_cfg)
        return cls(cfg)

    def forward(self, waveform: Tensor) -> Tensor:
        """
        Args:
            waveform: (batch_size, num_channels, num_samples)

        Returns:
            embeddings: (batch_size, time_frames, embedding_dim)
        """
        if waveform.dim() != 3:
            raise ValueError(f"SSLChoralEncoder expected waveform (B, C, T), got {waveform.shape}")

        # 1. Safe Mono Conversion
        # If stereo, average the channels. If already mono, squeeze the channel dim.
        if waveform.size(1) > 1:
            mono_waveform = waveform.mean(dim=1)  # (B, T)
        else:
            mono_waveform = waveform.squeeze(1)   # (B, T)

        # 2. Device-Aware Resampling
        if self.resample is not None:
            # Ensure the resampler's internal filter bank is on the same device as the waveform
            if self.resample.resample.kernel.device != mono_waveform.device:
                self.resample = self.resample.to(mono_waveform.device)
            
            mono_waveform = self.resample(mono_waveform)

        # 3. Extract Embeddings
        # If the model is not trainable, we enforce no_grad to save VRAM and compute,
        # acting as a safeguard even if the calling script forgets to wrap it.
        context_manager = torch.no_grad() if not self._cfg.trainable else torch.enable_grad()
        
        with context_manager:
            outputs = self._ssl(mono_waveform)
            embeddings: Tensor = outputs.last_hidden_state  # (B, T_ssl, D)

        return embeddings