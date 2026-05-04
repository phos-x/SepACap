# models/bs_roformer/bs_roformer.py (conceptual patch)

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn


class BSRoFormer(nn.Module):
    # existing __init__ ...

    def forward(
        self,
        mix_spec: Tensor,
        ssl_embeddings: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            mix_spec: (B, F, T)
            ssl_embeddings: optional (B, T_ssl, D)

        Returns:
            stems_spec: (B, num_stems, F, T)
        """
        if ssl_embeddings is not None:
            ssl_embeddings = self._align_ssl_to_spec(
                ssl_embeddings,
                target_frames=mix_spec.size(2),
            )

        return self._forward_with_optional_ssl(mix_spec, ssl_embeddings)

    def _align_ssl_to_spec(
        self,
        ssl_embeddings: Tensor,
        target_frames: int,
    ) -> Tensor:
        """
        Align SSL time dimension to spectrogram time dimension via interpolation.
        """
        if ssl_embeddings.dim() != 3:
            raise ValueError(f"Expected ssl_embeddings (B, T, D), got {ssl_embeddings.shape}")

        # (B, T, D) -> (B, D, T)
        ssl_t = ssl_embeddings.transpose(1, 2)

        ssl_resampled = nn.functional.interpolate(
            ssl_t,
            size=target_frames,
            mode="linear",
            align_corners=False,
        )

        # (B, D, T_spec) -> (B, T_spec, D)
        return ssl_resampled.transpose(1, 2)

    def _forward_with_optional_ssl(
        self,
        mix_spec: Tensor,
        ssl_embeddings: Optional[Tensor],
    ) -> Tensor:
        """
        Hook to integrate SSL embeddings into the existing BS-RoFormer pipeline.
        For now, you can ignore ssl_embeddings and keep baseline behavior,
        then gradually wire it into the band-split transformer.
        """
        # TODO: integrate ssl_embeddings into your internal blocks.
        return self._forward_baseline(mix_spec)

    def _forward_baseline(self, mix_spec: Tensor) -> Tensor:
        # Existing implementation
        raise NotImplementedError("Connect this to the current BS-RoFormer forward logic.")


##########################################
#_________________________________________#
###############################################

##patch for training loop 

from __future__ import annotations

from typing import Dict, Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from custom.embedding.factory import build_ssl_encoder
from models.bs_roformer.bs_roformer import BSRoFormer


def train_one_epoch(
    model: BSRoFormer,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg: Dict[str, Any],
) -> Dict[str, float]:
    model.train()

    ssl_encoder = build_ssl_encoder(cfg.get("ssl_encoder", {}))
    if ssl_encoder is not None:
        ssl_encoder.to(device)
        ssl_encoder.eval()  # freeze by default; change if training jointly

    epoch_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        mixture_waveform: Tensor = batch["mixture"].to(device)  # (B, C, T)
        stems_waveform: Tensor = batch["stems"].to(device)      # (B, num_stems, C, T)

        optimizer.zero_grad(set_to_none=True)

        ssl_embeddings = None
        if ssl_encoder is not None:
            with torch.no_grad():
                ssl_embeddings = ssl_encoder(mixture_waveform)  # (B, T_ssl, D)

        mix_spec: Tensor = waveform_to_spectrogram(mixture_waveform)  # use repo util

        stems_spec_pred: Tensor = model(
            mix_spec,
            ssl_embeddings=ssl_embeddings,
        )

        loss: Tensor = compute_separation_loss(stems_spec_pred, stems_waveform)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += float(loss.item())
        num_batches += 1

    return {"loss": epoch_loss / max(num_batches, 1)}

## gan training loop patch

critic = build_critic(cfg.get("critic", {}))
if critic is not None:
    critic.to(device)
    critic.train()

## during training loop:
# Forward separator
stems_spec_pred = separator(mix_spec, ssl_embeddings)

# Critic loss
if critic is not None:
    critic_score_fake = critic(mix_spec, stems_spec_pred.detach())
    critic_score_real = critic(mix_spec, stems_spec_gt)

    loss_g = generator_loss(critic_score_fake)
    loss_d = discriminator_loss(critic_score_real, critic_score_fake)

    total_loss = loss_recon + lambda_gan * loss_g
else:
    total_loss = loss_recon



########################################################################################################
#______________________________________________________________________________________________________#
#########################################################################################################

# agent training patch 

agent = build_agent(cfg.get("agent", {}))

for epoch in range(num_epochs):
    metrics = train_one_epoch(...)
    validate(...)

    if agent is not None and epoch % agent._cfg.interval == 0:
        snapshot = {
            "epoch": epoch,
            "metrics": metrics,
            "config": cfg,
            "model_state": {
                "grad_norm": grad_norm,
                "lr": optimizer.param_groups[0]["lr"],
            },
        }

        actions = agent.analyze(snapshot)
        apply_actions(actions, optimizer, critic)


def apply_actions(actions, optimizer, critic):
    for action in actions["actions"]:
        if action == "reduce_lr":
            optimizer.param_groups[0]["lr"] *= 0.8
        elif action == "increase_lr":
            optimizer.param_groups[0]["lr"] *= 1.1
        elif action == "freeze_critic" and critic is not None:
            for p in critic.parameters():
                p.requires_grad = False
        elif action == "unfreeze_critic" and critic is not None:
            for p in critic.parameters():
                p.requires_grad = True
        elif action == "run_data_check":
            run_data_alignment_check()
        elif action == "continue":
            pass


## loss/gan.py patch

from custom.gan.losses import discriminator_hinge_loss, generator_hinge_loss

# Forward separator
stems_spec_pred = separator(mix_spec, ssl_embeddings)

# Reconstruction loss
loss_recon = compute_separation_loss(stems_spec_pred, stems_spec_gt)

if critic is not None:
    critic_real = critic(mix_spec, stems_spec_gt)
    critic_fake = critic(mix_spec, stems_spec_pred.detach())

    loss_d = discriminator_hinge_loss(critic_real, critic_fake)

    critic_optimizer.zero_grad(set_to_none=True)
    loss_d.backward()
    critic_optimizer.step()

    critic_fake_for_g = critic(mix_spec, stems_spec_pred)
    loss_g = generator_hinge_loss(critic_fake_for_g)

    total_loss = loss_recon + lambda_gan * loss_g
else:
    total_loss = loss_recon


#### bs_roformer forward patch

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn


class BSRoFormer(nn.Module):
    # existing __init__ ...

    def forward(
        self,
        mix_spec: Tensor,
        ssl_embeddings: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            mix_spec: (B, F, T)
            ssl_embeddings: optional (B, T_ssl, D)

        Returns:
            stems_spec: (B, num_stems, F, T)
        """
        if ssl_embeddings is not None:
            ssl_embeddings = self._align_ssl_to_spec(
                ssl_embeddings,
                target_frames=mix_spec.size(2),
            )

        return self._forward_with_optional_ssl(mix_spec, ssl_embeddings)

    def _align_ssl_to_spec(
        self,
        ssl_embeddings: Tensor,
        target_frames: int,
    ) -> Tensor:
        """
        Align SSL time dimension to spectrogram time dimension via interpolation.
        """
        if ssl_embeddings.dim() != 3:
            raise ValueError(f"Expected ssl_embeddings (B, T, D), got {ssl_embeddings.shape}")

        ssl_t = ssl_embeddings.transpose(1, 2)  # (B, D, T_ssl)

        ssl_resampled = nn.functional.interpolate(
            ssl_t,
            size=target_frames,
            mode="linear",
            align_corners=False,
        )

        return ssl_resampled.transpose(1, 2)  # (B, T_spec, D)

    def _forward_with_optional_ssl(
        self,
        mix_spec: Tensor,
        ssl_embeddings: Optional[Tensor],
    ) -> Tensor:
        """
        Hook to integrate SSL embeddings into BS-RoFormer internals.
        For now, you can ignore ssl_embeddings and keep baseline behavior,
        then gradually wire it into the band-split transformer.
        """
        # TODO: integrate ssl_embeddings into internal blocks (e.g., FiLM, concat, cross-attention).
        return self._forward_baseline(mix_spec)

    def _forward_baseline(self, mix_spec: Tensor) -> Tensor:
        # Existing BS-RoFormer forward logic.
        raise NotImplementedError("Connect this to the current implementation.")


