import torch
import torch.nn as nn
from typing import Dict, Any

def extract_system_telemetry(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> Dict[str, Any]:
    """
    The Neurological MRI.
    Invoked periodically by the Orchestrator to give the Chief Scientist 
    a high-resolution map of the network's architecture and frozen state.
    """
    telemetry = {
        "health_status": "Nominal",
        "architecture_map": {},
        "frozen_layers": [],
        "active_layers": [],
        "optimizer_state": {
            "name": optimizer.__class__.__name__,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "weight_decay": optimizer.param_groups[0].get('weight_decay', 0.0)
        }
    }

    # Handle DDP wrapping
    base_model = model.module if hasattr(model, 'module') else model

    # Detect Sub-Networks
    w2v2_module = getattr(base_model, 'context_encoder', None)
    roformer_module = getattr(base_model, 'separator', None)

    # 1. Map Wav2Vec2 (Context Encoder)
    if w2v2_module is not None:
        telemetry["architecture_map"]["context_encoder"] = "Wav2Vec2 (SSL)"
        frozen_w2v2 = 0
        active_w2v2 = 0
        for name, param in w2v2_module.named_parameters():
            if not param.requires_grad:
                frozen_w2v2 += 1
                telemetry["frozen_layers"].append(f"context_encoder.{name}")
            else:
                active_w2v2 += 1
                telemetry["active_layers"].append(f"context_encoder.{name}")
        telemetry["architecture_map"]["context_encoder_frozen_params"] = frozen_w2v2
        telemetry["architecture_map"]["context_encoder_active_params"] = active_w2v2

    # 2. Map RoFormer (Separator)
    if roformer_module is not None:
        telemetry["architecture_map"]["separator"] = "BSRoFormer"
        frozen_roformer = 0
        active_roformer = 0
        for name, param in roformer_module.named_parameters():
            if not param.requires_grad:
                frozen_roformer += 1
                telemetry["frozen_layers"].append(f"separator.{name}")
            else:
                active_roformer += 1
                # We don't append every RoFormer layer to active_layers to save token space,
                # we just summarize unless they are explicitly frozen.
                
        telemetry["architecture_map"]["separator_frozen_params"] = frozen_roformer
        telemetry["architecture_map"]["separator_active_params"] = active_roformer

    # 3. Detect Critical Deadlocks (Sanity Check)
    if len(telemetry["active_layers"]) == 0 and active_roformer == 0:
        telemetry["health_status"] = "CRITICAL: ALL LAYERS FROZEN. NO GRADIENTS WILL FLOW."

    return telemetry