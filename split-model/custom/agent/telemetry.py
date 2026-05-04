import math
import torch
import logging

logger = logging.getLogger(__name__)

def get_module_health(module: torch.nn.Module, current_lr: float) -> dict:
    """Calculates ultra-dense physics metrics for a sub-network."""
    grad_norm_sq = 0.0
    weight_norm_sq = 0.0
    zero_grads = 0
    total_params = 0
    
    for p in module.parameters():
        if p.requires_grad and p.grad is not None:
            # Accumulate norms
            grad_norm_sq += p.grad.detach().data.norm(2).item() ** 2
            weight_norm_sq += p.data.norm(2).item() ** 2
            
            # Sparsity/Dead parameter check (close to zero)
            zero_grads += (p.grad.detach().abs() < 1e-8).sum().item()
            total_params += p.numel()
            
    grad_norm = grad_norm_sq ** 0.5
    weight_norm = weight_norm_sq ** 0.5
    
    # Calculate the Holy Grail: Update Ratio
    update_ratio = 0.0
    if weight_norm > 0:
        # How much of the weight are we changing this step?
        update_ratio = (current_lr * grad_norm) / weight_norm

    sparsity = (zero_grads / total_params * 100) if total_params > 0 else 0.0

    return {
        "grad_norm": round(grad_norm, 4),
        "update_ratio": float(f"{update_ratio:.2e}"), # Scientific notation for tokens
        "sparsity_pct": round(sparsity, 1)
    }

def extract_system_telemetry(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> dict:
    """
    The Ultimate Neurological MRI for the architecture.
    """
    telemetry = {
        "ssl_encoder": {},
        "attention_bridge": {},
        "roformer": {},
        "health_status": "Nominal"
    }
    
    try:
        # Get current LR for update ratio calculation
        current_lr = optimizer.param_groups[0]['lr']
        base_model = model.module if hasattr(model, 'module') else model
        
        # 1. Inspect Wav2Vec2
        if hasattr(base_model, 'context_encoder') and base_model.context_encoder is not None:
            telemetry["ssl_encoder"] = get_module_health(base_model.context_encoder, current_lr)
            
        # 2. Inspect Cross-Attention
        if hasattr(base_model, 'separator') and hasattr(base_model.separator, 'context_injector'):
            if base_model.separator.context_injector is not None:
                telemetry["attention_bridge"] = get_module_health(base_model.separator.context_injector, current_lr)
                
        # 3. Inspect Base RoFormer
        if hasattr(base_model, 'separator'):
            telemetry["roformer"] = get_module_health(base_model.separator, current_lr)

        # Advanced Anomaly Detection Rules Engine
        norms = [telemetry["ssl_encoder"].get("grad_norm", 0), 
                 telemetry["attention_bridge"].get("grad_norm", 0)]
        
        ratios = [telemetry["ssl_encoder"].get("update_ratio", 0), 
                  telemetry["roformer"].get("update_ratio", 0)]

        if any(math.isnan(n) or math.isinf(n) for n in norms):
            telemetry["health_status"] = "CRITICAL: INF/NaN Gradient Detected"
        elif any(r > 0.1 for r in ratios):
            telemetry["health_status"] = "WARNING: Catastrophic Forgetting Risk (Update Ratio > 10%)"
        elif any(r < 1e-6 and r > 0 for r in ratios):
            telemetry["health_status"] = "WARNING: Learning Stagnation (Update Ratio < 1e-6)"

    except Exception as e:
        logger.error(f"Telemetry extraction failed: {e}")
        telemetry["health_status"] = f"Telemetry Error: {e}"

    return telemetry