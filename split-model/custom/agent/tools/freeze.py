import logging
from .registry import registry

logger = logging.getLogger(__name__)

@registry.register(
    name="freeze_subnetwork", 
    description="Freezes a specific part of the architecture (e.g., 'context_encoder', 'separator', 'separator.context_injector'). Prevents it from learning so other parts are forced to adapt."
)
def freeze_subnetwork(model, module_name: str):
    """Surgically shuts off gradients for a specific PyTorch module."""
    target_module = None
    
    # Traverse the model to find the exact sub-network
    try:
        # Handle DDP module wrapping if present
        base_model = model.module if hasattr(model, 'module') else model
        
        if module_name == 'context_encoder' and hasattr(base_model, 'context_encoder'):
            target_module = base_model.context_encoder
        elif module_name == 'separator' and hasattr(base_model, 'separator'):
            target_module = base_model.separator
        elif module_name == 'context_injector' and hasattr(base_model.separator, 'context_injector'):
            target_module = base_model.separator.context_injector
        else:
            return f"Error: Could not locate sub-network '{module_name}' in the architecture."

        # Lock the gradients
        frozen_count = 0
        for param in target_module.parameters():
            if param.requires_grad:
                param.requires_grad = False
                frozen_count += 1
                
        target_module.eval() # Turn off dropout/batchnorm for this module
        
        success_msg = f"Successfully FROZE '{module_name}' ({frozen_count} tensors locked). It will no longer update."
        logger.info(f"🤖 TOOL EXECUTED: {success_msg}")
        return success_msg

    except Exception as e:
        return f"Failed to freeze '{module_name}': {str(e)}"


@registry.register(
    name="unfreeze_subnetwork", 
    description="Unfreezes a previously frozen sub-network so it can begin learning again."
)
def unfreeze_subnetwork(model, module_name: str):
    """Restores gradients for a specific PyTorch module."""
    target_module = None
    
    try:
        base_model = model.module if hasattr(model, 'module') else model
        
        if module_name == 'context_encoder' and hasattr(base_model, 'context_encoder'):
            target_module = base_model.context_encoder
        elif module_name == 'separator' and hasattr(base_model, 'separator'):
            target_module = base_model.separator
        elif module_name == 'context_injector' and hasattr(base_model.separator, 'context_injector'):
            target_module = base_model.separator.context_injector
        else:
            return f"Error: Could not locate sub-network '{module_name}'."

        # Unlock the gradients
        thawed_count = 0
        for param in target_module.parameters():
            if not param.requires_grad:
                param.requires_grad = True
                thawed_count += 1
                
        target_module.train() # Turn dropout/batchnorm back on
        
        success_msg = f"Successfully UNFROZE '{module_name}' ({thawed_count} tensors unlocked). It is now actively learning."
        logger.info(f"🤖 TOOL EXECUTED: {success_msg}")
        return success_msg

    except Exception as e:
        return f"Failed to unfreeze '{module_name}': {str(e)}"