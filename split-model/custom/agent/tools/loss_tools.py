import logging
from .registry import registry

logger = logging.getLogger(__name__)

# Map human-readable stems to their index in the ZFTurbo MultiLoss array
STEM_INDEX_MAP = {"soprano": 0, "alto": 1, "tenor": 2, "bass": 3}

@registry.register(
    name="adjust_stem_loss_weight", 
    description="Multiplies the loss weight for a specific instrument. E.g., if bass is failing, multiply bass weight by 2.0."
)
def adjust_stem_loss_weight(multi_loss, stem_name: str, multiplier: float):
    stem_name = stem_name.lower()
    if stem_name not in STEM_INDEX_MAP:
        logger.error(f"🤖 TOOL ERROR: Unknown stem '{stem_name}'")
        return

    idx = STEM_INDEX_MAP[stem_name]
    
    # Assuming multi_loss has a loss_weights attribute/tensor
    if hasattr(multi_loss, 'loss_weights'):
        old_weight = multi_loss.loss_weights[idx].item()
        multi_loss.loss_weights[idx] *= multiplier
        new_weight = multi_loss.loss_weights[idx].item()
        logger.info(f"🤖 TOOL EXECUTED: '{stem_name}' loss weight adjusted from {old_weight:.2f} to {new_weight:.2f}")
    else:
        logger.warning("🤖 TOOL FAILED: multi_loss does not have 'loss_weights' attribute exposed.")