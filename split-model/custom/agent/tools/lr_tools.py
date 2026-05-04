import logging
from .registry import registry

logger = logging.getLogger(__name__)

@registry.register(
    name="multiply_learning_rate", 
    description="Multiplies the current learning rate by a specific factor. Use < 1.0 to cool down, > 1.0 to heat up."
)
def multiply_learning_rate(optimizer, factor: float):
    for param_group in optimizer.param_groups:
        old_lr = param_group['lr']
        new_lr = old_lr * factor
        param_group['lr'] = new_lr
    logger.info(f"🤖 TOOL EXECUTED: Learning rate multiplied by {factor} (Now {new_lr})")

@registry.register(
    name="set_absolute_learning_rate", 
    description="Forces the learning rate to a specific absolute value. Use ONLY in emergencies."
)
def set_absolute_learning_rate(optimizer, target_lr: float):
    for param_group in optimizer.param_groups:
        param_group['lr'] = target_lr
    logger.info(f"🤖 TOOL EXECUTED: Learning rate hard-set to {target_lr}")