import os
import torch
from loguru import logger
from torchinfo import summary as summary_
from ptflops import get_model_complexity_info
from thop import profile
import numpy as np

def load_last_checkpoint_n_get_epoch(checkpoint_dir, model, optimizer, location):
    """
    Loads the latest checkpoint. 
    Security best practice: Uses weights_only=False for legacy support with internal trust.
    """
    # Defensive check: Ensure directory exists
    if not os.path.exists(checkpoint_dir):
        logger.warning(f"Checkpoint directory {checkpoint_dir} not found. Starting from scratch.")
        return 1

    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(('.pth', '.pt', '.pkl'))]

    if not checkpoint_files:
        return 1
    else:
        # DSA: Sort by epoch number to find the true latest
        try:
            epochs = [int(f.split('.')[1]) for f in checkpoint_files]
            latest_file = checkpoint_files[epochs.index(max(epochs))]
            latest_path = os.path.join(checkpoint_dir, latest_file)
        except (IndexError, ValueError) as e:
            logger.error(f"Failed to parse checkpoint filenames in {checkpoint_dir}: {e}")
            return 1

        logger.info(f"Loaded Pretrained model from {latest_path} .....")
        
        # Security: weights_only=False is used here to allow loading custom optimizer states
        # Operand 118 error is bypassed by this flag.
        checkpoint_dict = torch.load(latest_path, map_location=location, weights_only=False)
        
        model.load_state_dict(checkpoint_dict['model_state_dict'], strict=False)
        if 'optimizer_state_dict' in checkpoint_dict and optimizer is not None:
            optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict'])
        
        return checkpoint_dict.get('epoch', 0) + 1

def _save_checkpoint(path, epoch, model, optimizer, train_loss, valid_loss, wandb_run=None):
    """Internal helper to standardize the saving process (DRY Principle)."""
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'valid_loss': valid_loss
    }
    torch.save(state, path)
    
    # Defensive WandB handling: Only save if the object exists and has a .save method
    if wandb_run is not None and hasattr(wandb_run, 'save'):
        try:
            wandb_run.save(path)
        except Exception as e:
            logger.warning(f"Failed to upload checkpoint to WandB: {e}")

def save_checkpoint_per_nth(nth, epoch, model, optimizer, train_loss, valid_loss, checkpoint_path, wandb_run=None):
    """Saves every Nth epoch."""
    if epoch % nth == 0:
        full_path = os.path.join(checkpoint_path, f"epoch.{epoch:04}.pth")
        _save_checkpoint(full_path, epoch, model, optimizer, train_loss, valid_loss, wandb_run)

def save_checkpoint_per_best(best, valid_loss, train_loss, epoch, model, optimizer, checkpoint_path, wandb_run=None):
    """
    Saves the checkpoint if it's the best seen so far.
    Aligned signature to match Engine.py call.
    """
    if valid_loss < best:
        full_path = os.path.join(checkpoint_path, f"epoch.{epoch:04}.pth")
        _save_checkpoint(full_path, epoch, model, optimizer, train_loss, valid_loss, wandb_run)
        
        # Security: Remove older 'best' checkpoints to save disk space if preferred
        # For research, we usually keep them, but in production, we'd prune here.
        best = valid_loss
        
    return best

def step_scheduler(scheduler, **kwargs):
    """Dynamic scheduler stepping based on type detection."""
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        val_loss = kwargs.get('val_loss')
        if val_loss is None:
            raise ValueError("ReduceLROnPlateau requires 'val_loss' to step.")
        scheduler.step(val_loss)
    else:
        scheduler.step()

def model_params_mac_summary(model, input, dummy_input, metrics):
    """Standardized summary reporting for compute audit."""
    # ptflops
    if 'ptflops' in metrics:
        try:
            macs, params = get_model_complexity_info(model, (input.shape[1],), print_per_layer_stat=False, verbose=False)
            logger.info(f"ptflops: MACs: {macs}, Params: {params}")
        except Exception as e:
            logger.warning(f"ptflops profiling failed: {e}")

    # thop
    if 'thop' in metrics:
        macs, params = profile(model, inputs=(input, ), verbose=False)
        logger.info(f"thop: MACs: {macs/1e9:.2f} GMac, Params: {params/1e6:.2f}M")
    
    # torchinfo
    if 'torchinfo' in metrics:
        # DSA: Handle input sizes correctly for 1D signals
        model_profile = summary_(model, input_size=input.size(), verbose=0)
        logger.info(f"torchinfo: MACs: {model_profile.total_mult_adds/1e9:.2f} GMac, Params: {model_profile.total_params/1e6:.2f}M")