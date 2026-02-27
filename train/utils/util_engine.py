import os
import torch
from loguru import logger
from torchinfo import summary as summary_
from ptflops import get_model_complexity_info
from thop import profile
import numpy as np

def load_last_checkpoint_n_get_epoch(checkpoint_dir, model, optimizer, location):
    """
    Loads the latest checkpoint with Architecture Mismatch Protection.
    Ensures that 2-spk speech weights don't crash 6-spk singing models.
    """
    if not os.path.exists(checkpoint_dir):
        logger.warning(f"Checkpoint directory {checkpoint_dir} not found. Starting from scratch.")
        return 1

    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(('.pth', '.pt', '.pkl'))]

    if not checkpoint_files:
        return 1
    
    # Sort by epoch number to find the true latest (format: epoch.0001.pth)
    try:
        epochs = [int(f.split('.')[1]) for f in checkpoint_files]
        latest_file = checkpoint_files[epochs.index(max(epochs))]
        latest_path = os.path.join(checkpoint_dir, latest_file)
    except (IndexError, ValueError) as e:
        logger.error(f"Failed to parse checkpoint filenames in {checkpoint_dir}: {e}")
        return 1

    logger.info(f"Attempting to load checkpoint: {latest_path}")
    
    checkpoint_dict = torch.load(latest_path, map_location=location, weights_only=True)
    
    try:
        # Strict=False allows loading partial weights (e.g., just the encoder)
        model.load_state_dict(checkpoint_dict['model_state_dict'], strict=False)
        logger.info("Successfully loaded model state dict.")
    except RuntimeError as e:
        # Shield: Catch size mismatches (2-speaker vs 6-speaker weights)
        logger.warning("Architecture Mismatch Detected! Checkpoint and Model have different dimensions.")
        logger.warning(f"Error Details: {str(e)[:200]}...")
        logger.warning("Initializing incompatible layers from scratch while retaining compatible ones.")

    if 'optimizer_state_dict' in checkpoint_dict and optimizer is not None:
        try:
            optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict'])
        except Exception:
            logger.warning("Optimizer state mismatch. Resetting optimizer.")
    
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
    """Saves the checkpoint if validation loss improves."""
    if valid_loss < best:
        full_path = os.path.join(checkpoint_path, "epoch.best.pth")
        _save_checkpoint(full_path, epoch, model, optimizer, train_loss, valid_loss, wandb_run)
        return valid_loss
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
    """
    Standardized summary reporting for compute audit.
    Includes 'Type Shield' casting to prevent profiler branch errors.
    """
    # Ensure input is 1D for SepACap (Batch, Samples)
    if input.dim() == 3 and input.shape[1] == 1:
        input = input.squeeze(1)


    if 'ptflops' in metrics:
        try:
            macs, params = get_model_complexity_info(
                model, (input.shape[1],), 
                print_per_layer_stat=False, 
                verbose=False
            )
            logger.info(f"ptflops: MACs: {macs}, Params: {params}")
        except Exception as e:
            logger.warning(f"ptflops profiling failed: {e}")

    if 'thop' in metrics:
        try:
            macs, params = profile(model, inputs=(input, ), verbose=False)
            logger.info(f"thop: MACs: {macs/1e9:.2f} GMac, Params: {params/1e6:.2f}M")
        except Exception as e:
            logger.warning(f"thop profiling failed: {e}")
    
    if 'torchinfo' in metrics:
        try:
            model_profile = summary_(model, input_size=input.size(), verbose=0)
            logger.info(f"torchinfo: MACs: {model_profile.total_mult_adds/1e9:.2f} GMac, Params: {model_profile.total_params/1e6:.2f}M")
        except Exception as e:
            logger.warning(f"torchinfo failed: {e}")