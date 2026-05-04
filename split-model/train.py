# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo): https://github.com/ZFTurbo/'
__version__ = '1.0.5'

import argparse
import sys

import numpy as np
from tqdm.auto import tqdm
import torch
import wandb
import torch.nn as nn
from torch.utils.data import DataLoader
from ml_collections import ConfigDict
from typing import List, Callable, Union, Tuple
import torch.distributed as dist
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from utils.settings import get_scheduler, parse_args_train, initialize_environment_ddp, \
    initialize_environment, get_model_from_config, wandb_init
from utils.model_utils import save_weights, normalize_batch, \
    save_last_weights, initialize_model_and_device

from valid import valid_multi_gpu, valid

import warnings

# Try to import the custom agent factory gracefully
try:
    from custom.agent.agent_factory import build_agent
except ImportError:
    print(f"Custom agent factory not found. Proceeding without LLM agent. Error: {sys.exc_info()[0]}")
    build_agent = None

warnings.filterwarnings("ignore")

def forward_step(x, y, active_stem_ids, get_internal_loss, model, multi_loss, device_ids):
    if get_internal_loss:
        loss = model(x, y, active_stem_ids=active_stem_ids)
        if isinstance(device_ids, (list, tuple)):
            loss = loss.mean()
        return loss
    else:
        y_ = model(x)
        return multi_loss(y_, y, x)


def train_one_epoch(model: torch.nn.Module, config: ConfigDict, args: argparse.Namespace,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device, device_ids: List[int], epoch: int, use_amp: bool,
                    scaler: torch.cuda.amp.GradScaler,
                    scheduler,
                    gradient_accumulation_steps: int, train_loader: torch.utils.data.DataLoader,
                    multi_loss: Callable[[torch.Tensor, torch.Tensor, torch.Tensor,], torch.Tensor], all_losses=None, world_size=None, ema_model=None, safe_mode=None) -> float:
    """
    Train the model for one epoch.
    Returns:
        The average training loss for the epoch.
    """
    ddp = True if world_size else False
    should_print = not dist.is_initialized() or dist.get_rank() == 0
    model.train()
    if not ddp:
        model.to(device)
    if should_print:
        print(f'Train epoch: {epoch} Learning rate: {optimizer.param_groups[0]["lr"]}')
        sys.stdout.flush()
    loss_val = 0.
    total = 0
    all_losses[f'epoch_{epoch}'] = []

    normalize = getattr(config.training, 'normalize', False)

    get_internal_loss = (args.model_type in (
        'mel_band_roformer',
        'bs_roformer',
        'bs_mamba2',
        'mel_band_conformer',
        'bs_conformer'
    ) and not args.use_standard_loss)

    if ddp:
        pbar = tqdm(train_loader,
                    dynamic_ncols=True) if dist.get_rank() == 0 else train_loader
    else:
        pbar = tqdm(train_loader)

    for i, data in enumerate(pbar):
        if len(data)==3:
            batch, mixes, active_stem_ids = data
        elif len(data)==2:
            batch, mixes = data
            active_stem_ids = None
        else:
            raise ValueError(f'len data is {len(data)}')
        x = mixes.to(device)
        y = batch.to(device)

        if normalize:
            x, y = normalize_batch(x, y)
        if safe_mode:
            try:
                with torch.cuda.amp.autocast(enabled=use_amp):
                    loss = forward_step(x, y, active_stem_ids, get_internal_loss, model, multi_loss, device_ids)
            except Exception as e:
                print(f'Error: {e}')
                continue
        else:
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = forward_step(x, y, active_stem_ids, get_internal_loss, model, multi_loss, device_ids)
        loss /= gradient_accumulation_steps
        scaler.scale(loss).backward()

        if ((i + 1) % gradient_accumulation_steps == 0) or (i == len(train_loader) - 1):

            scaler.unscale_(optimizer)

            if config.training.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            if ema_model is not None:
                if ddp:
                    ema_model.update_parameters(model.module)
                else:
                    ema_model.update_parameters(model)

            if scheduler.name in ['linear_scheduler']:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            
        if ddp:
            with torch.no_grad():
                loss_copy = loss.detach().clone()
                dist.all_reduce(loss_copy, op=dist.ReduceOp.SUM)
                loss_copy /= dist.get_world_size()
            if dist.get_rank() == 0:
                li = loss_copy.item() * gradient_accumulation_steps
                all_losses[f'epoch_{epoch}'].append(li)
                loss_val += li
                total += 1
                pbar.set_postfix({'loss': 100 * li, 'avg_loss': 100 * loss_val / (i + 1)})
                sys.stdout.flush()
                wandb.log({'loss': 100 * li, 'avg_loss': 100 * loss_val / (i + 1), 'i': i})
        else:
            li = loss.item() * gradient_accumulation_steps
            all_losses[f'epoch_{epoch}'].append(li)
            loss_val += li
            total += 1
            pbar.set_postfix({'loss': 100 * li, 'avg_loss': 100 * loss_val / (i + 1)})
            wandb.log({'loss': 100 * li, 'avg_loss': 100 * loss_val / (i + 1), 'i': i})
            loss.detach()

    avg_train_loss = loss_val / max(total, 1)
    if should_print:
        print(f'Training loss: {avg_train_loss}')
        wandb.log({'train_loss': avg_train_loss, 'epoch': epoch, 'learning_rate': optimizer.param_groups[0]['lr']})

    return avg_train_loss


def compute_epoch_metrics(model: torch.nn.Module, args: argparse.Namespace, config: ConfigDict,
                          device: torch.device, device_ids: List[int], best_metric: float,
                          epoch: int, scheduler: torch.optim.lr_scheduler, optimizer,
                          all_time_all_metrics, all_losses,  world_size=None, metrics_avg=None, all_metrics=None) -> Tuple[float, float]:
    """
    Compute and log the metrics for the current epoch.
    Returns:
        (best_metric, current_metric_avg)
    """

    ddp = True if world_size else False
    should_print = not dist.is_initialized() or dist.get_rank() == 0
    if not ddp:
        if torch.cuda.is_available() and len(device_ids) > 1:
            metrics_avg, all_metrics = valid_multi_gpu(model, args, config, args.device_ids, verbose=False)
        else:
            metrics_avg, all_metrics = valid(model, args, config, device, verbose=False)
        all_time_all_metrics[f"epoch_{epoch}"] = all_metrics

    metric_avg = metrics_avg[args.metric_for_scheduler]
    if metric_avg > best_metric:

        if args.each_metrics_in_name:
            stem_parts = []
            for stem_name, values in all_metrics[args.metric_for_scheduler].items():
                stem_values = np.array(values)
                mean_val = stem_values.mean()
                std_val = stem_values.std()
                stem_parts.append(
                    f"{stem_name}_{args.metric_for_scheduler}_{mean_val:.4f}_std_{std_val:.4f}"
                )
            stem_info = "__".join(stem_parts)
            store_path = (
                f"{args.results_path}/model_{args.model_type}_ep_{epoch}_{stem_info}.ckpt"
            )
        else:
            store_path = (
                f"{args.results_path}/model_{args.model_type}_ep_{epoch}_{args.metric_for_scheduler}_{metric_avg:.4f}.ckpt"
            )
        if should_print:
            print(f'Store weights: {store_path}')
            save_weights(
                store_path=store_path,
                model=model,
                device_ids=device_ids,
                optimizer=optimizer,
                epoch=epoch,
                all_time_all_metrics=all_time_all_metrics,
                all_losses=all_losses,
                best_metric=best_metric,
                args=args,
                scheduler=scheduler
            )
        best_metric = metric_avg

    if args.save_weights_every_epoch:
        metric_string = ''
        for m in metrics_avg:
            metric_string += '_{}_{:.4f}'.format(m, metrics_avg[m])
        store_path = f'{args.results_path}/model_{args.model_type}_ep_{epoch}{metric_string}.ckpt'
        save_weights(
            store_path=store_path,
            model=model,
            device_ids=device_ids,
            optimizer=optimizer,
            epoch=epoch,
            all_time_all_metrics=all_time_all_metrics,
            all_losses=all_losses,
            best_metric=best_metric,
            args=args,
            scheduler=scheduler
        )

    if scheduler.name in ['ReduceLROnPlateau']:
        scheduler.step(metric_avg)

    if should_print:
        wandb.log({'metric_main': metric_avg, 'best_metric': best_metric})
        for metric_name in metrics_avg:
            wandb.log({f'metric_{metric_name}': metrics_avg[metric_name]})

    return best_metric, metric_avg


def train_model(args: Union[argparse.Namespace, None], rank=None, world_size=None) -> None:
    from utils.dataset import prepare_data
    from utils.model_utils import load_start_checkpoint
    from utils.model_utils import get_lora
    from utils.losses import choice_loss
    from torch.cuda.amp.grad_scaler import GradScaler
    from utils.model_utils import get_optimizer, log_model_info

    args = parse_args_train(args)
    ddp = True if world_size else False
    if ddp:
        initialize_environment_ddp(rank, world_size, args.seed, args.results_path)
    else:
        initialize_environment(args.seed, args.results_path)
        
    model, config = get_model_from_config(args.model_type, args.config_path)
    if 'model_type' in config.training:
        args.model_type = config.training.model_type
        
    # --- CUSTOM LLM AGENT INITIALIZATION ---
    agent = None
    if build_agent is not None and "agent" in config:
        agent = build_agent(config["agent"])
    # ---------------------------------------

    use_amp = getattr(config.training, 'use_amp', True)
    device_ids = args.device_ids
    if ddp:
        batch_size = config.training.batch_size
    else:
        batch_size = config.training.batch_size * len(device_ids)

    if not dist.is_initialized() or dist.get_rank() == 0:
        wandb_init(args, config, batch_size)

    train_loader = prepare_data(config, args, batch_size)

    if args.start_check_point:
        checkpoint = torch.load(args.start_check_point, weights_only=False, map_location='cpu')
        load_start_checkpoint(args, model, checkpoint, type_='train')
    model = get_lora(args, config, model)

    if args.freeze_layers is not None:
        freeze_layers = []
        train_layers = []
        for name, param in model.named_parameters():
            if any(name.startswith(prefix) for prefix in args.freeze_layers):
                freeze_layers.append(name)
                print('Freezing layer:', name)
                param.requires_grad = False
            else:
                train_layers.append(name)
        print('Trainable layers: {}'.format(len(train_layers)))
        print('Frozen layers: {}'.format(len(freeze_layers)))

    if ddp:
        device = torch.device(f'cuda:{rank}')
        model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)
        model_module = model.module
    else:
        device, model = initialize_model_and_device(model, args.device_ids)
        model_module = model.module if hasattr(model, 'module') else model

    ema_model = None
    if hasattr(config.training, 'ema_momentum') and config.training.ema_momentum > 0:
        from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"Initializing EMA with decay: {config.training.ema_momentum}")
        ema_model = AveragedModel(model_module, multi_avg_fn=get_ema_multi_avg_fn(config.training.ema_momentum))
        
    if args.pre_valid:
        model_to_valid = ema_model if ema_model is not None else model
        if ddp:
            valid_multi_gpu(model_to_valid, args, config, args.device_ids, verbose=False)
        else:
            if torch.cuda.is_available() and len(args.device_ids) > 1:
                valid_multi_gpu(model_to_valid, args, config, args.device_ids, verbose=True)
            else:
                valid(model_to_valid, args, config, device, verbose=True)

    gradient_accumulation_steps = int(getattr(config.training, 'gradient_accumulation_steps', 1))

    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)

    if args.start_check_point and "optimizer_state_dict" in checkpoint and args.load_optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if args.start_check_point and "scheduler_state_dict" in checkpoint and args.load_scheduler:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if args.start_check_point and "epoch" in checkpoint and args.load_epoch:
        start_epoch = checkpoint["epoch"] + 1
    else:
        start_epoch = 0

    if args.start_check_point and "best_metric" in checkpoint and args.load_best_metric:
        best_metric = checkpoint["best_metric"]
    else:
        best_metric = float('-inf')

    if args.start_check_point and "all_metrics" in checkpoint and args.load_all_metrics:
        all_time_all_metrics = checkpoint["all_metrics"]
    else:
        all_time_all_metrics = {}

    if args.start_check_point and "all_losses" in checkpoint and args.load_all_losses:
        all_losses = checkpoint["all_losses"]
    else:
        all_losses = {}

    multi_loss = choice_loss(args, config)
    scaler = GradScaler()

    if args.set_per_process_memory_fraction:
        torch.cuda.set_per_process_memory_fraction(1.0)
    torch.cuda.empty_cache()

    safe_mode = args.safe_mode

    should_print = not dist.is_initialized() or dist.get_rank() == 0

    if should_print:
        if world_size:
            batch_size = config.training.batch_size
            ef_batch_size = batch_size * gradient_accumulation_steps * world_size
            num_gpu = world_size
        else:
            device_ids = args.device_ids
            batch_size = config.training.batch_size * len(device_ids)
            ef_batch_size = batch_size * gradient_accumulation_steps
            num_gpu = len(device_ids)

        print(
            f"Instruments: {config.training.instruments}\n"
            f"Metrics for training: {args.metrics}. Metric for scheduler: {args.metric_for_scheduler}\n"
            f"Patience: {config.training.patience} "
            f"Reduce factor: {config.training.reduce_factor}\n"
            f"Batch size: {batch_size} "
            f"Grad accum steps: {gradient_accumulation_steps} "
            f"Num gpus: {num_gpu} "
            f"Effective batch size: {ef_batch_size}\n"
            f"Dataset type: {args.dataset_type}\n"
            f"Optimizer: {config.training.optimizer}"
        )

        print(f'Train for: {config.training.num_epochs} epochs')
        log_model_info(model, args.results_path)

    for epoch in range(start_epoch, config.training.num_epochs):
        if ddp:
            train_loader.sampler.set_epoch(epoch)

        train_loss = train_one_epoch(model, config, args, optimizer, device, device_ids, epoch,
                        use_amp, scaler, scheduler, gradient_accumulation_steps, train_loader, multi_loss, all_losses,
                        world_size, ema_model=ema_model, safe_mode=safe_mode)

        model_to_valid = ema_model if ema_model is not None else model
        metric_avg = 0.0

        if should_print:
            save_last_weights(args, model, device_ids, optimizer, epoch, all_time_all_metrics, best_metric, scheduler)
            
        if ddp:
            metrics_avg, all_metrics = valid_multi_gpu(model, args, config, args.device_ids, verbose=False)
            if rank == 0:
                all_time_all_metrics[f"epoch_{epoch}"] = all_metrics
                best_metric, metric_avg = compute_epoch_metrics(
                    model=model,
                    args=args,
                    config=config,
                    device=device,
                    device_ids=device_ids,
                    best_metric=best_metric,
                    epoch=epoch,
                    scheduler=scheduler,
                    optimizer=optimizer,
                    all_time_all_metrics=all_time_all_metrics,
                    all_losses=all_losses,
                    world_size=world_size,
                    metrics_avg=metrics_avg,
                    all_metrics=all_metrics
                )
        else:
            best_metric, metric_avg = compute_epoch_metrics(
                model=model,
                args=args,
                config=config,
                device=device,
                device_ids=device_ids,
                best_metric=best_metric,
                epoch=epoch,
                scheduler=scheduler,
                optimizer=optimizer,
                all_time_all_metrics=all_time_all_metrics,
                all_losses=all_losses,
            )

        # --- CUSTOM LLM AGENT HOOK ---
        # The agent acts as a Meta-Controller: analyzing logs and suggesting actions.
        if agent is not None and getattr(agent._cfg, 'enabled', True) and (epoch % agent._cfg.interval == 0):
            actions_list = [[]]
            
            # Rank 0 handles the actual API request to prevent hitting rate limits
            if should_print:
                snapshot = {
                    "epoch": epoch,
                    "train_loss": float(train_loss),
                    "val_metric": float(metric_avg),
                    "current_lr": optimizer.param_groups[0]['lr'],
                }
                
                agent_response = agent.analyze(snapshot)
                actions_list[0] = agent_response.get("actions", [])
                
                issues = agent_response.get("issues", [])
                if issues and issues != ["none"]:
                    print(f"🤖 LLM Agent flagged issues: {issues}")

            # Sync actions across all GPUs to prevent DDP desync
            if ddp:
                dist.broadcast_object_list(actions_list, src=0)

            # Apply identical actions on all ranks
            for action in actions_list[0]:
                if action == "reduce_lr":
                    if should_print: print("🤖 LLM Agent Action: Reducing LR by 0.5x")
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= 0.5
                elif action == "increase_lr":
                    if should_print: print("🤖 LLM Agent Action: Increasing LR by 1.5x")
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= 1.5
                elif action == "continue":
                    pass
        # -----------------------------


if __name__ == "__main__":
    train_model(None)