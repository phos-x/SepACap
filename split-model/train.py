# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo): https://github.com/ZFTurbo/'
__version__ = '1.0.6 - Asynchronous Panopticon Edition'

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
import math

from utils.settings import get_scheduler, parse_args_train, initialize_environment_ddp, \
    initialize_environment, get_model_from_config, wandb_init
from utils.model_utils import save_weights, normalize_batch, \
    save_last_weights, initialize_model_and_device

from valid import valid_multi_gpu, valid

import warnings
warnings.filterwarnings("ignore")

# --- ASYNCHRONOUS PANOPTICON IMPORTS ---
try:
    from custom.agent.agent_factory import build_agent
    from custom.agent.blackboard import global_blackboard
    from custom.agent.daemons import DaemonManager
    from custom.agent.ledger import DNALedger
    from custom.agent.telemetry import extract_system_telemetry
    HIVE_MIND_ACTIVE = True
except ImportError as e:
    print(f"Asynchronous Panopticon modules not found. Proceeding with standard training. Error: {e}")
    HIVE_MIND_ACTIVE = False
# ---------------------------------------

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
                    multi_loss: Callable[[torch.Tensor, torch.Tensor, torch.Tensor,], torch.Tensor], 
                    all_losses=None, world_size=None, ema_model=None, safe_mode=None, ledger=None) -> float:
    """
    Train the model for one epoch.
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
        'mel_band_roformer', 'bs_roformer', 'bs_mamba2', 'mel_band_conformer', 'bs_conformer'
    ) and not args.use_standard_loss)

    pbar = tqdm(train_loader, dynamic_ncols=True) if should_print else train_loader
    base_model = model.module if hasattr(model, 'module') else model

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
                ema_model.update_parameters(base_model)

            if scheduler.name in ['linear_scheduler']:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            
            # --- THE ASYNCHRONOUS PANOPTICON HOOK ---
            if HIVE_MIND_ACTIVE and should_print:
                # 1. MICRO-INTERRUPT KILL SWITCH
                if global_blackboard.panic.value:
                    global_blackboard.panic.value = False
                    print("\n🚨 AI MICRO-INTERRUPT TRIPPED! Severing forward pass and clearing VRAM.")
                    torch.cuda.empty_cache()
                
                # 2. WRITE HOLOGRAPHIC STATE TO DMZ (Every 10 steps to balance resolution and nanosecond speed)
                if i % 10 == 0:
                    loss_item = loss.item() * gradient_accumulation_steps
                    _loss = loss_item if not math.isnan(loss_item) else 999.0
                    global_blackboard.write_matrix(base_model, optimizer, x, y, _loss)

                # 3. CONSUME QUEUED ACTIONS FROM DAEMONS (Brains A & B)
                pending_actions = global_blackboard.consume_actions()
                for intent in pending_actions:
                    action_type = intent.get("action") or intent.get("tool_name")
                    val = intent.get("value")
                    reasoning = intent.get("ai_reasoning", "No AI reasoning provided.")
                    
                    print(f"\n⚡ HIVE MIND MUTATION: {action_type} -> {val}")
                    print(f"⚖️  REASONING: {reasoning}\n")
                    
                    # Execute supported tools
                    if action_type in ["set_lr", "set_learning_rate"]:
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = float(val)
                            
                    # Record to DNA Ledger
                    if ledger:
                        matrix_state = global_blackboard.read_matrix()
                        ledger.record_mutation(epoch, i, matrix_state, intent, reasoning)
            # ----------------------------------------
            
        # FAST DDP SYNC: Ensure if Rank 0 changed LR, all other GPUs get the update instantly
        if ddp:
            with torch.no_grad():
                # Sync Loss
                loss_copy = loss.detach().clone()
                dist.all_reduce(loss_copy, op=dist.ReduceOp.SUM)
                loss_copy /= dist.get_world_size()
                
                # Sync Learning Rate (Prevents AI-induced desyncs)
                lr_tensor = torch.tensor([optimizer.param_groups[0]['lr']], dtype=torch.float32, device=device)
                dist.broadcast(lr_tensor, src=0)
                if not should_print:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_tensor.item()
            
            if should_print:
                li = loss_copy.item() * gradient_accumulation_steps
                all_losses[f'epoch_{epoch}'].append(li)
                loss_val += li
                total += 1
                pbar.set_postfix({'loss': 100 * li, 'avg_loss': 100 * loss_val / (i + 1)})
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
    """ Compute and log the metrics for the current epoch. """
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
                stem_parts.append(f"{stem_name}_{args.metric_for_scheduler}_{mean_val:.4f}_std_{std_val:.4f}")
            stem_info = "__".join(stem_parts)
            store_path = f"{args.results_path}/model_{args.model_type}_ep_{epoch}_{stem_info}.ckpt"
        else:
            store_path = f"{args.results_path}/model_{args.model_type}_ep_{epoch}_{args.metric_for_scheduler}_{metric_avg:.4f}.ckpt"
            
        if should_print:
            print(f'Store weights: {store_path}')
            save_weights(
                store_path=store_path, model=model, device_ids=device_ids,
                optimizer=optimizer, epoch=epoch, all_time_all_metrics=all_time_all_metrics,
                all_losses=all_losses, best_metric=best_metric, args=args, scheduler=scheduler
            )
        best_metric = metric_avg

    if args.save_weights_every_epoch:
        metric_string = ''
        for m in metrics_avg:
            metric_string += '_{}_{:.4f}'.format(m, metrics_avg[m])
        store_path = f'{args.results_path}/model_{args.model_type}_ep_{epoch}{metric_string}.ckpt'
        save_weights(
            store_path=store_path, model=model, device_ids=device_ids,
            optimizer=optimizer, epoch=epoch, all_time_all_metrics=all_time_all_metrics,
            all_losses=all_losses, best_metric=best_metric, args=args, scheduler=scheduler
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
    from utils.model_utils import load_start_checkpoint, get_lora, get_optimizer, log_model_info
    from utils.losses import choice_loss
    from torch.cuda.amp.grad_scaler import GradScaler

    args = parse_args_train(args)
    ddp = True if world_size else False
    if ddp:
        initialize_environment_ddp(rank, world_size, args.seed, args.results_path)
    else:
        initialize_environment(args.seed, args.results_path)
        
    model, config = get_model_from_config(args.model_type, args.config_path)
    if 'model_type' in config.training:
        args.model_type = config.training.model_type

    should_print = not dist.is_initialized() or dist.get_rank() == 0

    # ==========================================
    # --- PANOPTICON DAEMON & LEDGER BOOTUP ---
    # ==========================================
    daemon_manager = None
    ledger = None
    llm_engine = None
    
    if HIVE_MIND_ACTIVE and should_print and "agent" in config:
        llm_engine = build_agent(config["agent"])
        ledger = DNALedger(filepath=f"{args.results_path}/dna_ledger.jsonl")
        
        # Write Genesis Block to lock environment determinism
        ledger.write_genesis_block(config=config, seed=args.seed)
        
        # Spin up Multiprocessing Brains
        daemon_manager = DaemonManager(llm_engine, global_blackboard, config.get("agent", {}))
        daemon_manager.start_all()
    # ==========================================

    try:
        use_amp = getattr(config.training, 'use_amp', True)
        device_ids = args.device_ids
        if ddp:
            batch_size = config.training.batch_size
        else:
            batch_size = config.training.batch_size * len(device_ids)

        if should_print:
            wandb_init(args, config, batch_size)

        train_loader = prepare_data(config, args, batch_size)

        if args.start_check_point:
            checkpoint = torch.load(args.start_check_point, weights_only=False, map_location='cpu')
            load_start_checkpoint(args, model, checkpoint, type_='train')
        model = get_lora(args, config, model)

        if args.freeze_layers is not None:
            freeze_layers, train_layers = [], []
            for name, param in model.named_parameters():
                if any(name.startswith(prefix) for prefix in args.freeze_layers):
                    freeze_layers.append(name)
                    param.requires_grad = False
                else:
                    train_layers.append(name)
            if should_print:
                print(f'Trainable layers: {len(train_layers)} | Frozen layers: {len(freeze_layers)}')

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
            if should_print:
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

        if args.start_check_point:
            if "optimizer_state_dict" in checkpoint and args.load_optimizer:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scheduler_state_dict" in checkpoint and args.load_scheduler:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = checkpoint["epoch"] + 1 if ("epoch" in checkpoint and args.load_epoch) else 0
            best_metric = checkpoint["best_metric"] if ("best_metric" in checkpoint and args.load_best_metric) else float('-inf')
            all_time_all_metrics = checkpoint["all_metrics"] if ("all_metrics" in checkpoint and args.load_all_metrics) else {}
            all_losses = checkpoint["all_losses"] if ("all_losses" in checkpoint and args.load_all_losses) else {}
        else:
            start_epoch = 0
            best_metric = float('-inf')
            all_time_all_metrics = {}
            all_losses = {}

        multi_loss = choice_loss(args, config)
        scaler = GradScaler()

        if args.set_per_process_memory_fraction:
            torch.cuda.set_per_process_memory_fraction(1.0)
        torch.cuda.empty_cache()

        if should_print:
            ef_batch_size = batch_size * gradient_accumulation_steps * (world_size if world_size else 1)
            num_gpu = world_size if world_size else len(device_ids)
            print(
                f"Instruments: {config.training.instruments}\n"
                f"Metrics for training: {args.metrics}. Metric for scheduler: {args.metric_for_scheduler}\n"
                f"Batch size: {batch_size} Grad accum steps: {gradient_accumulation_steps} Num gpus: {num_gpu}\n"
                f"Dataset type: {args.dataset_type}\n"
                f"Optimizer: {config.training.optimizer}"
            )
            print(f'Train for: {config.training.num_epochs} epochs')
            log_model_info(model, args.results_path)

        for epoch in range(start_epoch, config.training.num_epochs):
            if ddp:
                train_loader.sampler.set_epoch(epoch)

            train_loss = train_one_epoch(
                model, config, args, optimizer, device, device_ids, epoch,
                use_amp, scaler, scheduler, gradient_accumulation_steps, train_loader, 
                multi_loss, all_losses, world_size, ema_model=ema_model, safe_mode=args.safe_mode, ledger=ledger
            )

            metric_avg = 0.0
            if should_print:
                save_last_weights(args, model, device_ids, optimizer, epoch, all_time_all_metrics, best_metric, scheduler)
                
            if ddp:
                metrics_avg, all_metrics = valid_multi_gpu(model, args, config, args.device_ids, verbose=False)
                if rank == 0:
                    all_time_all_metrics[f"epoch_{epoch}"] = all_metrics
                    best_metric, metric_avg = compute_epoch_metrics(
                        model=model, args=args, config=config, device=device,
                        device_ids=device_ids, best_metric=best_metric, epoch=epoch,
                        scheduler=scheduler, optimizer=optimizer,
                        all_time_all_metrics=all_time_all_metrics, all_losses=all_losses,
                        world_size=world_size, metrics_avg=metrics_avg, all_metrics=all_metrics
                    )
            else:
                best_metric, metric_avg = compute_epoch_metrics(
                    model=model, args=args, config=config, device=device,
                    device_ids=device_ids, best_metric=best_metric, epoch=epoch,
                    scheduler=scheduler, optimizer=optimizer,
                    all_time_all_metrics=all_time_all_metrics, all_losses=all_losses,
                )

            # ==========================================
            # --- BRAIN C: CHIEF SCIENTIST MACRO-HOOK ---
            # ==========================================
            agent_interval = config.get('agent', {}).get('interval', 1)
            
            if HIVE_MIND_ACTIVE and should_print and llm_engine is not None and (epoch % agent_interval == 0):
                is_nan = math.isnan(train_loss) or math.isnan(metric_avg)
                
                epoch_metrics = all_time_all_metrics.get(f"epoch_{epoch}", {})
                sdr_dict = {}
                if 'sdr' in epoch_metrics:
                    sdr_dict = {stem: float(np.mean(vals)) for stem, vals in epoch_metrics['sdr'].items()}
                
                # Retrieve the Neurological MRI
                internal_telemetry = extract_system_telemetry(model, optimizer)
                
                snapshot = {
                    "epoch": epoch,
                    "global_metrics": {
                        "train_loss": float(train_loss) if not is_nan else "NaN",
                        "val_metric_avg": float(metric_avg) if not is_nan else "NaN",
                    },
                    "stem_metrics": sdr_dict,
                    "optimizer_state": {
                        "current_lr": optimizer.param_groups[0]['lr'],
                    },
                    "INTERNAL_TELEMETRY": internal_telemetry,
                    "SYSTEM_HEALTH": "CRITICAL NaN" if is_nan else internal_telemetry.get("health_status", "Nominal")
                }
                
                print(f"\n🧠 Waking Chief Scientist | System Health: {snapshot['SYSTEM_HEALTH']}")
                
                # Call Brain C
                agent_response = llm_engine.analyze(snapshot)
                
                debate = agent_response.get('debate', {})
                if debate:
                    print(f"🔥 ACCELERATOR OPINION: {debate.get('accelerator', 'None')}")
                    print(f"🛡️ STABILIZER OPINION: {debate.get('stabilizer', 'None')}")
                    
                print(f"⚖️ CHIEF SCIENTIST REASONING: {agent_response.get('reasoning')}")
                
                # --- THE MOTOR CORTEX ---
                actions = agent_response.get('actions_taken', [])
                if actions:
                    print(f"🤖 INITIATING ARCHITECTURAL SURGERY: {len(actions)} actions.")
                    base_model = model.module if hasattr(model, 'module') else model
                    
                    for action in actions:
                        tool = action.get("tool_name")
                        val = action.get("value")
                        
                        # Tool 1: Unfreeze Layer
                        if tool == "unfreeze_layer":
                            unfrozen_count = 0
                            for name, param in base_model.named_parameters():
                                if name.startswith(val):
                                    param.requires_grad = True
                                    unfrozen_count += 1
                            print(f"  🔓 UNFROZE {unfrozen_count} tensors in: {val}")
                            
                        # Tool 2: Freeze Layer
                        elif tool == "freeze_layer":
                            frozen_count = 0
                            for name, param in base_model.named_parameters():
                                if name.startswith(val):
                                    param.requires_grad = False
                                    frozen_count += 1
                            print(f"  🔒 FROZE {frozen_count} tensors in: {val}")
                            
                        # Hash the architectural mutation into the DNA Ledger
                        if ledger:
                            matrix_state = global_blackboard.read_matrix()
                            ledger.record_mutation(
                                epoch=epoch, 
                                step=-1, # -1 denotes an epoch-level macro change
                                matrix_state=matrix_state, 
                                action=action, 
                                reasoning=agent_response.get('reasoning', "Epoch Macro-Strategy")
                            )
                else:
                    print("🤖 NO SURGERY REQUIRED. Continuing nominal training.")
                # ------------------------

                # WandB Logging
                if wandb.run is not None:
                    wandb.log({
                        "agent/health": snapshot["SYSTEM_HEALTH"],
                        "agent/accelerator_opinion": wandb.Html(f"<p>{debate.get('accelerator', '')}</p>") if debate else "",
                        "agent/stabilizer_opinion": wandb.Html(f"<p>{debate.get('stabilizer', '')}</p>") if debate else "",
                        "agent/chief_reasoning": wandb.Html(f"<p>{agent_response.get('reasoning')}</p>"), 
                        "agent/actions_taken": str(actions)
                    })
            # ==========================================

    finally:
        # ==========================================
        # --- PANOPTICON GRACEFUL TEARDOWN ---
        # ==========================================
        if daemon_manager is not None:
            print("\n🛑 Training terminated. Halting Multiprocessing Daemons...")
            daemon_manager.stop_all()


if __name__ == "__main__":
    train_model(None)