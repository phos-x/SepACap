import os
import time
import torch
import torchaudio
from pathlib import Path
from loguru import logger
from tqdm import tqdm
from utils import util_engine, functions
from utils.decorators import logger_wraps
from torch.utils.tensorboard import SummaryWriter
import wandb

# Modern standard for audio metrics
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio, SignalDistortionRatio

@logger_wraps()
class Engine(object):
    def __init__(self, args, config, model, dataloaders, criterions, optimizers, schedulers, gpuid, device):
        
        self.engine_mode = args.engine_mode
        # Security & Best Practice: Use pathlib for robust path resolution
        self.out_wav_dir = Path(args.out_wav_dir).resolve() 
        self.out_wav_dir.mkdir(parents=True, exist_ok=True)
        
        self.config = config
        self.gpuid = gpuid
        self.device = device
        
        # Automatic Mixed Precision (AMP) Scaler for faster training
        self.scaler = torch.amp.GradScaler('cuda' if 'cuda' in str(self.device) else 'cpu')
        
        self.use_wandb = 'wandb' in self.config and self.config['wandb'].get('enabled', False)
        if self.use_wandb:
            wandb_cfg = self.config['wandb']
            wandb.init(
                project=wandb_cfg.get('project', 'SepACap'),
                group=wandb_cfg.get('group', None),
                config=self.config,
                job_type=self.engine_mode
            )
            self.wandb_run = wandb.run
            logger.info(f"W&B initialized securely. Dashboard URL: {self.wandb_run.url}")
        else:
            self.wandb_run = getattr(args, 'wandb_run', None)
            logger.info("W&B logging is disabled in configs.yaml.")
        
        self.model = model.to(self.device)
        self.dataloaders = dataloaders 
        self.PIT_SISNR_mag_loss, self.PIT_SISNR_time_loss, self.PIT_SISNRi_loss, self.PIT_SDRi_loss = criterions
        self.main_optimizer = optimizers[0]
        self.main_scheduler, self.warmup_scheduler = schedulers
        
        self.metric_sisdr = ScaleInvariantSignalDistortionRatio().to(self.device)
        self.metric_sdr = SignalDistortionRatio().to(self.device)
        
        base_path = Path(__file__).parent.resolve()
        self.pretrain_weights_path = base_path / "log" / "pretrain_weights"
        self.scratch_weights_path = base_path / "log" / "scratch_weights"
        self.pretrain_weights_path.mkdir(parents=True, exist_ok=True)
        self.scratch_weights_path.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_path = self.pretrain_weights_path if any(
            self.pretrain_weights_path.glob('*.[pp][th][l]*') # matches .pt, .pth, .pkl securely
        ) else self.scratch_weights_path
        
        # NOTE: util_engine.load_last_checkpoint_n_get_epoch MUST use weights_only=True internally 
        # in its torch.load() call to prevent arbitrary code execution vulnerabilities.
        self.start_epoch = util_engine.load_last_checkpoint_n_get_epoch(
            str(self.checkpoint_path), self.model, self.main_optimizer, location=self.device
        )
        
        dummy_len = self.config['check_computations']['dummy_len']
        util_engine.model_params_mac_summary(
            model=self.model, 
            input=torch.randn(1, dummy_len, device=self.device), 
            dummy_input=torch.rand(1, dummy_len, device=self.device), 
            metrics=['ptflops', 'thop', 'torchinfo']
        )
        
        logger.info(f"Clip gradient by 2-norm {self.config['engine']['clip_norm']}")

    def _apply_parallel(self, nnet_input):
        """Standardized wrapper for model execution to ensure alignment."""
        if len(self.gpuid) > 1:
            return torch.nn.parallel.data_parallel(self.model, nnet_input, device_ids=self.gpuid)
        return self.model(nnet_input)

    @logger_wraps()
    def _train(self, dataloader, epoch):
        self.model.train()
        tot_loss_freq = torch.zeros(self.model.num_stages, device=self.device)
        tot_loss_time, num_batch = 0.0, 0
        
        pbar = tqdm(total=len(dataloader), unit='batches', colour="YELLOW", dynamic_ncols=True)
        for input_sizes, mixture, src, _ in dataloader:
            num_batch += 1
            
            nnet_input = functions.apply_cmvn(mixture) if self.config['engine']['mvn'] else mixture
            nnet_input = nnet_input.to(self.device, non_blocking=True)
            src = [s.to(self.device, non_blocking=True) for s in src]
            
            if epoch == 1: self.warmup_scheduler.step()
            self.main_optimizer.zero_grad(set_to_none=True) # DSA Optimization: Faster than zero_grad()
            
            # AMP Context Manager for accelerated forward pass
            with torch.amp.autocast(device_type='cuda' if 'cuda' in str(self.device) else 'cpu'):
                estim_src, estim_src_bn = self._apply_parallel(nnet_input)
                
                cur_loss_s_bn = []
                for idx, estim_val in enumerate(estim_src_bn):
                    loss_f = self.PIT_SISNR_mag_loss(estims=estim_val, idx=idx, input_sizes=input_sizes, target_attr=src)
                    cur_loss_s_bn.append(loss_f)
                    tot_loss_freq[idx] += loss_f.detach() / self.config['model']['num_spks']
                
                cur_loss_s = self.PIT_SISNR_time_loss(estims=estim_src, input_sizes=input_sizes, target_attr=src)
                tot_loss_time += cur_loss_s.item() / self.config['model']['num_spks']
                
                alpha = 0.4 * 0.8**(1+(epoch-101)//5) if epoch > 100 else 0.4
                combined_loss = (1-alpha) * cur_loss_s + alpha * (sum(cur_loss_s_bn) / len(cur_loss_s_bn))
                final_loss = combined_loss / self.config['model']['num_spks']
            
            self.scaler.scale(final_loss).backward()
            
            if self.config['engine']['clip_norm']:
                self.scaler.unscale_(self.main_optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['engine']['clip_norm'])
            
            self.scaler.step(self.main_optimizer)
            self.scaler.update()
            
            dict_loss = {"T_Loss": tot_loss_time / num_batch}
            pbar.set_postfix(dict_loss)
            pbar.update(1)
            
        pbar.close()
        avg_freq = (tot_loss_freq.sum() / (len(tot_loss_freq) * num_batch)).item()
        return tot_loss_time / num_batch, avg_freq, num_batch

    @logger_wraps()
    def _validate(self, dataloader):
        self.model.eval()
        tot_loss_time, num_batch = 0.0, 0
        
        self.metric_sisdr.reset()
        self.metric_sdr.reset()
        
        pbar = tqdm(total=len(dataloader), unit='batches', colour="RED", dynamic_ncols=True)
        with torch.inference_mode(): # Faster and safer than torch.no_grad()
            for input_sizes, mixture, src, _ in dataloader:
                num_batch += 1
                nnet_input = functions.apply_cmvn(mixture) if self.config['engine']['mvn'] else mixture
                nnet_input = nnet_input.to(self.device, non_blocking=True)
                src = [s.to(self.device, non_blocking=True) for s in src]
                
                estim_src, _ = self._apply_parallel(nnet_input)
                
                loss_t = self.PIT_SISNR_time_loss(estims=estim_src, input_sizes=input_sizes, target_attr=src)
                tot_loss_time += loss_t.item() / self.config['model']['num_spks']
                
                # Stack targets and estimates for metric calculation
                # Assumes shape matches: (batch, time) per source
                stacked_src = torch.stack(src, dim=1) 
                stacked_est = torch.stack(estim_src, dim=1)
                
                self.metric_sisdr.update(stacked_est, stacked_src)
                self.metric_sdr.update(stacked_est, stacked_src)
                
                pbar.set_postfix({"T_Loss": tot_loss_time/num_batch})
                pbar.update(1)
                
        pbar.close()
        
        # Compute epoch-level metrics
        final_sisdr = self.metric_sisdr.compute().item()
        final_sdr = self.metric_sdr.compute().item()
        
        logger.info(f"Validation Metrics -> SI-SDR: {final_sisdr:.2f} dB | SDR: {final_sdr:.2f} dB")
        
        return tot_loss_time / num_batch, final_sisdr, final_sdr, num_batch

    @logger_wraps()
    def run(self):
        tb_path = Path(__file__).parent.resolve() / "log" / "tensorboard"
        writer = SummaryWriter(str(tb_path))
        
        valid_loss_best = float('inf')

        try:
            if "test" in self.engine_mode:
                self._test_cycle()
            else:
                if self.start_epoch > 1:
                    logger.info("Running baseline validation...")
                    v_t, v_sisdr, v_sdr, _ = self._validate(self.dataloaders['valid'])
                    valid_loss_best = v_t
                
                for epoch in range(self.start_epoch, self.config['engine']['max_epoch'] + 1):
                    t_start = time.time()
                    
                    t_loss_t, t_loss_f, _ = self._train(self.dataloaders['train'], epoch)
                    v_loss_t, v_sisdr, v_sdr, _ = self._validate(self.dataloaders['valid'])
                    
                    if epoch > self.config['engine']['start_scheduling']:
                        self.main_scheduler.step(v_loss_t)
                    
                    elapsed = time.time() - t_start
                    current_lr = self.main_optimizer.param_groups[0]['lr']
                    
                    logger.info(f"Epoch {epoch} | Train: {t_loss_t:.2f} | Valid Loss: {v_loss_t:.2f} | SI-SDR: {v_sisdr:.2f}dB | {elapsed:.1f}s")
                    
                    valid_loss_best = util_engine.save_checkpoint_per_best(
                        valid_loss_best, v_loss_t, t_loss_t, epoch, 
                        self.model, self.main_optimizer, str(self.checkpoint_path), self.wandb_run
                    )
                    
                    writer.add_scalars("Loss/Time", {"Train": t_loss_t, "Valid": v_loss_t}, epoch)
                    writer.add_scalars("Metrics/Quality", {"SI-SDR": v_sisdr, "SDR": v_sdr}, epoch)
                    writer.add_scalar("Meta/LR", current_lr, epoch)
                    writer.flush()

                    if self.use_wandb:
                        wandb.log({
                            "epoch": epoch,
                            "train/time_loss": t_loss_t,
                            "train/freq_loss": t_loss_f,
                            "valid/time_loss": v_loss_t,
                            "metrics/SI-SDR": v_sisdr,
                            "metrics/SDR": v_sdr,
                            "meta/learning_rate": current_lr,
                            "meta/epoch_time_seconds": elapsed
                        })
                        
        except Exception as e:
            logger.error(f"Training interrupted unexpectedly: {e}")
            raise e
        finally:
            writer.close()
            if self.use_wandb:
                wandb.finish()

    def _test_cycle(self):
        """Cleaned helper for the test routine."""
        logger.info("Starting test routine...")
        t_tloss, t_sisdr, t_sdr, _ = self._validate(self.dataloaders['test'])
        logger.success(f"Test Complete: SI-SDR: {t_sisdr:.2f} dB | SDR: {t_sdr:.2f} dB")