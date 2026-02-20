import os
import torch
import csv
import time
import soundfile as sf
import librosa
import numpy as np
from loguru import logger
from tqdm import tqdm
from utils import util_engine, functions
from utils.decorators import logger_wraps
from torch.utils.tensorboard import SummaryWriter

@logger_wraps()
class Engine(object):
    def __init__(self, args, config, model, dataloaders, criterions, optimizers, schedulers, gpuid, device):
        
        # 1. Platform Security: Basic setup
        self.engine_mode = args.engine_mode
        self.out_wav_dir = args.out_wav_dir
        self.config = config
        self.gpuid = gpuid
        self.device = device
        
        # 2. Defensive Attributes (Fixes wandb_run AttributeError)
        self.wandb_run = getattr(args, 'wandb_run', None)
        
        # 3. Model Orchestration
        # DSA best practice: Prefer DistributedDataParallel for modern PyTorch, 
        # but maintaining DataParallel for your specific config compatibility.
        self.model = model.to(self.device)
        self.dataloaders = dataloaders 
        self.PIT_SISNR_mag_loss, self.PIT_SISNR_time_loss, self.PIT_SISNRi_loss, self.PIT_SDRi_loss = criterions
        self.main_optimizer = optimizers[0]
        self.main_scheduler, self.warmup_scheduler = schedulers
        
        # 4. Secure Path Management
        base_path = os.path.dirname(os.path.abspath(__file__))
        self.pretrain_weights_path = os.path.join(base_path, "log", "pretrain_weights")
        self.scratch_weights_path = os.path.join(base_path, "log", "scratch_weights")
        os.makedirs(self.pretrain_weights_path, exist_ok=True)
        os.makedirs(self.scratch_weights_path, exist_ok=True)
        
        # 5. Checkpoint Logic
        self.checkpoint_path = self.pretrain_weights_path if any(
            f.endswith(('.pt', '.pth', '.pkl')) for f in os.listdir(self.pretrain_weights_path)
        ) else self.scratch_weights_path
        
        self.start_epoch = util_engine.load_last_checkpoint_n_get_epoch(
            self.checkpoint_path, self.model, self.main_optimizer, location=self.device
        )
        
        # 6. Pre-flight MACs/Params Summary
        dummy_len = self.config['check_computations']['dummy_len']
        util_engine.model_params_mac_summary(
            model=self.model, 
            input=torch.randn(1, dummy_len).to(self.device), 
            dummy_input=torch.rand(1, dummy_len).to(self.device), 
            metrics=['ptflops', 'thop', 'torchinfo']
        )
        
        logger.info(f"Clip gradient by 2-norm {self.config['engine']['clip_norm']}")

    def _apply_parallel(self, nnet_input):
        """Standardized wrapper for model execution to ensure alignment."""
        if len(self.gpuid) > 1:
            return torch.nn.parallel.data_parallel(self.model, nnet_input, device_ids=self.gpuid)
        else:
            return self.model(nnet_input)

    @logger_wraps()
    def _train(self, dataloader, epoch):
        self.model.train()
        tot_loss_freq = [0.0] * self.model.num_stages
        tot_loss_time, num_batch = 0.0, 0
        
        pbar = tqdm(total=len(dataloader), unit='batches', colour="YELLOW", dynamic_ncols=True)
        for input_sizes, mixture, src, _ in dataloader:
            num_batch += 1
            
            # CMVN & Device Transfer
            nnet_input = functions.apply_cmvn(mixture) if self.config['engine']['mvn'] else mixture
            nnet_input = nnet_input.to(self.device)
            
            # Warm-up Logic
            if epoch == 1: self.warmup_scheduler.step()
            
            self.main_optimizer.zero_grad()
            
            # Forward Pass
            estim_src, estim_src_bn = self._apply_parallel(nnet_input)
            
            # Frequency Domain Losses (BN Stages)
            cur_loss_s_bn = []
            for idx, estim_val in enumerate(estim_src_bn):
                loss_f = self.PIT_SISNR_mag_loss(estims=estim_val, idx=idx, input_sizes=input_sizes, target_attr=src)
                cur_loss_s_bn.append(loss_f)
                tot_loss_freq[idx] += loss_f.item() / self.config['model']['num_spks']
            
            # Time Domain Loss
            cur_loss_s = self.PIT_SISNR_time_loss(estims=estim_src, input_sizes=input_sizes, target_attr=src)
            tot_loss_time += cur_loss_s.item() / self.config['model']['num_spks']
            
            # Dynamic Alpha Weighting
            alpha = 0.4 * 0.8**(1+(epoch-101)//5) if epoch > 100 else 0.4
            combined_loss = (1-alpha) * cur_loss_s + alpha * (sum(cur_loss_s_bn) / len(cur_loss_s_bn))
            
            # Normalize and Backward
            (combined_loss / self.config['model']['num_spks']).backward()
            
            if self.config['engine']['clip_norm']:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['engine']['clip_norm'])
            
            self.main_optimizer.step()
            
            # UI Updates
            dict_loss = {"T_Loss": tot_loss_time / num_batch}
            dict_loss.update({f"F_{i}": v / num_batch for i, v in enumerate(tot_loss_freq)})
            pbar.set_postfix(dict_loss)
            pbar.update(1)
            
        pbar.close()
        avg_freq = sum(tot_loss_freq) / (len(tot_loss_freq) * num_batch)
        return tot_loss_time / num_batch, avg_freq, num_batch

    @logger_wraps()
    def _validate(self, dataloader):
        self.model.eval()
        tot_loss_freq = [0.0] * self.model.num_stages
        tot_loss_time, num_batch = 0.0, 0
        
        pbar = tqdm(total=len(dataloader), unit='batches', colour="RED", dynamic_ncols=True)
        with torch.inference_mode():
            for input_sizes, mixture, src, _ in dataloader:
                num_batch += 1
                nnet_input = functions.apply_cmvn(mixture) if self.config['engine']['mvn'] else mixture
                nnet_input = nnet_input.to(self.device)
                
                estim_src, estim_src_bn = self._apply_parallel(nnet_input)
                
                for idx, estim_val in enumerate(estim_src_bn):
                    loss_f = self.PIT_SISNR_mag_loss(estims=estim_val, idx=idx, input_sizes=input_sizes, target_attr=src)
                    tot_loss_freq[idx] += loss_f.item() / self.config['model']['num_spks']
                
                loss_t = self.PIT_SISNR_time_loss(estims=estim_src, input_sizes=input_sizes, target_attr=src)
                tot_loss_time += loss_t.item() / self.config['model']['num_spks']
                
                pbar.set_postfix({"T_Loss": tot_loss_time/num_batch})
                pbar.update(1)
                
        pbar.close()
        avg_freq = sum(tot_loss_freq) / (len(tot_loss_freq) * num_batch)
        return tot_loss_time / num_batch, avg_freq, num_batch

    @logger_wraps()
    def run(self):
        # Establish Tensorboard Writer
        tb_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log", "tensorboard")
        writer = SummaryWriter(tb_path)
        
        # Best loss tracker
        valid_loss_best = float('inf')

        try:
            if "test" in self.engine_mode:
                self._test_cycle()
            else:
                # 1. Initial Validation (Baseline)
                if self.start_epoch > 1:
                    logger.info("Running baseline validation...")
                    v_t, v_f, _ = self._validate(self.dataloaders['valid'])
                    valid_loss_best = v_t
                
                # 2. Main Training Loop
                for epoch in range(self.start_epoch, self.config['engine']['max_epoch'] + 1):
                    t_start = time.time()
                    
                    # Training Phase
                    t_loss_t, t_loss_f, t_num = self._train(self.dataloaders['train'], epoch)
                    
                    # Validation Phase
                    v_loss_t, v_loss_f, v_num = self._validate(self.dataloaders['valid'])
                    
                    # Scheduler Step
                    if epoch > self.config['engine']['start_scheduling']:
                        self.main_scheduler.step(v_loss_t)
                    
                    # Logging
                    elapsed = time.time() - t_start
                    logger.info(f"Epoch {epoch} | Train: {t_loss_t:.2f}dB | Valid: {v_loss_t:.2f}dB | {elapsed:.1f}s")
                    
                    # Safe Checkpoint Saving (using local wandb_run attribute)
                    valid_loss_best = util_engine.save_checkpoint_per_best(
                        valid_loss_best, v_loss_t, t_loss_t, epoch, 
                        self.model, self.main_optimizer, self.checkpoint_path, self.wandb_run
                    )
                    
                    # Tensorboard
                    writer.add_scalars("Loss/Time", {"Train": t_loss_t, "Valid": v_loss_t}, epoch)
                    writer.add_scalar("Meta/LR", self.main_optimizer.param_groups[0]['lr'], epoch)
                    writer.flush()
        finally:
            writer.close()

    def _test_cycle(self):
        """Cleaned helper for the test routine."""
        logger.info("Starting test routine...")
        t_sisnr, t_sdr, t_num = self._test(self.dataloaders['test'], self.out_wav_dir)
        logger.success(f"Test Complete: SISNRi: {t_sisnr:.2f} | SDRi: {t_sdr:.2f}")