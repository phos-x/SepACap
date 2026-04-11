import argparse
import gc
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

import optuna
import torch
import yaml
from optuna.trial import TrialState
from optuna.integration.wandb import WeightsAndBiasesCallback

# ---------------------------------------------------------------------------
# 1. ENTERPRISE LOGGING
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 2. OBJECT-ORIENTED OBJECTIVE (DSA Best Practice)
# ---------------------------------------------------------------------------
class SepACapObjective:
    """
    Encapsulates the Optuna objective function. 
    Using a class instead of a global function prevents namespace pollution 
    and securely passes dynamic arguments (like config paths) to the trial.
    """
    def __init__(self, config_path: str, args: argparse.Namespace):
        self.config_path = Path(config_path).resolve()
        self.args = args
        
        # Security: Validate config path before attempting to open
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found at {self.config_path}")

    def __call__(self, trial: optuna.Trial) -> float:
        """The core execution loop for a single Optuna Trial."""
        engine = None
        
        try:
            # A. Securely Load Configuration
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config: Dict[str, Any] = yaml.safe_load(f)

            # B. Suggest Hyperparameters (The "Settings")
            lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
            dropout = trial.suggest_float("dropout", 0.1, 0.5)
            batch_size = trial.suggest_categorical("batch_size", [2, 4, 8])

            # Override base config
            config['optim']['lr'] = lr
            config['model']['dropout'] = dropout
            config['data']['batch_size'] = batch_size

            logger.info(f"Trial {trial.number} starting | LR: {lr:.5f}, Dropout: {dropout:.2f}, Batch: {batch_size}")

            # C. Dynamic Module Loading (Matches run.py architecture)
            import importlib
            module_path = f"models.{self.args.model}.main"
            main_module = importlib.import_module(module_path)
            
            model, dataloaders, criterions, optimizers, schedulers, gpuid, device = main_module.build_components(config, self.args)

            # D. Initialize Engine
            from utils.util_engine import Engine
            engine = Engine(
                args=self.args, 
                config=config, 
                model=model, 
                dataloaders=dataloaders, 
                criterions=criterions, 
                optimizers=optimizers, 
                schedulers=schedulers, 
                gpuid=[0], 
                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            )

            # E. The Mini-Training Loop
            EPOCHS_TO_TEST = self.args.tune_epochs
            best_val_loss = float('inf')

            for epoch in range(1, EPOCHS_TO_TEST + 1):
                # Execute engine functions (matching your updated engine.py signature)
                t_loss_t, _, _ = engine._train(dataloaders['train'], epoch)
                v_loss_t, _, _ = engine._validate(dataloaders['valid'])

                if v_loss_t < best_val_loss:
                    best_val_loss = v_loss_t

                # F. Early Pruning Hook
                trial.report(v_loss_t, epoch)
                if trial.should_prune():
                    logger.info(f"Trial {trial.number} mathematically pruned at epoch {epoch}.")
                    raise optuna.exceptions.TrialPruned()

            return best_val_loss

        except Exception as e:
            if not isinstance(e, optuna.exceptions.TrialPruned):
                logger.error(f"Trial {trial.number} failed due to error: {e}")
            raise e
            
        finally:
            # G. DSA CRITICAL: Guaranteed Memory Cleanup
            # The finally block executes 100% of the time, even if the trial crashes or is pruned.
            if engine is not None:
                del engine
            
            # Force Python garbage collector to destroy unreferenced objects
            gc.collect() 
            
            # Flush the GPU VRAM so the next trial starts with a clean slate
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SepACap Automated Hyperparameter Tuning")
    parser.add_argument("--model", type=str, default="SepReformer_Base_WSJ0")
    parser.add_argument("--config", type=str, required=True, help="Path to configs.yaml")
    parser.add_argument("--tune-epochs", type=int, default=15, help="Number of epochs per trial")
    parser.add_argument("--trials", type=int, default=30, help="Total number of trials to run")
    parser.add_argument("--engine-mode", type=str, default="tune")
    
    # Ignore injected AWS SageMaker arguments
    args, unknown = parser.parse_known_args()
    
    # 4. Weights & Biases Callback Integration
    callbacks = []
    wandb_api_key = os.environ.get("WANDB_API_KEY")
    
    if wandb_api_key:
        logger.info("W&B API Key detected. Enabling cloud sweep tracking.")
        wandb_kwargs = {"project": "SepACap", "group": "Optuna-Sweeps"}
        wandbc = WeightsAndBiasesCallback(metric_name="val_loss", wandb_kwargs=wandb_kwargs)
        callbacks.append(wandbc)
    else:
        logger.warning("WANDB_API_KEY not found. W&B sweeping visualizations disabled.")

    # 5. Execute the Study
    objective = SepACapObjective(config_path=args.config, args=args)
    
    study = optuna.create_study(
        direction="minimize", 
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_epochs=3)
    )
    
    logger.info(f"Initiating Optuna Hyperparameter Search ({args.trials} trials)...")
    study.optimize(objective, n_trials=args.trials, callbacks=callbacks)

    # 6. Post-Study Analysis
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    logger.info("--- Study Complete ---")
    logger.info(f"Finished trials: {len(study.trials)}")
    logger.info(f"Pruned trials: {len(pruned_trials)}")
    logger.info(f"Complete trials: {len(complete_trials)}")

    best_trial = study.best_trial
    logger.info(f"Best Trial Value (Loss): {best_trial.value}")
    logger.info("Best Hyperparameters:")
    for key, value in best_trial.params.items():
        logger.info(f"  {key}: {value}")