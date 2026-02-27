import os
import torch
from loguru import logger
from typing import Tuple, Dict, Any

from .dataset import get_dataloaders
from .model import Model
from .engine import Engine
from utils.implements.criterions import PIT_SISNRi, PIT_SDRi
from utils import util_system, util_implement
from utils.decorators import logger_wraps

# ---------------------------------------------------------------------------
# 1. SYSTEM LOGGING SETUP
# ---------------------------------------------------------------------------
# Configure Loguru to write persistent logs for debugging AWS/Docker crashes
log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log/system_log.log")
logger.add(log_file_path, level="DEBUG", mode="w")

# ---------------------------------------------------------------------------
# 2. COMPONENT BUILDER (The Modular Core)
# ---------------------------------------------------------------------------
def build_components(config: Dict[str, Any], args: Any) -> Tuple:
    """
    Constructs and returns all PyTorch entities required for training/tuning.
    Extracting this allows external scripts (like tune.py) to dynamically 
    override the config, call this function, and instantly get a fresh model.
    """
    logger.info("Constructing PyTorch components...")

    # A. Hardware Routing
    gpuid_str = str(config["engine"].get("gpuid", "0"))
    gpuid = tuple(map(int, gpuid_str.split(',')))
    
    # Graceful fallback to CPU if CUDA is unavailable (crucial for CI/CD testing)
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{gpuid[0]}')
    else:
        device = torch.device('cpu')
        logger.warning("CUDA not found. Falling back to CPU compute.")

    # B. Dataloaders
    dataloaders = get_dataloaders(args, config["dataset"], config["dataloader"])

    # C. Model Construction
    model = Model(**config["model"]).to(device)

    # D. Criterions (Loss Functions & Metrics)
    raw_criterions = util_implement.CriterionFactory(config["criterion"], device).get_criterions()
    
    # Aligned Unpacking Logic:
    # SepACap uses one loss for both domain targets. We pad the list to length 4 to 
    # satisfy the Engine's: mag_loss, time_loss, sisnri_metric, sdri_metric unpacking.
    composite_loss = raw_criterions[0]
    criterions = [
        composite_loss,   # Assigned to mag_loss
        composite_loss,   # Assigned to time_loss
        PIT_SISNRi(),     # Metric placeholder 1
        PIT_SDRi()        # Metric placeholder 2
    ]

    # E. Optimizers & Schedulers
    optimizers = util_implement.OptimizerFactory(config["optimizer"], model.parameters()).get_optimizers()
    schedulers = util_implement.SchedulerFactory(config["scheduler"], optimizers).get_schedulers()

    logger.info("All PyTorch components successfully constructed.")
    return model, dataloaders, criterions, optimizers, schedulers, gpuid, device

# ---------------------------------------------------------------------------
# 3. MAIN EXECUTION PIPELINE
# ---------------------------------------------------------------------------
@logger_wraps()
def main(args):
    """
    The primary entry point when executed via run.py. 
    It parses the config, builds components, and hands them to the Engine.
    """
    # 1. Configuration Resolution
    # Prioritize the dynamic AWS --config argument, fallback to local configs.yaml
    yaml_path = getattr(args, 'config', None)
    if not yaml_path or not os.path.exists(yaml_path):
        yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs.yaml")
        
    logger.info(f"Loading configuration from: {yaml_path}")
    yaml_dict = util_system.parse_yaml(yaml_path)
    config = yaml_dict["config"]

    # 2. Build Components
    model, dataloaders, criterions, optimizers, schedulers, gpuid, device = build_components(config, args)

    # 3. Initialize the Engine
    engine = Engine(
        args=args, 
        config=config, 
        model=model, 
        dataloaders=dataloaders, 
        criterions=criterions, 
        optimizers=optimizers, 
        schedulers=schedulers, 
        gpuid=gpuid, 
        device=device
    )

    # 4. Route Execution based on CLI mode
    logger.info(f"Engine routing triggered for mode: {args.engine_mode}")
    if args.engine_mode == 'infer_sample':
        # Defensive check to ensure the sample file was provided
        if not hasattr(args, 'sample_file') or not args.sample_file:
            raise ValueError("--sample-file must be provided when running 'infer_sample' mode.")
        engine._inference_sample(args.sample_file)
    else:
        # Standard training/validation loop
        engine.run()