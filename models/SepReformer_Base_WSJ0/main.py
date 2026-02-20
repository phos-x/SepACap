import os
import torch
from loguru import logger
from .dataset import get_dataloaders
from .model import Model
from .engine import Engine
from utils.implements.criterions import PIT_SISNRi, PIT_SDRi # Import placeholders for padding
from utils import util_system, util_implement
from utils.decorators import *

# Setup logger
log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log/system_log.log")
logger.add(log_file_path, level="DEBUG", mode="w")

@logger_wraps()
def main(args):
    
    ''' Build Setting '''
    # Call configuration file (configs.yaml)
    yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs.yaml")
    yaml_dict = util_system.parse_yaml(yaml_path)
    
    # Get configuration from YAML
    config = yaml_dict["config"]
    
    # 1. Call DataLoader [train / valid / test / etc...]
    dataloaders = get_dataloaders(args, config["dataset"], config["dataloader"])
    
    ''' Build Model '''
    # 2. Call network model with Snake activation support
    model = Model(**config["model"])

    ''' Build Engine '''
    # Call gpu id & device
    gpuid = tuple(map(int, config["engine"]["gpuid"].split(',')))
    device = torch.device(f'cuda:{gpuid[0]}')
    
    # 3. Build Criterions with DSA Alignment
    # Get the primary SepACapCompositeLoss from the factory
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
    
    # 4. Build Optimizers & Schedulers
    optimizers = util_implement.OptimizerFactory(config["optimizer"], model.parameters()).get_optimizers()
    schedulers = util_implement.SchedulerFactory(config["scheduler"], optimizers).get_schedulers()
    
    # 5. Initialize & Run Engine
    engine = Engine(args, config, model, dataloaders, criterions, optimizers, schedulers, gpuid, device)
    
    if args.engine_mode == 'infer_sample':
        engine._inference_sample(args.sample_file)
    else:
        engine.run()