import argparse
import logging
import os
from pathlib import Path

import torch
import yaml
import onnx

from models.SepACap_base.model import Model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def validate_file_path(path_str: str, check_exists: bool = True) -> Path:
    """Validates that a file path is structurally sound and optionally exists."""
    path = Path(path_str).resolve()
    if check_exists and not path.exists():
        raise FileNotFoundError(f"Critical file not found: {path}")
    return path

def promote_model(config_path: Path, pth_path: Path, onnx_path: Path) -> None:
    """
    Converts a trained PyTorch model to a static ONNX C++ graph for production.
    """
    logger.info("Starting ONNX promotion process...")

    logger.info(f"Loading configuration from {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        try:
            config = yaml.safe_load(f)['config']
        except yaml.YAMLError as e:
            logger.error("Failed to parse YAML configuration.")
            raise e

    model_config = config.get('model', {})
    max_len = config.get('dataset', {}).get('max_len', 24000)

    device = torch.device('cpu')
    logger.info("Instantiating SepACap architecture on CPU...")
    model = Model(**model_config).to(device)

    logger.info(f"Loading model weights from {pth_path}")
    try:
        # SECURITY CRITICAL: weights_only=True prevents Arbitrary Code Execution (ACE)
        # Hackers can hide malware inside PyTorch Pickled .pth files. This blocks it.
        checkpoint = torch.load(pth_path, map_location=device, weights_only=True)
        
        # Extract state_dict (handles cases where optimizer states were saved alongside weights)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict, strict=False)
    except Exception as e:
        logger.error(f"Failed to securely load PyTorch weights: {e}")
        raise

    model.eval()

    # E. Prepare Dummy Input for Graph Tracing
    # Shape: [Batch, Channels, Time]
    logger.info(f"Generating dummy input tensor (Shape: [1, 1, {max_len}])...")
    dummy_input = torch.randn(1, 1, max_len, device=device)

    logger.info("Tracing and compiling model to ONNX...")
    
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            export_params=True,        # Embed the learned weights inside the file
            opset_version=14,          # Opset 14 supports modern Transformer ops
            
            # Constant Folding computes static mathematical operations 
            # (like fixed scaling factors) ahead of time, deleting them from the graph 
            # to make inference fundamentally faster (O(1) instead of O(N)).
            do_constant_folding=True,  
            
            input_names=['input_audio'],
            output_names=['separated_stems'],
            
            # Dynamic Axes prevent the model from crashing if a user 
            # uploads a 4-second song instead of a 3-second song.
            dynamic_axes={             
                'input_audio': {0: 'batch_size', 2: 'time_length'},
                'separated_stems': {0: 'batch_size', 2: 'time_length'}
            }
        )
    except Exception as e:
        logger.error(f"ONNX Export failed during graph tracing: {e}")
        raise

    # G. Validate the Generated Graph
    logger.info("Validating the generated ONNX mathematical graph...")
    try:
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        logger.info(f"🎉 SUCCESS: Valid ONNX model compiled and saved to {onnx_path}")
    except onnx.checker.ValidationError as e:
        logger.error(f"ONNX Model validation failed: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Promote SepACap PyTorch model to ONNX for Production.")
    parser.add_argument("--config", type=str, required=True, help="Path to the model configs.yaml")
    parser.add_argument("--weights", type=str, required=True, help="Path to the trained epoch.best.pth file")
    parser.add_argument("--output", type=str, required=True, help="Path to save the output .onnx file")
    
    args = parser.parse_args()

    try:
        cfg_file = validate_file_path(args.config, check_exists=True)
        pth_file = validate_file_path(args.weights, check_exists=True)
        onnx_file = validate_file_path(args.output, check_exists=False)
        
        promote_model(cfg_file, pth_file, onnx_file)
    except Exception as err:
        logger.critical(f"Promotion script aborted due to error: {err}")
        exit(1)