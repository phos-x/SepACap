import argparse
import importlib
import logging
import os
import re
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def sanitize_model_name(model_name: str) -> str:
    if not re.match(r'^[\w]+$', model_name):
        raise ValueError(f"Invalid model name: '{model_name}'. Only alphanumeric characters and '_' allowed.")
    return model_name

def main():
    parser = argparse.ArgumentParser(description="SepACap Master Execution Script (SageMaker Ready)")
    
    parser.add_argument(
        "--model",
        type=str,
        default="SepReformer_Base_WSJ0", 
        help="Name of the model directory inside the 'models' folder"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to the configs.yaml file (Injected by SageMaker)"
    )
    parser.add_argument(
        "--engine-mode",
        choices=["train", "test", "test_save", "infer_sample", "tune"], # Added 'tune' for Optuna compatibility
        default="train",
        help="Execution mode for the pipeline"
    )
    parser.add_argument(
        "--sample-file",
        type=str,
        default=None,
        help="Directory or path for sample audio (Used in infer_sample mode)"
    )
    parser.add_argument(
        "--out-wav-dir",
        type=str,
        default=None,
        help="Directory to save output wav files (Used in test/infer modes)"
    )
    
    args, unknown = parser.parse_known_args()
    
    if unknown:
        logger.warning(f"Ignored unknown arguments injected by environment: {unknown}")

    try:
        safe_model_name = sanitize_model_name(args.model)
        module_path = f"models.{safe_model_name}.main"
        
        logger.info(f"Dynamically importing target module: {module_path}")
        main_module = importlib.import_module(module_path)
        
        if not hasattr(main_module, 'main'):
            raise AttributeError(f"The module '{module_path}' does not have a 'main(args)' function.")
            
        logger.info(f"Starting execution in mode: [{args.engine_mode.upper()}]")
        main_module.main(args)
        logger.info("🎉 Pipeline execution completed successfully.")

    except ModuleNotFoundError:
        logger.critical(f"Could not find model module at '{module_path.replace('.', '/')}.py'. Check spelling.")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Fatal error during execution: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()