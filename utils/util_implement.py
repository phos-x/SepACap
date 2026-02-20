import torch
import importlib
import inspect
from dataclasses import dataclass
from typing import List, Type, Any, Callable, Optional, Union
from loguru import logger
from utils.decorators import *

@dataclass(slots=True)
class BaseFactory:
    config: dict

    def load_class(self, category: str, lib_check: Type[Any], name: str) -> Type[Any]:
        """Loading Classes from a Module with fallback to custom implementations."""
        try:
            # First check the standard library (torch.nn, torch.optim, etc)
            if hasattr(lib_check, name):
                return getattr(lib_check, name)
            
            # Fallback to custom research implementations
            # Expected path: models.SepReformer_Base_WSJ0.criterions (or similar)
            # Or the path provided in your sys.path: utils.implements.{category}
            module_path = f"utils.implements.{category}"
            module = importlib.import_module(module_path)
            return getattr(module, name)
            
        except (ImportError, AttributeError):
            # Final fallback: Try to load from the model's local directory if called from main
            try:
                # This handles cases where criterions are in the model folder
                module_path = f"models.SepReformer_Base_WSJ0.{category}"
                module = importlib.import_module(module_path)
                return getattr(module, name)
            except Exception as e:
                logger.error(f"Class {name} could not be located in {lib_check} or custom modules: {e}")
                raise

    def create_instance(self, cls: Type[Any], name: str, target: Any = None, target_key: str = None) -> Any:
        """
        Create Instance with Signature Inspection.
        Prevents TypeError by checking if the class actually accepts the target.
        """
        arg_dict = self.config.get(name, {}).copy()
        
        # Inspection logic: check if 'device' or 'params' is a valid argument for the __init__
        sig = inspect.signature(cls.__init__)
        params = sig.parameters
        
        # Prepare arguments
        if target is not None and target_key is not None:
            # If the class accepts the specific target key (e.g., 'device' or 'optimizer')
            if target_key in params or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
                arg_dict[target_key] = target
            else:
                # If it's a standard torch class (like L1Loss) that doesn't take 'device' in init
                logger.debug(f"Class {cls.__name__} does not accept {target_key}, skipping.")

        logger.info(f'Instantiating {cls.__name__} for "{name}"')
        
        # Determine if we should pass the target positionally (legacy) or as kwarg
        # Standard approach for this pipeline:
        try:
            return cls(**arg_dict)
        except TypeError as e:
            # Fallback for classes that require a positional target (like some Schedulers)
            if target is not None:
                return cls(target, **arg_dict)
            raise e

    def get_instances(self, category: str, lib_check: Type[Any], instance_creator: Callable) -> List[Any]:
        """Create Instance List"""
        # Ensure we don't crash if "name" is a string instead of a list
        names = self.config["name"]
        if isinstance(names, str):
            names = [names]
            
        return [instance_creator(self.load_class(category, lib_check, name), name) for name in names]


@logger_wraps()
@dataclass(slots=True)
class CriterionFactory(BaseFactory):
    device: torch.device
    
    def get_criterions(self) -> List[Any]:
        # We map 'device' to the target_key to ensure SepACapCompositeLoss receives it
        return self.get_instances(
            "criterions", 
            torch.nn, 
            lambda cls, name: self.create_instance(cls, name, target=self.device, target_key="device")
        )


@logger_wraps()
@dataclass(slots=True)
class OptimizerFactory(BaseFactory):
    parameters_policy: Any

    def get_optimizers(self) -> List[Any]:
        # Optimizers usually take params as the first positional argument
        return self.get_instances(
            "optimizers", 
            torch.optim, 
            lambda cls, name: self.create_instance(cls, name, target=self.parameters_policy, target_key="params")
        )


@logger_wraps()
@dataclass(slots=True)
class SchedulerFactory(BaseFactory):
    optimizers: List[Any]

    def get_schedulers(self) -> List[Any]:
        # Sync number of optimizers with number of requested schedulers
        requested_names = self.config["name"]
        if isinstance(requested_names, str):
            requested_names = [requested_names]
            
        opts = self.optimizers
        if len(opts) == 1 and len(requested_names) > 1:
            opts = opts * len(requested_names)

        return [
            self.create_instance(
                self.load_class("schedulers", torch.optim.lr_scheduler, name), 
                name, 
                target=opt, 
                target_key="optimizer"
            ) 
            for name, opt in zip(requested_names, opts)
        ]