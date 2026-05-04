from typing import Any, Dict, Optional
from .hive_mind import HiveMindOrchestrator

def build_agent(cfg: Dict[str, Any]) -> Optional[HiveMindOrchestrator]:
    """
    Builds and returns the Hive Mind Orchestrator if enabled in the config.
    Strictly config-driven: passes the entire parameters block to the engine.
    """
    if not cfg.get("enabled", False):
        return None
    
    # Pass the entire params block directly. Zero hardcoding.
    return HiveMindOrchestrator(cfg=cfg.get("params", {}))