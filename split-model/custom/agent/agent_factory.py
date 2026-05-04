from typing import Any, Dict, Optional
from .orchestrator import AutonomousOrchestrator

def build_agent(cfg: Dict[str, Any]) -> Optional[AutonomousOrchestrator]:
    """
    Builds and returns the Autonomous Orchestrator if enabled in the config.
    """
    if not cfg.get("enabled", False):
        return None

    params = cfg.get("params", {})
    
    # Extract config parameters with safe defaults for the Groq/Llama-3 setup
    model_name = params.get("model_name", "llama-3.1-8b-instant")
    api_key_env = params.get("api_key_env_var", "GROQ_API_KEY")
    
    # The Orchestrator manages its own memory ledger and tool registry natively
    return AutonomousOrchestrator(api_key_env=api_key_env, model=model_name)