import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

def build_agent(cfg: Dict[str, Any]) -> Optional[Any]:
    """
    Builds and returns the LLM Engine for the Asynchronous Panopticon.
    This engine is passed to the DaemonManager to evaluate threshold anomalies via Groq.
    """
    if not cfg.get("enabled", False):
        logger.info("⬛ Asynchronous Panopticon is disabled in config.")
        return None
    
    try:
        # We import locally to prevent crashes if Groq isn't installed
        from .llm_engine import PanopticonEngine 
        
        logger.info(f"🧠 Booting Panopticon LLM Engine: {cfg.get('model', 'llama-3.3-70b-versatile')}")
        return PanopticonEngine(cfg)
        
    except ImportError as e:
        logger.error(f"Failed to import PanopticonEngine. Is Groq installed? Error: {e}")
        return None