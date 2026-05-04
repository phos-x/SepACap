from typing import Any, Dict, Optional
from .llm_agent import LLMAgent, LLMAgentConfig
from .providers.openai_compat import OpenAICompatibleProvider

def build_agent(cfg: Dict[str, Any]) -> Optional[LLMAgent]:
    if not cfg.get("enabled", False):
        return None

    params = cfg.get("params", {})
    
    # 1. Build the Provider dynamically based on config
    provider_type = params.get("provider_type", "openai_compat")
    
    if provider_type == "openai_compat":
        provider = OpenAICompatibleProvider(
            base_url=params.get("base_url"),
            model_name=params.get("model_name"),
            api_key_env_var=params.get("api_key_env_var")
        )
    else:
        raise ValueError(f"Unknown provider_type: {provider_type}")
        
    # 2. Build the Agent config
    agent_cfg = LLMAgentConfig(
        interval=params.get("interval", 3),
        temperature=params.get("temperature", 0.1),
        action_whitelist=params.get("action_whitelist", ["continue"])
    )
    
    # 3. Inject the Provider into the Agent
    return LLMAgent(cfg=agent_cfg, provider=provider)