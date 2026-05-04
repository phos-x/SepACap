from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

from .base import AgentBase
from .providers.base import LLMProviderInterface

logger = logging.getLogger(__name__)

@dataclass
class LLMAgentConfig:
    interval: int = 3
    temperature: float = 0.1
    action_whitelist: List[str] = field(default_factory=lambda: [
        "reduce_lr", "increase_lr", "continue"
    ])


class LLMAgent(AgentBase):
    """
    Decoupled Meta-Controller. 
    Relies on Dependency Injection for its LLM Provider.
    """
    def __init__(self, cfg: LLMAgentConfig, provider: LLMProviderInterface) -> None:
        self._cfg = cfg
        self._provider = provider  # The injected interface
        logger.info("Initializing Decoupled LLMAgent")

    def analyze(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        user_prompt = self._build_user_prompt(snapshot)
        system_prompt = "You are an expert MLOps diagnostic agent. Always output valid JSON."
        
        try:
            # The agent relies entirely on the interface contract
            raw_response = self._provider.generate_json(
                system_prompt=system_prompt, 
                user_prompt=user_prompt, 
                temperature=self._cfg.temperature
            )
            return self._parse_and_sanitize(raw_response)
        except Exception as e:
            logger.error(f"LLMAgent analysis aborted: {e}")
            return {"actions": [], "issues": ["agent_api_failure"]}

    def _build_user_prompt(self, snapshot: Dict[str, Any]) -> str:
        return (
            "Analyze this JSON snapshot of the training state. Identify instability or plateauing.\n\n"
            f"ALLOWED ACTIONS: {self._cfg.action_whitelist}\n\n"
            "Respond ONLY with a valid JSON object:\n"
            "{\"issues\": [\"...\"], \"actions\": [\"...\"]}\n\n"
            f"SNAPSHOT:\n{json.dumps(snapshot, indent=2)}"
        )

    def _parse_and_sanitize(self, raw_response: str) -> Dict[str, Any]:
        try:
            parsed = json.loads(raw_response)
            proposed_actions = parsed.get("actions", [])
            safe_actions = [act for act in proposed_actions if act in self._cfg.action_whitelist]
            return {"actions": safe_actions, "issues": parsed.get("issues", [])}
        except json.JSONDecodeError:
            logger.error(f"LLM returned invalid JSON: {raw_response}")
            return {"actions": [], "issues": ["json_parse_error"]}