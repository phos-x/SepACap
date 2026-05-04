import os
import json
import logging
from typing import Dict, Any
import concurrent.futures
from openai import OpenAI

from .memory.json_ledger import JSONLedger
from .tools.registry import registry
from .tools import lr_tools, loss_tools, freeze

logger = logging.getLogger(__name__)

class HiveMindOrchestrator:
    """
    Level 6: Config-Driven Adversarial Mixture of Experts.
    Strictly adheres to dependency injection: no hardcoded URLs, models, or prompts.
    """
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        
        # Core API Configuration
        self.api_key = os.environ.get(self.cfg.get("api_key_env_var", "OPENAI_API_KEY"), "")
        self.base_url = self.cfg.get("base_url") # Falls back to default OpenAI client if None
        
        # Initialize the global client (can be overridden per-brain if needed in the future)
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        self.ledger = JSONLedger()
        
        # Load Brain Configurations
        self.brains = self.cfg.get("brains", {})
        self.accel_cfg = self.brains.get("accelerator", {})
        self.stab_cfg = self.brains.get("stabilizer", {})
        self.chief_cfg = self.brains.get("chief", {})
        
        # Threading Config
        self.max_workers = self.cfg.get("max_workers", 2)
        
        logger.info(f"🐝 Config-Driven Hive Mind Online. Chief Model: {self.chief_cfg.get('model', 'Unknown')}")

    def _get_expert_opinion(self, brain_cfg: Dict[str, Any], snapshot_str: str) -> str:
        """Helper to query an expert brain based strictly on its YAML config."""
        model = brain_cfg.get("model")
        temp = brain_cfg.get("temperature", 0.7)
        system_prompt = brain_cfg.get("system_prompt", "You are an AI assistant.")
        
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": snapshot_str}
            ],
            temperature=temp,
        )
        return response.choices[0].message.content

    def analyze(self, snapshot: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        context = context or {}
        snapshot_str = json.dumps(snapshot, indent=2)
        available_tools = registry.get_tool_prompt()

        # ---------------------------------------------------------
        # PHASE 1: PARALLEL ADVERSARIAL DEBATE
        # ---------------------------------------------------------
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_accel = executor.submit(self._get_expert_opinion, self.accel_cfg, snapshot_str)
            future_stab = executor.submit(self._get_expert_opinion, self.stab_cfg, snapshot_str)
            
            accel_opinion = future_accel.result()
            stab_opinion = future_stab.result()

        # ---------------------------------------------------------
        # PHASE 2: THE CHIEF SCIENTIST (Consensus & Execution)
        # ---------------------------------------------------------
        chief_system_prompt = self.chief_cfg.get("system_prompt", "You are the Chief Scientist.")
        chief_system_prompt += (
            f"\n\n--- AVAILABLE TOOLS ---\n{available_tools}\n\n"
            "Respond ONLY with a valid JSON object matching this schema:\n"
            "{\n"
            '  "reasoning": "Explain your final decision.",\n'
            '  "tool_calls": [{"name": "tool_name", "args": {"arg1": "value"}}]\n'
            "}"
        )

        user_chief = (
            f"--- TRAINING TELEMETRY ---\n{snapshot_str}\n\n"
            f"--- ACCELERATOR PROPOSAL ---\n{accel_opinion}\n\n"
            f"--- STABILIZER PROPOSAL ---\n{stab_opinion}\n\n"
            "Make the final executive decision. Output strict JSON."
        )

        try:
            response = self.client.chat.completions.create(
                model=self.chief_cfg.get("model"),
                messages=[
                    {"role": "system", "content": chief_system_prompt},
                    {"role": "user", "content": user_chief}
                ],
                temperature=self.chief_cfg.get("temperature", 0.1),
                response_format={"type": "json_object"}
            )
            payload = json.loads(response.choices[0].message.content)
            reasoning = payload.get("reasoning", "")
            tool_calls = payload.get("tool_calls", [])
            
        except Exception as e:
            logger.error(f"🤖 Hive Mind Collapse: {e}")
            return {"actions_taken": [], "reasoning": "API Failure"}

        # ---------------------------------------------------------
        # PHASE 3: EXECUTION
        # ---------------------------------------------------------
        actions_taken = []
        for call in tool_calls:
            t_name = call.get("name")
            t_args = call.get("args", {})
            result = registry.execute(t_name, t_args, context)
            actions_taken.append({t_name: result})

        self.ledger.log_intervention(snapshot.get("epoch", 0), snapshot, actions_taken, reasoning)
        
        return {
            "actions_taken": actions_taken, 
            "reasoning": reasoning,
            "debate": {"accelerator": accel_opinion, "stabilizer": stab_opinion}
        }