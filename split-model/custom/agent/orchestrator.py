import os
import json
import logging
from typing import Dict, Any
from openai import OpenAI

from .memory.json_ledger import JSONLedger
from .tools.registry import registry

# IMPORTANT: We must import the tool files here so the @registry.register decorators fire and load the tools
from .tools import lr_tools, loss_tools, freeze

logger = logging.getLogger(__name__)

class AutonomousOrchestrator:
    """
    The Level 4.5 AI Orchestrator. 
    It possesses long-term memory, single-shot tool execution, and self-healing logic.
    """
    def __init__(self, api_key_env: str = "GROQ_API_KEY", model: str = "llama-3.1-8b-instant"):
        self.model_name = model
        api_key = os.environ.get(api_key_env)
        
        if not api_key:
            logger.warning(f"⚠️ {api_key_env} is not set. Orchestrator will fail to connect.")

        self.client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=api_key
        )
        self.ledger = JSONLedger()
        logger.info(f"🧠 Autonomous Orchestrator Online. Model: {self.model_name}")

    def analyze(self, snapshot: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        The main AI loop: Observe -> Recall -> Think -> Act -> Remember.
        `context` contains the live PyTorch objects (optimizer, multi_loss) passed from train.py.
        """
        context = context or {}
        epoch = snapshot.get("epoch", 0)

        # 1. RECALL: Fetch the last few interventions
        history_context = self.ledger.get_recent_history_prompt()
        
        # 2. CAPABILITY: Get the live tool specification
        available_tools = registry.get_tool_prompt()

        # 3. THINK: Build the cognitive prompt
        system_prompt = (
            "You are an accurate autonomous AI ML Engineer orchestrating a complex PyTorch audio separation training loop.\n"
            "Your goal is to prevent overfitting, fix plateauing instruments, and avoid NaN catastrophic collapses.\n"
            "Respond ONLY with a valid JSON object matching this schema:\n"
            "{\n"
            '  "reasoning": "Detailed explanation of why you are taking these actions based on the current metrics and past history.",\n'
            '  "tool_calls": [\n'
            '    {"name": "tool_name", "args": {"arg1": value}}\n'
            '  ]\n'
            "}"
        )

        user_prompt = (
            f"--- CURRENT TRAINING SNAPSHOT ---\n{json.dumps(snapshot, indent=2)}\n\n"
            f"{history_context}\n\n"
            f"--- AVAILABLE TOOLS ---\n{available_tools}\n\n"
            "Analyze the state. If no action is needed, leave tool_calls empty. Do not hallucinate tools. review results multiple times before responding to ensure accuracy and relevance."
        )

        try:
            # Generate the action plan (Single-shot, $0 cost)
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1, # Keep it highly deterministic
                response_format={"type": "json_object"}
            )
            
            payload = json.loads(response.choices[0].message.content)
            reasoning = payload.get("reasoning", "No reasoning provided.")
            tool_calls = payload.get("tool_calls", [])
            
        except Exception as e:
            logger.error(f"🤖 Brain Fault: {e}")
            return {"actions_taken": [], "reasoning": f"API Failure: {e}"}

        # 4. ACT: Execute the physical tools
        actions_taken = []
        for call in tool_calls:
            t_name = call.get("name")
            t_args = call.get("args", {})
            
            # The registry handles injecting the live PyTorch objects
            result = registry.execute(t_name, t_args, context)
            actions_taken.append({t_name: result})
            logger.info(result)

        # 5. REMEMBER: Commit this intervention to the ledger
        self.ledger.log_intervention(epoch, snapshot, actions_taken, reasoning)

        return {"actions_taken": actions_taken, "reasoning": reasoning}  