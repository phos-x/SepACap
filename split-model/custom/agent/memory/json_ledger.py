import os
import json
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class JSONLedger:
    """
    A $0 local memory database for the AI Orchestrator.
    It reads and writes to a local JSON file to maintain state across training epochs.
    """
    def __init__(self, filepath: str = "agent_memory_ledger.json"):
        self.filepath = filepath
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        """Creates the JSON file if it doesn't exist."""
        if not os.path.exists(self.filepath):
            with open(self.filepath, 'w') as f:
                json.dump([], f)
            logger.info(f"🧠 MEMORY: Initialized new blank ledger at {self.filepath}")

    def log_intervention(self, epoch: int, metrics: Dict[str, Any], actions: List[Dict[str, Any]], reasoning: str):
        """
        Records what the AI saw, why it decided to act, and what it did.
        """
        # Load existing memory
        with open(self.filepath, 'r') as f:
            try:
                history = json.load(f)
            except json.JSONDecodeError:
                history = []

        # Create the new memory block
        entry = {
            "epoch": epoch,
            "state_snapshot": metrics,
            "ai_reasoning": reasoning,
            "actions_executed": actions
        }
        
        history.append(entry)

        # Save it back to disk
        with open(self.filepath, 'w') as f:
            json.dump(history, f, indent=2)

    def get_recent_history_prompt(self, limit: int = 3) -> str:
        """
        Retrieves the last N interventions.
        We limit this to keep the LLM token context small and free!
        """
        if not os.path.exists(self.filepath):
            return "No prior history. This is your first intervention."

        with open(self.filepath, 'r') as f:
            try:
                history = json.load(f)
            except json.JSONDecodeError:
                return "Memory corrupted or unreadable."

        if not history:
            return "No prior history. This is your first intervention."

        # Grab only the most recent N entries to save tokens
        recent_history = history[-limit:]
        
        prompt_text = "--- RECENT MEMORY LEDGER (Prior Interventions) ---\n"
        for entry in recent_history:
            prompt_text += f"Epoch {entry['epoch']}:\n"
            prompt_text += f"  - Saw State: {json.dumps(entry['state_snapshot'])}\n"
            prompt_text += f"  - Reasoning: {entry['ai_reasoning']}\n"
            prompt_text += f"  - Actions Taken: {json.dumps(entry['actions_executed'])}\n"
            prompt_text += "-" * 40 + "\n"
            
        prompt_text += "Analyze the CURRENT state and compare it to this history. Did your last actions work?\n"
        
        return prompt_text