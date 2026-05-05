import os
import json
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class PanopticonEngine:
    """
    The API Connector for the Asynchronous Hive Mind.
    Translates physical 4x4 matrices into LLM Prompts and returns strict JSON execution intents.
    """
    def __init__(self, config: Dict[str, Any]):
        try:
            from groq import Groq
            self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        except ImportError:
            raise ImportError("Please `pip install groq` to use the Panopticon Engine.")
            
        self.model = config.get("model", "llama-3.3-70b-versatile")
        self.temperature = config.get("temperature", 0.1) # Ruthless logic, zero creativity
        self.max_tokens = config.get("max_tokens", 512)

    def _call_llm_json(self, system_prompt: str, user_payload: dict) -> Dict[str, Any]:
        """A standardized, robust Groq caller enforcing JSON output."""
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(user_payload, indent=2)}
                ],
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"}
            )
            response_text = chat_completion.choices[0].message.content
            return json.loads(response_text)
        except Exception as e:
            logger.error(f"LLM API Failure: {e}")
            return {}

    def analyze_hardware_anomaly(self, payload: dict) -> dict:
        """
        Invoked by the Stabilizer Daemon (Brain B) when VRAM spikes or data is poisoned.
        """
        system_prompt = """
        You are the STABILIZER DAEMON for a PyTorch audio separation network.
        Your sole directive is survival. You monitor hardware (Silicon) and external data (Environment).
        
        The PyTorch loop has triggered an interrupt because a critical threshold was crossed.
        Read the provided payload containing the trigger reason and the state matrix.
        
        You must output a strict JSON object with EXACTLY three keys:
        1. "action": The string tool to use (e.g., "slash_batch_size", "halt_augmentations", "none").
        2. "value": The numeric or string parameter for the tool (e.g., 0.5 for batch size reduction).
        3. "ai_reasoning": A 1-sentence explanation of your logical deduction for the DNA Ledger.
        
        If VRAM > 90%, you MUST slash batch size.
        If Input Data has NaNs, you MUST halt augmentations.
        """
        return self._call_llm_json(system_prompt, payload)

    def analyze_physics_anomaly(self, payload: dict) -> dict:
        """
        Invoked by the Accelerator Daemon (Brain A) when learning plateaus or networks die.
        """
        system_prompt = """
        You are the ACCELERATOR DAEMON for a PyTorch audio separation network.
        Your directive is to optimize kinetic learning energy without causing catastrophic collapse.
        
        You monitor the internal math:
        - Mass: Network weight magnitude.
        - Velocity: Gradient update magnitude.
        - Variance: Fractal standard deviation of gradients (detects localized layer death).
        - Update Ratio / Entropy: Loss and learning momentum.
        
        The PyTorch loop has detected a physical anomaly (plateau, forgetting, or dead neurons).
        
        You must output a strict JSON object with EXACTLY three keys:
        1. "action": The tool to use (e.g., "set_learning_rate", "unfreeze_wav2vec2", "warm_restart").
        2. "value": The numeric parameter for the tool (e.g., 1e-5).
        3. "ai_reasoning": A concise, mathematical explanation of your deduction for the DNA Ledger.
        
        If Velocity is < 1e-4 but Entropy is high, spike the learning rate to escape the local minimum.
        If Variance is > 50.0 but Velocity is normal, localized death has occurred. Drop the learning rate.
        """
        return self._call_llm_json(system_prompt, payload)
    
    def analyze(self, snapshot: dict, live_context: dict = None) -> dict:
        """
        Invoked by the Orchestrator (train.py) at the end of an epoch.
        This is Brain C: The Chief Scientist.
        It analyzes the high-resolution Neurological MRI (telemetry) and global SDR scores.
        """
        system_prompt = """
        You are the CHIEF SCIENTIST for a PyTorch audio separation training run.
        You only wake up at the end of an epoch. 
        
        You are reviewing the 'Neurological MRI' (Telemetry) and the global validation metrics (SDR).
        Your job is MACRO-STRATEGY. 
        - Should we unfreeze more layers of the Context Encoder?
        - Do we need to alter the multi-loss function because the Bass stem is failing?
        
        Read the provided snapshot. Output a strict JSON object with EXACTLY three keys:
        1. "debate": A nested object with two strings: "accelerator" (arguments for aggressive optimization) and "stabilizer" (arguments for caution).
        2. "reasoning": A 2-sentence summary of your final strategic decision.
        3. "actions_taken": A list of JSON action objects (e.g., [{"tool_name": "unfreeze_layer", "value": "context_encoder.layer_10"}]). Keep empty if no action is needed.
        """
        
        # We don't send the live PyTorch objects (live_context) to the LLM, just the JSON snapshot
        try:
            return self._call_llm_json(system_prompt, snapshot)
        except Exception as e:
            logger.error(f"Chief Scientist Analysis Failed: {e}")
            return {"debate": {}, "reasoning": "API Error. Defaulting to nominal continuation.", "actions_taken": []}