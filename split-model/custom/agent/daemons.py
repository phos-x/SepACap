import multiprocessing as mp
import time
import logging
from typing import Dict, Any
from .blackboard import NeuralBlackboard

logger = logging.getLogger(__name__)

class BaseDaemon(mp.Process):
    """
    Abstract base class for all asynchronous Cognitive Daemons.
    Runs on entirely separate CPU cores to bypass the Python GIL.
    """
    def __init__(self, name: str, poll_interval: float, llm_engine: Any, blackboard: NeuralBlackboard):
        super().__init__(name=name, daemon=True) 
        self.poll_interval = poll_interval
        self.llm_engine = llm_engine
        self.blackboard = blackboard
        self._stop_event = mp.Event()

    def run(self):
        # We re-seed/re-init anything that might have broken across the process fork
        logger.info(f"🧠 Daemon [{self.name}] online. Core isolated. Polling: {self.poll_interval}s")
        while not self._stop_event.is_set():
            try:
                self._monitor_and_react()
            except Exception as e:
                logger.error(f"Daemon [{self.name}] encountered an error: {e}")
            time.sleep(self.poll_interval)

    def stop(self):
        self._stop_event.set()

    def _monitor_and_react(self):
        raise NotImplementedError("Subclasses must implement monitoring logic.")


class StabilizerDaemon(BaseDaemon):
    """
    Brain B: The Amygdala.
    Monitors Environment (Data) and Silicon (Hardware).
    Possesses the authority to trigger the Micro-Interrupt Kill Switch.
    """
    def __init__(self, llm_engine: Any, blackboard: NeuralBlackboard, vram_panic_threshold: float = 90.0):
        super().__init__(name="Stabilizer_Brain", poll_interval=2.0, llm_engine=llm_engine, blackboard=blackboard)
        self.vram_panic_threshold = vram_panic_threshold

    def _monitor_and_react(self):
        matrix = self.blackboard.read_matrix()
        env = matrix.get("environment", [0,0,0,0])
        silicon = matrix.get("silicon", [0,0,0,0])

        has_nan = env[2] > 0.5
        vram_pct = silicon[2]

        trigger_reason = None
        
        # 1. HARDWARE PANIC: Execute Micro-Interrupt
        if vram_pct > self.vram_panic_threshold:
            trigger_reason = f"CRITICAL: VRAM Allocation at {vram_pct}%. Impending OOM."
            logger.critical("🚨 STABILIZER FIRING MICRO-INTERRUPT. SEVERING GPU FORWARD PASS.")
            self.blackboard.trigger_panic() # Drops GPU tensors instantly in the PyTorch loop
            
        # 2. DATA POISONING
        elif has_nan:
            trigger_reason = "CRITICAL: Poisoned Data (NaN) detected in Environment."

        # PING LLM
        if trigger_reason:
            logger.warning(f"🚨 Stabilizer Triggered: {trigger_reason}")
            
            payload = {
                "trigger": trigger_reason,
                "matrix_rows": {"environment": env, "silicon": silicon}
            }
            
            # The LLM MUST return a JSON with: {"action": "...", "value": "...", "ai_reasoning": "..."}
            intent = self.llm_engine.analyze_hardware_anomaly(payload)
            
            if intent and "ai_reasoning" in intent:
                self.blackboard.post_action(intent)
                
            time.sleep(10.0) # Cooldown to let PyTorch recover from the interrupt


class AcceleratorDaemon(BaseDaemon):
    """
    Brain A: The Dopamine System.
    Monitors Physics and Fractal Variance.
    """
    def __init__(self, llm_engine: Any, blackboard: NeuralBlackboard):
        super().__init__(name="Accelerator_Brain", poll_interval=10.0, llm_engine=llm_engine, blackboard=blackboard)
        self.plateau_counter = 0

    def _monitor_and_react(self):
        matrix = self.blackboard.read_matrix()
        w2v2 = matrix.get("wav2vec2", [0,0,0,0])
        roformer = matrix.get("roformer", [0,0,0,0])

        w2v2_update_ratio = w2v2[3]
        
        roformer_vel = roformer[1]
        roformer_var = roformer[2] # The Fractal Variance
        roformer_entropy = roformer[3]

        trigger_reason = None
        
        # 1. FRACTAL VARIANCE PANIC: Localized Sub-network Death
        if roformer_var > 50.0 and roformer_vel < 100.0:
            # Velocity is normal, but variance is massive. One attention head just exploded or died.
            trigger_reason = f"CRITICAL ANOMALY: Fractal Variance spiked to {roformer_var}. Localized network death detected despite normal global velocity."

        # 2. CATASTROPHIC FORGETTING
        elif w2v2_update_ratio > 0.1:
            trigger_reason = f"WARNING: Wav2Vec2 update ratio at {w2v2_update_ratio:.2e}. Pre-trained weights collapsing."
            
        # 3. LOCAL MINIMA / PLATEAUS
        elif roformer_vel < 1e-4 and roformer_entropy > 0.5:
            self.plateau_counter += 1
            if self.plateau_counter >= 3:
                trigger_reason = "STRATEGY: Network trapped in local minimum. Velocity dead. Entropy static."
                self.plateau_counter = 0
        else:
            self.plateau_counter = 0

        # PING LLM
        if trigger_reason:
            logger.info(f"🔥 Accelerator Triggered: {trigger_reason}")
            
            payload = {
                "trigger": trigger_reason,
                "matrix_rows": {"wav2vec2": w2v2, "roformer": roformer}
            }
            
            intent = self.llm_engine.analyze_physics_anomaly(payload)
            
            if intent and "ai_reasoning" in intent:
                self.blackboard.post_action(intent)
                
            time.sleep(30.0) 


class DaemonManager:
    """
    Orchestrates the Multiprocessing Daemons.
    """
    def __init__(self, llm_engine: Any, blackboard: NeuralBlackboard, config: Dict[str, Any]):
        self.daemons = [
            StabilizerDaemon(llm_engine, blackboard, vram_panic_threshold=config.get("vram_panic_threshold", 90.0)),
            AcceleratorDaemon(llm_engine, blackboard)
        ]

    def start_all(self):
        for d in self.daemons:
            d.start()

    def stop_all(self):
        logger.info("🛑 Shutting down Multiprocessing Daemons...")
        for d in self.daemons:
            d.stop()
        for d in self.daemons:
            d.join(timeout=3.0)
            if d.is_alive():
                logger.warning(f"Daemon {d.name} hung. Terminating aggressively.")
                d.terminate()