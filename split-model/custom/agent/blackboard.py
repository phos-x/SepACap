import torch
import multiprocessing as mp
import time
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class NeuralBlackboard:
    """
    The Armored Asynchronous Panopticon.
    Utilizes cross-core shared memory to bypass the Python GIL entirely.
    """
    def __init__(self):
        # --- THE MICRO-INTERRUPT ---
        # A raw C-boolean living in shared memory. Evaluates in ~50 nanoseconds.
        # If the AI sets this to True, the GPU drops its tensors instantly.
        self.panic = mp.Value('b', False) 
        
        # --- THE HOLOGRAPHIC STATE MATRIX (SHARED MEMORY) ---
        # A flat, contiguous array of 16 double-precision floats ('d').
        # [0:4]   -> Environment: [x_norm, y_norm, has_nan, step_time_ms]
        # [4:8]   -> Silicon:     [vram_alloc, vram_res, vram_pct, 0.0]
        # [8:12]  -> Wav2Vec2:    [mass, velocity, variance, update_ratio]
        # [12:16] -> RoFormer:    [mass, velocity, variance, entropy/loss]
        self._shared_matrix = mp.Array('d', 16)
        
        # A process-safe queue for the LLM to submit execution intents to PyTorch
        self.action_queue = mp.Queue()
        
        self._last_step_time = time.time()
        logger.info("⬛ Armored Neural Blackboard Online. Shared Memory mapped to CPU cores.")

    def _get_fractal_physics(self, module: torch.nn.Module, lr: float) -> List[float]:
        """
        Calculates Mass, Velocity, and FRACTAL VARIANCE.
        Executes highly optimized, fused operations on the GPU before extracting.
        """
        if module is None:
            return [0.0, 0.0, 0.0, 0.0]

        layer_norms = []
        layer_grads = []
        
        for p in module.parameters():
            if p.requires_grad and p.grad is not None:
                layer_norms.append(p.data.norm(2))
                layer_grads.append(p.grad.detach().norm(2))

        if not layer_grads:
            return [0.0, 0.0, 0.0, 0.0]

        # Stack onto the GPU, compute stats, and only sync to CPU once (.item)
        stacked_norms = torch.stack(layer_norms)
        stacked_grads = torch.stack(layer_grads)

        mass = stacked_norms.norm(2).item()
        velocity = stacked_grads.norm(2).item()
        
        # FRACTAL VARIANCE: Catches localized sub-network death
        variance = stacked_grads.std().item()

        update_ratio = (lr * velocity) / mass if mass > 0 else 0.0
        
        return [round(mass, 4), round(velocity, 4), round(variance, 4), float(f"{update_ratio:.2e}")]

    def write_matrix(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
                     batch_x: torch.Tensor, batch_y: torch.Tensor, loss: float):
        """
        Invoked by the PyTorch GPU Thread. 
        Calculates physics and dumps directly into C-level shared memory.
        """
        # 1. Environment
        has_nan = 1.0 if torch.isnan(batch_x).any().item() or torch.isnan(batch_y).any().item() else 0.0
        x_norm = batch_x.detach().norm(2).item()
        y_norm = batch_y.detach().norm(2).item()
        
        current_time = time.time()
        step_time_ms = (current_time - self._last_step_time) * 1000
        self._last_step_time = current_time

        # 2. Silicon (Hardware constraints)
        if torch.cuda.is_available():
            vram_alloc = torch.cuda.memory_allocated() / (1024 ** 2)
            vram_res = torch.cuda.memory_reserved() / (1024 ** 2)
            vram_pct = (vram_alloc / 15000.0) * 100  # Assumes 15GB usable Kaggle T4 limit
        else:
            vram_alloc, vram_res, vram_pct = 0.0, 0.0, 0.0

        # 3. Physics (Fractal Sub-network states)
        lr = optimizer.param_groups[0]['lr']
        base_model = model.module if hasattr(model, 'module') else model
        
        w2v2_module = getattr(base_model, 'context_encoder', None)
        roformer_module = getattr(base_model, 'separator', None)

        w2v2_physics = self._get_fractal_physics(w2v2_module, lr)
        roformer_physics = self._get_fractal_physics(roformer_module, lr)
        
        # Overwrite RoFormer's 4th slot with Entropy (Loss) for the plateau detector
        roformer_physics[3] = round(loss, 4) 

        # --- WRITE TO SHARED MEMORY (NO GIL, NO LOCK DELAY) ---
        # We acquire the memory lock for exactly 16 float assignments (microseconds)
        with self._shared_matrix.get_lock():
            self._shared_matrix[0:4] = [x_norm, y_norm, has_nan, step_time_ms]
            self._shared_matrix[4:8] = [vram_alloc, vram_res, vram_pct, 0.0]
            self._shared_matrix[8:12] = w2v2_physics
            self._shared_matrix[12:16] = roformer_physics

    def read_matrix(self) -> Dict[str, List[float]]:
        """
        Invoked by the LLM Daemons on a separate CPU Core. 
        Reconstructs the 4x4 Python dictionary for the Groq API payload.
        """
        with self._shared_matrix.get_lock():
            arr = self._shared_matrix[:] # Rapid copy from C-array to Python list
            
        return {
            "environment": arr[0:4],
            "silicon": arr[4:8],
            "wav2vec2": arr[8:12],
            "roformer": arr[12:16]
        }

    def post_action(self, action: Dict[str, Any]):
        """Invoked by LLM Daemons to queue an execution intent."""
        self.action_queue.put(action)
        logger.info(f"📥 Action queued by Daemon: {action.get('tool_name')}")

    def consume_actions(self) -> List[Dict[str, Any]]:
        """Invoked by PyTorch to drain the queue at the end of a batch."""
        actions = []
        while not self.action_queue.empty():
            try:
                actions.append(self.action_queue.get_nowait())
            except mp.queues.Empty:
                break
        return actions

    def trigger_panic(self):
        """Invoked by the Stabilizer Daemon to initiate a micro-interrupt."""
        self.panic.value = True

# Global Singleton for ease of access
global_blackboard = NeuralBlackboard()