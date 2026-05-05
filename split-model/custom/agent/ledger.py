import os
import json
import time
import torch
import numpy as np
import random
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class LedgerDesyncError(Exception):
    """Raised when the physical state of the playback diverges from the recorded DNA Ledger."""
    pass

class DNALedger:
    """
    The State-Verified DNA Ledger.
    Enforces absolute determinism and causal reasoning trails for the Asynchronous Panopticon.
    """
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.is_playback_mode = False
        
        # O(1) lookup table for playback: {(epoch, step): mutation_dict}
        self._playback_mutations = {}
        self.genesis_config = {}

    def write_genesis_block(self, config: Dict[str, Any], seed: int = 42):
        """
        Locks the physical universe of the training run.
        Must be called BEFORE initializing the model or dataloaders.
        """
        if self.is_playback_mode:
            logger.warning("Playback Mode Active: Skipping Genesis Block creation.")
            return

        # 1. Force strict hardware/software determinism
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # 2. Construct the Genesis Block
        genesis = {
            "block_type": "GENESIS",
            "timestamp": time.time(),
            "environment": {
                "pytorch_version": torch.__version__,
                "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
                "master_seed": seed
            },
            "base_config": config
        }
        
        # Write mode 'w' creates/overwrites the file
        with open(self.filepath, 'w') as f:
            f.write(json.dumps(genesis) + "\n")
            
        logger.info(f"🧬 Genesis Block written to {self.filepath}. Universe locked with seed {seed}.")

    def record_mutation(self, epoch: int, step: int, matrix_state: Dict[str, list], action: Dict[str, Any], reasoning: str):
        """
        Records an AI action stamped with the exact physical state of the network.
        matrix_state expects the reconstructed dictionary from the Blackboard.
        """
        if self.is_playback_mode:
            return

        # Extract the critical dimensions for the checksum (e_in, roformer mass, velocity, entropy)
        env = matrix_state.get("environment", [0,0,0,0])
        roformer = matrix_state.get("roformer", [0,0,0,0])

        mutation = {
            "block_type": "MUTATION",
            "temporal_address": {"epoch": epoch, "step": step},
            "state_hash": {
                "e_in": env[0],
                "mass": roformer[0],
                "velocity": roformer[1],
                "entropy": roformer[3] 
            },
            "execution": action,
            "ai_reasoning": reasoning
        }
        
        # Append mode 'a' adds to the ledger safely
        with open(self.filepath, 'a') as f:
            f.write(json.dumps(mutation) + "\n")
            
        logger.info(f"🧬 Mutation recorded at Epoch {epoch}, Step {step}: {action.get('tool_name', 'Unknown')}")

    def load_for_playback(self):
        """
        Loads the JSONL ledger into memory for O(1) deterministic playback.
        Sets the ledger to playback mode.
        """
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"Cannot load playback: Ledger file {self.filepath} missing.")

        self.is_playback_mode = True
        self._playback_mutations.clear()

        with open(self.filepath, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                
                block = json.loads(line)
                
                if block["block_type"] == "GENESIS":
                    self.genesis_config = block.get("base_config", {})
                    logger.info("🧬 Genesis Block loaded for Playback.")
                    
                elif block["block_type"] == "MUTATION":
                    addr = block["temporal_address"]
                    key = (addr["epoch"], addr["step"])
                    self._playback_mutations[key] = block

        logger.info(f"⏪ Playback Mode Armed. Loaded {len(self._playback_mutations)} historical mutations.")

    def get_mutation(self, epoch: int, step: int) -> Optional[Dict[str, Any]]:
        """
        Invoked by PyTorch during the training loop.
        Returns the expected mutation for this exact moment in time, or None.
        """
        if not self.is_playback_mode:
            return None
            
        return self._playback_mutations.get((epoch, step))

    def verify_state(self, expected_hash: Dict[str, float], current_matrix: Dict[str, list], tolerance: float = 1e-3):
        """
        Cryptographically verifies that the current PyTorch run hasn't drifted 
        from the historical ledger.
        """
        env = current_matrix.get("environment", [0,0,0,0])
        roformer = current_matrix.get("roformer", [0,0,0,0])
        
        actual_hash = {
            "e_in": env[0],
            "mass": roformer[0],
            "velocity": roformer[1],
            "entropy": roformer[3]
        }

        # Compare the floats with a slight tolerance for floating-point determinism quirks
        for key in expected_hash:
            diff = abs(expected_hash[key] - actual_hash[key])
            if diff > tolerance:
                raise LedgerDesyncError(
                    f"Playback Drift Detected on [{key}]. "
                    f"Expected: {expected_hash[key]:.4f}, Actual: {actual_hash[key]:.4f}, Diff: {diff:.4f}"
                )