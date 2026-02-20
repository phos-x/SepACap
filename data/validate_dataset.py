import os
import torch
import librosa
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("DatasetValidator")

def validate_sepacap_standard(scp_dir: str, test_list_path: str, target_sr=8000):
    """
    Enhanced validation for jaCappella/SepACap standards.
    Ensures signal periodicity support and silence-aware loss compatibility.
    """
    scp_path = Path(scp_dir)
    # The 7-septuplet standard for SepACap
    stems = ["alto", "bass", "lead_vocal", "soprano", "tenor", "vocal_percussion", "mixture"]
    partitions = ["tr", "cv", "tt"]
    
    # 1. Load Test Song List for Leakage Check
    test_songs = set()
    if os.path.exists(test_list_path):
        with open(test_list_path, 'r') as f:
            test_songs = {line.strip() for line in f if line.strip()}
    
    errors = 0

    for part in partitions:
        logger.info(f"--- Validating Partition: {part} ---")
        song_registry = {} 
        
        for stem in stems:
            scp_file = scp_path / f"{part}_{stem}.scp"
            if not scp_file.exists():
                logger.error(f"Critical Error: Missing manifest {scp_file}")
                errors += 1
                continue
            
            with open(scp_file, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
                
            for line in tqdm(lines, desc=f"Checking {stem}", leave=False):
                # Using maxsplit=1 to handle potential spaces in jaCappella paths
                parts = line.split(maxsplit=1)
                if len(parts) < 2:
                    logger.warning(f"Malformed line skipped: {line}")
                    continue
                
                song_id, audio_path = parts
                
                # Check A: Evaluation Integrity (Leakage)
                if part != "tt" and song_id in test_songs:
                    logger.error(f"Leakage Error: '{song_id}' is in {part} but must be TT only.")
                    errors += 1
                
                # Check B: File System Reachability
                if not os.path.exists(audio_path):
                    logger.error(f"IO Error: File not found -> {audio_path}")
                    errors += 1
                    continue
                
                # Check C: Technical Signal Specifications (only once per song_id to save time)
                if song_id not in song_registry:
                    try:
                        # Load header to verify SR and Channels
                        # Use soundfile backend via librosa for speed
                        sr = librosa.get_samplerate(audio_path)
                        
                        # 1. Sampling Rate Check (Requirement: 8kHz)
                        if sr != target_sr:
                            logger.error(f"SR Mismatch: {audio_path} is {sr}Hz, needs {target_sr}Hz")
                            errors += 1
                        
                        # 2. Channel Check (Requirement: Mono)
                        # Waveform-domain SepACap expects (1, S) tensors
                        y, _ = librosa.load(audio_path, sr=target_sr, mono=False)
                        if len(y.shape) > 1 and y.shape[0] > 1:
                            logger.error(f"Channel Error: {audio_path} is Stereo. Must be Mono.")
                            errors += 1
                        
                        samples = y.shape[-1]
                        
                        # 3. Platform Length Floor (Requirement: >1024)
                        # Prevents STFT kernel-size runtime errors
                        if samples < 1024:
                            logger.error(f"Size Error: {audio_path} too short ({samples} samples)")
                            errors += 1
                            
                        # 4. Stride Alignment Check (Requirement: % 4 == 0)
                        # SepReformer stages downsample by 4; mismatch causes dimension drift
                        if samples % 4 != 0:
                            logger.warning(f"Stride Warning: {audio_path} length {samples} is not divisible by 4.")
                            # This is a warning because dataset.py usually crops/pads this later.
                        
                        # 5. Silence Sanity Check (Power Set Support)
                        # Ensure Clean Stems aren't accidentally empty in the training set
                        if part == "tr" and stem != "mixture" and np.max(np.abs(y)) < 1e-6:
                            logger.warning(f"Silence Warning: Stem {stem} for {song_id} is digital silence.")

                    except Exception as e:
                        logger.error(f"Corruption Error: Could not process {audio_path}: {e}")
                        errors += 1
                
                # Track for Parallel Alignment check
                if song_id not in song_registry:
                    song_registry[song_id] = set()
                song_registry[song_id].add(stem)

        # Check D: Septuplet Alignment (Parallelism)
        # Every song must have exactly the 7 required files to allow subset mixing
        for sid, found_stems in song_registry.items():
            if len(found_stems) != len(stems):
                missing = set(stems) - found_stems
                logger.error(f"Alignment Error: Song {sid} in {part} is missing manifests for: {missing}")
                errors += 1

    if errors == 0:
        logger.info("✅ Dataset Validation Passed: SepACap Standards Met.")
        return True
    else:
        logger.error(f"❌ Dataset Validation Failed with {errors} errors.")
        return False

if __name__ == '__main__':
    # Update these paths to match your Kaggle structure
    SCP_DIR = "/kaggle/working/SepACap/data/scp_ss_8k_jacappella"
    TEST_LIST = "/kaggle/working/jaCappella/test_song_list_for_vocal_ensemble_separation.txt"
    
    validate_sepacap_standard(SCP_DIR, TEST_LIST)