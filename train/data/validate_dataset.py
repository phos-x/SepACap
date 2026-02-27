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
    Finalized validation for the jaCappella-SepACap pipeline.
    Validates the 7-file septuplet (1 mix + 6 stems) required for Power Set Augmentation.
    """
    scp_path = Path(scp_dir)
    stems = ["alto", "bass", "lead_vocal", "soprano", "tenor", "vocal_percussion", "mixture"]
    partitions = ["tr", "cv", "tt"]
    
    test_songs = set()
    if os.path.exists(test_list_path):
        with open(test_list_path, 'r', encoding='utf-8') as f:
            test_songs = {line.strip() for line in f if line.strip()}
    
    errors = 0

    for part in partitions:
        logger.info(f"--- Validating Partition: {part} ---")
        song_registry = {} 
        
        for stem in stems:
            scp_file = scp_path / f"{part}_{stem}.scp"
            if not scp_file.exists():
                logger.error(f"Critical Path Error: Missing manifest -> {scp_file.name}")
                errors += 1
                continue
            
            with open(scp_file, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
                
            for line in tqdm(lines, desc=f"Checking {stem}", leave=False):
                parts = line.split(maxsplit=1)
                if len(parts) < 2:
                    logger.warning(f"Malformed manifest line (skipped): {line}")
                    continue
                
                song_id, audio_path = parts
                
                if part != "tt" and song_id in test_songs:
                    logger.error(f"Leakage: Song '{song_id}' found in {part} but is reserved for Test set.")
                    errors += 1
                
                if not os.path.exists(audio_path):
                    logger.error(f"Missing Audio: {audio_path}")
                    errors += 1
                    continue
                
                if song_id not in song_registry:
                    try:
                        sr = librosa.get_samplerate(audio_path)
                        
                        # 1. Sampling Rate Check (Requirement: 8000Hz)
                        if sr != target_sr:
                            logger.error(f"SR Mismatch: {audio_path} is {sr}Hz, must be {target_sr}Hz")
                            errors += 1
                        
                        # 2. Channel Check (Requirement: Mono)
                        y, _ = librosa.load(audio_path, sr=target_sr, mono=False)
                        if len(y.shape) > 1 and y.shape[0] > 1:
                            logger.error(f"Channel Error: {audio_path} is Stereo. Waveform-domain separation requires Mono.")
                            errors += 1
                        
                        samples = y.shape[-1]
                        
                        # 3. Kernel Size Floor (Requirement: >= 1024)
                        if samples < 1024:
                            logger.error(f"Audio Too Short: {audio_path} ({samples} samples). Min 1024 for STFT kernels.")
                            errors += 1
                        
                        # 4. Stride Alignment (Requirement: % 4 == 0 for SepReformer downsampling)
                        if samples % 4 != 0:
                            logger.warning(f"Stride Warning: {song_id} length {samples} is not a multiple of 4.")

                    except Exception as e:
                        logger.error(f"Corruption: Could not read {audio_path}: {e}")
                        errors += 1
                
                # Register the stem to the song_id to check parallelism later
                if song_id not in song_registry:
                    song_registry[song_id] = set()
                song_registry[song_id].add(stem)

        # Check D: Parallel Septuplet Alignment
        # This prevents the "num_samples=0" DataLoader error by ensuring every song has all 7 files
        for sid, found_stems in song_registry.items():
            if len(found_stems) != len(stems):
                missing = set(stems) - found_stems
                logger.error(f"Alignment Error: Song '{sid}' in {part} manifest is missing: {missing}")
                errors += 1

    if errors == 0:
        logger.info("✅ Dataset Validation Passed: SepACap Standard Alignment Confirmed.")
        return True
    else:
        logger.error(f"❌ Dataset Validation Failed with {errors} errors. Fix manifests before training.")
        return False

if __name__ == '__main__':
    SCP_DIR = "/kaggle/working/SepACap/data/scp_ss_jacappella"
    TEST_LIST = "/kaggle/working/jaCappella/test_song_list_for_vocal_ensemble_separation.txt"
    
    validate_sepacap_standard(SCP_DIR, TEST_LIST)