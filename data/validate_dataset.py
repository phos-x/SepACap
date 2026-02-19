import os
import torch
import librosa
import logging
from pathlib import Path
from tqdm import tqdm

# DevOps Visibility Configuration
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("DatasetValidator")

def validate_sepacap_standard(scp_dir: str, test_list_path: str, target_sr=8000):
    """
    Comprehensive validation of jaCappella manifests and audio integrity.
    """
    scp_path = Path(scp_dir)
    stems = ["alto", "bass", "lead_vocal", "soprano", "tenor", "vocal_percussion", "mixture"]
    partitions = ["tr", "cv", "tt"]
    
    # 1. Load Test Song List for Leakage Check
    test_songs = set()
    if os.path.exists(test_list_path):
        with open(test_list_path, 'r') as f:
            test_songs = {line.strip() for line in f if line.strip()}
    
    errors = 0
    manifest_data = {}

    # 2. Manifest and Signal Integrity Loop
    for part in partitions:
        logger.info(f"--- Validating Partition: {part} ---")
        song_registry = {} # To ensure parallel alignment
        
        for stem in stems:
            scp_file = scp_path / f"{part}_{stem}.scp"
            if not scp_file.exists():
                logger.error(f"Missing manifest: {scp_file}")
                errors += 1
                continue
            
            with open(scp_file, 'r') as f:
                lines = f.readlines()
                
            for line in tqdm(lines, desc=f"Checking {stem}", leave=False):
                parts = line.strip().split(maxsplit=1)
                if len(parts) < 2: continue
                
                song_id, audio_path = parts
                
                # Check A: Leakage
                if part != "tt" and song_id in test_songs:
                    logger.error(f"Leakage Error: {song_id} found in {part} manifest but belongs to test set.")
                    errors += 1
                
                # Check B: File Reachability
                if not os.path.exists(audio_path):
                    logger.error(f"File Not Found: {audio_path}")
                    errors += 1
                    continue
                
                # Check C: Signal Standards (Sampling Rate & Length)
                # Performance optimization: only check sr once per song_id
                if song_id not in song_registry:
                    try:
                        # Load just header to be fast
                        duration = librosa.get_duration(path=audio_path)
                        sr = librosa.get_samplerate(audio_path)
                        samples = duration * sr
                        
                        if sr != target_sr:
                            logger.error(f"SR Mismatch: {audio_path} is {sr}Hz, needs {target_sr}Hz")
                            errors += 1
                        
                        if samples < 1024:
                            logger.error(f"Size Error: {audio_path} is too short ({samples} samples)")
                            errors += 1
                            
                    except Exception as e:
                        logger.error(f"Corruption Error: Could not read {audio_path}: {e}")
                        errors += 1
                
                # Check D: Index-Matching (Triplet/Septuplet alignment)
                if song_id not in song_registry:
                    song_registry[song_id] = set()
                song_registry[song_id].add(stem)

        # Check E: Manifest Parallelism
        # Every song in this partition must have exactly the same stems
        first_song_stems = None
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
    SCP_DIR = "/kaggle/working/data/scp_ss_8k_jacappella"
    TEST_LIST = "/kaggle/working/jaCappella/test_song_list_for_vocal_ensemble_separation.txt"
    
    validate_sepacap_standard(SCP_DIR, TEST_LIST)