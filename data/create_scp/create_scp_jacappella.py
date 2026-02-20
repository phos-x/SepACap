import os
import logging
from pathlib import Path
from typing import Set, Dict, List

# Set up logging for DevOps visibility and error tracking
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("SCP_Generator")

def generate_jacappella_manifests(root_dir: str, output_dir: str, test_list_path: str):
    """
    Parses the jaCappella directory structure and generates parallel .scp manifests.
    Ensures that only complete septuplets (mixture + 6 stems) are indexed. 
    """
    root = Path(root_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Define the 6-stem standard required for a cappella separation 
    # These match the vocal parts identified in the JaCappella corpus [cite: 164]
    stems = ["alto", "bass", "lead_vocal", "soprano", "tenor", "vocal_percussion"]
    
    # 2. Security Check: Load official test song list to ensure partition integrity 
    test_songs: Set[str] = set()
    if os.path.exists(test_list_path):
        with open(test_list_path, 'r', encoding='utf-8') as f:
            test_songs = {line.strip() for line in f if line.strip()}
    else:
        logger.warning(f"Test list not found at {test_list_path}. All songs will go to Train/Valid.")

    # 3. Initialize Manifest Buffers
    # We create 21 manifests: (mixture + 6 stems) * (tr, cv, tt) [cite: 44]
    manifests: Dict[str, Dict[str, List[str]]] = {
        partition: {s: [] for s in stems + ["mixture"]} 
        for partition in ["tr", "cv", "tt"]
    }

    # 4. Traversal Logic: Navigate jaCappella/{subset}/{title_in_en}/ [cite: 71, 72]
    song_dirs = [d for d in root.glob("*/*") if d.is_dir()]
    
    for s_dir in sorted(song_dirs):
        song_id = s_dir.name
        
        # DSA Best Practice: Strict Validation of Septuplet Completeness
        # We only index songs that have every required voice part 
        missing = [s for s in stems if not (s_dir / f"{s}.wav").exists()]
        if missing:
            logger.debug(f"Skipping {song_id}: Missing stems {missing}")
            continue
        if not (s_dir / "mixture.wav").exists():
            logger.debug(f"Skipping {song_id}: Missing mixture.wav")
            continue

        # Partitioning Logic based on the JaCappella experimental setup [cite: 71, 163]
        if song_id in test_songs:
            partition = "tt"
        else:
            # Deterministic 10% cross-validation split for the 'cv' partition
            partition = "cv" if len(manifests["tr"]["mixture"]) % 10 == 0 else "tr"

        # 5. Buffer Assembly: Using absolute paths for platform portability
        # mixture key-path pair (The Input) 
        manifest_line_mix = f"{song_id} {(s_dir / 'mixture.wav').absolute()}"
        manifests[partition]["mixture"].append(manifest_line_mix)
        
        # Stem key-path pairs (The 6 Targets) 
        for s in stems:
            manifest_line_stem = f"{song_id} {(s_dir / f'{s}.wav').absolute()}"
            manifests[partition][s].append(manifest_line_stem)

    # 6. Atomic Write: Writing all 21 manifest files (.scp)
    for part, data in manifests.items():
        for category, lines in data.items():
            file_name = out_path / f"{part}_{category}.scp"
            try:
                with open(file_name, 'w', encoding='utf-8') as f:
                    # Defensive formatting: ensures no trailing empty lines or parsing errors
                    f.write("\n".join(lines) + "\n")
                logger.info(f"Generated {file_name.name} with {len(lines)} entries.")
            except IOError as e:
                logger.error(f"Failed to write manifest {file_name}: {e}")

if __name__ == '__main__':
    # Kaggle-specific paths based on your current workspace
    DATA_ROOT = "/kaggle/working/jaCappella"
    SCP_OUT = "/kaggle/working/SepACap/data/scp_ss_jacappella"
    # Official test list song titles provided by the researchers 
    TEST_LIST = "/kaggle/working/jaCappella/test_song_list_for_vocal_ensemble_separation.txt"

    generate_jacappella_manifests(DATA_ROOT, SCP_OUT, TEST_LIST)