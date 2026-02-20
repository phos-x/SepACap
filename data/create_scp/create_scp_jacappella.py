import os
import logging
from pathlib import Path
from typing import Set, Dict, List

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

    stems = ["alto", "bass", "lead_vocal", "soprano", "tenor", "vocal_percussion"]
    
    test_songs: Set[str] = set()
    if os.path.exists(test_list_path):
        with open(test_list_path, 'r', encoding='utf-8') as f:
            test_songs = {line.strip() for line in f if line.strip()}
    else:
        logger.warning(f"Test list not found at {test_list_path}. All songs will go to Train/Valid.")

    manifests: Dict[str, Dict[str, List[str]]] = {
        partition: {s: [] for s in stems + ["mixture"]} 
        for partition in ["tr", "cv", "tt"]
    }

    song_dirs = sorted([d for d in root.glob("*/*") if d.is_dir()])
    
    train_val_idx = 0 
    
    for s_dir in song_dirs:
        song_id = s_dir.name
        
        missing = [s for s in stems if not (s_dir / f"{s}.wav").exists()]
        if missing or not (s_dir / "mixture.wav").exists():
            logger.debug(f"Skipping {song_id}: Incomplete septuplet.")
            continue

        if song_id in test_songs:
            partition = "tt"
        else:
            partition = "cv" if train_val_idx % 10 == 0 else "tr"
            train_val_idx += 1

        manifests[partition]["mixture"].append(f"{song_id} {(s_dir / 'mixture.wav').absolute()}")
        
        for s in stems:
            manifests[partition][s].append(f"{song_id} {(s_dir / f'{s}.wav').absolute()}")

    for part, data in manifests.items():
        for category, lines in data.items():
            file_name = out_path / f"{part}_{category}.scp"
            try:
                with open(file_name, 'w', encoding='utf-8') as f:
                    f.write("\n".join(lines) + "\n")
                logger.info(f"Generated {file_name.name} with {len(lines)} entries.")
            except IOError as e:
                logger.error(f"Failed to write manifest {file_name}: {e}")

if __name__ == '__main__':
    DATA_ROOT = "/kaggle/working/jaCappella"
    SCP_OUT = "/kaggle/working/SepACap/data/scp_ss_jacappella"
    TEST_LIST = "/kaggle/working/jaCappella/test_song_list_for_vocal_ensemble_separation.txt"

    generate_jacappella_manifests(DATA_ROOT, SCP_OUT, TEST_LIST)