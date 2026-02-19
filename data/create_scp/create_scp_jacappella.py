import os
from pathlib import Path
import logging

# Set up logging for DevOps visibility
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def generate_jacappella_manifests(root_dir: str, output_dir: str, test_list_path: str):
    root = Path(root_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Define the 6-stem standard required by SepACap 
    stems = ["alto", "bass", "lead_vocal", "soprano", "tenor", "vocal_percussion"]
    
    # 2. Security: Load the official test song list to prevent data leakage [cite: 163]
    test_songs = set()
    if os.path.exists(test_list_path):
        with open(test_list_path, 'r') as f:
            test_songs = {line.strip() for line in f if line.strip()}

    # Initialize manifest buffers
    manifests = {partition: {s: [] for s in stems + ["mixture"]} 
                 for partition in ["tr", "cv", "tt"]}

    # 3. Traversal: Walk through subset/song_title/ [cite: 71, 72]
    # Structure: jaCappella/{subset}/{title_in_en}/{voice_part}.wav
    song_dirs = [d for d in root.glob("*/*") if d.is_dir()]
    
    for s_dir in sorted(song_dirs):
        song_id = s_dir.name
        
        # Check if all 6 stems + mixture exist in this folder
        missing = [s for s in stems if not (s_dir / f"{s}.wav").exists()]
        if missing or not (s_dir / "mixture.wav").exists():
            continue

        # Logic: Determine partition based on the official test song list
        partition = "tt" if song_id in test_songs else "tr"
        
        # Validation subset logic: simple 10% split of training data
        if partition == "tr" and len(manifests["tr"]["mixture"]) % 10 == 0:
            partition = "cv"

        # Add absolute paths to buffers
        manifests[partition]["mixture"].append(f"{song_id} {(s_dir / 'mixture.wav').absolute()}")
        for s in stems:
            manifests[partition][s].append(f"{song_id} {(s_dir / f'{s}.wav').absolute()}")

    # 4. Atomic Write: Write all 21 manifest files (.scp)
    for part, data in manifests.items():
        for category, lines in data.items():
            file_name = out_path / f"{part}_{category}.scp"
            with open(file_name, 'w', encoding='utf-8') as f:
                f.write("\n".join(lines) + "\n")
            logging.info(f"Wrote {len(lines)} entries to {file_name.name}")

if __name__ == '__main__':
    # Platform-specific paths
    DATA_ROOT = "/kaggle/working/jaCappella"
    SCP_OUT = "/kaggle/working/data/scp_ss_8k_jacappella"
    TEST_LIST = "/kaggle/working/jaCappella/test_song_list_for_vocal_ensemble_separation.txt"

    generate_jacappella_manifests(DATA_ROOT, SCP_OUT, TEST_LIST)