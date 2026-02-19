import os
import librosa
import soundfile as sf
from pathlib import Path
from joblib import Parallel, delayed
from tqdm import tqdm

def resample_file(file_path, target_sr=8000):
    """Resamples a single file and overwrites it to save space."""
    try:
        # Load audio (resampling during load is memory-efficient)
        y, _ = librosa.load(file_path, sr=target_sr)
        # Overwrite the 48kHz file with the 8kHz version
        sf.write(file_path, y, target_sr)
        return True
    except Exception as e:
        print(f"Error resampling {file_path}: {e}")
        return False

def batch_resample_dataset(root_dir, target_sr=8000):
    root = Path(root_dir)
    # Find all .wav files in the nested structure
    all_wavs = list(root.rglob("*.wav"))
    
    print(f"Starting batch resampling of {len(all_wavs)} files to {target_sr}Hz...")
    
    # Parallel execution to save time on Kaggle's multi-core CPU
    results = Parallel(n_jobs=-1)(
        delayed(resample_file)(str(p), target_sr) for p in tqdm(all_wavs)
    )
    
    success_count = sum(results)
    print(f"Done! Successfully resampled {success_count}/{len(all_wavs)} files.")

if __name__ == '__main__':
    DATASET_ROOT = "/kaggle/working/jaCappella"
    batch_resample_dataset(DATASET_ROOT)