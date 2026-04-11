import os
import random
import itertools
import math
import torch
import torchaudio
import torchaudio.functional as F
from pathlib import Path
from loguru import logger
from torch.utils.data import Dataset, DataLoader
from utils import util_dataset
from utils.decorators import logger_wraps

@logger_wraps()
def get_dataloaders(args, dataset_config, loader_config):    
    partitions = ["test"] if "test" in args.engine_mode else ["train", "valid", "test"]
    dataloaders = {}
    scp_dir = Path(dataset_config["scp_dir"]).resolve() 

    for partition in partitions:
        mix_scp = scp_dir / dataset_config[partition]['mixture']
        src_scps = [scp_dir / dataset_config[partition][k] for k in dataset_config[partition] if k.startswith('spk')]
        
        is_train = (partition == 'train')
        dataset = SepACapDataset(
            max_len=dataset_config['max_len'],
            fs=dataset_config['sampling_rate'],
            partition=partition,
            wave_scp_srcs=src_scps,
            wave_scp_mix=mix_scp,
            dynamic_mixing=dataset_config[partition].get("dynamic_mixing", True) if is_train else False
        )

        dataloaders[partition] = DataLoader(
            dataset=dataset,
            batch_size=1 if partition == 'test' else loader_config["batch_size"],
            shuffle=is_train, 
            num_workers=loader_config.get("num_workers", 8),
            collate_fn=_collate,
            pin_memory=True, #Accelerates CPU-to-GPU transfer
            drop_last=is_train
        )
    return dataloaders

def _collate(batch):
    """Ensures variable-length audio in a batch is padded correctly."""
    batch.sort(key=lambda x: x['num_sample'], reverse=True)
    
    input_sizes = torch.tensor([d['num_sample'] for d in batch], dtype=torch.long)
    mixture = torch.nn.utils.rnn.pad_sequence([d['mix'] for d in batch], batch_first=True)
    keys = [d['key'] for d in batch]
    
    num_spks = len(batch[0]['src'])
    srcs = [
        torch.nn.utils.rnn.pad_sequence([d['src'][i] for d in batch], batch_first=True)
        for i in range(num_spks)
    ]
        
    return input_sizes, mixture, srcs, keys

class SepACapDataset(Dataset):
    def __init__(self, max_len, fs, partition, wave_scp_srcs, wave_scp_mix, dynamic_mixing=False):
        self.max_len = max_len
        self.fs = fs
        self.partition = partition
        self.dynamic_mixing = dynamic_mixing

        self.raw_mix_dict = util_dataset.parse_scps(str(wave_scp_mix))
        self.wave_keys = sorted(list(self.raw_mix_dict.keys()))
        
        self.raw_src_dicts = [util_dataset.parse_scps(str(scp_src)) for scp_src in wave_scp_srcs]

        if not self.wave_keys:
            raise RuntimeError(f"No samples found for {partition}. Check manifest paths.")
            
        self.num_stems = len(self.raw_src_dicts)
        self.power_set_subsets = self._get_power_set_indices(self.num_stems)

    def _get_power_set_indices(self, n):
        """Generates all possible combinations of active singers."""
        indices = list(range(n))
        subsets = []
        for r in range(1, n + 1):
            subsets.extend(list(itertools.combinations(indices, r)))
        return subsets

    def _apply_phase_1_augmentations(self, samps_src_list):
        """Applies synchronous Phase 1 Supervised Augmentations."""
        # 1. Gain Scaling: -6 dB to +6 dB 
        gain_db = random.uniform(-6.0, 6.0)
        gain_linear = 10 ** (gain_db / 20.0)
        samps_src_list = [src * gain_linear for src in samps_src_list]

        # 2. Pitch Shift: ±1 to 2 semitones
        if random.random() < 0.5:
            # Bound strictly to prevent destroying SATB formants
            pitch_shift_steps = random.choice([-2, -1, 1, 2])
            samps_src_list = [
                F.pitch_shift(src, self.fs, n_steps=pitch_shift_steps) 
                for src in samps_src_list
            ]

        # 3. Time Stretch: 0.95x to 1.05x
        if random.random() < 0.5:
            stretch_factor = random.uniform(0.95, 1.05)
            # torchaudio phase vocoder is much faster than librosa
            samps_src_list = [
                self._fast_time_stretch(src, stretch_factor) 
                for src in samps_src_list
            ]
            
        return samps_src_list

    def _fast_time_stretch(self, waveform, rate):
        """Helper for torchaudio time stretching."""
        # Note: In production, pre-compute the complex spectrogram for speed.
        # This acts as a functional placeholder for torchaudio's PhaseVocoder.
        n_fft = 1024
        hop_length = 256
        spec = torch.stft(waveform, n_fft=n_fft, hop_length=hop_length, return_complex=True)
        stretch_spec = F.phase_vocoder(spec, rate=rate, phase_advance=torch.tensor([math.pi/4]))
        return torch.istft(stretch_spec, n_fft=n_fft, hop_length=hop_length)

    def _load_audio_snippet(self, file_path):
        """Loads a fixed-size random snippet directly from disk to save memory/I/O."""
        info = torchaudio.info(file_path)
        total_frames = info.num_frames
        
        if self.partition != "test" and total_frames > self.max_len:
            frame_offset = random.randint(0, total_frames - self.max_len)
            num_frames = self.max_len
        else:
            frame_offset = 0
            num_frames = total_frames

        waveform, _ = torchaudio.load(file_path, frame_offset=frame_offset, num_frames=num_frames)
        return waveform[0] # Convert to 1D tensor

    def __getitem__(self, index):
        key = self.wave_keys[index]
        samps_src_list = []

        # ---------------------------------------------------------
        # PHASE 2: Cross-Track Remixing (Dissonance Engine)
        # ---------------------------------------------------------
        is_cross_track = self.dynamic_mixing and self.partition == "train" and random.random() < 0.3

        if is_cross_track:
            for src_dict in self.raw_src_dicts:
                # Randomly sample this stem from a completely different track
                random_key = random.choice(self.wave_keys)
                stem = self._load_audio_snippet(src_dict[random_key])
                
                # Apply independent random gain per stem prior to mixing
                ind_gain = 10 ** (random.uniform(-3.0, 3.0) / 20.0)
                samps_src_list.append(stem * ind_gain)
                
            # Align lengths of cross-track stems (they may differ slightly due to snippet logic)
            min_len = min(len(s) for s in samps_src_list)
            samps_src_list = [s[:min_len] for s in samps_src_list]
            
        else:
            # Standard synchronous load
            samps_src_list = [self._load_audio_snippet(src_dict[key]) for src_dict in self.raw_src_dicts]

        # ---------------------------------------------------------
        # PHASE 1 & Power-Set Augmentation
        # ---------------------------------------------------------
        if self.dynamic_mixing and self.partition == "train":
            # Apply synchronous phase-1 tweaks to all stems exactly the same way
            samps_src_list = self._apply_phase_1_augmentations(samps_src_list)
            
            # Power Set Logic: Dropout entire stems to teach the model silence
            chosen_subset = random.choice(self.power_set_subsets)
            new_srcs = [torch.zeros_like(samps_src_list[0]) for _ in range(self.num_stems)]
            for i in chosen_subset:
                new_srcs[i] = samps_src_list[i]
            samps_src_list = new_srcs

        # Mathematically reconstruct the mixture (x_synth)
        s_mix = torch.stack(samps_src_list, dim=0).sum(dim=0)

        # Ensure lengths are a multiple of 4 for encoder/decoder striding
        length = (len(s_mix) // 4) * 4
        s_mix = s_mix[:length]
        samps_src_list = [s[:length] for s in samps_src_list]

        return {
            "num_sample": len(s_mix), 
            "mix": s_mix, 
            "src": samps_src_list, 
            "key": key if not is_cross_track else f"synth_{key}"
        }

    def __len__(self):
        return len(self.wave_keys)