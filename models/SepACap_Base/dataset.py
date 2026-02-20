import os
import torch
import random
import numpy as np
import librosa as audio_lib
import itertools
from pathlib import Path
from loguru import logger
from torch.utils.data import Dataset, DataLoader
from utils import util_dataset
from utils.decorators import logger_wraps

@logger_wraps()
def get_dataloaders(args, dataset_config, loader_config):    
    partitions = ["test"] if "test" in args.engine_mode else ["train", "valid", "test"]
    dataloaders = {}
    scp_dir = Path(dataset_config["scp_dir"])

    for partition in partitions:
        mix_scp = scp_dir / dataset_config[partition]['mixture']
        # Dynamically load all spk manifests as defined in configs.yaml
        src_scps = [scp_dir / dataset_config[partition][k] for k in dataset_config[partition] if k.startswith('spk')]
        
        is_train = (partition == 'train')
        dataset = SepACapDataset(
            max_len=dataset_config['max_len'],
            fs=dataset_config['sampling_rate'],
            partition=partition,
            wave_scp_srcs=src_scps,
            wave_scp_mix=mix_scp,
            # Enables the joint separation and detection strategy 
            dynamic_mixing=dataset_config[partition].get("dynamic_mixing", True) if is_train else False
        )

        dataloaders[partition] = DataLoader(
            dataset=dataset,
            batch_size=1 if partition == 'test' else loader_config["batch_size"],
            shuffle=is_train, 
            num_workers=loader_config.get("num_workers", 12),
            collate_fn=_collate # Crucial for sequence padding
        )
    return dataloaders

def _collate(batch):
    """Ensures variable-length audio in a batch is padded correctly."""
    batch = sorted(batch, key=lambda x: x['num_sample'], reverse=True)
    input_sizes = torch.tensor([d['num_sample'] for d in batch], dtype=torch.long)
    mixture = torch.nn.utils.rnn.pad_sequence([torch.from_numpy(d['mix']) for d in batch], batch_first=True)
    keys = [d['key'] for d in batch]
    
    num_spks = len(batch[0]['src'])
    srcs = []
    for i in range(num_spks):
        srcs.append(torch.nn.utils.rnn.pad_sequence([torch.from_numpy(d['src'][i]) for d in batch], batch_first=True))
        
    return input_sizes, mixture, srcs, keys

class SepACapDataset(Dataset):
    def __init__(self, max_len, fs, partition, wave_scp_srcs, wave_scp_mix, dynamic_mixing=False):
        self.max_len, self.fs = max_len, fs
        self.partition, self.dynamic_mixing = partition, dynamic_mixing

        # Load and sort manifests to ensure parallel septuplet alignment
        raw_mix_dict = util_dataset.parse_scps(str(wave_scp_mix))
        self.wave_keys = sorted(raw_mix_dict.keys())
        self.wave_list_mix = [raw_mix_dict[k] for k in self.wave_keys]

        self.wave_list_srcs = []
        for scp_src in wave_scp_srcs:
            raw_src_dict = util_dataset.parse_scps(str(scp_src))
            self.wave_list_srcs.append(raw_src_dict)

        if len(self.wave_keys) == 0:
            raise RuntimeError(f"No samples found for {partition}. Check manifest paths.")

    def _get_power_set_indices(self, n):
        """Generates all possible combinations of active singers."""
        indices = list(range(n))
        subsets = []
        for r in range(1, n + 1):
            subsets.extend(list(itertools.combinations(indices, r)))
        return subsets

    def _process_audio(self, samps_mix, samps_src_list):
        """Standardizes audio length and applies SepACap data augmentation[cite: 21, 48]."""
        # Floor logic: Prevents STFT window overlap errors (min 1024 samples)
        if len(samps_mix) < 1024:
            pad = 1024 - len(samps_mix)
            samps_mix = np.pad(samps_mix, (0, pad))
            samps_src_list = [np.pad(s, (0, pad)) for s in samps_src_list]

        # Power Set Augmentation: Handles subsets of stems for robust detection 
        if self.dynamic_mixing and self.partition == "train":
            subsets = self._get_power_set_indices(len(samps_src_list))
            chosen = random.choice(subsets)
            samps_mix = np.zeros_like(samps_src_list[0])
            new_srcs = [np.zeros_like(s) for s in samps_src_list]
            for i in chosen:
                samps_mix += samps_src_list[i]
                new_srcs[i] = samps_src_list[i]
            samps_src_list = new_srcs

        # Stride alignment: Multiple of 4 for encoder/decoder downsampling
        length = (len(samps_mix) // 4) * 4
        samps_mix = samps_mix[:length]
        samps_src_list = [s[:length] for s in samps_src_list]

        # Snippet segmentation: 4-second fixed snippets [cite: 69, 75]
        if self.partition != "test" and length > self.max_len:
            start = random.randint(0, length - self.max_len)
            samps_mix = samps_mix[start:start + self.max_len]
            samps_src_list = [s[start:start + self.max_len] for s in samps_src_list]
            
        return samps_mix, samps_src_list

    def __getitem__(self, index):
        key = self.wave_keys[index]
        s_mix, _ = audio_lib.load(self.wave_list_mix[index], sr=self.fs)
        # Fetch each stem for the current key from parallel manifests
        s_srcs = [audio_lib.load(src_dict[key], sr=self.fs)[0] for src_dict in self.wave_list_srcs]
        
        p_mix, p_srcs = self._process_audio(s_mix, s_srcs)
        
        return {
            "num_sample": len(p_mix), 
            "mix": p_mix.astype(np.float32), 
            "src": [s.astype(np.float32) for s in p_srcs], 
            "key": key
        }

    def __len__(self):
        return len(self.wave_list_mix)