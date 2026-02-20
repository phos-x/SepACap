import os
import torch
import random
import numpy as np
import librosa as audio_lib
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
        src_scps = [scp_dir / dataset_config[partition][k] for k in dataset_config[partition] if k.startswith('spk')]
        
        is_train = (partition == 'train')
        dataset = MyDataset(
            max_len=dataset_config['max_len'],
            fs=dataset_config['sampling_rate'],
            partition=partition,
            wave_scp_srcs=src_scps,
            wave_scp_mix=mix_scp,
            dynamic_mixing=dataset_config[partition].get("dynamic_mixing", False) if is_train else False
        )

        dataloaders[partition] = DataLoader(
            dataset=dataset,
            batch_size=1 if partition == 'test' else loader_config["batch_size"],
            shuffle=is_train, 
            pin_memory=loader_config.get("pin_memory", False),
            num_workers=loader_config.get("num_workers", 4),
            drop_last=loader_config.get("drop_last", False),
            collate_fn=_collate
        )
    return dataloaders

def _collate(batch):
    batch = sorted(batch, key=lambda x: x['num_sample'], reverse=True)
    keys = [d['key'] for d in batch]
    input_sizes = torch.tensor([d['num_sample'] for d in batch], dtype=torch.long)
    mixture = torch.nn.utils.rnn.pad_sequence([torch.from_numpy(d['mix']) for d in batch], batch_first=True)
    
    num_spks = len(batch[0]['src'])
    srcs = []
    for i in range(num_spks):
        srcs.append(torch.nn.utils.rnn.pad_sequence([torch.from_numpy(d['src'][i]) for d in batch], batch_first=True))
        
    return input_sizes, mixture, srcs, keys

class MyDataset(Dataset):
    def __init__(self, max_len, fs, partition, wave_scp_srcs, wave_scp_mix, dynamic_mixing=False):
        self.max_len = max_len
        self.fs = fs
        self.partition = partition
        self.dynamic_mixing = dynamic_mixing

        raw_mix_dict = util_dataset.parse_scps(str(wave_scp_mix))
        self.wave_list_mix = [raw_mix_dict[k] for k in sorted(raw_mix_dict.keys())]
        self.wave_keys = sorted(raw_mix_dict.keys())

        self.wave_list_srcs = []
        for scp_src in wave_scp_srcs:
            raw_src_dict = util_dataset.parse_scps(str(scp_src))
            self.wave_list_srcs.append([raw_src_dict[k] for k in sorted(raw_src_dict.keys())])

        logger.info(f"Initialized {partition} set with {len(self.wave_list_mix)}Aligned index-matching samples.")

    def _process_audio(self, samps_mix, samps_src):
        # 1. Platform Engineering Floor: Audio must be longer than the STFT kernel (512)
        # We set it to 1024 to be safe.
        min_required = 1024
        if len(samps_mix) < min_required:
            pad_len = min_required - len(samps_mix)
            samps_mix = np.pad(samps_mix, (0, pad_len), mode='constant')
            samps_src = [np.pad(s, (0, pad_len), mode='constant') for s in samps_src]

        # 2. Stride Alignment: Length must be divisible by 4
        length = (len(samps_mix) // 4) * 4
        samps_mix = samps_mix[:length]
        samps_src = [s[:length] for s in samps_src]

        # 3. Training Crop
        if self.partition != "test" and length > self.max_len:
            start = random.randint(0, length - self.max_len)
            samps_mix = samps_mix[start:start + self.max_len]
            samps_src = [s[start:start + self.max_len] for s in samps_src]
            
        return samps_mix, samps_src

    def _direct_load(self, index):
        samps_mix, _ = audio_lib.load(self.wave_list_mix[index], sr=self.fs)
        samps_src = [audio_lib.load(src_list[index], sr=self.fs)[0] for src_list in self.wave_list_srcs]
        return self._process_audio(samps_mix, samps_src)

    def __len__(self):
        return len(self.wave_list_mix)

    def __getitem__(self, index):
        samps_mix, samps_src = self._direct_load(index)
        return {
            "num_sample": len(samps_mix),
            "mix": samps_mix.astype(np.float32),
            "src": [s.astype(np.float32) for s in samps_src],
            "key": self.wave_keys[index]
        }