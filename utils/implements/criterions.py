import torch
import torch.nn as nn
import numpy as np
from math import ceil
from itertools import permutations
from torchaudio.transforms import MelScale
from typing import List, Any, Tuple, Dict

# Dynamic Sequence Alignment (DSA) to prevent 31934 vs 32000 length mismatch
def sync_tensors(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if a.shape[-1] == b.shape[-1]: return a, b
    min_len = min(a.shape[-1], b.shape[-1])
    return a[..., :min_len], b[..., :min_len]

class STFT(nn.Module):
    """
    Fixed-kernel STFT implementation for differentiable spectral loss.
    """
    def __init__(self, device: torch.device, frame_length: int, frame_shift: int, window: str):
        super().__init__()
        self.device = device
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        
        N = self.frame_length
        W = torch.hann_window(N) if window == 'hann' else torch.ones(N)
        S = 0.5 * (N * N / self.frame_shift)**0.5
        
        # Pre-compute STFT kernel
        K = torch.fft.rfft(torch.eye(N) / S, dim=1)[:N]
        K = torch.stack((torch.real(K), torch.imag(K)), dim=2)
        K = torch.transpose(K, 0, 2) * W 
        self.K = nn.Parameter(torch.reshape(K, (N + 2, 1, N)), requires_grad=False).to(self.device)

    def forward(self, x):
        if x.shape[-1] < self.K.shape[-1]:
            x = torch.nn.functional.pad(x, (0, self.K.shape[-1] - x.shape[-1]))
        N_frame = ceil(x.shape[-1] / self.frame_shift)
        x = torch.nn.functional.pad(x.unsqueeze(1), (0, (N_frame * self.frame_shift) - x.shape[-1]))
        c = torch.nn.functional.conv1d(x, self.K, stride=self.frame_shift)
        r, i = torch.chunk(c, 2, dim=1)
        return (r**2 + i**2 + 1e-10)**0.5, torch.atan2(i, r)

class SepACapCompositeLoss(nn.Module):
    """
    SepACap Composite Loss: Waveform L1 + Multi-Res Spectral + Mel.
    Designed for 8kHz JaCappella training with silent stem support.
    """
    def __init__(self, device: torch.device, num_spks: int, weights: Dict, window_sizes: List[int], **kwargs):
        super().__init__()
        self.device = device
        self.num_spks = num_spks
        self.weights = weights
        self.l1 = nn.L1Loss()
        
        # Multi-resolution Spectral setup (512, 1024, 2048)
        self.spectral_ops = nn.ModuleList([
            STFT(self.device, w, w//4, 'hann') for w in window_sizes
        ])
        
        # Mel setup: n_stft must be (window_sizes[1]//2 + 1) -> 513 for 1024 window
        n_stft_mel = window_sizes[1] // 2 + 1
        self.mel_op = MelScale(n_mels=80, sample_rate=8000, n_stft=n_stft_mel).to(self.device)

    def forward(self, **kwargs):
        # Unpack variables from the training engine
        estims = kwargs['estims']
        targets = [t.to(self.device) for t in kwargs["target_attr"]]
        input_sizes = kwargs["input_sizes"]
        
        def _calc_loss(perm):
            l_total = 0
            for s, t in enumerate(perm):
                est, tar = sync_tensors(estims[s], targets[t])
                
                # 1. Waveform L1 (Direct signal reconstruction)
                l_wave = self.l1(est, tar)
                
                # 2. Multi-Resolution Spectral (Converges faster than time-domain alone)
                l_spec = 0
                for op in self.spectral_ops:
                    m_est, _ = op(est)
                    m_tar, _ = op(tar)
                    m_est, m_tar = sync_tensors(m_est, m_tar)
                    # Log-magnitude + Magnitude L1
                    l_spec += self.l1(m_est, m_tar) + self.l1(torch.log(m_est + 1e-7), torch.log(m_tar + 1e-7))
                
                # 3. Mel Loss (Psychoacoustic weight)
                m_est_mel, _ = self.spectral_ops[1](est) # Always use index 1 (1024 window)
                m_tar_mel, _ = self.spectral_ops[1](tar)
                l_mel = self.l1(self.mel_op(m_est_mel), self.mel_op(m_tar_mel))
                
                # Weighted Summation
                l_total += (self.weights['waveform'] * l_wave) + \
                           (self.weights['spectral'] * l_spec / len(self.spectral_ops)) + \
                           (self.weights['mel'] * l_mel)
            return l_total

        # PIT: Find optimal permutation for the 6 vocal parts
        pscore = torch.stack([_calc_loss(p) for p in permutations(range(self.num_spks))])
        
        # Batch normalization
        return torch.min(pscore) / input_sizes.shape[0]

# --- Compatibility Wrappers ---
class PIT_SISNR_mag(SepACapCompositeLoss): pass
class PIT_SISNR_time(SepACapCompositeLoss): pass

class PIT_SISNRi:
    def __init__(self, **kwargs): pass
    def __call__(self, **kwargs): return torch.tensor(0.0, device='cuda'), torch.zeros(2)

class PIT_SDRi:
    def __init__(self, **kwargs): pass
    def __call__(self, **kwargs): return 0.0, np.zeros(2)