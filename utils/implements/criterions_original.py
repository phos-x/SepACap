import torch
import numpy as np
from math import ceil
from itertools import permutations
from torchaudio.transforms import MelScale
from dataclasses import dataclass, field
from typing import List, Any, Tuple
from loguru import logger
from mir_eval.separation import bss_eval_sources

# --- DSA: Defensive Shape Alignment ---
def sync_tensors(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if a.shape[-1] == b.shape[-1]:
        return a, b
    min_len = min(a.shape[-1], b.shape[-1])
    return a[..., :min_len], b[..., :min_len]

def l2norm(mat, keepdim=False):
    return torch.norm(mat, dim=-1, keepdim=keepdim)

@dataclass(slots=True)
class STFTBase(torch.nn.Module):
    device: torch.device
    frame_length: int
    frame_shift: int
    window: str
    K: torch.nn.Parameter = field(init=False)
    num_bins: int = field(init=False)

    def __post_init__(self):
        super(STFTBase, self).__init__()
        K = self._init_kernel(self.frame_length, self.frame_shift)
        self.K = torch.nn.Parameter(K, requires_grad=False).to(self.device)
        self.num_bins = self.K.shape[0] // 2
    
    def _init_kernel(self, frame_len, frame_hop):
        N = frame_len
        W = torch.hann_window(frame_len) if self.window == 'hann' else torch.ones(frame_len)
        if N//4 == frame_hop: W = (2/3)**0.5 * W
        elif N//2 == frame_hop: W = W**0.5
        S = 0.5 * (N * N / frame_hop)**0.5
        K = torch.fft.rfft(torch.eye(N) / S, dim=1)[:frame_len]
        K = torch.stack((torch.real(K), torch.imag(K)), dim=2)
        K = torch.transpose(K, 0, 2) * W 
        return torch.reshape(K, (N + 2, 1, frame_len))

@dataclass(slots=True)
class STFT(STFTBase):
    def forward(self, x, cplx=False):
        # Platform Safety Check: Kernel size vs Input size
        if x.shape[-1] < self.K.shape[-1]:
            x = torch.nn.functional.pad(x, (0, self.K.shape[-1] - x.shape[-1]))

        N_frame = ceil(x.shape[-1] / self.frame_shift)
        len_padded = N_frame * self.frame_shift
        
        if x.dim() == 2:
            pad = torch.zeros(x.shape[0], len_padded - x.shape[-1], device=x.device)
            x = torch.cat((x, pad), dim=-1).unsqueeze(1)
            c = torch.nn.functional.conv1d(x, self.K, stride=self.frame_shift)
            r, i = torch.chunk(c, 2, dim=1)
        else:        
            pad = torch.zeros(x.shape[0], x.shape[1], len_padded - x.shape[-1], device=x.device)
            x = torch.cat((x, pad), dim=-1)
            N, C, S = x.shape
            x = x.reshape(N * C, 1, S)
            c = torch.nn.functional.conv1d(x, self.K, stride=self.frame_shift)
            c = c.reshape(N, C, -1, c.shape[-1])
            r, i = torch.chunk(c, 2, dim=2)

        if cplx: return r, i
        return (r**2 + i**2 + 1e-10)**0.5, torch.atan2(i, r)

@dataclass(slots=True)
class PIT_SISNR_mag:
    device: torch.device
    frame_length: int
    frame_shift: int
    window: str
    num_stages: int
    num_spks: int
    scale_inv: bool
    mel_opt: bool
    stft: List[Any] = field(init=False)
    mel_fb: Any = field(init=False)
    
    def __post_init__(self):
        self.stft = [STFT(self.device, self.frame_length, self.frame_shift, self.window) for _ in range(self.num_stages)]
        self.mel_fb = MelScale(n_mels=80, sample_rate=16000, n_stft=self.frame_length//2+1).to(self.device) if self.mel_opt else lambda x: x

    def __call__(self, **kwargs):
        estims, idx, sizes = kwargs['estims'], kwargs['idx'], kwargs["input_sizes"].to(self.device)
        targets = [t.to(self.device) for t in kwargs["target_attr"]]
        
        def _loss(perm, eps=1e-12):
            l_perm = []
            for s, t in enumerate(perm):
                m, src = sync_tensors(estims[s], targets[t])
                m_zm, s_zm = m - m.mean(-1, True), src - src.mean(-1, True)
                if self.scale_inv:
                    scale = torch.sum(m_zm * s_zm, -1, True) / (l2norm(s_zm, True)**2 + eps)
                    s_zm = torch.clamp(scale, min=1e-2) * s_zm
                m_mag = self.stft[idx](m_zm.to(self.device))[0]
                s_mag = self.stft[idx](s_zm.to(self.device))[0]
                m_mag, s_mag = sync_tensors(m_mag, s_mag)
                if self.mel_opt: m_mag, s_mag = self.mel_fb(m_mag), self.mel_fb(s_mag)
                l_perm.append(-20 * torch.log10(eps + l2norm(l2norm(s_mag)) / (l2norm(l2norm(m_mag - s_mag)) + eps)))
            return sum(l_perm)
        
        pscore = torch.stack([_loss(p) for p in permutations(range(self.num_spks))])
        return torch.min(pscore, dim=0)[0].sum() / sizes.shape[0]

@dataclass(slots=True)
class PIT_SISNR_time:
    device: torch.device
    num_spks: int
    scale_inv: bool

    def __call__(self, **kwargs):
        estims, sizes = kwargs['estims'], kwargs["input_sizes"].to(self.device)
        targets = [target.to(self.device) for target in kwargs["target_attr"]]
        def _loss(perm, eps=1e-8):
            l_perm = []
            for s, t in enumerate(perm):
                m, src = sync_tensors(estims[s], targets[t])
                m_zm, s_zm = m - m.mean(-1, True), src - src.mean(-1, True)
                if self.scale_inv:
                    scale = torch.sum(m_zm * s_zm, -1, True) / (l2norm(s_zm, True)**2 + eps)
                    s_zm = scale * s_zm
                l_perm.append(torch.clamp(-20 * torch.log10(eps + l2norm(s_zm) / (l2norm(m_zm - s_zm) + eps)), min=-30))
            return sum(l_perm)
        return torch.min(torch.stack([_loss(p) for p in permutations(range(self.num_spks))]), dim=0)[0].sum() / sizes.shape[0]

@dataclass(slots=True)
class PIT_SISNRi:
    device: torch.device
    num_spks: int
    scale_inv: bool

    def __call__(self, **kwargs):
        estims, sizes, eps = kwargs['estims'], kwargs["input_sizes"].to(self.device), kwargs['eps']
        targets = [t.to(self.device) for t in kwargs["target_attr"]]
        raw_zm = kwargs['mixture'].to(self.device) - kwargs['mixture'].to(self.device).mean(-1, True)
        def _loss(perm):
            l_perm = []
            for s, t in enumerate(perm):
                est, src = sync_tensors(estims[s], targets[t])
                e_zm, s_zm = est - est.mean(-1, True), src - src.mean(-1, True)
                m_sync, _ = sync_tensors(raw_zm, s_zm)
                s_est = torch.sum(e_zm * s_zm, -1, True) / (l2norm(s_zm, True)**2 + eps) * s_zm
                s_mix = torch.sum(m_sync * s_zm, -1, True) / (l2norm(s_zm, True)**2 + eps) * s_zm
                l_perm.append(20 * torch.log10(eps + l2norm(s_est) / (l2norm(e_zm - s_est) + eps)) - 20 * torch.log10(eps + l2norm(s_mix) / (l2norm(m_sync - s_mix) + eps)))
            return torch.tensor(l_perm)
        score = torch.stack([_loss(p) for p in permutations(range(self.num_spks))], 0)
        max_v, max_idx = torch.max(score.sum(-1), 0)
        return max_v.sum() / sizes.shape[0], score[max_idx]

@dataclass(slots=True)
class PIT_SDRi:
    device: torch.device
    dump: int
    def __call__(self, **kwargs):
        estims = torch.stack(kwargs['estims'], 0).squeeze(1).cpu().numpy()
        targets = torch.stack([t.to(self.device) for t in kwargs["target_attr"]], 0).squeeze(1).cpu().numpy()
        mix = torch.cat([kwargs['mixture'], kwargs['mixture']], 0).cpu().numpy()
        out_sdr = bss_eval_sources(targets, estims)[0]
        in_sdr = bss_eval_sources(targets, mix)[0]
        return np.sum(out_sdr - in_sdr) / kwargs["input_sizes"].shape[0], out_sdr - in_sdr