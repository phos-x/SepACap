import sys
sys.path.append('../')

import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')

from utils.decorators import *
from .network import *

# --- New: Periodic Activation for SepACap ---
class Snake(nn.Module):
    """
    Snake Activation Function: x + (1/a) * sin^2(ax)
    Specifically designed for periodic signal extrapolation (singing voices).
    """
    def __init__(self, in_features, alpha=1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, in_features, 1) * alpha)

    def forward(self, x):
        return x + (1.0 / self.alpha) * torch.pow(torch.sin(self.alpha * x), 2)

def get_activation(act_name: str, channels: int):
    """Factory to return the specified activation function."""
    if act_name == "SNAKE":
        return Snake(channels)
    elif act_name == "GELU":
        return nn.GELU()
    elif act_name == "PReLU":
        return nn.PReLU(channels)
    return nn.ReLU()

# --- Updated Modules ---

class AudioEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, bias, activation="GELU"):
        super().__init__()
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride, groups, bias)
        self.act = get_activation(activation, out_channels)
    
    def forward(self, x: torch.Tensor):
        x = x.unsqueeze(1) if x.dim() == 2 else x.unsqueeze(0).unsqueeze(0)
        x = self.conv1d(x)
        x = self.act(x)
        return x
    
class FeatureProjector(nn.Module):
    def __init__(self, num_channels, in_channels, out_channels, kernel_size, bias):
        super().__init__()
        self.norm = nn.GroupNorm(1, num_channels, eps=1e-8)
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, bias=bias)
    
    def forward(self, x: torch.Tensor):
        x = self.norm(x)
        x = self.conv1d(x)
        return x

class Separator(nn.Module):
    def __init__(self, num_stages, relative_positional_encoding, enc_stage, spk_split_stage, simple_fusion, dec_stage, activation="SNAKE"):
        super().__init__()
        self.activation_type = activation
        
        # Internal Class Definitions
        class RelativePositionalEncoding(nn.Module):
            def __init__(self, in_channels, num_heads, maxlen, embed_v=False):
                super().__init__()
                self.embedding_dim = in_channels // num_heads
                self.maxlen = maxlen
                self.pe_k = nn.Embedding(2*maxlen, self.embedding_dim)
                self.pe_v = nn.Embedding(2*maxlen, self.embedding_dim) if embed_v else None
            
            def forward(self, pos_seq: torch.Tensor):
                pos_seq = pos_seq.clamp(-self.maxlen, self.maxlen - 1) + self.maxlen
                return self.pe_k(pos_seq), (self.pe_v(pos_seq) if self.pe_v else None)

        class DownConvLayer(nn.Module):
            def __init__(self, in_channels, samp_kernel_size, activation="SNAKE"):
                super().__init__()
                self.down_conv = nn.Conv1d(in_channels, in_channels, samp_kernel_size, stride=2, padding=(samp_kernel_size-1)//2, groups=in_channels)
                self.BN = nn.BatchNorm1d(in_channels)
                self.act = get_activation(activation, in_channels)
            
            def forward(self, x: torch.Tensor):
                return self.act(self.BN(self.down_conv(x)))

        class SepEncStage(nn.Module):
            def __init__(self, global_blocks, local_blocks, down_conv_layer, down_conv=True, activation="SNAKE"):
                super().__init__()
                self.g_block_1 = GlobalBlock(**global_blocks)
                self.l_block_1 = LocalBlock(**local_blocks)
                self.g_block_2 = GlobalBlock(**global_blocks)
                self.l_block_2 = LocalBlock(**local_blocks)
                self.downconv = DownConvLayer(**down_conv_layer, activation=activation) if down_conv else None
            
            def forward(self, x, pos_k):
                x = self.g_block_1(x, pos_k)
                x = self.l_block_1(x.transpose(1, 2)).transpose(1, 2)
                x = self.g_block_2(x, pos_k)
                x = self.l_block_2(x.transpose(1, 2)).transpose(1, 2)
                skip = x
                if self.downconv:
                    x = self.downconv(x.transpose(1, 2)).transpose(1, 2)
                return x, skip

        # Main Structure initialization
        self.num_stages = num_stages
        self.pos_emb = RelativePositionalEncoding(**relative_positional_encoding)
        
        # Clean config dictionaries to avoid "multiple values for keyword argument 'activation'"
        enc_clean = {k: v for k, v in enc_stage.items() if k != 'activation'}
        dec_clean = {k: v for k, v in dec_stage.items() if k != 'activation'}

        self.enc_stages = nn.ModuleList([
            SepEncStage(**enc_clean, down_conv=True, activation=self.activation_type) 
            for _ in range(num_stages)
        ])
        
        self.bottleneck_G = SepEncStage(**enc_clean, down_conv=False, activation=self.activation_type)
        self.spk_split_block = SpkSplitStage(**spk_split_stage)
        
        self.simple_fusion = nn.ModuleList([
            nn.Conv1d(simple_fusion['out_channels']*2, simple_fusion['out_channels'], 1) 
            for _ in range(num_stages)
        ])
        
        self.dec_stages = nn.ModuleList([
            SepDecStage(**dec_clean, activation=self.activation_type) 
            for _ in range(num_stages)
        ])
    
    def forward(self, input: torch.Tensor):
        x, _ = self.pad_signal(input)
        len_x = x.shape[-1]
        
        pos_seq = torch.arange(0, len_x // 2**self.num_stages).long().to(x.device)
        pos_seq = pos_seq[:, None] - pos_seq[None, :]
        pos_k, _ = self.pos_emb(pos_seq)
        
        skip = []
        for idx in range(self.num_stages):
            x, skip_ = self.enc_stages[idx](x.transpose(1, 2), pos_k)
            skip.append(self.spk_split_block(skip_.transpose(1, 2)))
            x = x.transpose(1, 2)
        
        x, _ = self.bottleneck_G(x.transpose(1, 2), pos_k)
        x = self.spk_split_block(x.transpose(1, 2))
        
        each_stage_outputs = []
        for idx in range(self.num_stages):
            each_stage_outputs.append(x)
            idx_en = self.num_stages - (idx + 1)
            x = F.interpolate(x, size=skip[idx_en].shape[-1], mode='linear', align_corners=False)
            x = self.simple_fusion[idx](torch.cat([x, skip[idx_en]], dim=1))
            x, _ = self.dec_stages[idx](x.transpose(1, 2), pos_k)
            x = x.transpose(1, 2)
        
        return x, each_stage_outputs

    def pad_signal(self, input: torch.Tensor):
        if input.dim() == 2: input = input.unsqueeze(1)
        L = 2**self.num_stages
        nframe = input.size(2)
        padded_len = (nframe // L + 1) * L
        rest = 0 if nframe % L == 0 else padded_len - nframe
        if rest > 0:
            input = F.pad(input, (0, rest))
        return input, rest

class SpkSplitStage(nn.Module):
    def __init__(self, in_channels, num_spks):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Conv1d(in_channels, 4*in_channels*num_spks, 1),
            nn.GLU(dim=-2),
            nn.Conv1d(2*in_channels*num_spks, in_channels*num_spks, 1))
        self.norm = nn.GroupNorm(1, in_channels, eps=1e-8)
        self.num_spks = num_spks
                
    def forward(self, x: torch.Tensor):
        x = self.linear(x)
        B, _, T = x.shape
        x = self.norm(x.view(B*self.num_spks, -1, T))
        return x

class SepDecStage(nn.Module):
    def __init__(self, num_spks, global_blocks, local_blocks, spk_attention, activation="SNAKE"):
        super().__init__()
        self.g_block_1 = GlobalBlock(**global_blocks)
        self.l_block_1 = LocalBlock(**local_blocks)
        self.spk_attn_1 = SpkAttention(**spk_attention)
        self.g_block_2 = GlobalBlock(**global_blocks)
        self.l_block_2 = LocalBlock(**local_blocks)
        self.spk_attn_2 = SpkAttention(**spk_attention)
        self.g_block_3 = GlobalBlock(**global_blocks)
        self.l_block_3 = LocalBlock(**local_blocks)
        self.spk_attn_3 = SpkAttention(**spk_attention)
        self.num_spk = num_spks
    
    def forward(self, x, pos_k):
        for gb, lb, sa in [(self.g_block_1, self.l_block_1, self.spk_attn_1), 
                           (self.g_block_2, self.l_block_2, self.spk_attn_2), 
                           (self.g_block_3, self.l_block_3, self.spk_attn_3)]:
            x = gb(x, pos_k)
            x = lb(x.transpose(1, 2)).transpose(1, 2)
            x = sa(x.transpose(1, 2), self.num_spk).transpose(1, 2)
        return x, x

class OutputLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_spks, masking=False):
        super().__init__()
        self.masking = masking
        self.num_spks = num_spks
        self.spe_block = Masking(in_channels, Activation_mask="ReLU")
        self.end_conv1x1 = nn.Sequential(
            nn.Linear(out_channels, 4*out_channels),
            nn.GLU(),
            nn.Linear(2*out_channels, in_channels))
            
    def forward(self, x, input):
        x = self.end_conv1x1(x[..., :input.shape[-1]].transpose(1, 2)).transpose(1, 2)
        B = x.shape[0] // self.num_spks
        if self.masking:
            inp_v = input.expand(self.num_spks, B, -1, -1).transpose(0, 1).reshape(B*self.num_spks, -1, x.shape[-1])
            x = self.spe_block(x, inp_v)
        return x.view(B, self.num_spks, -1, x.shape[-1]).transpose(0, 1)

class AudioDecoder(nn.ConvTranspose1d):
    def forward(self, x):
        x = super().forward(x if x.dim() == 3 else x.unsqueeze(1))
        return x.squeeze(1) if x.shape[1] == 1 else x