import sys
sys.path.append('../')

import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')

from utils.decorators import *
from .network import *

# --- Periodic Activation for SepACap Harmonic Modeling ---
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


class AudioEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, bias, activation="GELU"):
        super().__init__()
        # TYPE SHIELD: Explicitly cast arguments to prevent tuple/bool type errors in F.conv1d
        self.conv1d = nn.Conv1d(
            in_channels=int(in_channels), 
            out_channels=int(out_channels), 
            kernel_size=int(kernel_size), 
            stride=int(stride), 
            groups=int(groups), 
            bias=bool(bias)
        )
        self.act = get_activation(activation, out_channels)
    
    def forward(self, x: torch.Tensor):
        x = x.unsqueeze(1) if x.dim() == 2 else x.unsqueeze(0).unsqueeze(0)
        x = self.conv1d(x)
        x = self.act(x)
        return x
    
class FeatureProjector(nn.Module):
    def __init__(self, num_channels, in_channels, out_channels, kernel_size, bias):
        super().__init__()
        self.norm = nn.GroupNorm(1, int(num_channels), eps=1e-8)
        self.conv1d = nn.Conv1d(int(in_channels), int(out_channels), int(kernel_size), bias=bool(bias))
    
    def forward(self, x: torch.Tensor):
        x = self.norm(x)
        x = self.conv1d(x)
        return x

class Separator(nn.Module):
    def __init__(self, num_stages, relative_positional_encoding, enc_stage, spk_split_stage, simple_fusion, dec_stage, activation="SNAKE"):
        super().__init__()
        self.activation_type = activation
        
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
                self.down_conv = nn.Conv1d(int(in_channels), int(in_channels), int(samp_kernel_size), stride=2, padding=(int(samp_kernel_size)-1)//2, groups=int(in_channels))
                self.BN = nn.BatchNorm1d(int(in_channels))
                self.act = get_activation(activation, in_channels)
            
            def forward(self, x: torch.Tensor):
                # Strict adherence to original permute wrapping
                x = x.permute([0, 2, 1])
                x = self.down_conv(x)
                x = self.BN(x)
                x = self.act(x)
                x = x.permute([0, 2, 1])
                return x

        class SepEncStage(nn.Module):
            def __init__(self, global_blocks, local_blocks, down_conv_layer, down_conv=True, activation="SNAKE"):
                super().__init__()
                self.g_block_1 = GlobalBlock(**global_blocks)
                self.l_block_1 = LocalBlock(**local_blocks)
                self.g_block_2 = GlobalBlock(**global_blocks)
                self.l_block_2 = LocalBlock(**local_blocks)
                self.downconv = DownConvLayer(**down_conv_layer, activation=activation) if down_conv else None
            
            def forward(self, x, pos_k):
                # Strict adherence to original architecture permutes
                x = self.g_block_1(x, pos_k)
                x = x.permute(0, 2, 1).contiguous()
                x = self.l_block_1(x)
                x = x.permute(0, 2, 1).contiguous()
                
                x = self.g_block_2(x, pos_k)
                x = x.permute(0, 2, 1).contiguous()
                x = self.l_block_2(x)
                x = x.permute(0, 2, 1).contiguous()
                
                skip = x
                if self.downconv:
                    x = x.permute(0, 2, 1).contiguous()
                    x = self.downconv(x)
                    x = x.permute(0, 2, 1).contiguous()
                return x, skip

        self.num_stages = num_stages
        self.pos_emb = RelativePositionalEncoding(**relative_positional_encoding)
        
        enc_clean = {k: v for k, v in enc_stage.items() if k != 'activation'}
        dec_clean = {k: v for k, v in dec_stage.items() if k != 'activation'}

        self.enc_stages = nn.ModuleList([
            SepEncStage(**enc_clean, down_conv=True, activation=self.activation_type) 
            for _ in range(num_stages)
        ])
        
        self.bottleneck_G = SepEncStage(**enc_clean, down_conv=False, activation=self.activation_type)
        self.spk_split_block = SpkSplitStage(**spk_split_stage)
        
        self.simple_fusion = nn.ModuleList([
            nn.Conv1d(int(simple_fusion['out_channels']*2), int(simple_fusion['out_channels']), 1) 
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
            x, skip_ = self.enc_stages[idx](x, pos_k)
            skip_ = self.spk_split_block(skip_)
            skip.append(skip_)
            
        x, _ = self.bottleneck_G(x, pos_k)
        x = self.spk_split_block(x)
        
        each_stage_outputs = []
        for idx in range(self.num_stages):
            each_stage_outputs.append(x)
            idx_en = self.num_stages - (idx + 1)
            x = F.interpolate(x, size=skip[idx_en].shape[-1], mode='linear', align_corners=False)
            x = torch.cat([x, skip[idx_en]], dim=1)
            x = self.simple_fusion[idx](x)
            x, _ = self.dec_stages[idx](x, pos_k)
            
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
            nn.Conv1d(int(in_channels), int(4*in_channels*num_spks), 1),
            nn.GLU(dim=-2),
            nn.Conv1d(int(2*in_channels*num_spks), int(in_channels*num_spks), 1))
        self.norm = nn.GroupNorm(1, int(in_channels), eps=1e-8)
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
            x = x.permute(0, 2, 1).contiguous()
            x = lb(x)
            x = x.permute(0, 2, 1).contiguous()
            x = sa(x, self.num_spk)
        return x, x

class OutputLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_spks, masking=False):
        super().__init__()
        self.masking = masking
        self.num_spks = num_spks
        self.spe_block = Masking(int(in_channels), Activation_mask="ReLU", concat_opt=None)
        self.end_conv1x1 = nn.Sequential(
            nn.Linear(int(out_channels), int(4*out_channels)),
            nn.GLU(),
            nn.Linear(int(2*out_channels), int(in_channels)))
            
    def forward(self, x, input):
        x = x[..., :input.shape[-1]]
        x = x.permute([0, 2, 1])
        x = self.end_conv1x1(x)
        x = x.permute([0, 2, 1])
        B, N, L = x.shape
        B = B // self.num_spks
        
        if self.masking:
            inp_v = input.expand(self.num_spks, B, N, L).transpose(0, 1).contiguous()
            inp_v = inp_v.view(B*self.num_spks, N, L)
            x = self.spe_block(x, inp_v)
            
        x = x.view(B, self.num_spks, N, L)
        return x.transpose(0, 1)

class AudioDecoder(nn.ConvTranspose1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride, bias):
        super().__init__(
            in_channels=int(in_channels), 
            out_channels=int(out_channels), 
            kernel_size=int(kernel_size), 
            stride=int(stride), 
            bias=bool(bias)
        )
    def forward(self, x):
        x = super().forward(x if x.dim() == 3 else x.unsqueeze(1))
        return x.squeeze(1) if x.shape[1] == 1 else x