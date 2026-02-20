import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
from loguru import logger
from utils.decorators import logger_wraps
from .modules.module import AudioEncoder, FeatureProjector, Separator, OutputLayer, AudioDecoder

@logger_wraps()
class Model(nn.Module):
    """
    SepACap: An adaptation of SepReformer for multi-singer separation.
    Implements periodicity-aware modeling using SNAKE activations and 
    direct waveform reconstruction.
    """
    def __init__(self, 
                 num_stages: int, 
                 num_spks: int, 
                 module_audio_enc: Dict, 
                 module_feature_projector: Dict, 
                 module_separator: Dict, 
                 module_output_layer: Dict, 
                 module_audio_dec: Dict,
                 activation: str = "ReLU"): # Added to fix TypeError
        super().__init__()
        
        # 1. Parameter Validation
        if num_stages <= 0 or num_spks <= 0:
            raise ValueError(f"Invalid model dimensions: stages={num_stages}, speakers={num_spks}")
            
        self.num_stages = num_stages
        self.num_spks = num_spks
        self.activation_type = activation

        # 2. Inject Activation Choice into Separator Config
        # This ensures the Transformers/Convolutions inside the separator use SNAKE
        if "activation" not in module_separator:
            module_separator["activation"] = self.activation_type

        # 3. Main Processing Chain
        self.audio_encoder = AudioEncoder(**module_audio_enc)
        self.feature_projector = FeatureProjector(**module_feature_projector)
        self.separator = Separator(**module_separator)
        self.out_layer = OutputLayer(**module_output_layer)
        self.audio_decoder = AudioDecoder(**module_audio_dec)
        
        # 4. Auxiliary Loss Modules (Multi-Resolution Supervision)
        # ModuleList for bottleneck supervision at each transformer stage
        self.out_layer_bn = nn.ModuleList([
            OutputLayer(**module_output_layer, masking=True) for _ in range(self.num_stages)
        ])
        self.decoder_bn = nn.ModuleList([
            AudioDecoder(**module_audio_dec) for _ in range(self.num_stages)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        """
        Processes raw audio mixtures. 
        Targeting (Batch, Samples) -> (num_spks, Batch, Samples)
        """
        # A. Feature Extraction
        encoder_output = self.audio_encoder(x)
        projected_feature = self.feature_projector(encoder_output)
        
        # B. Separation Logic
        last_stage_output, each_stage_outputs = self.separator(projected_feature)
        
        # C. Primary Source Reconstruction
        out_layer_output = self.out_layer(last_stage_output, encoder_output)
        
        # Ensure audio outputs are synced with input length
        target_len = x.shape[-1]
        each_spk_output = [out_layer_output[idx] for idx in range(self.num_spks)]
        audio = [self.audio_decoder(out)[..., :target_len] for out in each_spk_output]
        
        # D. Auxiliary Supervision
        audio_aux = []
        for idx, stage_out in enumerate(each_stage_outputs):
            # Upsample stage features to match encoder resolution for masking
            upsampled = F.interpolate(
                stage_out, 
                size=encoder_output.shape[-1], 
                mode='linear', 
                align_corners=False
            )
            
            stage_masks = self.out_layer_bn[idx](upsampled, encoder_output)
            
            stage_audio_list = []
            for spk_idx in range(self.num_spks):
                decoded = self.decoder_bn[idx](stage_masks[spk_idx])
                stage_audio_list.append(decoded[..., :target_len])
            
            audio_aux.append(stage_audio_list)
            
        return audio, audio_aux

    def extra_repr(self) -> str:
        return (f'num_stages={self.num_stages}, num_spks={self.num_spks}, '
                f'activation={self.activation_type}, mode=SepACap_Waveform')