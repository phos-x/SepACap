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
    SepACap: An adaptation of SepReformer for multi-singer separation[cite: 22, 40].
    Utilizes direct waveform modeling and SNR-independent composite objectives[cite: 36, 42].
    """
    def __init__(self, 
                 num_stages: int, 
                 num_spks: int, 
                 module_audio_enc: Dict, 
                 module_feature_projector: Dict, 
                 module_separator: Dict, 
                 module_output_layer: Dict, 
                 module_audio_dec: Dict):
        super().__init__()
        
        # 1. Parameter Validation (Security/DSA Best Practice)
        if num_stages <= 0 or num_spks <= 0:
            raise ValueError(f"Invalid model dimensions: stages={num_stages}, speakers={num_spks}")
            
        self.num_stages = num_stages
        self.num_spks = num_spks
        
        # 2. Main Processing Chain
        # Note: Separator sub-modules should be initialized with Snake activations 
        # as noted in the ETH research for improved harmonic extrapolation.
        self.audio_encoder = AudioEncoder(**module_audio_enc)
        self.feature_projector = FeatureProjector(**module_feature_projector)
        self.separator = Separator(**module_separator)
        self.out_layer = OutputLayer(**module_output_layer)
        self.audio_decoder = AudioDecoder(**module_audio_dec)
        
        # 3. Auxiliary Loss Modules (Multi-Resolution Supervision)
        # We use ModuleList for proper device registration and state tracking[cite: 42].
        self.out_layer_bn = nn.ModuleList([
            OutputLayer(**module_output_layer, masking=True) for _ in range(self.num_stages)
        ])
        self.decoder_bn = nn.ModuleList([
            AudioDecoder(**module_audio_dec) for _ in range(self.num_stages)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        """
        Processes raw audio mixtures through direct waveform modeling.
        Args:
            x: Input mixture tensor of shape (Batch, Samples).
        Returns:
            audio: Final separated sources (List of num_spks tensors).
            audio_aux: Auxiliary outputs for each separator stage.
        """
        # A. Feature Extraction
        # Encoder translates 1D waveform to latent 2D space[cite: 36].
        encoder_output = self.audio_encoder(x)
        projected_feature = self.feature_projector(encoder_output)
        
        # B. Separation Logic
        # Separator captures spectral structure using Transformers[cite: 33].
        last_stage_output, each_stage_outputs = self.separator(projected_feature)
        
        # C. Primary Source Reconstruction
        out_layer_output = self.out_layer(last_stage_output, encoder_output)
        
        # DSA: List comprehensions for clean, traceable source isolation.
        each_spk_output = [out_layer_output[idx] for idx in range(self.num_spks)]
        audio = [self.audio_decoder(out) for out in each_spk_output]
        
        # D. Auxiliary Supervison (Bottleneck Supervision)
        # Critical for ensuring deep layers learn useful features when some stems are silent[cite: 49].
        audio_aux = []
        target_len = x.shape[-1]
        
        for idx, stage_out in enumerate(each_stage_outputs):
            # Upsample stage features to match encoder resolution.
            upsampled = F.interpolate(stage_out, size=encoder_output.shape[-1], mode='linear', align_corners=False)
            
            stage_masks = self.out_layer_bn[idx](upsampled, encoder_output)
            
            stage_audio_list = []
            for spk_idx in range(self.num_spks):
                # Apply decoder and strictly sync length with input mixture (DSA Shield).
                decoded = self.decoder_bn[idx](stage_masks[spk_idx])
                stage_audio_list.append(decoded[..., :target_len])
            
            audio_aux.append(stage_audio_list)
            
        return audio, audio_aux

    def extra_repr(self) -> str:
        return f'num_stages={self.num_stages}, num_spks={self.num_spks}, mode=SepACap_Waveform'