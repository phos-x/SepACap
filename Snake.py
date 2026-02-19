import torch
import torch.nn as nn

class SnakeActivation(nn.Module):
    """
    Snake activation function for periodic signals.
    'a' controls the frequency of the periodic component.
    """
    def __init__(self, in_features, layer_index=0, total_stages=4):
        super().__init__()
        self.in_features = in_features
        # DSA Best Practice: Initialize alpha based on the receptive field depth
        # Early stages (low index) focus on broader harmonics; late stages on fine detail.
        init_val = 1.0 + (layer_index / total_stages) * 5.0 
        self.alpha = nn.Parameter(torch.ones(1, in_features) * init_val)


    def forward(self, x):
        # DSA best practice: vectorized math for GPU efficiency
        # formula: x + (1/alpha) * sin^2(alpha * x)
        return x + (1.0 / (self.alpha + 1e-9)) * torch.pow(torch.sin(self.alpha * x), 2)