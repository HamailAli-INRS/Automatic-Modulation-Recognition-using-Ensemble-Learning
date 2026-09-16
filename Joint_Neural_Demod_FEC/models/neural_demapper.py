"""
Neural Demapper Architecture for 2D OFDM Resource Grids.
Ingests multi-carrier Time-Frequency grids [B, 2, Subcarriers, Symbols] (Real and Imaginary channels)
and outputs soft Log-Likelihood Ratios (LLRs) for each coded bit plane.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock2D(nn.Module):
    """2D Convolutional Block with Group Normalization and GELU activation."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True)
        self.norm = nn.GroupNorm(num_groups=1, num_channels=out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class ResBlock2D(nn.Module):
    """2D Residual Convolutional Block to preserve fine time-frequency channel features."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        self.norm1 = nn.GroupNorm(num_groups=1, num_channels=channels)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        self.norm2 = nn.GroupNorm(num_groups=1, num_channels=channels)
        self.act2 = nn.GELU()

    def forward(self, x):
        residual = x
        out = self.act1(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = self.act2(out + residual)
        return out

class ResCNNDemapper(nn.Module):
    """
    Joint 2D Time-Frequency Neural Demapper.
    
    Parameters:
    -----------
    num_bits_per_symbol : int
        Modulation order bit depth (e.g. 2 for QPSK, 4 for 16-QAM, 6 for 64-QAM).
    hidden_dim : int
        Number of hidden feature channels in intermediate CNN layers.
    num_res_blocks : int
        Number of residual convolutional blocks.
    temperature : float
        Learnable or fixed temperature scaling factor for LLR calibration.
    """
    def __init__(self, num_bits_per_symbol=4, hidden_dim=64, num_res_blocks=3, temperature=1.0):
        super().__init__()
        self.num_bits_per_symbol = num_bits_per_symbol
        self.hidden_dim = hidden_dim
        
        # Ingest Real and Imaginary (2 channels)
        self.input_proj = ConvBlock2D(in_channels=2, out_channels=hidden_dim)
        
        # Residual feature extraction across subcarriers and OFDM symbols
        self.res_blocks = nn.ModuleList([
            ResBlock2D(channels=hidden_dim) for _ in range(num_res_blocks)
        ])
        
        # Output layer producing soft LLR logits for each bit of the constellation
        self.output_conv = nn.Conv2d(hidden_dim, num_bits_per_symbol, kernel_size=3, padding=1, bias=True)
        
        # Learnable calibration parameter for optimal Belief Propagation decoding
        self.scale = nn.Parameter(torch.tensor(temperature, dtype=torch.float32))

    def forward(self, y_grid):
        """
        Forward pass.
        
        Parameters:
        -----------
        y_grid : torch.Tensor
            Complex received OFDM resource grid of shape [Batch, Subcarriers, Symbols]
            or real-valued tensor of shape [Batch, 2, Subcarriers, Symbols].
            
        Returns:
        --------
        llrs : torch.Tensor
            Soft LLRs of shape [Batch, Total_Coded_Bits] (flattened to feed downstream LDPC decoder).
        """
        if torch.is_complex(y_grid):
            x = torch.stack([y_grid.real, y_grid.imag], dim=1).float()
        else:
            x = y_grid
            
        feat = self.input_proj(x)
        for block in self.res_blocks:
            feat = block(feat)
            
        # [Batch, num_bits_per_symbol, Subcarriers, Symbols]
        raw_logits = self.output_conv(feat)
        
        # Calibrated LLRs
        calibrated_llrs = raw_logits * self.scale
        
        # Reshape to [Batch, Subcarriers * Symbols * num_bits_per_symbol]
        b, m, sc, sym = calibrated_llrs.shape
        llrs = calibrated_llrs.permute(0, 2, 3, 1).contiguous().view(b, -1)
        return llrs
