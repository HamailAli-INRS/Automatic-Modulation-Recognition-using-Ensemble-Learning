"""
2D ResCNN Neural Demapper Architecture for 2D Time-Frequency OFDM Resource Grids.
Supports arbitrary subcarrier counts (e.g. 64, 128, 256), symbols (e.g. 14),
and modulation bit depths (e.g. QPSK=2, 16-QAM=4, 64-QAM=6).
Includes theoretical and empirical FLOPs and parameter counters.
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

class ResCNNDemapper2D(nn.Module):
    """
    2D Residual CNN Neural Demapper.
    
    Parameters:
    -----------
    num_bits_per_symbol : int
        Modulation order bit depth (2 for QPSK, 4 for 16-QAM, 6 for 64-QAM).
    hidden_dim : int
        Number of hidden feature channels in intermediate CNN layers.
    num_res_blocks : int
        Number of residual convolutional blocks.
    temperature : float
        Initial learnable temperature scaling factor for LLR calibration.
    """
    def __init__(self, num_bits_per_symbol=4, hidden_dim=64, num_res_blocks=3, temperature=1.0):
        super().__init__()
        self.num_bits_per_symbol = num_bits_per_symbol
        self.hidden_dim = hidden_dim
        self.num_res_blocks = num_res_blocks
        
        # Ingest Real and Imaginary (2 channels)
        self.input_proj = ConvBlock2D(in_channels=2, out_channels=hidden_dim)
        
        # Residual feature extraction across subcarriers and OFDM symbols
        self.res_blocks = nn.ModuleList([
            ResBlock2D(channels=hidden_dim) for _ in range(num_res_blocks)
        ])
        
        # Output layer producing soft LLR logits for each bit of the constellation
        self.output_conv = nn.Conv2d(hidden_dim, num_bits_per_symbol, kernel_size=3, padding=1, bias=True)
        
        # Learnable calibration parameter for optimal downstream Belief Propagation decoding
        self.scale = nn.Parameter(torch.tensor(temperature, dtype=torch.float32))

    def forward(self, y_grid, data_mask=None):
        """
        Forward pass.
        
        Parameters:
        -----------
        y_grid : torch.Tensor
            Complex received OFDM resource grid of shape [Batch, Symbols, Subcarriers]
            or real-valued tensor of shape [Batch, 2, Symbols, Subcarriers].
        data_mask : torch.Tensor (optional)
            Boolean mask [Symbols, Subcarriers] where True indicates data REs and False indicates pilots.
            
        Returns:
        --------
        llrs : torch.Tensor
            Soft LLRs of shape [Batch, Total_Data_Coded_Bits] (flattened to feed LDPC decoder).
        """
        if torch.is_complex(y_grid):
            x = torch.stack([y_grid.real, y_grid.imag], dim=1).float()
        else:
            x = y_grid
            
        feat = self.input_proj(x)
        for block in self.res_blocks:
            feat = block(feat)
            
        # [Batch, num_bits_per_symbol, Symbols, Subcarriers]
        raw_logits = self.output_conv(feat)
        
        # Calibrated LLRs
        calibrated_llrs = raw_logits * self.scale
        
        # Permute to [Batch, Symbols, Subcarriers, num_bits_per_symbol]
        llrs_grid = calibrated_llrs.permute(0, 2, 3, 1)
        b, sym, sc, m = llrs_grid.shape
        
        if data_mask is not None:
            # Flatten spatial grid to [Batch, Symbols*Subcarriers, num_bits_per_symbol]
            llrs_spatial_flat = llrs_grid.view(b, sym * sc, m)
            mask_flat = data_mask.view(-1)
            # Extract only data resource elements [Batch, Num_Data_Symbols, num_bits_per_symbol]
            data_llrs = llrs_spatial_flat[:, mask_flat, :]
            llrs = data_llrs.contiguous().view(b, -1)
        else:
            llrs = llrs_grid.contiguous().view(b, -1)
            
        return llrs

    def count_parameters(self):
        """Returns total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def calculate_flops(self, num_subcarriers=64, num_ofdm_symbols=14):
        """
        Analytical FLOP calculation for a single forward pass of 2D ResCNN.
        
        Each Conv2D(Cin, Cout, K=3) on (H, W) grid requires:
        FLOPs ≈ 2 * Cin * Cout * K^2 * H * W
        """
        h, w = num_subcarriers, num_ofdm_symbols
        flops = 0
        
        # 1. Input Projection: Conv2D(2, hidden_dim, 3x3)
        flops += 2 * 2 * self.hidden_dim * 9 * h * w
        # GroupNorm & GELU
        flops += 4 * self.hidden_dim * h * w
        
        # 2. Residual Blocks: each has 2 Conv2D(hidden_dim, hidden_dim, 3x3)
        for _ in range(self.num_res_blocks):
            # Conv 1
            flops += 2 * self.hidden_dim * self.hidden_dim * 9 * h * w
            flops += 4 * self.hidden_dim * h * w # GN + GELU
            # Conv 2
            flops += 2 * self.hidden_dim * self.hidden_dim * 9 * h * w
            flops += 4 * self.hidden_dim * h * w # GN + Residual Add + GELU
            
        # 3. Output Projection: Conv2D(hidden_dim, num_bits_per_symbol, 3x3)
        flops += 2 * self.hidden_dim * self.num_bits_per_symbol * 9 * h * w
        # Temperature scaling
        flops += self.num_bits_per_symbol * h * w
        
        return flops
