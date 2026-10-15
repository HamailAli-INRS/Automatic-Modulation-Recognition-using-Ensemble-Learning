"""
2D Deep ResCNN Neural LLR Corrector & Demapper for 64-QAM over 3GPP TDL Channels.
Directly learns to correct non-Gaussian and multipath estimation mismatch in LLRs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Neural64QAMDemapper(nn.Module):
    """
    Neural LLR Corrector & Demapper.
    Ingests 2D time-frequency equalized symbols [B, 2, 12, 64] and initial soft LLRs [B, 6, 12, 64]
    to output optimized, calibrated LLRs for downstream 5G NR LDPC decoding.
    """
    def __init__(self, hidden_dim=64, num_res_blocks=3, temperature=1.0):
        super().__init__()
        # Input channels: 2 (real, imag of equalized symbols) + 6 (initial LLRs) = 8
        self.input_conv = nn.Sequential(
            nn.Conv2d(8, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(1, hidden_dim),
            nn.GELU()
        )
        
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.GroupNorm(1, hidden_dim),
                nn.GELU(),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.GroupNorm(1, hidden_dim)
            ) for _ in range(num_res_blocks)
        ])
        self.act = nn.GELU()
        
        self.output_conv = nn.Conv2d(hidden_dim, 6, kernel_size=3, padding=1)
        self.scale = nn.Parameter(torch.tensor(temperature, dtype=torch.float32))

    def forward(self, x_hat_symbols, raw_llrs):
        """
        Forward pass.
        x_hat_symbols: [B, 1, 1, 768] complex
        raw_llrs: [B, 1, 1, 768, 6] or [B, 4608]
        """
        b = x_hat_symbols.shape[0]
        x_2d = x_hat_symbols.view(b, 12, 64)
        x_real_imag = torch.stack([x_2d.real, x_2d.imag], dim=1).float() # [B, 2, 12, 64]
        
        llr_2d = raw_llrs.view(b, 12, 64, 6).permute(0, 3, 1, 2).float() # [B, 6, 12, 64]
        
        inp = torch.cat([x_real_imag, llr_2d], dim=1) # [B, 8, 12, 64]
        feat = self.input_conv(inp)
        for block in self.res_blocks:
            feat = self.act(feat + block(feat))
            
        delta_llr = self.output_conv(feat)
        
        # Residual LLR refinement with temperature scaling
        refined_llrs_2d = (llr_2d + 0.5 * delta_llr) * self.scale
        
        # Flatten to [B, 4608]
        out_llrs = refined_llrs_2d.permute(0, 2, 3, 1).contiguous().view(b, -1)
        return out_llrs
