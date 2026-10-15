"""
Classical 5G NR Baseline: LS Channel Estimation + LMMSE Equalization + Max-Log APP Demapping + LDPC Decoder.
Supports both Pilot-Assisted (DMRS) and Pilot-Free modes across AWGN, Rayleigh, and TDL channels.
"""

import torch
import numpy as np

class ClassicalBaseline:
    """
    Classical Physical Layer Demapping & Decoding Baseline.
    """
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.device = pipeline.device

    @torch.no_grad()
    def evaluate(self, batch_size=64, ebno_db=10.0):
        """
        Evaluate Classical Baseline across a batch of transmissions.
        
        Returns:
        --------
        raw_ber : float
        post_fec_ber : float
        bler : float
        """
        u_true, c_true, _, no, rx_grid_raw = self.pipeline.transmit(batch_size, ebno_db)
        
        # Classical Demapping and LDPC Decoding
        llrs, u_hat, c_hat = self.pipeline.classical_demap_and_decode(rx_grid_raw, no)
        
        # Uncoded / Raw BER
        pred_bits = (llrs > 0).float()
        # Flatten and truncate/match dimensions if needed
        pred_bits_flat = pred_bits.view(batch_size, -1)[:, :self.pipeline.n]
        raw_ber = (pred_bits_flat != c_true).float().mean().item()
        
        # Post-FEC BER and BLER
        u_hat_flat = u_hat.view(batch_size, -1)[:, :self.pipeline.k]
        post_fec_ber = (u_hat_flat != u_true).float().mean().item()
        
        frame_errors = (u_hat_flat != u_true).float().sum(dim=-1) > 0
        bler = frame_errors.float().mean().item()
        
        return raw_ber, post_fec_ber, bler
