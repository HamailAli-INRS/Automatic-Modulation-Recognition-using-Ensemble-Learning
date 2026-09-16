"""
5G NR OFDM + LDPC Transmission Chain with Zero-Pilot (Pilot-Free) Support.
Implements end-to-end forward error correction (LDPC) and modulation mapped over 2D resource grids.
"""

import torch
import torch.nn as nn
import numpy as np

import sionna
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.phy.mapping import Mapper, Demapper, Constellation
from sionna.phy.ofdm import ResourceGrid, ResourceGridMapper
from sionna.phy.channel import OFDMChannel, RayleighBlockFading, AWGN
from sionna.phy.utils import ebnodb2no

class OFDMFECPipeline:
    """
    Standard-compliant 5G NR OFDM and LDPC Physical Layer Pipeline.
    
    Parameters:
    -----------
    num_ofdm_symbols : int
        Number of OFDM symbols in a slot (e.g. 14 for 5G NR).
    fft_size : int
        Number of subcarriers (e.g. 64 or 76).
    subcarrier_spacing : float
        Subcarrier spacing in Hz (default: 15 kHz).
    cyclic_prefix_length : int
        Cyclic prefix sample length.
    num_bits_per_symbol : int
        Modulation order (2 for QPSK, 4 for 16-QAM, 6 for 64-QAM).
    coderate : float
        LDPC code rate (e.g., 0.5, 0.67, 0.75).
    num_ldpc_iter : int
        Number of Belief Propagation iterations in LDPC decoder.
    channel_type : str
        'rayleigh' for multipath fading channel or 'awgn' for additive white Gaussian noise.
    """
    def __init__(self,
                 num_ofdm_symbols=14,
                 fft_size=64,
                 subcarrier_spacing=15e3,
                 cyclic_prefix_length=16,
                 num_bits_per_symbol=4,
                 coderate=0.5,
                 num_ldpc_iter=20,
                 channel_type="rayleigh",
                 device="cpu"):
        
        self.num_ofdm_symbols = num_ofdm_symbols
        self.fft_size = fft_size
        self.num_bits_per_symbol = num_bits_per_symbol
        self.coderate = coderate
        self.channel_type = channel_type.lower()
        self.device = torch.device(device)
        
        # 1. Pilot-Free Resource Grid (100% data symbols, 0% pilot overhead)
        self.rg = ResourceGrid(
            num_ofdm_symbols=num_ofdm_symbols,
            fft_size=fft_size,
            subcarrier_spacing=subcarrier_spacing,
            num_tx=1,
            num_streams_per_tx=1,
            cyclic_prefix_length=cyclic_prefix_length,
            pilot_pattern=None # ZERO PILOT OVERHEAD
        )
        
        self.num_data_symbols = self.rg.num_data_symbols # e.g. 14 * 64 = 896
        self.n = self.num_data_symbols * self.num_bits_per_symbol # Total coded bits: 3584
        self.k = int(self.n * self.coderate) # Info bits: 1792
        
        # 2. 5G NR LDPC Encoder & Decoder
        self.encoder = LDPC5GEncoder(k=self.k, n=self.n).to(self.device)
        self.decoder = LDPC5GDecoder(self.encoder, num_iter=num_ldpc_iter).to(self.device)
        
        # 3. Constellation & Mapper
        self.constellation = Constellation("qam", num_bits_per_symbol=self.num_bits_per_symbol).to(self.device)
        self.mapper = Mapper(constellation=self.constellation).to(self.device)
        self.classical_demapper = Demapper("app", constellation=self.constellation).to(self.device)
        self.rg_mapper = ResourceGridMapper(self.rg).to(self.device)
        
        # 4. Realistic Channel
        self.awgn = AWGN().to(self.device)
        if self.channel_type == "rayleigh":
            self.fading_model = RayleighBlockFading(num_rx=1, num_rx_ant=1, num_tx=1, num_tx_ant=1).to(self.device)
            self.ofdm_channel = OFDMChannel(self.fading_model, self.rg).to(self.device)
        else:
            self.ofdm_channel = None

    def transmit(self, batch_size, ebno_db):
        """
        Generate, encode, modulate, and transmit an OFDM frame batch over the channel.
        
        Returns:
        --------
        u : torch.Tensor [B, k] (ground truth information bits)
        c : torch.Tensor [B, n] (ground truth encoded bits)
        y_grid : torch.Tensor [B, 2, num_ofdm_symbols, fft_size] (received distorted I/Q grid)
        no : float (noise power spectral density)
        """
        # Generate random info bits
        u = torch.randint(0, 2, (batch_size, self.k), dtype=torch.float32, device=self.device)
        
        # LDPC Encode
        c = self.encoder(u) # [B, n]
        
        # QAM Modulate
        # Reshape coded bits to [B, 1, 1, num_data_symbols * num_bits_per_symbol]
        c_reshaped = c.view(batch_size, 1, 1, self.n)
        symbols = self.mapper(c_reshaped) # [B, 1, 1, num_data_symbols]
        
        # Resource Grid Mapping -> [B, 1, 1, num_ofdm_symbols, fft_size]
        tx_grid = self.rg_mapper(symbols)
        
        # Compute Noise Variance
        no = ebnodb2no(ebno_db, num_bits_per_symbol=self.num_bits_per_symbol, coderate=self.coderate)
        
        # Apply Multipath Channel & Noise
        if self.channel_type == "rayleigh" and self.ofdm_channel is not None:
            faded_grid = self.ofdm_channel(tx_grid)
        else:
            faded_grid = tx_grid
            
        rx_grid = self.awgn(faded_grid, no) # [B, 1, 1, num_ofdm_symbols, fft_size]
        
        # Squeeze down to [B, num_ofdm_symbols, fft_size]
        y_complex = rx_grid.squeeze(1).squeeze(1)
        
        # Format as 2-channel tensor [B, 2, num_ofdm_symbols, fft_size]
        y_tensor = torch.stack([y_complex.real, y_complex.imag], dim=1).float()
        
        return u, c, y_tensor, no

    def decode_ldpc(self, llrs):
        """
        Decode soft LLRs with 5G NR LDPC Belief Propagation.
        
        Parameters:
        -----------
        llrs : torch.Tensor [B, n]
        
        Returns:
        --------
        u_hat : torch.Tensor [B, k] (decoded info bits)
        c_hat : torch.Tensor [B, n] (re-encoded parity-consistent codeword)
        """
        u_hat = self.decoder(llrs)
        c_hat = self.encoder(u_hat)
        return u_hat, c_hat
