"""
Standard-Compliant 5G NR OFDM + LDPC Physical Layer Pipeline.
Supports:
- Scalable Info Bit Lengths: k=1792 (Nsc=64), k=3584 (Nsc=128), k=7168 (Nsc=256)
- Multipath Channel Dynamics: AWGN, Rayleigh Block/Multipath Fading, 3GPP TR 38.901 TDL / Ray-Tracing
- Dual Pilot Modes: Pilot-Free (0% overhead, 100% data) vs Pilot-Assisted (DMRS grid)
"""

import torch
import torch.nn as nn
import numpy as np

import sionna
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.phy.mapping import Mapper, Demapper, Constellation
from sionna.phy.ofdm import ResourceGrid, ResourceGridMapper, LSChannelEstimator, LMMSEEqualizer
from sionna.phy.mimo import StreamManagement
from sionna.phy.channel import OFDMChannel, RayleighBlockFading, AWGN
from sionna.phy.channel.tr38901 import TDL
from sionna.phy.utils import ebnodb2no

class OFDMChannelPipeline:
    """
    Unified Physical Layer Pipeline supporting AWGN, Rayleigh, and TDL/Ray-Tracing channels.
    
    Parameters:
    -----------
    num_ofdm_symbols : int
        Number of OFDM symbols per slot (default: 14).
    fft_size : int
        Number of subcarriers (64 -> k=1792, 128 -> k=3584, 256 -> k=7168 for 16-QAM R=0.5 pilot-free).
    subcarrier_spacing : float
        SCS in Hz (default: 15 kHz).
    cyclic_prefix_length : int
        CP sample length (default: 16).
    num_bits_per_symbol : int
        Modulation order bit depth (2 for QPSK, 4 for 16-QAM, 6 for 64-QAM).
    coderate : float
        LDPC code rate (default: 0.5).
    num_ldpc_iter : int
        Belief Propagation iterations.
    channel_type : str
        'awgn', 'rayleigh', or 'tdl_raytracing'.
    pilot_mode : str
        'pilot_free' or 'pilot_assisted'.
    speed : float
        Doppler speed in m/s (for Rayleigh/TDL ray-tracing, default 10.0 m/s = 36 km/h).
    delay_spread : float
        RMS delay spread in seconds (default 30e-9 s).
    device : str
        'cpu' or 'cuda'.
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
                 pilot_mode="pilot_free",
                 speed=10.0,
                 delay_spread=30e-9,
                 carrier_frequency=3.5e9,
                 device="cpu"):
        
        self.num_ofdm_symbols = num_ofdm_symbols
        self.fft_size = fft_size
        self.num_bits_per_symbol = num_bits_per_symbol
        self.coderate = coderate
        self.channel_type = channel_type.lower()
        self.pilot_mode = pilot_mode.lower()
        self.device = torch.device(device)
        self.speed = speed
        self.delay_spread = delay_spread
        self.carrier_frequency = carrier_frequency
        
        # Configure Resource Grid
        if self.pilot_mode == "pilot_assisted":
            self.pilot_symbol_indices = [2, 11]
            self.rg = ResourceGrid(
                num_ofdm_symbols=num_ofdm_symbols,
                fft_size=fft_size,
                subcarrier_spacing=subcarrier_spacing,
                num_tx=1,
                num_streams_per_tx=1,
                cyclic_prefix_length=cyclic_prefix_length,
                pilot_pattern='kronecker',
                pilot_ofdm_symbol_indices=self.pilot_symbol_indices
            )
            self.num_pilot_symbols = self.rg.num_pilot_symbols
        else: # pilot_free (Zero Pilots)
            self.pilot_symbol_indices = []
            self.rg = ResourceGrid(
                num_ofdm_symbols=num_ofdm_symbols,
                fft_size=fft_size,
                subcarrier_spacing=subcarrier_spacing,
                num_tx=1,
                num_streams_per_tx=1,
                cyclic_prefix_length=cyclic_prefix_length,
                pilot_pattern=None
            )
            self.num_pilot_symbols = 0
            
        self.num_data_symbols = self.rg.num_data_symbols
        self.num_total_symbols = self.num_ofdm_symbols * self.fft_size
        self.pilot_overhead = self.num_pilot_symbols / self.num_total_symbols
        
        self.n = self.num_data_symbols * self.num_bits_per_symbol
        self.k = int(self.n * self.coderate)
        
        # 5G NR LDPC Codec
        self.encoder = LDPC5GEncoder(k=self.k, n=self.n).to(self.device)
        self.decoder = LDPC5GDecoder(self.encoder, num_iter=num_ldpc_iter).to(self.device)
        
        # Modulation & Mapping
        self.constellation = Constellation("qam", num_bits_per_symbol=self.num_bits_per_symbol).to(self.device)
        self.mapper = Mapper(constellation=self.constellation).to(self.device)
        self.classical_demapper = Demapper("app", constellation=self.constellation).to(self.device)
        self.rg_mapper = ResourceGridMapper(self.rg).to(self.device)
        
        # Classical Equalizer & Estimator
        self.sm = StreamManagement(np.array([[1]]), 1)
        self.awgn = AWGN().to(self.device)
        
        if self.pilot_mode == "pilot_assisted":
            self.ls_estimator = LSChannelEstimator(self.rg, interpolation_type="nn")
            self.lmmse_equalizer = LMMSEEqualizer(self.rg, self.sm)
            self.data_mask = (self.rg.pilot_pattern.mask.squeeze(0).squeeze(0) == 0).to(self.device)
        else:
            self.ls_estimator = None
            self.lmmse_equalizer = None
            self.data_mask = None
            
        # Channel Instantiation
        if self.channel_type == "rayleigh":
            self.fading_model = RayleighBlockFading(num_rx=1, num_rx_ant=1, num_tx=1, num_tx_ant=1).to(self.device)
            self.channel = OFDMChannel(self.fading_model, self.rg, add_awgn=True).to(self.device)
        elif self.channel_type in ["tdl", "tdl_raytracing", "raytracing", "rician"]:
            self.tdl_model = TDL(
                model="A",
                delay_spread=self.delay_spread,
                carrier_frequency=self.carrier_frequency,
                min_speed=self.speed,
                max_speed=self.speed
            )
            self.channel = OFDMChannel(self.tdl_model, self.rg, add_awgn=True).to(self.device)
        else: # awgn
            self.channel = None

    def transmit(self, batch_size, ebno_db):
        """
        Generate information bits, encode with LDPC, modulate, and transmit over configured channel.
        
        Returns:
        --------
        u : torch.Tensor [B, k] (ground truth information bits)
        c : torch.Tensor [B, n] (ground truth LDPC coded bits)
        y_grid_2ch : torch.Tensor [B, 2, num_ofdm_symbols, fft_size] (received I/Q channel grid for ResCNN)
        no : float (noise variance)
        rx_grid_raw : torch.Tensor [B, 1, 1, num_ofdm_symbols, fft_size] (complex received grid)
        """
        u = torch.randint(0, 2, (batch_size, self.k), dtype=torch.float32, device=self.device)
        c = self.encoder(u)
        
        c_reshaped = c.view(batch_size, 1, 1, self.n)
        symbols = self.mapper(c_reshaped)
        tx_grid = self.rg_mapper(symbols)
        
        no = ebnodb2no(ebno_db, num_bits_per_symbol=self.num_bits_per_symbol, coderate=self.coderate)
        
        if self.channel is not None:
            rx_grid = self.channel(tx_grid, no)
        else: # pure AWGN
            rx_grid = self.awgn(tx_grid, no)
            
        # Squeeze to [B, num_ofdm_symbols, fft_size]
        y_complex = rx_grid.squeeze(1).squeeze(1)
        
        # Format as 2-channel tensor [B, 2, num_ofdm_symbols, fft_size]
        y_grid_2ch = torch.stack([y_complex.real, y_complex.imag], dim=1).float()
        
        return u, c, y_grid_2ch, no, rx_grid

    def decode_ldpc(self, llrs):
        """Decode soft LLRs via 5G NR LDPC Belief Propagation."""
        b = llrs.shape[0]
        llrs_2d = llrs.view(b, -1)[:, :self.n]
        u_hat = self.decoder(llrs_2d)
        c_hat = self.encoder(u_hat)
        return u_hat, c_hat

    def classical_demap_and_decode(self, rx_grid, no):
        """
        Perform classical Pilot-assisted LS/LMMSE Equalization and Max-Log APP Demapping followed by LDPC decode.
        """
        if self.pilot_mode == "pilot_assisted" and self.ls_estimator is not None:
            h_hat, err_var = self.ls_estimator(rx_grid, no)
            x_hat, no_eff = self.lmmse_equalizer(rx_grid, h_hat, err_var, no)
            llrs = self.classical_demapper(x_hat, no_eff)
        else:
            # Blind demapping without pilots (assumes perfect identity channel or raw demapping)
            llrs = self.classical_demapper(rx_grid, no)
            
        b = rx_grid.shape[0]
        llrs_flat = llrs.view(b, -1)[:, :self.n]
        u_hat, c_hat = self.decode_ldpc(llrs_flat)
        return llrs_flat, u_hat, c_hat
