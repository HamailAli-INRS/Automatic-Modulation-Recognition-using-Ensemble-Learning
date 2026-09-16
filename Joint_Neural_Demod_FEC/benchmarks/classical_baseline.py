"""
Classical 5G NR Transmission Benchmark (Pilot-Assisted with LS Estimation & LDPC).
Serves as the conventional state-of-the-art reference baseline for the paper.
Features 14.3% pilot overhead, Least-Squares Channel Estimation, LMMSE Equalization,
and Belief Propagation LDPC Decoding.
"""

import torch
import numpy as np

from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.phy.mapping import Mapper, Demapper, Constellation
from sionna.phy.ofdm import ResourceGrid, ResourceGridMapper, LSChannelEstimator, LMMSEEqualizer
from sionna.phy.mimo import StreamManagement
from sionna.phy.channel import OFDMChannel, RayleighBlockFading, AWGN
from sionna.phy.utils import ebnodb2no

class ClassicalPilotBaseline:
    """
    Standard Pilot-Assisted 5G NR Baseline System.
    """
    def __init__(self,
                 num_ofdm_symbols=14,
                 fft_size=64,
                 subcarrier_spacing=15e3,
                 cyclic_prefix_length=16,
                 pilot_symbol_indices=[2, 11],
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
        
        # 1. Pilot-Assisted Resource Grid (Kronecker pilots at symbols 2 and 11)
        self.rg = ResourceGrid(
            num_ofdm_symbols=num_ofdm_symbols,
            fft_size=fft_size,
            subcarrier_spacing=subcarrier_spacing,
            num_tx=1,
            num_streams_per_tx=1,
            cyclic_prefix_length=cyclic_prefix_length,
            pilot_pattern='kronecker',
            pilot_ofdm_symbol_indices=pilot_symbol_indices
        )
        
        self.num_data_symbols = self.rg.num_data_symbols # 768 data symbols
        self.num_pilot_symbols = self.rg.num_pilot_symbols # 128 pilot symbols (14.3% overhead)
        self.pilot_overhead = self.num_pilot_symbols / (self.num_data_symbols + self.num_pilot_symbols)
        
        self.n = self.num_data_symbols * self.num_bits_per_symbol # Coded bits: 3072
        self.k = int(self.n * self.coderate) # Info bits: 1536
        
        # 2. 5G NR LDPC
        self.encoder = LDPC5GEncoder(k=self.k, n=self.n).to(self.device)
        self.decoder = LDPC5GDecoder(self.encoder, num_iter=num_ldpc_iter).to(self.device)
        
        # 3. Modulation & Demapping
        self.constellation = Constellation("qam", num_bits_per_symbol=self.num_bits_per_symbol).to(self.device)
        self.mapper = Mapper(constellation=self.constellation).to(self.device)
        self.demapper = Demapper("app", constellation=self.constellation).to(self.device)
        self.rg_mapper = ResourceGridMapper(self.rg).to(self.device)
        
        # 4. Estimation & Equalization
        self.sm = StreamManagement(np.array([[1]]), 1)
        self.ls_estimator = LSChannelEstimator(self.rg, interpolation_type="nn")
        self.lmmse_equalizer = LMMSEEqualizer(self.rg, self.sm)
        
        # 5. Channel
        self.awgn = AWGN().to(self.device)
        if self.channel_type == "rayleigh":
            self.fading_model = RayleighBlockFading(num_rx=1, num_rx_ant=1, num_tx=1, num_tx_ant=1).to(self.device)
            self.ofdm_channel = OFDMChannel(self.fading_model, self.rg).to(self.device)
        else:
            self.ofdm_channel = None

    @torch.no_grad()
    def evaluate(self, batch_size=128, ebno_db=10.0):
        """
        Run full transmission, channel estimation, demapping, and LDPC decoding.
        
        Returns:
        --------
        raw_ber : float
        post_fec_ber : float
        bler : float
        """
        u = torch.randint(0, 2, (batch_size, self.k), dtype=torch.float32, device=self.device)
        c = self.encoder(u)
        
        c_reshaped = c.view(batch_size, 1, 1, self.n)
        symbols = self.mapper(c_reshaped)
        tx_grid = self.rg_mapper(symbols)
        
        no = ebnodb2no(ebno_db, num_bits_per_symbol=self.num_bits_per_symbol, coderate=self.coderate)
        
        if self.channel_type == "rayleigh" and self.ofdm_channel is not None:
            faded_grid = self.ofdm_channel(tx_grid)
        else:
            faded_grid = tx_grid
            
        rx_grid = self.awgn(faded_grid, no)
        
        # Channel Estimation & Equalization
        h_hat, err_var = self.ls_estimator(rx_grid, no)
        x_hat, no_eff = self.lmmse_equalizer(rx_grid, h_hat, err_var, no)
        
        # Soft Demapping to LLRs
        llrs = self.demapper(x_hat, no_eff) # [B, 1, 1, n]
        llrs_flat = llrs.view(batch_size, -1)
        
        # Raw BER
        pred_c_bits = (llrs_flat > 0).float()
        raw_ber = (pred_c_bits != c).float().mean().item()
        
        # Post-FEC LDPC Decoding
        u_hat = self.decoder(llrs_flat)
        post_fec_ber = (u_hat != u).float().mean().item()
        
        # Frame / Block Error Rate
        frame_errors = (u_hat != u).float().sum(dim=-1) > 0
        bler = frame_errors.float().mean().item()
        
        return raw_ber, post_fec_ber, bler
