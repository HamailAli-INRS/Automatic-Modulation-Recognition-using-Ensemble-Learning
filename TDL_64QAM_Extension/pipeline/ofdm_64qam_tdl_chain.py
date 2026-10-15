"""
5G NR OFDM + 64-QAM Transmission Chain over 3GPP TR 38.901 TDL Channels.
"""

import torch
import torch.nn as nn
import numpy as np

from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.phy.mapping import Mapper, Demapper, Constellation
from sionna.phy.ofdm import ResourceGrid, ResourceGridMapper, LSChannelEstimator, LMMSEEqualizer
from sionna.phy.mimo import StreamManagement
from sionna.phy.channel.tr38901 import TDL
from sionna.phy.channel import OFDMChannel
from sionna.phy.utils import ebnodb2no

class OFDM64QAMTDLPipeline:
    def __init__(self,
                 num_ofdm_symbols=14,
                 fft_size=64,
                 subcarrier_spacing=15e3,
                 cyclic_prefix_length=16,
                 pilot_symbol_indices=[2, 11],
                 num_bits_per_symbol=6,
                 coderate=0.5,
                 num_ldpc_iter=20,
                 tdl_model="A",
                 delay_spread=30e-9,
                 carrier_frequency=3.5e9,
                 speed=10.0,
                 device="cpu"):
        
        self.num_ofdm_symbols = num_ofdm_symbols
        self.fft_size = fft_size
        self.num_bits_per_symbol = num_bits_per_symbol
        self.coderate = coderate
        self.device = torch.device(device)
        
        # Pilot-Assisted Resource Grid (Kronecker pilots at symbols 2 and 11)
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
        self.num_pilot_symbols = self.rg.num_pilot_symbols # 128 pilots (14.3% overhead)
        self.n = self.num_data_symbols * self.num_bits_per_symbol # 4608 coded bits
        self.k = int(self.n * self.coderate) # 2304 info bits
        
        # 5G NR LDPC Codec
        self.encoder = LDPC5GEncoder(k=self.k, n=self.n).to(self.device)
        self.decoder = LDPC5GDecoder(self.encoder, num_iter=num_ldpc_iter).to(self.device)
        
        # 64-QAM Modulation
        self.constellation = Constellation("qam", num_bits_per_symbol=self.num_bits_per_symbol).to(self.device)
        self.mapper = Mapper(constellation=self.constellation).to(self.device)
        self.classical_demapper = Demapper("app", constellation=self.constellation).to(self.device)
        self.rg_mapper = ResourceGridMapper(self.rg).to(self.device)
        
        # Classical Estimator & Equalizer
        self.sm = StreamManagement(np.array([[1]]), 1)
        self.ls_estimator = LSChannelEstimator(self.rg, interpolation_type="nn")
        self.lmmse_equalizer = LMMSEEqualizer(self.rg, self.sm)
        
        # 3GPP TDL Channel
        self.tdl = TDL(
            model=tdl_model,
            delay_spread=delay_spread,
            carrier_frequency=carrier_frequency,
            min_speed=speed,
            max_speed=speed
        )
        self.channel = OFDMChannel(channel_model=self.tdl, resource_grid=self.rg, add_awgn=True).to(self.device)

    def transmit_and_receive(self, batch_size, ebno_db):
        """Transmit over 3GPP TDL, equalize, and compute initial classical LLRs."""
        u = torch.randint(0, 2, (batch_size, self.k), dtype=torch.float32, device=self.device)
        c = self.encoder(u)
        
        c_reshaped = c.view(batch_size, 1, 1, self.n)
        symbols = self.mapper(c_reshaped)
        tx_grid = self.rg_mapper(symbols)
        
        no = ebnodb2no(ebno_db, num_bits_per_symbol=self.num_bits_per_symbol, coderate=self.coderate)
        rx_grid = self.channel(tx_grid, no)
        
        h_hat, err_var = self.ls_estimator(rx_grid, no)
        x_hat, no_eff = self.lmmse_equalizer(rx_grid, h_hat, err_var, no)
        classical_llrs = self.classical_demapper(x_hat, no_eff)
        
        return u, c, x_hat, classical_llrs, no

    def decode_ldpc(self, llrs):
        u_hat = self.decoder(llrs)
        c_hat = self.encoder(u_hat)
        return u_hat, c_hat
