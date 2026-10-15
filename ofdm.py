import torch
import sionna
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sionna.phy.channel import AWGN
from sionna.phy.ofdm import ResourceGrid, ResourceGridMapper, OFDMModulator, OFDMDemodulator
from sionna.phy.mapping import Mapper, Demapper, Constellation
from sionna.phy.utils import ebnodb2no

NUM_OFDM_SYMBOLS    = 14
FFT_SIZE            = 76
CYCLIC_PREFIX_LEN   = 6
BATCH_SIZE          = 64
NUM_BITS_PER_SYMBOL = 4
PILOT_SYMBOL_IDX    = [2, 11]

rg = ResourceGrid(
    num_ofdm_symbols=NUM_OFDM_SYMBOLS,
    fft_size=FFT_SIZE,
    subcarrier_spacing=15e3,
    num_tx=1,
    num_streams_per_tx=1,
    cyclic_prefix_length=CYCLIC_PREFIX_LEN,
    pilot_pattern='kronecker',
    pilot_ofdm_symbol_indices=PILOT_SYMBOL_IDX
)

print('[OK] Resource Grid created', flush=True)
print('     OFDM symbols  :', rg.num_ofdm_symbols, flush=True)
print('     Subcarriers   :', rg.fft_size, flush=True)
print('     Data symbols  :', rg.num_data_symbols, flush=True)
print('     Pilot symbols :', rg.num_pilot_symbols, flush=True)

constellation = Constellation('qam', num_bits_per_symbol=NUM_BITS_PER_SYMBOL)
mapper        = Mapper(constellation=constellation)
demapper      = Demapper('app', constellation=constellation)
rg_mapper     = ResourceGridMapper(rg)
ofdm_mod      = OFDMModulator(cyclic_prefix_length=CYCLIC_PREFIX_LEN)
ofdm_demod    = OFDMDemodulator(FFT_SIZE, 0, CYCLIC_PREFIX_LEN)
awgn_channel  = AWGN()

def run_ofdm_awgn(ebno_db):
    num_data_bits = rg.num_data_symbols * NUM_BITS_PER_SYMBOL
    bits    = torch.randint(0, 2, (BATCH_SIZE, 1, 1, num_data_bits), dtype=torch.float32)
    symbols = mapper(bits)
    x_rg    = rg_mapper(symbols)
    x_time  = ofdm_mod(x_rg)
    no      = ebnodb2no(ebno_db, num_bits_per_symbol=NUM_BITS_PER_SYMBOL, coderate=1.0)
    y_time  = awgn_channel(x_time, no)
    y_freq  = ofdm_demod(y_time)
    return bits, x_rg, y_freq

print('[OK] Running simulation at Eb/N0 = 10 dB ...', flush=True)
bits, x_rg, y_freq = run_ofdm_awgn(10.0)
print('[OK] Simulation successful!', flush=True)
print('     bits   shape:', list(bits.shape), flush=True)
print('     x_rg   shape:', list(x_rg.shape), flush=True)
print('     y_freq shape:', list(y_freq.shape), flush=True)

print('[OK] Sweeping Eb/N0 for BER ...', flush=True)
ebno_range = list(range(-5, 21, 2))
bers = []

for ebno_db in ebno_range:
    bits, x_rg, y_freq = run_ofdm_awgn(float(ebno_db))
    x_real = x_rg[:, 0, 0, :, :].real.reshape(BATCH_SIZE, -1)
    y_real = y_freq[:, 0, 0, :, :].real.reshape(BATCH_SIZE, -1)
    ber = (torch.sign(x_real) != torch.sign(y_real)).float().mean().item()
    bers.append(ber)
    print(f'   Eb/N0 = {ebno_db:4d} dB  |  BER = {ber:.5f}', flush=True)

plt.figure(figsize=(8, 5))
plt.semilogy(ebno_range, [max(b, 1e-6) for b in bers], 'b-o', linewidth=2, label='OFDM 16-QAM AWGN')
plt.xlabel('Eb/N0 (dB)')
plt.ylabel('Bit Error Rate (BER)')
plt.title('OFDM System - BER vs SNR  (Sionna 2.0)')
plt.legend()
plt.grid(True, which='both', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('ber_curve.png', dpi=150)

print('[OK] BER curve saved -> ber_curve.png', flush=True)
print('Done!', flush=True)
