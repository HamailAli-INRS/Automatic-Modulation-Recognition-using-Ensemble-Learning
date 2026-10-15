import torch
import torch.nn as nn
import torch.optim as optim
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
NUM_BITS_PER_SYMBOL = 4  # 16-QAM
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

constellation = Constellation('qam', num_bits_per_symbol=NUM_BITS_PER_SYMBOL)
mapper        = Mapper(constellation=constellation)
rg_mapper     = ResourceGridMapper(rg)
ofdm_mod      = OFDMModulator(cyclic_prefix_length=CYCLIC_PREFIX_LEN)
ofdm_demod    = OFDMDemodulator(FFT_SIZE, 0, CYCLIC_PREFIX_LEN)
awgn_channel  = AWGN()

NUM_DATA_SYMBOLS = rg.num_data_symbols
NUM_DATA_BITS    = NUM_DATA_SYMBOLS * NUM_BITS_PER_SYMBOL

def generate_batch(batch_size, ebno_db):
    bits    = torch.randint(0, 2, (batch_size, 1, 1, NUM_DATA_BITS), dtype=torch.float32)
    symbols = mapper(bits)
    x_rg    = rg_mapper(symbols)
    x_time  = ofdm_mod(x_rg)
    no      = ebnodb2no(ebno_db, num_bits_per_symbol=NUM_BITS_PER_SYMBOL, coderate=1.0)
    y_time  = awgn_channel(x_time, no)
    y_freq  = ofdm_demod(y_time)
    
    # Extract data subcarrier positions from received grid
    data_ind = rg_mapper._data_ind
    y_data = y_freq[:, data_ind[:,0], data_ind[:,1], data_ind[:,2], data_ind[:,3]]
    return bits, y_data

# Subcarrier-Wise Neural Demodulator
class SubcarrierDemodulator(nn.Module):
    def __init__(self, num_bits_per_symbol):
        super().__init__()
        # Input: Real & Imag part of a complex subcarrier symbol (2 values)
        # Output: Probabilities for bits per symbol (4 for 16-QAM)
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_bits_per_symbol)
        )

    def forward(self, y_subcarriers):
        # y_subcarriers: [N_total, 2] -> [Real, Imag]
        return self.net(y_subcarriers)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('[OK] Using device:', device, flush=True)

model = SubcarrierDemodulator(NUM_BITS_PER_SYMBOL).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

print('[OK] Starting Subcarrier Neural Demodulator Training ...', flush=True)
EPOCHS = 1000

for epoch in range(1, EPOCHS + 1):
    model.train()
    random_ebno = float(np.random.uniform(2.0, 18.0))
    bits, y_data = generate_batch(BATCH_SIZE, random_ebno)
    
    # Reshape bits: [BATCH_SIZE, NUM_DATA_SYMBOLS, NUM_BITS_PER_SYMBOL] -> [-1, 4]
    target_bits = bits.reshape(-1, NUM_BITS_PER_SYMBOL).to(device)
    
    # Reshape subcarriers: [BATCH_SIZE, NUM_DATA_SYMBOLS] -> [-1, 2] (Real, Imag)
    y_flat = y_data.reshape(-1)
    input_samples = torch.stack([y_flat.real, y_flat.imag], dim=-1).to(device)
    
    optimizer.zero_grad()
    logits = model(input_samples)
    loss = criterion(logits, target_bits)
    loss.backward()
    optimizer.step()
    
    if epoch % 200 == 0 or epoch == 1:
        print('Epoch', epoch, '/', EPOCHS, '| Loss:', round(loss.item(), 5), '| SNR:', round(random_ebno, 1), 'dB', flush=True)

print('[OK] Training complete!', flush=True)
model.eval()
print('[OK] Evaluating Subcarrier AI Demodulator vs Classical ...', flush=True)

ebno_range = list(range(-5, 21, 2))
ai_bers = []
classical_bers = []

with torch.no_grad():
    for ebno_db in ebno_range:
        bits, y_data = generate_batch(256, float(ebno_db))
        
        # Ground truth bits
        true_bits = bits.reshape(-1, NUM_BITS_PER_SYMBOL)
        
        # Subcarrier inputs
        y_flat = y_data.reshape(-1)
        input_samples = torch.stack([y_flat.real, y_flat.imag], dim=-1).to(device)
        
        # AI Predictions
        logits = model(input_samples)
        pred_bits = (torch.sigmoid(logits) > 0.5).float().cpu()
        
        ai_ber = (pred_bits != true_bits).float().mean().item()
        ai_bers.append(ai_ber)
        print('   Eb/N0 =', ebno_db, 'dB | AI Demod BER =', round(ai_ber, 5), flush=True)

plt.figure(figsize=(8, 5))
plt.semilogy(ebno_range, [max(b, 1e-5) for b in ai_bers], 'r-s', linewidth=2, label='Subcarrier AI Demodulator')
plt.xlabel('Eb/N0 (dB)')
plt.ylabel('Bit Error Rate (BER)')
plt.title('Subcarrier AI Demodulator Performance (Sionna 2.0)')
plt.legend()
plt.grid(True, which='both', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('ai_demod_ber.png', dpi=150)
print('[OK] Updated BER plot saved -> ai_demod_ber.png', flush=True)
print('Done!', flush=True)
